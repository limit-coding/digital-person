"""Validate historical replay episodes and enforce temporal isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


CHAT_LOCATOR_RE = re.compile(r"message\.localId=(\d+)(?:\.\.(\d+))?")


class EpisodeValidationError(ValueError):
    """Raised when an episode would produce an invalid or leaky evaluation."""


def parse_timestamp(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise EpisodeValidationError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise EpisodeValidationError(f"{field} must include a timezone")
    return parsed


def validate_episode(episode: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "episode_id",
        "domain",
        "split",
        "leakage_group",
        "decision",
        "persona_state",
        "evidence",
        "labels",
    }
    missing = sorted(required - episode.keys())
    if missing:
        return [f"missing required field: {field}" for field in missing]

    try:
        cutoff = parse_timestamp(episode["decision"]["cutoff_at"], "decision.cutoff_at")
    except (KeyError, TypeError, EpisodeValidationError) as exc:
        return [str(exc)]

    option_ids = [item.get("option_id") for item in episode["decision"].get("options", [])]
    if len(option_ids) < 2:
        errors.append("decision.options must contain at least two options")
    if len(option_ids) != len(set(option_ids)):
        errors.append("decision option IDs must be unique")

    evidence_ids: list[str] = []
    for index, evidence in enumerate(episode.get("evidence", [])):
        evidence_id = evidence.get("evidence_id")
        evidence_ids.append(evidence_id)
        try:
            observed_at = parse_timestamp(
                evidence.get("observed_at"), f"evidence[{index}].observed_at"
            )
        except EpisodeValidationError as exc:
            errors.append(str(exc))
            continue
        if evidence.get("evidence_role") == "input" and observed_at > cutoff:
            errors.append(
                f"evidence[{index}] is future information but marked as model input"
            )
        if evidence.get("evidence_role") == "input" and evidence.get("author_role") == "ai":
            errors.append(
                f"evidence[{index}] is AI-authored input; use only if it truly existed before the decision"
            )
    if len(evidence_ids) != len(set(evidence_ids)):
        errors.append("evidence IDs must be unique")

    labels = episode.get("labels", {})
    deliberation = labels.get("contemporaneous_deliberation", {})
    referenced_options = set(deliberation.get("considered_option_ids", []))
    referenced_options.update(deliberation.get("option_probabilities", {}).keys())
    for label_name in ("actual_judgement", "actual_action"):
        option_id = labels.get(label_name, {}).get("option_id")
        if option_id is not None:
            referenced_options.add(option_id)
        recorded_at = labels.get(label_name, {}).get("recorded_at")
        if recorded_at:
            try:
                if parse_timestamp(recorded_at, f"labels.{label_name}.recorded_at") < cutoff:
                    errors.append(f"labels.{label_name}.recorded_at is before the cutoff")
            except EpisodeValidationError as exc:
                errors.append(str(exc))
    unknown_options = sorted(referenced_options - set(option_ids))
    if unknown_options:
        errors.append(f"labels reference unknown option IDs: {unknown_options}")

    probabilities = deliberation.get("option_probabilities", {})
    if probabilities:
        total = sum(probabilities.values())
        if abs(total - 1.0) > 1e-6:
            errors.append(f"option probabilities must sum to 1.0, got {total:.6f}")
        if any(value < 0 or value > 1 for value in probabilities.values()):
            errors.append("option probabilities must be between 0 and 1")
    return errors


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_chat_role(messages: list[dict[str, Any]]) -> str:
    roles = {
        "self" if message.get("isSend") == 1 else "other"
        for message in messages
        if message.get("isSend") in {0, 1}
    }
    if len(roles) > 1:
        return "mixed"
    return next(iter(roles), "unknown")


def verify_evidence_sources(
    episode: dict[str, Any], workspace: Path
) -> list[str]:
    """Cross-check source hashes and aggregate chat windows against raw exports."""
    errors: list[str] = []
    cache: dict[Path, Any] = {}
    cutoff = parse_timestamp(episode["decision"]["cutoff_at"], "decision.cutoff_at")
    for index, evidence in enumerate(episode.get("evidence", [])):
        source_path = workspace / evidence["source_relative_path"]
        if not source_path.exists():
            errors.append(f"evidence[{index}] source file does not exist")
            continue
        if source_sha256(source_path) != evidence["source_sha256"]:
            errors.append(f"evidence[{index}] source SHA-256 does not match")
            continue
        locator = evidence.get("record_locator") or ""
        match = CHAT_LOCATOR_RE.fullmatch(locator)
        if not match:
            continue
        if source_path not in cache:
            cache[source_path] = json.loads(source_path.read_text(encoding="utf-8"))
        start_id = int(match.group(1))
        end_id = int(match.group(2) or start_id)
        selected = [
            message
            for message in cache[source_path].get("messages", [])
            if isinstance(message, dict)
            and isinstance(message.get("localId"), int)
            and start_id <= message["localId"] <= end_id
        ]
        expected_count = end_id - start_id + 1
        if len(selected) != expected_count:
            errors.append(
                f"evidence[{index}] locator resolves to {len(selected)} messages, expected {expected_count}"
            )
            continue
        try:
            timestamps = [
                parse_timestamp(
                    message["formattedTime"] + "+08:00", "chat.formattedTime"
                )
                for message in selected
                if isinstance(message.get("formattedTime"), str)
            ]
        except EpisodeValidationError as exc:
            errors.append(f"evidence[{index}] {exc}")
            continue
        if len(timestamps) != len(selected):
            errors.append(f"evidence[{index}] has chat messages without timestamps")
            continue
        observed_at = parse_timestamp(
            evidence["observed_at"], f"evidence[{index}].observed_at"
        )
        latest = max(timestamps)
        if observed_at != latest:
            errors.append(
                f"evidence[{index}] observed_at is not the latest time in its chat window"
            )
        expected_role = expected_chat_role(selected)
        if evidence["author_role"] != expected_role:
            errors.append(
                f"evidence[{index}] author_role should be {expected_role}, got {evidence['author_role']}"
            )
        if evidence["evidence_role"] == "input" and latest > cutoff:
            errors.append(f"evidence[{index}] chat window contains future input")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    episode = json.loads(args.episode.read_text(encoding="utf-8"))
    errors = validate_episode(episode)
    if args.workspace:
        errors.extend(verify_evidence_sources(episode, args.workspace.resolve()))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"OK: {episode['episode_id']}")


if __name__ == "__main__":
    main()
