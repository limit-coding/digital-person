"""Create a content-free audit manifest for personal digital-mirror sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Policy:
    source_roots: tuple[str, ...]
    allowed_extensions: frozenset[str]
    excluded_path_fragments: tuple[str, ...]
    credential_filename_fragments: tuple[str, ...]
    ai_attribution_fragments: tuple[str, ...]
    authored_candidate_groups: frozenset[str]
    sensitivity_patterns: dict[str, re.Pattern[str]]

    @classmethod
    def load(cls, path: Path) -> "Policy":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            source_roots=tuple(raw["source_roots"]),
            allowed_extensions=frozenset(ext.lower() for ext in raw["allowed_extensions"]),
            excluded_path_fragments=tuple(raw["excluded_path_fragments"]),
            credential_filename_fragments=tuple(raw["credential_filename_fragments"]),
            ai_attribution_fragments=tuple(raw["ai_attribution_fragments"]),
            authored_candidate_groups=frozenset(raw["authored_candidate_groups"]),
            sensitivity_patterns={
                name: re.compile(pattern)
                for name, pattern in raw["sensitivity_patterns"].items()
            },
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_group(relative_path: Path) -> str:
    parts = relative_path.parts
    joined = "/".join(parts)
    if parts and parts[0] == "texts":
        return "chats"
    mappings = (
        ("iCloud/传记/", "biography"),
        ("iCloud/机器模式训练/", "machine_rules"),
        ("iCloud/社交图谱分析/", "social_graph"),
        ("iCloud/学习篇/", "learning_essays"),
        ("iCloud/30天自我/", "daily_reflection"),
        ("iCloud/Notes/", "notes"),
        ("iCloud/agent训练/", "agent_notes"),
    )
    for prefix, group in mappings:
        if joined.startswith(prefix):
            return group
    return "other"


def estimate_tokens(text: str) -> int:
    cjk_chars = len(re.findall(r"[\u3400-\u9fff]", text))
    non_cjk_words = len(re.findall(r"[A-Za-z0-9_]+", text))
    other_chars = max(0, len(text) - cjk_chars)
    return cjk_chars + non_cjk_words + other_chars // 4


def sensitivity_counts(text: str, policy: Policy) -> dict[str, int]:
    return {
        name: len(pattern.findall(text))
        for name, pattern in policy.sensitivity_patterns.items()
        if pattern.search(text)
    }


def classify_policy(
    relative_path: Path,
    group: str,
    flags: dict[str, int],
    policy: Policy,
) -> tuple[str, list[str]]:
    path_text = relative_path.as_posix()
    filename = relative_path.name
    reasons: list[str] = []

    if any(fragment in path_text for fragment in policy.excluded_path_fragments):
        return "exclude", ["excluded_path"]
    if any(fragment.casefold() in filename.casefold() for fragment in policy.credential_filename_fragments):
        return "exclude", ["credential_or_account_filename"]

    ai_attributed = any(
        fragment.casefold() in filename.casefold()
        for fragment in policy.ai_attribution_fragments
    )
    if ai_attributed:
        reasons.append("ai_attribution_in_filename")
    if flags:
        reasons.append("potential_sensitive_content")
    if group == "chats":
        reasons.append("contains_third_party_messages")
    if group == "social_graph":
        reasons.append("contains_third_party_profile")

    if flags:
        return "review_sensitive", reasons
    if ai_attributed:
        return "review_provenance", reasons
    if group in {"chats", "social_graph"}:
        return "include_with_redaction", reasons
    if group in policy.authored_candidate_groups:
        return "include_candidate", reasons
    return "review_scope", reasons or ["authorship_or_relevance_unconfirmed"]


def chat_metadata(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or not isinstance(raw.get("messages"), list):
        return {"valid_chat_export": False}
    messages = raw["messages"]
    type_counts: Counter[str] = Counter()
    self_sent = 0
    received = 0
    timestamps: list[int] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        type_counts[str(message.get("type", "unknown"))] += 1
        if message.get("isSend") == 1:
            self_sent += 1
        elif message.get("isSend") == 0:
            received += 1
        timestamp = message.get("createTime")
        if isinstance(timestamp, int):
            timestamps.append(timestamp)
    return {
        "valid_chat_export": True,
        "message_count": len(messages),
        "self_sent_count": self_sent,
        "received_count": received,
        "message_type_counts": dict(sorted(type_counts.items())),
        "first_timestamp": min(timestamps) if timestamps else None,
        "last_timestamp": max(timestamps) if timestamps else None,
    }


def audit_file(path: Path, workspace: Path, policy: Policy) -> dict[str, Any]:
    relative = path.relative_to(workspace)
    group = source_group(relative)
    text = path.read_text(encoding="utf-8", errors="replace")
    flags = sensitivity_counts(text, policy)
    status, reasons = classify_policy(relative, group, flags, policy)
    record: dict[str, Any] = {
        "relative_path": relative.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "modified_at": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "extension": path.suffix.lower(),
        "source_group": group,
        "policy_status": status,
        "policy_reasons": reasons,
        "sensitivity_flag_counts": flags,
        "text_characters": len(text),
        "estimated_tokens": estimate_tokens(text),
    }
    if group == "chats" and path.suffix.lower() == ".json":
        try:
            record["chat_metadata"] = chat_metadata(json.loads(text))
        except json.JSONDecodeError:
            record["chat_metadata"] = {"valid_chat_export": False}
            record["policy_status"] = "review_parse_error"
            record["policy_reasons"] = reasons + ["invalid_json"]
    return record


def iter_source_files(workspace: Path, policy: Policy) -> Iterable[Path]:
    for root_name in policy.source_roots:
        root = workspace / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in policy.allowed_extensions:
                yield path


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_group = Counter(record["source_group"] for record in records)
    by_status = Counter(record["policy_status"] for record in records)
    chat_messages = sum(
        record.get("chat_metadata", {}).get("message_count", 0)
        for record in records
    )
    chat_self_sent = sum(
        record.get("chat_metadata", {}).get("self_sent_count", 0)
        for record in records
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "estimated_tokens": sum(record["estimated_tokens"] for record in records),
        "files_by_source_group": dict(sorted(by_group.items())),
        "files_by_policy_status": dict(sorted(by_status.items())),
        "chat_message_count": chat_messages,
        "chat_self_sent_count": chat_self_sent,
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Data audit report",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        f"- Files: {summary['file_count']}",
        f"- Bytes: {summary['total_bytes']}",
        f"- Estimated tokens: {summary['estimated_tokens']}",
        f"- Chat messages: {summary['chat_message_count']}",
        f"- Self-sent chat messages: {summary['chat_self_sent_count']}",
        "",
        "## Files by source group",
        "",
    ]
    lines.extend(
        f"- `{key}`: {value}"
        for key, value in summary["files_by_source_group"].items()
    )
    lines.extend(["", "## Files by policy status", ""])
    lines.extend(
        f"- `{key}`: {value}"
        for key, value in summary["files_by_policy_status"].items()
    )
    lines.extend(
        [
            "",
            "The report and manifest contain metadata and counts only; no source content or secret values are copied.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(workspace: Path, policy_path: Path, output: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    policy = Policy.load(policy_path.resolve())
    records = [audit_file(path, workspace, policy) for path in iter_source_files(workspace, policy)]
    summary = build_summary(records)
    output.mkdir(parents=True, exist_ok=True)
    manifest_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    (output / "manifest.jsonl").write_text(manifest_text, encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_audit(args.workspace, args.policy, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
