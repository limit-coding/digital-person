"""Compile leak-free historical replay cases and score model predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .episode import EpisodeValidationError, parse_timestamp, validate_episode


def compile_replay_case(episode: dict[str, Any]) -> dict[str, Any]:
    errors = validate_episode(episode)
    if errors:
        raise EpisodeValidationError("; ".join(errors))

    cutoff = parse_timestamp(episode["decision"]["cutoff_at"], "decision.cutoff_at")
    visible_evidence: list[dict[str, Any]] = []
    for evidence in episode["evidence"]:
        if evidence["evidence_role"] != "input":
            continue
        observed_at = parse_timestamp(evidence["observed_at"], "evidence.observed_at")
        if observed_at > cutoff:
            raise EpisodeValidationError(
                f"future evidence cannot be compiled: {evidence['evidence_id']}"
            )
        visible_evidence.append(
            {
                "evidence_id": evidence["evidence_id"],
                "observed_at": evidence["observed_at"],
                "author_role": evidence["author_role"],
                "summary": evidence["summary"],
            }
        )

    return {
        "schema_version": 1,
        "episode_id": episode["episode_id"],
        "task_mode": episode["decision"]["task_mode"],
        "cutoff_at": episode["decision"]["cutoff_at"],
        "domain": episode["domain"],
        "question": episode["decision"]["question"],
        "options": episode["decision"]["options"],
        "persona_state": episode["persona_state"],
        "evidence": visible_evidence,
        "instructions": [
            "只使用截止时间前给出的证据，不假设知道后来结果。",
            "分别预测当时的判断与实际行动；二者允许不同。",
            "为所有候选选项给出总和为 1 的概率。",
            "还原竞争选项、冲突轴、未知信息与纠结强度。",
            "每个结论引用实际使用的 evidence_id；没有证据时降低置信度。",
        ],
        "required_output_fields": [
            "episode_id",
            "predicted_judgement",
            "predicted_action",
            "option_probabilities",
            "considered_option_ids",
            "tensions",
            "unknowns",
            "deliberation_intensity",
            "confidence",
            "evidence_ids",
        ],
    }


def validate_prediction(case: dict[str, Any], prediction: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = set(case["required_output_fields"])
    missing = sorted(required - prediction.keys())
    if missing:
        return [f"missing prediction field: {field}" for field in missing]
    if prediction["episode_id"] != case["episode_id"]:
        errors.append("prediction episode_id does not match the replay case")

    option_ids = {option["option_id"] for option in case["options"]}
    for field in ("predicted_judgement", "predicted_action"):
        if prediction[field] not in option_ids:
            errors.append(f"{field} references an unknown option")
    probabilities = prediction["option_probabilities"]
    if set(probabilities) != option_ids:
        errors.append("option_probabilities must contain every and only declared option")
    elif abs(sum(probabilities.values()) - 1.0) > 1e-6:
        errors.append("option_probabilities must sum to 1.0")
    if any(value < 0 or value > 1 for value in probabilities.values()):
        errors.append("option probabilities must be between 0 and 1")

    for field in ("deliberation_intensity", "confidence"):
        value = prediction[field]
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            errors.append(f"{field} must be between 0 and 1")
    visible_ids = {evidence["evidence_id"] for evidence in case["evidence"]}
    visible_ids.update(
        history["history_id"] for history in case.get("personal_history", [])
    )
    unknown_evidence = set(prediction["evidence_ids"]) - visible_ids
    if unknown_evidence:
        errors.append(f"prediction cites unavailable evidence: {sorted(unknown_evidence)}")
    return errors


def set_metrics(predicted: list[str], actual: list[str]) -> dict[str, float]:
    predicted_set = set(predicted)
    actual_set = set(actual)
    overlap = len(predicted_set & actual_set)
    precision = overlap / len(predicted_set) if predicted_set else 0.0
    recall = overlap / len(actual_set) if actual_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def score_prediction(
    episode: dict[str, Any],
    prediction: dict[str, Any],
    replay_case: dict[str, Any] | None = None,
) -> dict[str, Any]:
    canonical_case = compile_replay_case(episode)
    case = replay_case or canonical_case
    for field in (
        "episode_id",
        "cutoff_at",
        "question",
        "options",
        "evidence",
        "required_output_fields",
    ):
        if case.get(field) != canonical_case.get(field):
            raise EpisodeValidationError(
                f"scoring replay case does not match episode field: {field}"
            )
    cutoff = parse_timestamp(case["cutoff_at"], "cutoff_at")
    for history in case.get("personal_history", []):
        observed_at = parse_timestamp(
            history.get("occurred_at"), "personal_history.occurred_at"
        )
        if observed_at >= cutoff:
            raise EpisodeValidationError(
                f"personal history is not strictly pre-cutoff: {history.get('history_id')}"
            )
    errors = validate_prediction(case, prediction)
    if errors:
        raise EpisodeValidationError("; ".join(errors))

    labels = episode["labels"]
    actual_judgement = labels["actual_judgement"]["option_id"]
    actual_action = labels["actual_action"]["option_id"]
    option_ids = [option["option_id"] for option in case["options"]]
    brier = None
    if actual_judgement is not None:
        brier = sum(
            (
                prediction["option_probabilities"][option_id]
                - (1.0 if option_id == actual_judgement else 0.0)
            )
            ** 2
            for option_id in option_ids
        ) / len(option_ids)
    label_intensity = labels["contemporaneous_deliberation"].get(
        "deliberation_intensity"
    )
    return {
        "episode_id": episode["episode_id"],
        "annotation_status": episode.get("annotation", {}).get("status", "unknown"),
        "judgement_correct": (
            prediction["predicted_judgement"] == actual_judgement
            if actual_judgement is not None
            else None
        ),
        "action_correct": (
            prediction["predicted_action"] == actual_action
            if actual_action is not None
            else None
        ),
        "judgement_brier_score": brier,
        "considered_options": set_metrics(
            prediction["considered_option_ids"],
            labels["contemporaneous_deliberation"]["considered_option_ids"],
        ),
        "deliberation_intensity_absolute_error": (
            abs(prediction["deliberation_intensity"] - label_intensity)
            if label_intensity is not None
            else None
        ),
        "manual_review_required": [
            "tensions",
            "unknowns",
            "evidence_support",
        ],
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("episode", type=Path)
    compile_parser.add_argument("--output", type=Path, required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("episode", type=Path)
    score_parser.add_argument("prediction", type=Path)
    score_parser.add_argument(
        "--case",
        type=Path,
        help="Exact leak-free replay case shown to the model (required for history citations).",
    )
    score_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "compile":
        case = compile_replay_case(load_json(args.episode))
        write_json(args.output, case)
        print(f"WROTE: {args.output}")
        return
    result = score_prediction(
        load_json(args.episode),
        load_json(args.prediction),
        load_json(args.case) if args.case else None,
    )
    if args.output:
        write_json(args.output, result)
        print(f"WROTE: {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
