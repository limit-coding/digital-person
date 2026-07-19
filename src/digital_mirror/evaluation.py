"""Aggregate historical replay results and compare simple action calibration policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .replay import load_json, write_json


def policy_correct(
    prediction: dict[str, Any], actual_action: str, policy: str
) -> bool:
    predicted = (
        prediction["predicted_action"]
        if policy == "raw_action"
        else prediction["predicted_judgement"]
    )
    return predicted == actual_action


def aggregate(
    episodes: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    brier_values: list[float] = []
    intensity_errors: list[float] = []
    judgement_correct = 0
    judgement_count = 0
    for episode in episodes:
        episode_id = episode["episode_id"]
        prediction = predictions[episode_id]
        labels = episode["labels"]
        actual_judgement = labels["actual_judgement"]["option_id"]
        actual_action = labels["actual_action"]["option_id"]
        if actual_judgement is not None:
            judgement_count += 1
            judgement_correct += prediction["predicted_judgement"] == actual_judgement
            option_ids = [option["option_id"] for option in episode["decision"]["options"]]
            brier_values.append(
                sum(
                    (
                        prediction["option_probabilities"][option_id]
                        - (1.0 if option_id == actual_judgement else 0.0)
                    )
                    ** 2
                    for option_id in option_ids
                )
                / len(option_ids)
            )
        intensity = labels["contemporaneous_deliberation"].get("deliberation_intensity")
        if intensity is not None:
            intensity_errors.append(abs(prediction["deliberation_intensity"] - intensity))
        rows.append(
            {
                "episode_id": episode_id,
                "annotation_status": episode.get("annotation", {}).get("status", "unknown"),
                "actual_judgement": actual_judgement,
                "actual_action": actual_action,
                "predicted_judgement": prediction["predicted_judgement"],
                "predicted_action": prediction["predicted_action"],
            }
        )

    action_rows = [row for row in rows if row["actual_action"] is not None]
    policies = ("raw_action", "judgement_as_action")
    policy_results = {
        policy: {
            "correct": sum(
                policy_correct(predictions[row["episode_id"]], row["actual_action"], policy)
                for row in action_rows
            ),
            "count": len(action_rows),
        }
        for policy in policies
    }

    loo_correct = 0
    loo_choices: list[dict[str, Any]] = []
    for held_out in action_rows:
        training = [row for row in action_rows if row is not held_out]
        training_scores = {
            policy: sum(
                policy_correct(predictions[row["episode_id"]], row["actual_action"], policy)
                for row in training
            )
            for policy in policies
        }
        selected = max(policies, key=lambda policy: (training_scores[policy], policy == "raw_action"))
        correct = policy_correct(
            predictions[held_out["episode_id"]], held_out["actual_action"], selected
        )
        loo_correct += correct
        loo_choices.append(
            {
                "episode_id": held_out["episode_id"],
                "selected_policy": selected,
                "correct": correct,
                "training_scores": training_scores,
            }
        )

    return {
        "schema_version": 1,
        "annotation_warning": "Metrics include machine_draft labels unless filtered externally.",
        "episode_count": len(rows),
        "judgement": {
            "correct": judgement_correct,
            "count": judgement_count,
            "mean_brier_score": sum(brier_values) / len(brier_values) if brier_values else None,
        },
        "deliberation": {
            "mean_intensity_absolute_error": (
                sum(intensity_errors) / len(intensity_errors) if intensity_errors else None
            )
        },
        "action_policy_comparison": policy_results,
        "leave_one_out_policy_selection": {
            "correct": loo_correct,
            "count": len(action_rows),
            "choices": loo_choices,
        },
        "episodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episodes = [load_json(path) for path in sorted(args.episode_root.glob("*.json"))]
    predictions = {
        episode["episode_id"]: load_json(
            args.prediction_root / f"{episode['episode_id']}.json"
        )
        for episode in episodes
    }
    result = aggregate(episodes, predictions)
    write_json(args.output, result)
    print(f"WROTE: {args.output} ({result['episode_count']} episodes)")


if __name__ == "__main__":
    main()
