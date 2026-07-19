import copy
import json
import unittest
from pathlib import Path

from digital_mirror.episode import EpisodeValidationError
from digital_mirror.replay import compile_replay_case, score_prediction


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        template_path = Path(__file__).parents[1] / "templates" / "historical_episode.json"
        cls.episode = json.loads(template_path.read_text(encoding="utf-8"))
        cls.episode["evidence"] = [
            {
                "evidence_id": "synthetic_input",
                "source_relative_path": "sources/example.md",
                "source_sha256": "a" * 64,
                "record_locator": "paragraph:1",
                "observed_at": "2026-01-01T11:00:00+08:00",
                "author_role": "self",
                "evidence_role": "input",
                "summary": "A synthetic situation requires a choice.",
            },
            {
                "evidence_id": "synthetic_future_result",
                "source_relative_path": "sources/example.md",
                "source_sha256": "a" * 64,
                "record_locator": "paragraph:2",
                "observed_at": "2026-01-01T13:00:00+08:00",
                "author_role": "self",
                "evidence_role": "target",
                "summary": "The synthetic action was recorded after the cutoff.",
            },
        ]
        cls.episode["labels"]["contemporaneous_deliberation"] = {
            "considered_option_ids": ["wait", "clarify"],
            "tensions": ["certainty vs delay"],
            "unknowns": ["future response"],
            "option_probabilities": {"wait": 0.2, "clarify": 0.7, "withdraw": 0.1},
            "deliberation_intensity": 0.6,
            "confidence": 0.8,
        }
        cls.episode["labels"]["actual_judgement"] = {
            "option_id": "clarify",
            "recorded_at": "2026-01-01T12:01:00+08:00",
            "notes": "Synthetic label.",
        }
        cls.episode["labels"]["actual_action"] = {
            "option_id": "clarify",
            "recorded_at": "2026-01-01T12:05:00+08:00",
            "notes": "Synthetic label.",
        }

    def valid_prediction(self, evidence_ids=None):
        return {
            "episode_id": self.episode["episode_id"],
            "predicted_judgement": "clarify",
            "predicted_action": "clarify",
            "option_probabilities": {"wait": 0.2, "clarify": 0.7, "withdraw": 0.1},
            "considered_option_ids": ["wait", "clarify"],
            "tensions": [],
            "unknowns": [],
            "deliberation_intensity": 0.6,
            "confidence": 0.8,
            "evidence_ids": evidence_ids or ["synthetic_input"],
        }

    def test_compiled_case_contains_no_future_or_labels(self):
        case = compile_replay_case(self.episode)
        serialized = json.dumps(case, ensure_ascii=False)
        self.assertNotIn("labels", case)
        self.assertNotIn("ai_references", case)
        self.assertNotIn("outcome", serialized)
        self.assertNotIn("synthetic_future_result", serialized)
        self.assertEqual([item["evidence_id"] for item in case["evidence"]], ["synthetic_input"])

    def test_perfect_structured_prediction_scores_correctly(self):
        score = score_prediction(self.episode, self.valid_prediction())
        self.assertTrue(score["judgement_correct"])
        self.assertTrue(score["action_correct"])
        self.assertEqual(score["considered_options"]["f1"], 1.0)
        self.assertEqual(score["deliberation_intensity_absolute_error"], 0.0)

    def test_personal_history_is_valid_citable_evidence(self):
        case = compile_replay_case(self.episode)
        case["personal_history"] = [
            {
                "history_id": "history:synthetic_prior",
                "occurred_at": "2025-12-31T12:00:00+08:00",
                "text": "Synthetic prior behavior.",
            }
        ]
        score = score_prediction(
            self.episode,
            self.valid_prediction(["history:synthetic_prior"]),
            case,
        )
        self.assertTrue(score["judgement_correct"])

    def test_personal_history_must_be_strictly_pre_cutoff(self):
        case = compile_replay_case(self.episode)
        case["personal_history"] = [
            {
                "history_id": "history:synthetic_future",
                "occurred_at": case["cutoff_at"],
                "text": "Synthetic future behavior.",
            }
        ]
        with self.assertRaises(EpisodeValidationError):
            score_prediction(
                self.episode,
                self.valid_prediction(["history:synthetic_future"]),
                case,
            )


if __name__ == "__main__":
    unittest.main()
