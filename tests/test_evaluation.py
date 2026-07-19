import unittest

from digital_mirror.evaluation import aggregate


class EvaluationTests(unittest.TestCase):
    def test_judgement_as_action_policy_is_measured_separately(self):
        episodes = []
        predictions = {}
        for index, actual in enumerate(("a", "a", "b")):
            episode_id = f"e{index}"
            episodes.append(
                {
                    "episode_id": episode_id,
                    "annotation": {"status": "machine_draft"},
                    "decision": {"options": [{"option_id": "a"}, {"option_id": "b"}]},
                    "labels": {
                        "actual_judgement": {"option_id": actual},
                        "actual_action": {"option_id": actual},
                        "contemporaneous_deliberation": {"deliberation_intensity": 0.5},
                    },
                }
            )
            predictions[episode_id] = {
                "predicted_judgement": actual,
                "predicted_action": "b" if actual == "a" else "a",
                "option_probabilities": {"a": 0.5, "b": 0.5},
                "deliberation_intensity": 0.5,
            }
        result = aggregate(episodes, predictions)
        self.assertEqual(result["action_policy_comparison"]["raw_action"]["correct"], 0)
        self.assertEqual(
            result["action_policy_comparison"]["judgement_as_action"]["correct"], 3
        )
        self.assertEqual(result["leave_one_out_policy_selection"]["correct"], 3)


if __name__ == "__main__":
    unittest.main()
