import json
import unittest
from pathlib import Path

from digital_mirror.local_baseline import build_prompt, run_with_retries
from digital_mirror.replay import compile_replay_case


class LocalBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        episode_path = Path(__file__).parents[1] / "templates" / "historical_episode.json"
        cls.case = compile_replay_case(json.loads(episode_path.read_text(encoding="utf-8")))
        schema_path = Path(__file__).parents[1] / "schemas" / "replay_prediction.schema.json"
        cls.schema = json.loads(schema_path.read_text(encoding="utf-8"))

    def valid_prediction(self):
        return {
            "episode_id": self.case["episode_id"],
            "predicted_judgement": "clarify",
            "predicted_action": "wait",
            "option_probabilities": {
                "wait": 0.2,
                "clarify": 0.7,
                "withdraw": 0.1,
            },
            "considered_option_ids": ["wait", "clarify"],
            "tensions": ["time", "hunger"],
            "unknowns": ["schedule", "energy"],
            "deliberation_intensity": 0.3,
            "confidence": 0.7,
            "evidence_ids": [],
        }

    def test_prompt_contains_options_but_no_answers(self):
        prompt = build_prompt(self.case)
        self.assertIn("clarify", prompt)
        self.assertNotIn("actual_judgement", prompt)
        self.assertNotIn("outcome", prompt)

    def test_format_retry_does_not_need_labels(self):
        responses = iter(
            [
                json.dumps({"episode_id": self.case["episode_id"]}),
                json.dumps(self.valid_prediction()),
            ]
        )
        prompts = []

        def generate(prompt, _schema):
            prompts.append(prompt)
            return next(responses)

        prediction, retries = run_with_retries(self.case, generate, self.schema)
        self.assertEqual(retries, 1)
        self.assertEqual(prediction["predicted_action"], "wait")
        self.assertIn("missing prediction field", prompts[1])
        self.assertNotIn("actual_action", prompts[1])


if __name__ == "__main__":
    unittest.main()
