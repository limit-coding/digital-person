import os
import unittest
from unittest.mock import patch

from digital_mirror.cloud_baseline import cloud_safety_errors, deepseek_generator


class CloudBaselineTests(unittest.TestCase):
    def test_cloud_safety_rejects_labels_paths_and_keys(self):
        self.assertTrue(cloud_safety_errors({"labels": {}, "episode_id": "x"}))
        self.assertTrue(cloud_safety_errors({"episode_id": "x", "note": "/Users/me/file"}))
        self.assertTrue(cloud_safety_errors({"episode_id": "x", "note": "sk-secretvalue123"}))

    def test_cloud_safety_accepts_compiled_summary(self):
        self.assertEqual(
            cloud_safety_errors(
                {
                    "episode_id": "health_test",
                    "question": "先跑步还是吃饭？",
                    "evidence": [{"evidence_id": "e1", "summary": "想跑步"}],
                }
            ),
            [],
        )

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"})
    def test_thinking_payload_omits_temperature(self):
        captured = {}

        def request(url, key, payload, timeout):
            captured.update(payload)
            return {"choices": [{"message": {"content": "{}"}}]}

        generator = deepseek_generator(
            "deepseek-v4-pro", "enabled", "high", 0.2, 30, request=request
        )
        generator("prompt", {"type": "object"})
        self.assertNotIn("temperature", captured)
        self.assertEqual(captured["thinking"], {"type": "enabled"})

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"})
    def test_nonthinking_payload_sets_temperature(self):
        captured = {}

        def request(url, key, payload, timeout):
            captured.update(payload)
            return {"choices": [{"message": {"content": "{}"}}]}

        generator = deepseek_generator(
            "deepseek-v4-pro", "disabled", "high", 0.2, 30, request=request
        )
        generator("prompt", {"type": "object"})
        self.assertEqual(captured["temperature"], 0.2)


if __name__ == "__main__":
    unittest.main()
