import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from digital_mirror.episode import validate_episode, verify_evidence_sources


class EpisodeValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        template_path = Path(__file__).parents[1] / "templates" / "historical_episode.json"
        cls.template = json.loads(template_path.read_text(encoding="utf-8"))

    def test_template_is_valid(self):
        self.assertEqual(validate_episode(copy.deepcopy(self.template)), [])

    def test_synthetic_source_cross_check(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source = workspace / "sources" / "example.md"
            source.parent.mkdir()
            source.write_text("synthetic evidence", encoding="utf-8")
            episode = copy.deepcopy(self.template)
            episode["evidence"] = [
                {
                    "evidence_id": "synthetic_input",
                    "source_relative_path": "sources/example.md",
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "record_locator": "paragraph:1",
                    "observed_at": "2026-01-01T11:00:00+08:00",
                    "author_role": "self",
                    "evidence_role": "input",
                    "summary": "Synthetic pre-decision evidence.",
                }
            ]
            self.assertEqual(validate_episode(episode), [])
            self.assertEqual(verify_evidence_sources(episode, workspace), [])

    def test_future_input_is_rejected(self):
        episode = copy.deepcopy(self.template)
        episode["evidence"].append(
            {
                "evidence_id": "future",
                "source_relative_path": "sources/example.md",
                "source_sha256": "a" * 64,
                "record_locator": "paragraph:1",
                "observed_at": "2026-01-02T12:00:00+08:00",
                "author_role": "self",
                "evidence_role": "input",
                "summary": "future result",
            }
        )
        self.assertTrue(any("future information" in error for error in validate_episode(episode)))

    def test_probabilities_must_sum_to_one(self):
        episode = copy.deepcopy(self.template)
        episode["labels"]["contemporaneous_deliberation"]["option_probabilities"] = {
            "wait": 0.8,
            "clarify": 0.8,
        }
        self.assertTrue(any("sum to 1.0" in error for error in validate_episode(episode)))

    def test_judgement_and_action_can_differ(self):
        episode = copy.deepcopy(self.template)
        episode["labels"]["actual_judgement"] = {
            "option_id": "clarify",
            "recorded_at": "2026-01-01T12:01:00+08:00",
            "notes": "",
        }
        episode["labels"]["actual_action"] = {
            "option_id": "wait",
            "recorded_at": "2026-01-01T13:00:00+08:00",
            "notes": "",
        }
        self.assertEqual(validate_episode(episode), [])


if __name__ == "__main__":
    unittest.main()
