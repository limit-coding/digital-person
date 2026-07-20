import json
import unittest

from digital_mirror.personal_profile import build_personal_snapshot


class PersonalProfileTests(unittest.TestCase):
    def test_snapshot_aggregates_without_exposing_source_paths(self):
        episodes = []
        for index, (domain, judgement, action) in enumerate(
            [("learning", "build", "build"), ("career", "apply", "wait")],
            start=1,
        ):
            episode = {
                "labels": {
                    "contemporaneous_deliberation": {
                        "deliberation_intensity": 0.4 + index * 0.1,
                        "considered_option_ids": ["a", "b", "c"],
                        "tensions": ["速度 vs 完整"],
                    },
                    "actual_judgement": {"option_id": judgement},
                    "actual_action": {"option_id": action},
                    "outcome": None,
                }
            }
            case = {
                "episode_id": f"synthetic-{index}",
                "domain": domain,
                "cutoff_at": f"2026-01-0{index}T12:00:00+08:00",
                "question": "合成测试问题",
                "evidence": [{"evidence_id": f"e-{index}"}],
            }
            episodes.append((episode, case))

        snapshot = build_personal_snapshot(episodes)
        self.assertEqual(snapshot["coverage"]["event_count"], 2)
        self.assertEqual(snapshot["coverage"]["domain_count"], 2)
        self.assertEqual(snapshot["decision_metrics"]["pivot_count"], 1)
        self.assertEqual(snapshot["decision_metrics"]["judgement_action_alignment"], 0.5)
        serialized = json.dumps(snapshot, ensure_ascii=False)
        self.assertNotIn("source_relative_path", serialized)
        self.assertNotIn("source_sha256", serialized)


if __name__ == "__main__":
    unittest.main()
