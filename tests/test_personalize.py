import json
import tempfile
import unittest
from pathlib import Path

from digital_mirror.personalize import augment_case, text_similarity, windows_overlap
from digital_mirror.retrieval import build_index


class PersonalizationTests(unittest.TestCase):
    def test_overlapping_windows_are_recognized_as_redundant(self):
        self.assertGreater(
            text_similarity(
                "[SELF] 我准备学习神经网络\n[OTHER] 可以做项目",
                "[OTHER] 可以做项目\n[SELF] 直接跑代码",
            ),
            0.1,
        )
        self.assertTrue(windows_overlap("window:761..772", "window:767..778"))
        self.assertFalse(windows_overlap("window:761..772", "window:773..784"))

    def test_only_pre_cutoff_self_behavior_history_is_added(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records = []
            for record_id, role, occurred_at, uses in (
                ("valid", "self", "2026-01-01T11:00:00+08:00", ["behavior_context"]),
                ("future", "self", "2026-01-01T13:00:00+08:00", ["behavior_context"]),
                ("other", "other", "2026-01-01T11:00:00+08:00", ["interaction_context"]),
                ("undated", "self", None, ["behavior_context"]),
            ):
                records.append(
                    {
                        "record_id": record_id,
                        "source_relative_path": "chat.json",
                        "locator": f"message:{record_id}",
                        "source_group": "chats",
                        "record_kind": "chat_message",
                        "author_role": role,
                        "occurred_at": occurred_at,
                        "text": "跑步训练计划",
                        "allowed_uses": uses,
                        "provenance_flags": [],
                    }
                )
            records_path = root / "records.jsonl"
            records_path.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
                encoding="utf-8",
            )
            database = root / "index.sqlite"
            build_index(records_path, database)
            case = {
                "episode_id": "test",
                "cutoff_at": "2026-01-01T12:00:00+08:00",
                "instructions": [],
            }
            augmented = augment_case(case, database, ["跑步训练"])
            self.assertEqual(
                [item["history_id"] for item in augmented["personal_history"]],
                ["history:valid"],
            )
            self.assertNotIn("labels", augmented)

    def test_raw_episode_with_labels_is_rejected(self):
        with self.assertRaises(ValueError):
            augment_case(
                {"cutoff_at": "2026-01-01T12:00:00+08:00", "labels": {}},
                Path("missing.sqlite"),
                ["跑步训练"],
            )


if __name__ == "__main__":
    unittest.main()
