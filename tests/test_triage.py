import json
import tempfile
import unittest
from pathlib import Path

from digital_mirror.triage import (
    normalize_review_for_item,
    prepare_queue,
    safe_context_text,
    validate_reviews,
)


class TriageTests(unittest.TestCase):
    def test_forwarded_content_is_not_copied(self):
        self.assertEqual(
            safe_context_text({"content": "[转发的聊天记录]secret", "isSend": 0}),
            "[FORWARDED_CONTENT]",
        )
        self.assertEqual(
            safe_context_text({"content": "long copied text " * 100, "isSend": 1}),
            "[LONG_OR_PASTED_CONTENT]",
        )

    def test_prepare_queue_excludes_covered_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "texts").mkdir()
            messages = {
                "messages": [
                    {"localId": 1, "formattedTime": "2026-01-01 12:00:00", "isSend": 0, "type": "文本消息", "content": "去吗"},
                    {"localId": 2, "formattedTime": "2026-01-01 12:01:00", "isSend": 1, "type": "文本消息", "content": "我决定去"},
                ]
            }
            (root / "texts" / "chat.json").write_text(json.dumps(messages), encoding="utf-8")
            episodes = root / "episodes"
            episodes.mkdir()
            (episodes / "covered.json").write_text(
                json.dumps({"evidence": [{"source_relative_path": "texts/chat.json", "record_locator": "message.localId=2"}]}),
                encoding="utf-8",
            )
            candidate = {
                "candidate_id": "c1",
                "source_relative_path": "texts/chat.json",
                "anchor_local_id": 2,
                "anchor_time": "2026-01-01T12:01:00+08:00",
                "inferred_domain": "other",
                "decision_subject": "self",
                "replay_readiness_score": 10,
            }
            self.assertEqual(prepare_queue(root, [candidate], episodes)["items"], [])

    def test_review_cutoff_must_precede_anchor(self):
        queue = {"items": [{"candidate_id": "c1", "anchor_local_id": 10}]}
        result = {"reviews": [{"candidate_id": "c1", "suggested_cutoff_local_id": 10}]}
        self.assertTrue(validate_reviews(queue, result))

    def test_leaky_cutoff_is_downgraded(self):
        item = {"candidate_id": "c1", "anchor_local_id": 10}
        review = {
            "candidate_id": "c1",
            "suggested_cutoff_local_id": 10,
            "disposition": "replay_ready",
            "has_pre_cutoff_context": True,
            "reason": "looks useful",
            "confidence": 0.9,
        }
        normalized = normalize_review_for_item(item, review)
        self.assertIsNone(normalized["suggested_cutoff_local_id"])
        self.assertEqual(normalized["disposition"], "needs_more_context")
        self.assertFalse(normalized["has_pre_cutoff_context"])


if __name__ == "__main__":
    unittest.main()
