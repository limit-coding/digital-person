import unittest
from datetime import datetime

from digital_mirror.candidates import (
    collapse_nearby,
    decision_subject,
    infer_domain,
    marker_categories,
    raw_candidates,
    redact_excerpt,
    self_authored_segment,
    likely_pasted_ai,
    replay_readiness_score,
)


class CandidateMiningTests(unittest.TestCase):
    def test_decision_markers(self):
        categories = marker_categories("我还在纠结要不要去面试，明天必须决定")
        self.assertIn("deliberation", categories)
        self.assertIn("commitment", categories)

    def test_domain_inference(self):
        self.assertEqual(infer_domain("我准备去参加实习面试"), "career")
        self.assertEqual(infer_domain("要不要继续跑步训练"), "health")

    def test_decision_subject_filters_questions_and_advice(self):
        self.assertEqual(decision_subject("我决定退出这个竞赛"), "self")
        self.assertEqual(decision_subject("你打算去哪啊"), "other")
        self.assertEqual(decision_subject("我要是你我就直接表白"), "hypothetical_other")

    def test_replay_readiness_rewards_pre_conflict_and_observed_action(self):
        plain = replay_readiness_score(3, "self", ["commitment"], 0, 0, False)
        replayable = replay_readiness_score(
            3, "self", ["commitment", "alternatives"], 1, 1, False
        )
        self.assertGreater(replayable, plain)
        self.assertLess(
            replay_readiness_score(3, "other", ["commitment"], 0, 0, False),
            plain,
        )

    def test_excerpt_redacts_common_direct_identifiers(self):
        excerpt = redact_excerpt("电话13800138000 邮箱a@example.com https://example.com")
        self.assertNotIn("13800138000", excerpt)
        self.assertNotIn("a@example.com", excerpt)
        self.assertNotIn("https://example.com", excerpt)

    def test_quoted_and_forwarded_content_is_not_self_authored(self):
        self.assertEqual(
            self_authored_segment("我不同意[引用 对方：我决定退出]"),
            "我不同意",
        )
        self.assertEqual(self_authored_segment("[转发的聊天记录]某人：我决定退出"), "")

    def test_likely_ai_paste_is_detected(self):
        pasted = "作为一个基于很多记录分析的 AI，以下是针对这些问题的分析：" + "**核心**" * 100
        self.assertTrue(likely_pasted_ai(pasted))

    def test_raw_candidate_contains_locators_not_other_content(self):
        messages = [
            {
                "localId": 1,
                "createTime": int(datetime(2026, 1, 1, 12).timestamp()),
                "isSend": 0,
                "type": "文本消息",
                "content": "third-party secret",
            },
            {
                "localId": 2,
                "createTime": int(datetime(2026, 1, 1, 12, 1).timestamp()),
                "isSend": 1,
                "type": "文本消息",
                "content": "我决定去面试",
            },
        ]
        candidates = raw_candidates(messages, "texts/test.json", "a" * 64, True)
        self.assertEqual(len(candidates), 1)
        self.assertNotIn("third-party secret", str(candidates))
        self.assertEqual(candidates[0]["anchor_local_id"], 2)
        self.assertEqual(candidates[0]["decision_subject"], "self")

    def test_decision_inside_quote_does_not_create_candidate(self):
        messages = [
            {
                "localId": 1,
                "createTime": int(datetime(2026, 1, 1, 12).timestamp()),
                "isSend": 1,
                "type": "引用消息",
                "content": "确实[引用 对方：我决定退出]",
            }
        ]
        self.assertEqual(raw_candidates(messages, "texts/test.json", "a" * 64, True), [])

    def test_long_pasted_post_does_not_create_candidate(self):
        messages = [
            {
                "localId": 1,
                "createTime": int(datetime(2026, 1, 1, 12).timestamp()),
                "isSend": 1,
                "type": "文本消息",
                "content": "我决定" + "这是粘贴的长文" * 100,
            }
        ]
        self.assertEqual(raw_candidates(messages, "texts/test.json", "a" * 64, True), [])

    def test_nearby_anchors_collapse(self):
        candidates = [
            {
                "anchor_time": "2026-01-01T12:00:00+08:00",
                "score": 3,
                "future_outcome_marker_count": 0,
                "anchor_local_id": 1,
                "marker_categories": ["commitment"],
            },
            {
                "anchor_time": "2026-01-01T12:05:00+08:00",
                "score": 4,
                "future_outcome_marker_count": 1,
                "anchor_local_id": 2,
                "marker_categories": ["deliberation"],
            },
        ]
        collapsed = collapse_nearby(candidates)
        self.assertEqual(len(collapsed), 1)
        self.assertEqual(collapsed[0]["anchor_local_id"], 2)
        self.assertEqual(collapsed[0]["cluster_anchor_count"], 2)


if __name__ == "__main__":
    unittest.main()
