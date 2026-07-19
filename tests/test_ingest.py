import unittest

from digital_mirror.ingest import (
    SECRET_ASSIGNMENT_RE,
    chat_role,
    markdown_author_role,
    likely_ai_response,
    redact_text,
    strip_quoted_suffix,
    ingest_chat_windows,
)


class IngestTests(unittest.TestCase):
    def test_sent_forward_is_not_self_style(self):
        role, flags = chat_role(
            {"isSend": 1, "type": "文本消息"},
            "[转发的聊天记录] 微信ClawBot: 一段AI回答",
        )
        self.assertEqual(role, "mixed_forwarded")
        self.assertIn("forwarded_content", flags)

    def test_sent_plain_text_is_self(self):
        role, _ = chat_role({"isSend": 1, "type": "文本消息"}, "这是我的话")
        self.assertEqual(role, "self")

    def test_long_ai_like_paste_is_not_self_style(self):
        text = "作为一个 AI，以下是针对这些问题的分析：" + "**核心建议**" * 120
        role, flags = chat_role({"isSend": 1, "type": "文本消息"}, text)
        self.assertEqual(role, "mixed_forwarded")
        self.assertIn("likely_ai_response", flags)

    def test_quote_is_removed_from_self_text(self):
        text, removed = strip_quoted_suffix("我的回答[引用 对方：这是对方的话]")
        self.assertTrue(removed)
        self.assertEqual(text, "我的回答")

    def test_sensitive_values_are_not_copied(self):
        text, flags = redact_text("联系 13800138000 或 a@example.com wxid_abc123")
        self.assertNotIn("13800138000", text)
        self.assertNotIn("a@example.com", text)
        self.assertNotIn("wxid_abc123", text)
        self.assertIn("redacted_phone", flags)

    def test_secret_assignment_is_detected_for_record_exclusion(self):
        self.assertIsNotNone(SECRET_ASSIGNMENT_RE.search("api_key = private-value"))

    def test_ai_attributed_filename_is_reference(self):
        role, _ = markdown_author_role(
            "一些文字",
            {"policy_status": "review_provenance", "source_group": "biography"},
        )
        self.assertEqual(role, "ai_reference")

    def test_chat_window_preserves_roles_but_is_not_style_training(self):
        messages = [
            {
                "localId": 1,
                "createTime": 100,
                "formattedTime": "2026-01-01 12:00:00",
                "isSend": 0,
                "type": "文本消息",
                "content": "几点吃饭",
            },
            {
                "localId": 2,
                "createTime": 110,
                "formattedTime": "2026-01-01 12:00:10",
                "isSend": 1,
                "type": "文本消息",
                "content": "我先跑步",
            },
        ]
        audit = {"sha256": "a" * 64}
        records = list(ingest_chat_windows(messages, "texts/test.json", audit))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["author_role"], "mixed")
        self.assertIn("[OTHER] 几点吃饭", records[0]["text"])
        self.assertIn("[SELF] 我先跑步", records[0]["text"])
        self.assertIn("behavior_context", records[0]["allowed_uses"])
        self.assertNotIn("style_candidate", records[0]["allowed_uses"])

    def test_mixed_ai_chunk_is_not_style_candidate(self):
        role, flags = markdown_author_role(
            "Claude 对这个事件的点评如下",
            {"policy_status": "include_candidate", "source_group": "biography"},
        )
        self.assertEqual(role, "unknown")
        self.assertIn("possible_ai_attributed_chunk", flags)

    def test_structured_ai_like_chunk_is_not_style_candidate(self):
        chunk = "作为一个 AI，以下是针对这些问题的分析：" + "**核心建议**" * 120
        role, flags = markdown_author_role(
            chunk,
            {"policy_status": "include_candidate", "source_group": "machine_rules"},
        )
        self.assertEqual(role, "unknown")
        self.assertIn("likely_ai_response", flags)

    def test_unattributed_second_person_ai_reply_is_detected(self):
        text = (
            "这是你给系统加的最好规则。"
            + "**补丁** 我给两个建议。" * 40
            + "总评很好。说句收尾，你等于给系统装了保险丝。"
        )
        self.assertTrue(likely_ai_response(text))

    def test_numbered_second_person_ai_analysis_is_detected(self):
        text = (
            "从这个视角看，你的做法是这样。\n"
            "1. **拿回主动权** 你把不确定性关闭。\n"
            "2. **解决内耗** 你减少了单向输出。\n"
            "3. **结果评估** 你得到了明确结果。\n"
            + "你需要重新获得自己的精力。" * 80
        )
        self.assertTrue(likely_ai_response(text))


if __name__ == "__main__":
    unittest.main()
