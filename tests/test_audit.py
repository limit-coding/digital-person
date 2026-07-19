import json
import tempfile
import unittest
from pathlib import Path

from digital_mirror.audit import (
    Policy,
    chat_metadata,
    classify_policy,
    estimate_tokens,
    source_group,
)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        policy_path = Path(self.temp_dir.name) / "policy.json"
        policy_path.write_text(
            json.dumps(
                {
                    "source_roots": ["iCloud", "texts"],
                    "allowed_extensions": [".md", ".json"],
                    "excluded_path_fragments": ["Recently Deleted"],
                    "credential_filename_fragments": ["账号", "手机号"],
                    "ai_attribution_fragments": ["Claude", "Gemini"],
                    "authored_candidate_groups": ["biography", "machine_rules"],
                    "sensitivity_patterns": {
                        "phone": "(?<!\\d)1[3-9]\\d{9}(?!\\d)"
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.policy = Policy.load(policy_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_source_groups(self):
        self.assertEqual(source_group(Path("iCloud/传记/大一.md")), "biography")
        self.assertEqual(source_group(Path("texts/chat.json")), "chats")

    def test_hard_exclusions(self):
        status, _ = classify_policy(
            Path("iCloud/Recently Deleted/a.md"), "other", {}, self.policy
        )
        self.assertEqual(status, "exclude")
        status, _ = classify_policy(
            Path("iCloud/海外生活/雅思账号.md"), "other", {}, self.policy
        )
        self.assertEqual(status, "exclude")

    def test_ai_provenance_is_reviewed(self):
        status, reasons = classify_policy(
            Path("iCloud/传记/Claude点评.md"), "biography", {}, self.policy
        )
        self.assertEqual(status, "review_provenance")
        self.assertIn("ai_attribution_in_filename", reasons)

    def test_chats_require_redaction(self):
        status, reasons = classify_policy(
            Path("texts/chat.json"), "chats", {}, self.policy
        )
        self.assertEqual(status, "include_with_redaction")
        self.assertIn("contains_third_party_messages", reasons)

    def test_chat_metadata_does_not_copy_identity_or_content(self):
        metadata = chat_metadata(
            {
                "session": {"nickname": "private"},
                "messages": [
                    {"isSend": 1, "type": "文本消息", "content": "secret", "createTime": 2},
                    {"isSend": 0, "type": "图片消息", "content": "[图片]", "createTime": 4},
                ],
            }
        )
        self.assertEqual(metadata["self_sent_count"], 1)
        self.assertEqual(metadata["received_count"], 1)
        self.assertNotIn("nickname", metadata)
        self.assertNotIn("content", json.dumps(metadata, ensure_ascii=False))

    def test_token_estimate_is_positive(self):
        self.assertGreater(estimate_tokens("中文 text 123"), 0)


if __name__ == "__main__":
    unittest.main()
