import json
import tempfile
import unittest
from pathlib import Path

from digital_mirror.retrieval import build_index, search


class RetrievalTests(unittest.TestCase):
    def test_chinese_trigram_search_and_filters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_path = root / "records.jsonl"
            records = [
                {
                    "record_id": "a",
                    "source_relative_path": "a.md",
                    "locator": "chunk:0",
                    "source_group": "machine_rules",
                    "record_kind": "markdown_chunk",
                    "author_role": "self",
                    "occurred_at": None,
                    "text": "机器模式的目标是减少内耗并稳定决策。",
                    "allowed_uses": ["memory", "decision_candidate"],
                    "provenance_flags": [],
                },
                {
                    "record_id": "b",
                    "source_relative_path": "b.md",
                    "locator": "chunk:0",
                    "source_group": "notes",
                    "record_kind": "markdown_chunk",
                    "author_role": "unknown",
                    "occurred_at": None,
                    "text": "普通技术笔记与数据库。",
                    "allowed_uses": ["memory_review"],
                    "provenance_flags": [],
                },
            ]
            records_path.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
                encoding="utf-8",
            )
            database = root / "index.sqlite"
            build_index(records_path, database)
            results = search(
                database,
                "机器模式",
                allowed_use="decision_candidate",
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["record_id"], "a")

    def test_historical_filter_excludes_undated_documents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_path = root / "records.jsonl"
            base = {
                "source_relative_path": "source.md",
                "locator": "chunk:0",
                "source_group": "machine_rules",
                "record_kind": "markdown_chunk",
                "author_role": "self",
                "text": "历史决策信息",
                "allowed_uses": ["memory"],
                "provenance_flags": [],
            }
            records = [
                {**base, "record_id": "dated", "occurred_at": "2026-01-01T10:00:00+08:00"},
                {**base, "record_id": "undated", "occurred_at": None},
            ]
            records_path.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
                encoding="utf-8",
            )
            database = root / "index.sqlite"
            build_index(records_path, database)
            results = search(database, "历史决策", before="2026-02-01 00:00:00")
            self.assertEqual([item["record_id"] for item in results], ["dated"])

    def test_historical_filter_rejects_same_day_future_with_mixed_iso_separators(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_path = root / "records.jsonl"
            base = {
                "source_relative_path": "chat.json",
                "locator": "message:1",
                "source_group": "chats",
                "record_kind": "chat_message",
                "author_role": "self",
                "text": "未来消息内容",
                "allowed_uses": ["behavior_context"],
                "provenance_flags": [],
            }
            records = [
                {**base, "record_id": "past", "occurred_at": "2026-01-01T11:00:00+08:00"},
                {**base, "record_id": "future", "occurred_at": "2026-01-01T13:00:00+08:00"},
            ]
            records_path.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
                encoding="utf-8",
            )
            database = root / "index.sqlite"
            build_index(records_path, database)
            results = search(database, "未来消息", before="2026-01-01T12:00:00+08:00")
            self.assertEqual([item["record_id"] for item in results], ["past"])


if __name__ == "__main__":
    unittest.main()
