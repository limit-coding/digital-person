"""Build provenance-aware, locally redacted corpus records from audited sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .candidates import likely_pasted_ai


REDACTIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"(?i)[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}")),
    ("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("long_number", re.compile(r"(?<!\d)\d{15,19}(?!\d)")),
    ("wxid", re.compile(r"(?i)wxid_[a-z0-9_]+")),
    ("url", re.compile(r"(?i)https?://\S+")),
)

FORWARDED_MARKERS = (
    "[转发的聊天记录]",
    "转发的聊天记录",
    "微信ClawBot:",
    "与文件传输助手的聊天记录",
)

AI_ATTRIBUTION_RE = re.compile(
    r"(?i)(Claude|Gemini|ChatGPT|GPT|Kimi|Opus).{0,20}(点评|评论|回答|回复|分析)"
)
AI_RESPONSE_PHRASES = (
    "我给两个",
    "我给三个",
    "总评",
    "说句收尾",
    "一句话总结",
    "我来拆",
    "你等于",
    "要不要我帮你",
    "一个实际的提议",
    "核心矛盾",
    "真正的问题",
)
QUOTE_MARKER_RE = re.compile(r"\[引用\s+[^：\]]+：")
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(password|passwd|api[_ -]?key|access[_ -]?token|secret)\s*[:=]"
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def redact_text(text: str) -> tuple[str, list[str]]:
    flags: list[str] = []
    redacted = text
    for name, pattern in REDACTIONS:
        if pattern.search(redacted):
            flags.append(f"redacted_{name}")
            redacted = pattern.sub(f"[{name.upper()}]", redacted)
    return redacted, flags


def strip_quoted_suffix(text: str) -> tuple[str, bool]:
    match = QUOTE_MARKER_RE.search(text)
    if not match:
        return text, False
    return text[: match.start()].rstrip(), True


def is_forwarded(text: str) -> bool:
    return any(marker in text for marker in FORWARDED_MARKERS)


def likely_ai_response(text: str) -> bool:
    if likely_pasted_ai(text):
        return True
    phrase_count = sum(phrase in text for phrase in AI_RESPONSE_PHRASES)
    bold_count = text.count("**")
    second_person_count = text.count("你")
    numbered_sections = len(re.findall(r"(?:^|\n)\s*\d+[.、]", text))
    if len(text) >= 350 and bold_count >= 4 and phrase_count >= 2:
        return True
    if len(text) >= 800 and bold_count >= 8 and second_person_count >= 8:
        return True
    return (
        len(text) >= 700
        and bold_count >= 4
        and second_person_count >= 5
        and (numbered_sections >= 2 or phrase_count >= 1)
    )


def split_text(text: str, max_chars: int = 1200, overlap_chars: int = 120) -> list[str]:
    text = text.strip()
    if not text:
        return []
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        pieces = [
            paragraph[index : index + max_chars]
            for index in range(0, len(paragraph), max_chars)
        ]
        for piece in pieces:
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if current and len(candidate) > max_chars:
                chunks.append(current)
                overlap = current[-overlap_chars:] if overlap_chars else ""
                current = f"{overlap}\n{piece}".strip()
            else:
                current = candidate
            if len(current) >= max_chars:
                chunks.append(current[:max_chars])
                current = current[max(0, max_chars - overlap_chars) :]
    if current.strip():
        chunks.append(current.strip())
    return chunks


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                records[item["relative_path"]] = item
    return records


def markdown_author_role(chunk: str, audit: dict[str, Any]) -> tuple[str, list[str]]:
    if audit["policy_status"] == "review_provenance":
        return "ai_reference", ["ai_attribution_in_filename"]
    if AI_ATTRIBUTION_RE.search(chunk):
        return "unknown", ["possible_ai_attributed_chunk"]
    if likely_ai_response(chunk):
        return "unknown", ["likely_ai_response"]
    if audit["source_group"] in {
        "biography",
        "machine_rules",
        "learning_essays",
        "daily_reflection",
    }:
        return "self", []
    return "unknown", ["authorship_unconfirmed"]


def allowed_uses(
    author_role: str,
    group: str,
    record_kind: str,
    provenance_flags: list[str],
) -> list[str]:
    if "conversation_window" in provenance_flags:
        return ["memory", "behavior_context", "interaction_context"]
    if author_role == "self":
        uses = ["memory"]
        if "non_linguistic_message" not in provenance_flags:
            uses.append("style_candidate")
        if group == "machine_rules":
            uses.append("decision_candidate")
        if group in {"biography", "chats"}:
            uses.append("behavior_context")
        return uses
    if author_role == "other" and record_kind == "chat_message":
        return ["interaction_context"]
    if author_role in {"ai_reference", "mixed_forwarded"}:
        return ["reference"]
    return ["memory_review"]


def make_record(
    *,
    source_relative_path: str,
    source_sha256: str,
    locator: str,
    source_group: str,
    record_kind: str,
    author_role: str,
    occurred_at: str | None,
    text: str,
    provenance_flags: list[str],
) -> dict[str, Any]:
    redacted, redaction_flags = redact_text(text.strip())
    content_sha = sha256_text(redacted)
    record_key = (
        f"{source_relative_path}\0{source_sha256}\0{locator}\0{content_sha}"
    )
    return {
        "schema_version": 1,
        "record_id": hashlib.sha256(record_key.encode("utf-8")).hexdigest()[:24],
        "source_relative_path": source_relative_path,
        "source_sha256": source_sha256,
        "locator": locator,
        "source_group": source_group,
        "record_kind": record_kind,
        "author_role": author_role,
        "occurred_at": occurred_at,
        "text": redacted,
        "content_sha256": content_sha,
        "allowed_uses": allowed_uses(
            author_role, source_group, record_kind, provenance_flags
        ),
        "privacy": "local_sensitive",
        "provenance_flags": sorted(set(provenance_flags + redaction_flags)),
    }


def ingest_markdown(
    path: Path, relative_path: str, audit: dict[str, Any]
) -> Iterable[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    for index, chunk in enumerate(split_text(text)):
        role, flags = markdown_author_role(chunk, audit)
        yield make_record(
            source_relative_path=relative_path,
            source_sha256=audit["sha256"],
            locator=f"chunk:{index}",
            source_group=audit["source_group"],
            record_kind="markdown_chunk",
            author_role=role,
            occurred_at=None,
            text=chunk,
            provenance_flags=flags,
        )


def chat_role(message: dict[str, Any], text: str) -> tuple[str, list[str]]:
    if message.get("type") == "系统消息":
        return "unknown", ["system_message"]
    if is_forwarded(text) or likely_ai_response(text):
        flag = "forwarded_content" if is_forwarded(text) else "likely_ai_response"
        return "mixed_forwarded", [flag]
    if message.get("isSend") == 1:
        flags = []
        if message.get("type") not in {"文本消息", "引用消息", "链接消息"}:
            flags.append("non_linguistic_message")
        return "self", flags
    if message.get("isSend") == 0:
        return "other", []
    return "unknown", ["sender_role_unknown"]


def ingest_chat(
    path: Path, relative_path: str, audit: dict[str, Any]
) -> Iterable[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    messages = [
        message for message in raw.get("messages", []) if isinstance(message, dict)
    ]
    for index, message in enumerate(messages):
        text = str(message.get("content") or "").strip()
        if not text:
            continue
        if SECRET_ASSIGNMENT_RE.search(text):
            continue
        role, flags = chat_role(message, text)
        if role == "self":
            text, quote_removed = strip_quoted_suffix(text)
            if quote_removed:
                flags.append("quoted_content_removed")
            if not text:
                continue
        local_id = message.get("localId", index)
        formatted_time = message.get("formattedTime")
        occurred_at = (
            formatted_time.replace(" ", "T") + "+08:00"
            if isinstance(formatted_time, str)
            else None
        )
        for part_index, part in enumerate(split_text(text)):
            yield make_record(
                source_relative_path=relative_path,
                source_sha256=audit["sha256"],
                locator=f"message:{local_id}:part:{part_index}",
                source_group="chats",
                record_kind="chat_message",
                author_role=role,
                occurred_at=occurred_at,
                text=part,
                provenance_flags=flags,
            )
    yield from ingest_chat_windows(messages, relative_path, audit)


def window_message_text(message: dict[str, Any]) -> tuple[str, str] | None:
    text = str(message.get("content") or "").strip()
    if not text or SECRET_ASSIGNMENT_RE.search(text):
        return None
    if is_forwarded(text) or likely_ai_response(text):
        return None
    if message.get("isSend") == 1:
        text, _ = strip_quoted_suffix(text)
        role = "SELF"
    elif message.get("isSend") == 0:
        role = "OTHER"
    else:
        return None
    if not text:
        return None
    return role, text


def chat_sessions(
    messages: list[dict[str, Any]], gap_seconds: int = 30 * 60
) -> list[list[dict[str, Any]]]:
    timestamped = [
        message
        for message in messages
        if isinstance(message.get("createTime"), int)
        and window_message_text(message) is not None
    ]
    if not timestamped:
        return []
    sessions: list[list[dict[str, Any]]] = [[timestamped[0]]]
    for message in timestamped[1:]:
        if message["createTime"] - sessions[-1][-1]["createTime"] > gap_seconds:
            sessions.append([message])
        else:
            sessions[-1].append(message)
    return sessions


def ingest_chat_windows(
    messages: list[dict[str, Any]],
    relative_path: str,
    audit: dict[str, Any],
    window_size: int = 12,
    stride: int = 6,
) -> Iterable[dict[str, Any]]:
    for session in chat_sessions(messages):
        if len(session) < 2:
            continue
        starts = list(range(0, len(session), stride))
        if starts and starts[-1] + 1 >= len(session):
            starts.pop()
        for start in starts:
            window = session[start : start + window_size]
            if len(window) < 2:
                continue
            lines: list[str] = []
            roles: set[str] = set()
            for message in window:
                parsed = window_message_text(message)
                if parsed is None:
                    continue
                role, text = parsed
                roles.add(role)
                lines.append(f"[{role}] {text}")
            if len(lines) < 2:
                continue
            start_id = window[0].get("localId", start)
            end_id = window[-1].get("localId", start + len(window) - 1)
            formatted_time = window[-1].get("formattedTime")
            occurred_at = (
                formatted_time.replace(" ", "T") + "+08:00"
                if isinstance(formatted_time, str)
                else None
            )
            author_role = "mixed" if len(roles) > 1 else "self"
            yield make_record(
                source_relative_path=relative_path,
                source_sha256=audit["sha256"],
                locator=f"window:{start_id}..{end_id}",
                source_group="chats",
                record_kind="chat_window",
                author_role=author_role,
                occurred_at=occurred_at,
                text="\n".join(lines),
                provenance_flags=["conversation_window"],
            )


def build_corpus(
    workspace: Path, manifest_path: Path, output_dir: Path
) -> dict[str, Any]:
    workspace = workspace.resolve()
    manifest = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    skipped_statuses: Counter[str] = Counter()
    for relative_path, audit in sorted(manifest.items()):
        status = audit["policy_status"]
        if status in {"exclude", "review_parse_error"} or (
            status == "review_sensitive" and audit["source_group"] != "chats"
        ):
            skipped_statuses[status] += 1
            continue
        path = workspace / relative_path
        if not path.exists():
            skipped_statuses["missing"] += 1
            continue
        if audit["source_group"] == "chats" and path.suffix.lower() == ".json":
            records.extend(ingest_chat(path, relative_path, audit))
        elif path.suffix.lower() in {".md", ".txt"}:
            records.extend(ingest_markdown(path, relative_path, audit))

    (output_dir / "records.jsonl").write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "record_count": len(records),
        "records_by_author_role": dict(
            sorted(Counter(record["author_role"] for record in records).items())
        ),
        "records_by_kind": dict(
            sorted(Counter(record["record_kind"] for record in records).items())
        ),
        "records_by_allowed_use": dict(
            sorted(
                Counter(
                    use for record in records for use in record["allowed_uses"]
                ).items()
            )
        ),
        "skipped_files_by_status": dict(sorted(skipped_statuses.items())),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_corpus(args.workspace, args.manifest, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
