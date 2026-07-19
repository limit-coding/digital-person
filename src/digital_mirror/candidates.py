"""Mine timestamped decision-event candidates from local chat exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .audit import sha256_file


MARKERS: dict[str, re.Pattern[str]] = {
    "deliberation": re.compile(r"纠结|犹豫|要不要|怎么办|怎么选|选哪个|拿不准"),
    "commitment": re.compile(r"决定|打算|准备|计划|必须|下决心|all[ -]?in", re.I),
    "withdrawal": re.compile(r"算了|不去了|不做了|放弃|拉黑|删除|免打扰|拒绝|退出"),
    "uncertainty": re.compile(r"不确定|不知道|不清楚|不一定|可能|估计|感觉"),
    "alternatives": re.compile(r"还是|或者|要么|不然|如果.*就"),
    "outcome": re.compile(r"结果|后来|最后|已经|果然|成功|失败|通过|没过|拿到|被.{0,8}(拒绝|删除|拉黑)"),
}

STRONG_MARKERS = {"deliberation", "commitment", "withdrawal"}

DOMAIN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("relationship", re.compile(r"女生|男生|喜欢|对象|恋爱|感情|拉黑|小作文|回消息|关系")),
    ("family", re.compile(r"爸|妈|父母|家庭|家里")),
    ("career", re.compile(r"实习|面试|导师|工作|求职|offer|公司|字节|英伟达", re.I)),
    ("learning", re.compile(r"学习|课程|考试|成绩|刷题|作业|复习|保研|论文|竞赛|神经网络")),
    ("health", re.compile(r"跑步|训练|睡觉|身体|医院|生病|受伤|配速|运动")),
    ("finance", re.compile(r"买|钱|价格|订阅|银行卡|付款|预算|加密货币")),
)

URL_RE = re.compile(r"https?://\S+", re.I)
PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
EMAIL_RE = re.compile(r"(?i)[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}")
QUOTE_MARKERS = ("[引用 ", "[引用 ")

OTHER_DIRECTED_RE = re.compile(
    r"(?:^|[，。！？?\s])你(?:.{0,10})(?:打算|准备|决定|要不要|怎么办|怎么选|去哪|住哪)"
)
HYPOTHETICAL_ADVICE_RE = re.compile(r"我要是你|如果我是你|换[作做]是我|换我(?:就|会)")
RETROSPECTIVE_RE = re.compile(r"原本打算|原来打算|当初|以前|那时候|后来|已经")
SELF_DECISION_RE = re.compile(
    r"(?:我|我们|咱们)(?:.{0,14})(?:打算|准备|决定|要不要|还是|算了|放弃|退出|拒绝|必须|就去|先去)"
    r"|^(?:打算|准备|决定|还是|算了|放弃|退出|拒绝|必须)"
)
ACTION_EVIDENCE_RE = re.compile(
    r"(?:我|我们|咱们)?(?:已经|刚刚?|正在|开始|最终|最后)?(?:去了|到了|发了|投了|退了|退出了|取消了|跑完|完事了|做完|没去|没做|没发|放弃了|拉伸|刚出来|放松.{0,6}肌肉)"
)


def self_authored_segment(text: str) -> str:
    """Remove quoted/forwarded bodies that were transmitted but not authored by self."""
    if text.lstrip().startswith("[转发的聊天记录]"):
        return ""
    cut = len(text)
    for marker in QUOTE_MARKERS:
        position = text.find(marker)
        if position >= 0:
            cut = min(cut, position)
    return text[:cut].strip()


def likely_pasted_ai(text: str) -> bool:
    if len(text) < 400:
        return False
    direct_signals = (
        "作为一个基于",
        "作为一个 AI",
        "作为AI",
        "以下是为您整理",
        "以下是针对这些",
    )
    if any(signal in text for signal in direct_signals):
        return True
    markdown_density = text.count("**") + text.count("##") * 2
    structured_signals = sum(
        signal in text for signal in ("总结", "核心", "建议", "分析", "概率")
    )
    return len(text) >= 900 and markdown_density >= 6 and structured_signals >= 2


def marker_categories(text: str) -> list[str]:
    return [name for name, pattern in MARKERS.items() if pattern.search(text)]


def candidate_score(categories: list[str], text: str) -> int:
    score = sum(3 if category in STRONG_MARKERS else 1 for category in categories)
    if len(text) >= 20:
        score += 1
    if len(text) >= 80:
        score += 1
    return score


def infer_domain(text: str) -> str:
    for domain, pattern in DOMAIN_PATTERNS:
        if pattern.search(text):
            return domain
    return "other"


def redact_excerpt(text: str) -> str:
    text = URL_RE.sub("[URL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    return text[:240]


def decision_subject(text: str) -> str:
    """Classify whose decision the anchor most likely describes."""
    if HYPOTHETICAL_ADVICE_RE.search(text):
        return "hypothetical_other"
    if OTHER_DIRECTED_RE.search(text):
        return "other"
    if SELF_DECISION_RE.search(text):
        return "self"
    return "unclear"


def replay_readiness_score(
    base_score: int,
    subject: str,
    categories: list[str],
    pre_deliberation_marker_count: int,
    future_action_marker_count: int,
    retrospective: bool,
) -> int:
    """Prioritize anchors that can become chronological judgement/action tests."""
    score = base_score
    score += 3 if subject == "self" else -5 if subject in {"other", "hypothetical_other"} else -1
    score += min(pre_deliberation_marker_count, 2) * 2
    score += min(future_action_marker_count, 2) * 2
    if "deliberation" in categories or "alternatives" in categories:
        score += 2
    if retrospective:
        score -= 3
    return score


def parse_chat(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    messages = raw.get("messages", []) if isinstance(raw, dict) else []
    return [message for message in messages if isinstance(message, dict)]


def parse_chat_time(message: dict[str, Any]) -> datetime | None:
    timestamp = message.get("createTime")
    if isinstance(timestamp, int):
        return datetime.fromtimestamp(timestamp)
    formatted = message.get("formattedTime")
    if isinstance(formatted, str):
        try:
            return datetime.fromisoformat(formatted)
        except ValueError:
            return None
    return None


def raw_candidates(
    messages: list[dict[str, Any]],
    relative_path: str,
    source_sha256: str,
    include_excerpts: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if message.get("isSend") != 1 or message.get("type") not in {"文本消息", "引用消息"}:
            continue
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            continue
        text = self_authored_segment(text)
        # Candidate anchors should be concise contemporaneous utterances. Long posts and
        # pasted model outputs need a separate provenance-aware document pipeline.
        if not text or len(text) > 600 or likely_pasted_ai(text):
            continue
        categories = marker_categories(text)
        if not STRONG_MARKERS.intersection(categories):
            continue
        timestamp = parse_chat_time(message)
        if timestamp is None:
            continue
        prior_self_texts = [
            self_authored_segment(item["content"])
            for item in messages[max(0, index - 30) : index]
            if item.get("isSend") == 1 and isinstance(item.get("content"), str)
            and parse_chat_time(item) is not None
            and (timestamp - parse_chat_time(item)).total_seconds() <= 30 * 60
        ]
        pre_deliberation_marker_count = sum(
            bool(
                value
                and any(
                    MARKERS[name].search(value)
                    for name in ("deliberation", "uncertainty", "alternatives")
                )
            )
            for value in prior_self_texts
        )
        window = messages[max(0, index - 8) : min(len(messages), index + 13)]
        self_count = sum(item.get("isSend") == 1 for item in window)
        other_count = sum(item.get("isSend") == 0 for item in window)
        following_self_texts: list[str] = []
        for item in messages[index + 1 : index + 61]:
            item_time = parse_chat_time(item)
            if item_time is None or (item_time - timestamp).total_seconds() > 24 * 60 * 60:
                continue
            if item.get("isSend") == 1 and isinstance(item.get("content"), str):
                following_self_texts.append(self_authored_segment(item["content"]))
        future_outcome_markers = sum(
            bool(text_value and MARKERS["outcome"].search(text_value))
            for text_value in following_self_texts
        )
        future_action_marker_count = sum(
            bool(text_value and ACTION_EVIDENCE_RE.search(text_value))
            for text_value in following_self_texts
        )
        subject = decision_subject(text)
        retrospective = bool(RETROSPECTIVE_RE.search(text))
        readiness = replay_readiness_score(
            candidate_score(categories, text),
            subject,
            categories,
            pre_deliberation_marker_count,
            future_action_marker_count,
            retrospective,
        )
        local_id = message.get("localId", index)
        candidate_id = hashlib.sha256(
            f"{relative_path}:{local_id}".encode("utf-8")
        ).hexdigest()[:16]
        record: dict[str, Any] = {
            "candidate_id": candidate_id,
            "source_relative_path": relative_path,
            "source_sha256": source_sha256,
            "anchor_local_id": local_id,
            "anchor_time": timestamp.astimezone().isoformat(),
            "marker_categories": categories,
            "score": candidate_score(categories, text),
            "inferred_domain": infer_domain(text),
            "window_start_local_id": window[0].get("localId") if window else None,
            "window_end_local_id": window[-1].get("localId") if window else None,
            "window_self_message_count": self_count,
            "window_other_message_count": other_count,
            "future_outcome_marker_count": future_outcome_markers,
            "future_action_marker_count": future_action_marker_count,
            "pre_deliberation_marker_count": pre_deliberation_marker_count,
            "decision_subject": subject,
            "retrospective": retrospective,
            "replay_readiness_score": readiness,
        }
        if include_excerpts:
            record["anchor_excerpt"] = redact_excerpt(text)
        results.append(record)
    return results


def collapse_nearby(candidates: list[dict[str, Any]], minutes: int = 30) -> list[dict[str, Any]]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: item["anchor_time"])
    clusters: list[list[dict[str, Any]]] = [[ordered[0]]]
    for candidate in ordered[1:]:
        previous_time = datetime.fromisoformat(clusters[-1][-1]["anchor_time"])
        current_time = datetime.fromisoformat(candidate["anchor_time"])
        if (current_time - previous_time).total_seconds() <= minutes * 60:
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])
    collapsed: list[dict[str, Any]] = []
    for cluster in clusters:
        best = max(
            cluster,
            key=lambda item: (
                item["score"],
                item["future_outcome_marker_count"],
                -int(item["anchor_local_id"] or 0),
            ),
        ).copy()
        best["cluster_anchor_count"] = len(cluster)
        best["cluster_marker_categories"] = sorted(
            {category for item in cluster for category in item["marker_categories"]}
        )
        collapsed.append(best)
    return collapsed


def mine_chats(
    workspace: Path, chat_paths: Iterable[Path], include_excerpts: bool = False
) -> list[dict[str, Any]]:
    all_candidates: list[dict[str, Any]] = []
    for path in chat_paths:
        relative_path = path.relative_to(workspace).as_posix()
        raw = raw_candidates(
            parse_chat(path),
            relative_path,
            sha256_file(path),
            include_excerpts,
        )
        all_candidates.extend(collapse_nearby(raw))
    return sorted(
        all_candidates,
        key=lambda item: (
            item["replay_readiness_score"],
            item["score"],
            item["future_action_marker_count"],
        ),
        reverse=True,
    )


def summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_count": len(candidates),
        "by_domain": dict(sorted(Counter(item["inferred_domain"] for item in candidates).items())),
        "by_marker": dict(
            sorted(
                Counter(
                    category
                    for item in candidates
                    for category in item["cluster_marker_categories"]
                ).items()
            )
        ),
        "by_decision_subject": dict(
            sorted(Counter(item["decision_subject"] for item in candidates).items())
        ),
        "replay_ready_count": sum(
            item["decision_subject"] == "self"
            and item["replay_readiness_score"] >= 8
            for item in candidates
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--chat-root", type=Path, default=Path("texts"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-excerpts", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    chat_root = workspace / args.chat_root
    paths = sorted(chat_root.glob("*.json"))
    candidates = mine_chats(workspace, paths, args.include_excerpts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in candidates),
        encoding="utf-8",
    )
    print(json.dumps(summary(candidates), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
