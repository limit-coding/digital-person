"""Prepare and locally review decision candidates for historical replay suitability."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .candidates import (
    likely_pasted_ai,
    parse_chat,
    parse_chat_time,
    redact_excerpt,
    self_authored_segment,
)
from .local_baseline import ollama_generator
from .replay import load_json, write_json


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def safe_context_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, str):
        return f"[{message.get('type', 'non-text')}]"
    if content.lstrip().startswith("[转发的聊天记录]"):
        return "[FORWARDED_CONTENT]"
    if len(content) > 600 or likely_pasted_ai(content):
        return "[LONG_OR_PASTED_CONTENT]"
    if message.get("isSend") == 1:
        content = self_authored_segment(content)
    return redact_excerpt(content.replace("\n", " ")) or "[NON_LINGUISTIC]"


def context_item(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "local_id": message.get("localId"),
        "occurred_at": message.get("formattedTime"),
        "role": "self" if message.get("isSend") == 1 else "other",
        "text": safe_context_text(message),
    }


def covered_locators(episode_root: Path) -> dict[str, list[tuple[int, int]]]:
    covered: dict[str, list[tuple[int, int]]] = {}
    import re

    pattern = re.compile(r"message\.localId=(\d+)(?:\.\.(\d+))?")
    for path in episode_root.glob("*.json"):
        episode = load_json(path)
        for evidence in episode.get("evidence", []):
            match = pattern.fullmatch(evidence.get("record_locator") or "")
            if not match:
                continue
            start = int(match.group(1))
            end = int(match.group(2) or start)
            covered.setdefault(evidence["source_relative_path"], []).append((start, end))
    return covered


def prepare_queue(
    workspace: Path,
    candidates: list[dict[str, Any]],
    episode_root: Path,
    limit: int = 24,
) -> dict[str, Any]:
    coverage = covered_locators(episode_root)
    chat_cache: dict[str, list[dict[str, Any]]] = {}
    items: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.get("decision_subject") != "self":
            continue
        if candidate.get("replay_readiness_score", 0) < 8:
            continue
        source = candidate["source_relative_path"]
        anchor_id = candidate["anchor_local_id"]
        is_covered = any(start <= anchor_id <= end for start, end in coverage.get(source, []))
        if is_covered:
            continue
        messages = chat_cache.setdefault(source, parse_chat(workspace / source))
        try:
            anchor_index = next(
                index for index, message in enumerate(messages) if message.get("localId") == anchor_id
            )
        except StopIteration:
            continue
        anchor_time = parse_chat_time(messages[anchor_index])
        if anchor_time is None:
            continue
        prior = []
        for message in messages[max(0, anchor_index - 16) : anchor_index]:
            message_time = parse_chat_time(message)
            if message_time and (anchor_time - message_time).total_seconds() <= 60 * 60:
                prior.append(context_item(message))
        following = []
        for message in messages[anchor_index + 1 : anchor_index + 31]:
            message_time = parse_chat_time(message)
            if message_time and (message_time - anchor_time).total_seconds() <= 24 * 60 * 60:
                following.append(context_item(message))
        items.append(
            {
                "candidate_id": candidate["candidate_id"],
                "source_relative_path": source,
                "anchor_local_id": anchor_id,
                "anchor_time": candidate["anchor_time"],
                "inferred_domain": candidate["inferred_domain"],
                "heuristic_score": candidate["replay_readiness_score"],
                "pre_context": prior,
                "anchor": context_item(messages[anchor_index]),
                "post_context": following,
            }
        )
        if len(items) >= limit:
            break
    return {
        "schema_version": 1,
        "privacy": "local_sensitive",
        "purpose": "candidate_triage_not_model_input",
        "items": items,
    }


def build_triage_prompt(
    queue: dict[str, Any], validation_errors: list[str] | None = None
) -> str:
    candidate_ids = [item["candidate_id"] for item in queue["items"]]
    lines = [
            "你是数字镜像历史回放数据集的审稿器，不是在预测事件结果。",
            "你可以查看候选事件的前文、决策锚点和后文，只判断该候选能否制作成严格时间截断的测验。",
            "replay_ready 要求：锚点确实是本人判断，锚点前有足够情境，后文有与同一决策直接相关的实际行动证据。",
            "judgement_only：判断明确，但行动没有被直接观察到。",
            "needs_more_context：可能有价值，但当前窗口不足以确定情境或行动。",
            "reject：询问别人、替别人建议、玩笑、回顾、已完成动作、过于琐碎或并非真实选择。",
            "不要把普通聊天继续发生当成行动证据；行动必须和锚点所述选择语义一致。",
            "suggested_cutoff_local_id 应是锚点之前最后一个仍可作为模型输入的消息；无法确定则为 null。",
            f"reviews 必须恰好返回这些 candidate_id，每个一次：{', '.join(candidate_ids)}。",
            "只输出符合给定 JSON Schema 的 JSON，不输出 Markdown。",
        ]
    if validation_errors:
        lines.append("上一次输出未通过校验，请只修正这些问题：")
        lines.extend(f"- {error}" for error in validation_errors)
    lines.append("\n候选队列：\n" + json.dumps(queue, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines)


def validate_reviews(queue: dict[str, Any], result: dict[str, Any]) -> list[str]:
    expected = {item["candidate_id"] for item in queue["items"]}
    reviews = result.get("reviews", [])
    actual = [review.get("candidate_id") for review in reviews]
    errors: list[str] = []
    if set(actual) != expected or len(actual) != len(expected):
        errors.append("reviews must contain each queued candidate exactly once")
    for review in reviews:
        cutoff = review.get("suggested_cutoff_local_id")
        item = next((x for x in queue["items"] if x["candidate_id"] == review.get("candidate_id")), None)
        if item and cutoff is not None and cutoff >= item["anchor_local_id"]:
            errors.append(f"cutoff must be before anchor: {review.get('candidate_id')}")
    return errors


def normalize_review_for_item(
    item: dict[str, Any], review: dict[str, Any]
) -> dict[str, Any]:
    """Conservatively downgrade a review that proposes a leaky cutoff."""
    normalized = dict(review)
    normalized["candidate_id"] = item["candidate_id"]
    cutoff = normalized.get("suggested_cutoff_local_id")
    if cutoff is not None and cutoff >= item["anchor_local_id"]:
        normalized["suggested_cutoff_local_id"] = None
        normalized["has_pre_cutoff_context"] = False
        if normalized.get("disposition") == "replay_ready":
            normalized["disposition"] = "needs_more_context"
        normalized["reason"] = (
            str(normalized.get("reason", ""))
            + "；模型建议的截止点会包含答案，已自动置空并降级。"
        ).lstrip("；")
        normalized["confidence"] = min(float(normalized.get("confidence", 0)), 0.5)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--workspace", type=Path, default=Path.cwd())
    prepare.add_argument("--candidates", type=Path, required=True)
    prepare.add_argument("--episode-root", type=Path, required=True)
    prepare.add_argument("--limit", type=int, default=24)
    prepare.add_argument("--output", type=Path, required=True)
    review = subparsers.add_parser("review")
    review.add_argument("queue", type=Path)
    review.add_argument("--schema", type=Path, default=Path("schemas/candidate_triage.schema.json"))
    review.add_argument("--model", default="qwen3.5:9b")
    review.add_argument("--temperature", type=float, default=0.1)
    review.add_argument("--seed", type=int, default=20260719)
    review.add_argument("--timeout-seconds", type=int, default=600)
    review.add_argument("--batch-size", type=int, default=4)
    review.add_argument("--max-format-retries", type=int, default=2)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--metadata-output", type=Path)
    args = parser.parse_args()

    if args.command == "prepare":
        workspace = args.workspace.resolve()
        queue = prepare_queue(
            workspace,
            load_jsonl(args.candidates),
            workspace / args.episode_root,
            args.limit,
        )
        write_json(args.output, queue)
        print(f"WROTE: {args.output} ({len(queue['items'])} candidates)")
        return

    queue = load_json(args.queue)
    schema = load_json(args.schema)
    generator = ollama_generator(args.model, args.temperature, args.seed, args.timeout_seconds)
    all_reviews: list[dict[str, Any]] = []
    items = queue["items"]
    for start in range(0, len(items), args.batch_size):
        batch = dict(queue)
        batch["items"] = items[start : start + args.batch_size]
        errors: list[str] | None = None
        batch_result: dict[str, Any] | None = None
        for _ in range(args.max_format_retries + 1):
            raw = generator(build_triage_prompt(batch, errors), schema)
            try:
                batch_result = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors = [f"response is not valid JSON: {exc.msg}"]
                continue
            errors = validate_reviews(batch, batch_result)
            if not errors:
                break
        available: dict[str, dict[str, Any]] = {}
        if batch_result is not None:
            expected_ids = {item["candidate_id"] for item in batch["items"]}
            for review_item in batch_result.get("reviews", []):
                candidate_id = review_item.get("candidate_id")
                if candidate_id in expected_ids and candidate_id not in available:
                    available[candidate_id] = review_item
        for item in batch["items"]:
            candidate_id = item["candidate_id"]
            review_item = available.get(candidate_id)
            if review_item is None:
                single = dict(queue)
                single["items"] = [item]
                raw = generator(
                    build_triage_prompt(
                        single,
                        ["上一批遗漏了该候选；本次只返回这一条。"],
                    ),
                    schema,
                )
                single_result = json.loads(raw)
                review_item = next(
                    (
                        value
                        for value in single_result.get("reviews", [])
                        if value.get("candidate_id") == candidate_id
                    ),
                    None,
                )
            if review_item is None:
                review_item = {
                    "candidate_id": candidate_id,
                    "disposition": "needs_more_context",
                    "is_personal_decision": True,
                    "has_pre_cutoff_context": False,
                    "judgement_observable": False,
                    "action_observable": "uncertain",
                    "judgement_action_gap_possible": False,
                    "suggested_cutoff_local_id": None,
                    "reason": "本地模型未返回可校验结果，必须人工复核。",
                    "confidence": 0.0,
                }
            all_reviews.append(normalize_review_for_item(item, review_item))
        print(
            f"REVIEWED: {min(start + args.batch_size, len(items))}/{len(items)}",
            flush=True,
        )
    result = {"schema_version": 1, "reviews": all_reviews}
    final_errors = validate_reviews(queue, result)
    if final_errors:
        raise ValueError("; ".join(final_errors))
    write_json(args.output, result)
    if args.metadata_output:
        write_json(
            args.metadata_output,
            {
                "model": args.model,
                "runtime": "ollama-local",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "temperature": args.temperature,
                "seed": args.seed,
                "cloud_upload": False,
                "queue_path": args.queue.as_posix(),
                "batch_size": args.batch_size,
                "batch_count": (len(items) + args.batch_size - 1) // args.batch_size,
            },
        )
    print(f"WROTE: {args.output} ({len(result['reviews'])} reviews)")


if __name__ == "__main__":
    main()
