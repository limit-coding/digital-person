"""Augment a leak-free replay case with only pre-cutoff personal history."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .replay import load_json, write_json
from .retrieval import search


def character_ngrams(text: str, size: int = 3) -> set[str]:
    normalized = "".join(text.split())
    return {
        normalized[index : index + size]
        for index in range(max(0, len(normalized) - size + 1))
    }


def text_similarity(left: str, right: str) -> float:
    left_grams = character_ngrams(left)
    right_grams = character_ngrams(right)
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def windows_overlap(left: str, right: str, threshold: float = 0.5) -> bool:
    pattern = re.compile(r"^window:(\d+)\.\.(\d+)$")
    left_match = pattern.match(left)
    right_match = pattern.match(right)
    if not left_match or not right_match:
        return False
    left_start, left_end = map(int, left_match.groups())
    right_start, right_end = map(int, right_match.groups())
    overlap = max(0, min(left_end, right_end) - max(left_start, right_start) + 1)
    shorter = min(left_end - left_start + 1, right_end - right_start + 1)
    return shorter > 0 and overlap / shorter >= threshold


def retrieve_history(
    database: Path,
    queries: list[str],
    cutoff_at: str,
    per_query: int = 8,
    total_limit: int = 8,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    reciprocal_rank: defaultdict[str, float] = defaultdict(float)
    matched_queries: defaultdict[str, list[str]] = defaultdict(list)
    for query in queries:
        results = search(
            database,
            query,
            limit=per_query,
            allowed_use="behavior_context",
            before=cutoff_at,
        )
        for rank, result in enumerate(results, start=1):
            record_id = result["record_id"]
            if len(result["text"].strip()) < 6:
                continue
            by_id[record_id] = result
            reciprocal_rank[record_id] += 1.0 / (60 + rank)
            matched_queries[record_id].append(query)

    ordered_ids = sorted(
        by_id,
        key=lambda record_id: (
            -(
                reciprocal_rank[record_id]
                + (0.001 if by_id[record_id]["record_kind"] == "chat_window" else 0.0)
            ),
            -len(by_id[record_id]["text"]),
            by_id[record_id]["occurred_at"] or "",
        ),
    )
    selected_ids: list[str] = []
    for record_id in ordered_ids:
        candidate = by_id[record_id]
        redundant = any(
            candidate["source_relative_path"] == by_id[selected]["source_relative_path"]
            and (
                windows_overlap(candidate["locator"], by_id[selected]["locator"])
                or
                candidate["text"] in by_id[selected]["text"]
                or by_id[selected]["text"] in candidate["text"]
                or text_similarity(candidate["text"], by_id[selected]["text"]) >= 0.55
            )
            for selected in selected_ids
        )
        if redundant:
            continue
        selected_ids.append(record_id)
        if len(selected_ids) >= total_limit:
            break
    return [
        {
            "history_id": f"history:{record_id}",
            "occurred_at": by_id[record_id]["occurred_at"],
            "text": by_id[record_id]["text"],
            "source_relative_path": by_id[record_id]["source_relative_path"],
            "locator": by_id[record_id]["locator"],
            "record_kind": by_id[record_id]["record_kind"],
            "matched_queries": matched_queries[record_id],
            "rrf_score": reciprocal_rank[record_id],
        }
        for record_id in selected_ids
    ]


def augment_case(
    case: dict[str, Any],
    database: Path,
    queries: list[str],
    per_query: int = 8,
    total_limit: int = 8,
) -> dict[str, Any]:
    if "labels" in case or "ai_references" in case:
        raise ValueError("personalization requires an already compiled leak-free case")
    history = retrieve_history(
        database,
        queries,
        case["cutoff_at"],
        per_query=per_query,
        total_limit=total_limit,
    )
    augmented = dict(case)
    augmented["personalization_mode"] = "pre_cutoff_self_history_fts"
    augmented["history_queries"] = queries
    augmented["personal_history"] = history
    augmented["instructions"] = list(case["instructions"]) + [
        "personal_history 仅是截止时间前的本人历史，可能相关也可能无关；不得把历史偏好机械套用到当前情境。",
        "不得从 personal_history 推断截止时间后的事件结果。",
    ]
    return augmented


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--per-query", type=int, default=8)
    parser.add_argument("--total-limit", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    augmented = augment_case(
        load_json(args.case),
        args.database,
        args.query,
        per_query=args.per_query,
        total_limit=args.total_limit,
    )
    write_json(args.output, augmented)
    print(f"WROTE: {args.output} ({len(augmented['personal_history'])} history records)")


if __name__ == "__main__":
    main()
