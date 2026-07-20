"""Build an auditable snapshot of the private personal mirror."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Iterable


DOMAIN_LABELS = {
    "career": "职业",
    "finance": "财务",
    "health": "健康",
    "learning": "学习",
    "relationship": "关系",
}


def _ratio(numerator: int, denominator: int) -> float | None:
    if not denominator:
        return None
    return round(numerator / denominator, 2)


def _confidence(sample_count: int) -> dict[str, Any]:
    score = min(0.85, round(0.2 + sample_count * 0.08, 2))
    if sample_count < 4:
        label = "样本不足"
    elif sample_count < 10:
        label = "探索性"
    else:
        label = "初步稳定"
    return {
        "label": label,
        "score": score,
        "note": f"基于 {sample_count} 个已整理决策切片；新增样本可能改变当前判断。",
    }


def _signal(
    signal_id: str,
    label: str,
    value: str,
    note: str,
    support: str,
) -> dict[str, str]:
    return {
        "signal_id": signal_id,
        "label": label,
        "value": value,
        "note": note,
        "support": support,
    }


def build_personal_snapshot(
    events: Iterable[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    """Aggregate private episodes without copying source paths or raw text."""

    pairs = list(events)
    domain_counts: Counter[str] = Counter()
    evidence_count = 0
    judgements = 0
    observed_actions = 0
    aligned_actions = 0
    pivots = 0
    outcome_count = 0
    deliberation_values: list[float] = []
    option_counts: list[int] = []
    dated_events: list[tuple[datetime, dict[str, Any]]] = []
    tension_rows: list[dict[str, str]] = []

    for episode, case in pairs:
        domain = str(case.get("domain") or "unknown")
        domain_counts[domain] += 1
        evidence_count += len(case.get("evidence") or [])
        labels = episode.get("labels") or {}
        deliberation = labels.get("contemporaneous_deliberation") or {}
        judgement = (labels.get("actual_judgement") or {}).get("option_id")
        action = (labels.get("actual_action") or {}).get("option_id")
        if judgement:
            judgements += 1
        if action:
            observed_actions += 1
            if judgement == action:
                aligned_actions += 1
            elif judgement:
                pivots += 1
        if labels.get("outcome"):
            outcome_count += 1
        intensity = deliberation.get("deliberation_intensity")
        if isinstance(intensity, (int, float)):
            deliberation_values.append(float(intensity))
        considered = deliberation.get("considered_option_ids") or []
        if considered:
            option_counts.append(len(considered))
        for tension in (deliberation.get("tensions") or [])[:2]:
            tension_rows.append(
                {
                    "domain": DOMAIN_LABELS.get(domain, domain),
                    "text": str(tension),
                    "episode_id": str(case.get("episode_id") or ""),
                }
            )
        try:
            occurred_at = datetime.fromisoformat(str(case["cutoff_at"]))
        except (KeyError, TypeError, ValueError):
            continue
        dated_events.append(
            (
                occurred_at,
                {
                    "episode_id": str(case.get("episode_id") or ""),
                    "domain": domain,
                    "domain_label": DOMAIN_LABELS.get(domain, domain),
                    "date": occurred_at.date().isoformat(),
                    "question": str(case.get("question") or ""),
                },
            )
        )

    event_count = len(pairs)
    average_deliberation = (
        round(sum(deliberation_values) / len(deliberation_values), 2)
        if deliberation_values
        else None
    )
    average_options = (
        round(sum(option_counts) / len(option_counts), 1) if option_counts else None
    )
    alignment = _ratio(aligned_actions, observed_actions)
    action_coverage = _ratio(observed_actions, judgements)

    if alignment is None:
        alignment_signal = _signal(
            "alignment",
            "判断 → 行动",
            "等待样本",
            "尚无同时记录判断与行动的切片。",
            "0 个可比较事件",
        )
    elif alignment >= 0.75:
        alignment_signal = _signal(
            "alignment",
            "判断 → 行动",
            f"{round(alignment * 100)}% 一致",
            "在已经观察到行动的样本中，多数最初判断最终得到执行。",
            f"{aligned_actions}/{observed_actions} 个可比较事件",
        )
    else:
        alignment_signal = _signal(
            "alignment",
            "判断 → 行动",
            f"{round(alignment * 100)}% 一致",
            "判断与执行之间存在较多重新评估，行动预测需要单独建模。",
            f"{aligned_actions}/{observed_actions} 个可比较事件",
        )

    if average_deliberation is None:
        deliberation_note = "还没有可用的审议强度标注。"
        deliberation_value = "等待样本"
    elif average_deliberation < 0.4:
        deliberation_note = "现有样本里多数决策收敛较快，但不能据此推断所有场景。"
        deliberation_value = f"{round(average_deliberation * 100)} / 100"
    elif average_deliberation < 0.65:
        deliberation_note = "现有样本表现为会权衡多个选项，但通常能在中等强度内收敛。"
        deliberation_value = f"{round(average_deliberation * 100)} / 100"
    else:
        deliberation_note = "现有样本包含较多高强度权衡，需要继续区分领域和状态。"
        deliberation_value = f"{round(average_deliberation * 100)} / 100"

    dated_events.sort(key=lambda item: item[0], reverse=True)
    dates = [item[0].date().isoformat() for item in dated_events]
    domain_rows = [
        {
            "domain": domain,
            "label": DOMAIN_LABELS.get(domain, domain),
            "count": count,
            "share": round(count / event_count, 2) if event_count else 0.0,
        }
        for domain, count in sorted(
            domain_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]

    return {
        "mirror_id": "personal",
        "title": "你的决策镜像",
        "subtitle": "从真实选择里缓慢长出来，而不是从人格标签里一次生成",
        "status": "growing" if event_count else "empty",
        "confidence": _confidence(event_count),
        "coverage": {
            "event_count": event_count,
            "domain_count": len(domain_counts),
            "evidence_count": evidence_count,
            "judgement_count": judgements,
            "observed_action_count": observed_actions,
            "outcome_count": outcome_count,
            "date_from": min(dates) if dates else None,
            "date_to": max(dates) if dates else None,
        },
        "decision_metrics": {
            "average_deliberation": average_deliberation,
            "average_considered_options": average_options,
            "judgement_action_alignment": alignment,
            "action_observation_rate": action_coverage,
            "pivot_count": pivots,
        },
        "signals": [
            alignment_signal,
            _signal(
                "deliberation",
                "权衡强度",
                deliberation_value,
                deliberation_note,
                f"{len(deliberation_values)} 个强度标注",
            ),
            _signal(
                "option-breadth",
                "选项广度",
                f"平均 {average_options} 项" if average_options is not None else "等待样本",
                "记录的是当时认真考虑过的选项，不等同于犹豫或拖延。",
                f"{len(option_counts)} 个有选项记录的事件",
            ),
            _signal(
                "revisions",
                "临场改道",
                f"{pivots} 次",
                "判断和行动不一致的事件会被保留，而不是被平均值抹掉。",
                f"{observed_actions} 个已观察行动",
            ),
        ],
        "domains": domain_rows,
        "active_tensions": tension_rows[:8],
        "recent_events": [item[1] for item in dated_events[:6]],
        "data_boundary": {
            "private": True,
            "historical_profiles_used": False,
            "raw_source_text_exposed": False,
            "note": "只聚合私人事件中的结构化标签，不向公开人物接口或浏览器暴露源文件路径。",
        },
    }
