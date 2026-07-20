"""Public, evidence-linked catalog of isolated historical figure mirrors."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


FIGURE_ID_RE = re.compile(r"^[a-z0-9-]{1,80}$")
PERIOD_ID_RE = re.compile(r"^[a-z0-9-]{1,120}$")


def catalog_root() -> Path:
    return Path(__file__).with_name("historical_figures")


def load_catalog(root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load one self-contained profile per directory.

    Keeping every figure in a separate directory makes the isolation boundary
    visible on disk as well as in the public API.
    """

    root = root or catalog_root()
    figures: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return figures
    for profile_path in sorted(root.glob("*/profile.json")):
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        figure_id = profile.get("figure_id")
        if not isinstance(figure_id, str) or not FIGURE_ID_RE.fullmatch(figure_id):
            continue
        if profile_path.parent.name != figure_id or figure_id in figures:
            continue
        profile["periods"] = periods_for(profile)
        figures[figure_id] = profile
    return figures


def periods_for(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Return explicit life periods, or derive a safe period from each slice."""

    explicit = profile.get("periods")
    if isinstance(explicit, list) and explicit:
        return [
            period
            for period in explicit
            if isinstance(period, dict)
            and isinstance(period.get("period_id"), str)
            and PERIOD_ID_RE.fullmatch(period["period_id"])
        ]

    derived: list[dict[str, Any]] = []
    for slice_ in profile.get("slices") or []:
        slice_id = slice_.get("slice_id")
        if not isinstance(slice_id, str) or not PERIOD_ID_RE.fullmatch(slice_id):
            continue
        derived.append(
            {
                "period_id": slice_id,
                "label": slice_.get("date_label") or slice_.get("date") or "历史切片",
                "range": slice_.get("date_label") or slice_.get("date") or "",
                "headline": slice_.get("title") or "决策切片",
                "context": slice_.get("context") or "",
                "anchors": list(slice_.get("known_at_cutoff") or []),
                "unknowns": list(slice_.get("unknown_at_cutoff") or []),
                "evidence": list(slice_.get("evidence") or []),
                "source_slice_ids": [slice_id],
            }
        )
    return derived


def period_from_profile(
    profile: dict[str, Any], period_id: str
) -> dict[str, Any] | None:
    if not PERIOD_ID_RE.fullmatch(period_id):
        return None
    return next(
        (
            period
            for period in profile.get("periods") or []
            if period.get("period_id") == period_id
        ),
        None,
    )


def public_summary(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        key: profile[key]
        for key in (
            "figure_id",
            "name",
            "native_name",
            "life",
            "field",
            "tagline",
            "accent",
            "archive_scale",
            "status",
        )
    } | {
        "slice_count": len(profile.get("slices", [])),
        "period_count": len(profile.get("periods", [])),
    }
