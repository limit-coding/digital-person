"""Public, evidence-linked catalog of isolated historical figure mirrors."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


FIGURE_ID_RE = re.compile(r"^[a-z0-9-]{1,80}$")


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
        figures[figure_id] = profile
    return figures


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
    } | {"slice_count": len(profile.get("slices", []))}
