"""Run a consent-gated, de-identified historical replay baseline via DeepSeek API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .local_baseline import build_prompt, run_with_retries
from .replay import load_json, write_json


FORBIDDEN_KEYS = {
    "labels",
    "ai_references",
    "source_relative_path",
    "source_sha256",
    "record_locator",
    "outcome",
    "hindsight",
}
SENSITIVE_PATTERNS = (
    re.compile(r"/Users/|(?:^|[\s\"'])iCloud/|(?:^|[\s\"'])texts/"),
    re.compile(r"wxid_", re.I),
    re.compile(r"(?i)[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}"),
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)


def walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in walk_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in walk_keys(item)}
    return set()


def cloud_safety_errors(case: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    forbidden = sorted(walk_keys(case) & FORBIDDEN_KEYS)
    if forbidden:
        errors.append(f"forbidden fields present: {forbidden}")
    serialized = json.dumps(case, ensure_ascii=False, sort_keys=True)
    for index, pattern in enumerate(SENSITIVE_PATTERNS):
        if pattern.search(serialized):
            errors.append(f"sensitive pattern {index} matched")
    if case.get("personal_history"):
        errors.append("personal_history is not allowed in the first cloud benchmark")
    return errors


def case_sha256(case: dict[str, Any]) -> str:
    payload = json.dumps(case, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def curl_json_request(
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    """Use system curl without exposing the key or request body in process arguments."""
    with tempfile.TemporaryDirectory(prefix="digital-mirror-deepseek-") as temp_dir:
        root = Path(temp_dir)
        body_path = root / "request.json"
        response_path = root / "response.json"
        config_path = root / "curl.conf"
        body_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        config_path.write_text(
            "\n".join(
                [
                    f'url = "{url}"',
                    'request = "POST"',
                    f'header = "Authorization: Bearer {api_key}"',
                    'header = "Content-Type: application/json"',
                    f'data-binary = "@{body_path}"',
                    f'output = "{response_path}"',
                    "silent",
                    "show-error",
                    "fail-with-body",
                    "http1.1",
                    f"max-time = {timeout_seconds}",
                    "connect-timeout = 30",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        os.chmod(body_path, 0o600)
        os.chmod(config_path, 0o600)
        completed = subprocess.run(
            ["curl", "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 10,
            check=False,
        )
        if completed.returncode != 0:
            detail = response_path.read_text(encoding="utf-8") if response_path.exists() else ""
            raise RuntimeError(
                f"DeepSeek API request failed ({completed.returncode}): "
                f"{completed.stderr.strip()} {detail[:500]}"
            )
        return json.loads(response_path.read_text(encoding="utf-8"))


def deepseek_generator(
    model: str,
    thinking: str,
    reasoning_effort: str,
    temperature: float,
    timeout_seconds: int,
    request: Callable[[str, str, dict[str, Any], int], dict[str, Any]] = curl_json_request,
) -> Callable[[str, dict[str, Any]], str]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    def generate(prompt: str, schema: dict[str, Any]) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                    + "\n\n输出必须符合以下 JSON Schema：\n"
                    + json.dumps(schema, ensure_ascii=False, sort_keys=True),
                }
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": thinking},
            "max_tokens": 8000,
        }
        if thinking == "enabled":
            payload["reasoning_effort"] = reasoning_effort
        else:
            payload["temperature"] = temperature
        response = request(
            "https://api.deepseek.com/chat/completions",
            api_key,
            payload,
            timeout_seconds,
        )
        return response["choices"][0]["message"]["content"]

    return generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--schema", type=Path, default=Path("schemas/replay_prediction.schema.json"))
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--thinking", choices=["enabled", "disabled"], required=True)
    parser.add_argument("--reasoning-effort", choices=["high", "max"], default="high")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--max-format-retries", type=int, default=2)
    parser.add_argument("--allow-cloud-upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    args = parser.parse_args()

    case = load_json(args.case)
    errors = cloud_safety_errors(case)
    if errors:
        raise SystemExit("cloud safety check failed: " + "; ".join(errors))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "safe": True,
                    "case_sha256": case_sha256(case),
                    "episode_id": case["episode_id"],
                    "personal_history_included": False,
                },
                ensure_ascii=False,
            )
        )
        return
    if not args.allow_cloud_upload:
        raise SystemExit("refusing cloud request without --allow-cloud-upload")
    if not args.output:
        raise SystemExit("--output is required unless --dry-run is used")

    schema = load_json(args.schema)
    prediction, retry_count = run_with_retries(
        case,
        deepseek_generator(
            args.model,
            args.thinking,
            args.reasoning_effort,
            args.temperature,
            args.timeout_seconds,
        ),
        schema,
        args.max_format_retries,
    )
    write_json(args.output, prediction)
    if args.metadata_output:
        write_json(
            args.metadata_output,
            {
                "provider": "deepseek",
                "model": args.model,
                "thinking": args.thinking,
                "reasoning_effort": args.reasoning_effort if args.thinking == "enabled" else None,
                "temperature": args.temperature if args.thinking == "disabled" else None,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "cloud_upload": True,
                "upload_scope": "deidentified_compiled_replay_case_only",
                "personal_history_included": False,
                "case_sha256": case_sha256(case),
                "format_retry_count": retry_count,
            },
        )
    print(f"WROTE: {args.output}")


if __name__ == "__main__":
    main()
