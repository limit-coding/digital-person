"""Run a reproducible leak-free replay baseline against a local Ollama model."""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .episode import EpisodeValidationError
from .replay import load_json, validate_prediction, write_json


Generator = Callable[[str, dict[str, Any]], str]


def build_prompt(case: dict[str, Any], format_errors: list[str] | None = None) -> str:
    option_ids = [option["option_id"] for option in case["options"]]
    constraints = [
        "你正在进行严格的历史回放盲测。",
        "只使用输入 JSON 中截止时间前的证据，不假设后来发生了什么。",
        "只输出一个符合 JSON Schema 的 JSON 对象，不要输出 Markdown 或额外解释。",
        f"predicted_judgement 和 predicted_action 必须是以下 option_id 之一：{', '.join(option_ids)}。",
        f"option_probabilities 必须恰好包含这些 option_id 且总和为 1：{', '.join(option_ids)}。",
        "tensions 与 unknowns 分别给出至少两条简短描述。",
    ]
    if format_errors:
        constraints.append("上一次输出只有格式问题；请修正以下问题，不改变证据边界：")
        constraints.extend(f"- {error}" for error in format_errors)
    return "\n".join(constraints) + "\n\n历史回放输入：\n" + json.dumps(
        case, ensure_ascii=False, sort_keys=True
    )


def run_with_retries(
    case: dict[str, Any],
    generate: Generator,
    schema: dict[str, Any],
    max_format_retries: int = 2,
) -> tuple[dict[str, Any], int]:
    errors: list[str] | None = None
    for retry in range(max_format_retries + 1):
        raw = generate(build_prompt(case, errors), schema)
        try:
            prediction = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors = [f"response is not valid JSON: {exc.msg}"]
            continue
        errors = validate_prediction(case, prediction)
        if not errors:
            return prediction, retry
    raise EpisodeValidationError(
        "local model did not produce a valid prediction after format retries: "
        + "; ".join(errors or ["unknown formatting failure"])
    )


def ollama_generator(
    model: str,
    temperature: float,
    seed: int,
    timeout_seconds: int,
) -> Generator:
    def generate(prompt: str, schema: dict[str, Any]) -> str:
        body = {
            "model": model,
            "prompt": prompt,
            "format": schema,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "seed": seed},
        }
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result = json.load(response)
        return result["response"]

    return generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--schema", type=Path, default=Path("schemas/replay_prediction.schema.json"))
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--max-format-retries", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    args = parser.parse_args()

    case = load_json(args.case)
    schema = load_json(args.schema)
    prediction, retry_count = run_with_retries(
        case,
        ollama_generator(args.model, args.temperature, args.seed, args.timeout_seconds),
        schema,
        args.max_format_retries,
    )
    write_json(args.output, prediction)
    if args.metadata_output:
        write_json(
            args.metadata_output,
            {
                "model": args.model,
                "runtime": "ollama-local",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "temperature": args.temperature,
                "seed": args.seed,
                "thinking": False,
                "cloud_upload": False,
                "format_retry_count": retry_count,
                "case_path": args.case.as_posix(),
            },
        )
    print(f"WROTE: {args.output}")


if __name__ == "__main__":
    main()
