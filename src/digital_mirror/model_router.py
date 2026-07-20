"""Model routing and evidence-bounded historical question answering."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


AnswerMode = Literal["grounded", "counterfactual"]


class ModelUnavailableError(ValueError):
    """Raised when the requested provider has not been configured."""


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str
    provider: str
    model: str
    available: bool
    local: bool
    note: str

    def public(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "label": self.label,
            "provider": self.provider,
            "model": self.model,
            "available": self.available,
            "local": self.local,
            "note": self.note,
        }


def model_specs() -> list[ModelSpec]:
    specs = [
        ModelSpec(
            model_id="evidence-synthesis",
            label="本地证据骨架",
            provider="builtin",
            model="deterministic",
            available=True,
            local=True,
            note="不调用大模型；用于显示证据边界和待回答部分。",
        )
    ]
    deepseek_model = os.environ.get("DIGITAL_MIRROR_DEEPSEEK_MODEL", "deepseek-chat")
    specs.append(
        ModelSpec(
            model_id="deepseek",
            label="DeepSeek",
            provider="deepseek",
            model=deepseek_model,
            available=bool(os.environ.get("DEEPSEEK_API_KEY")),
            local=False,
            note="使用时期材料生成结构化回答。",
        )
    )
    openai_model = os.environ.get("DIGITAL_MIRROR_OPENAI_MODEL", "")
    specs.append(
        ModelSpec(
            model_id="openai",
            label="OpenAI",
            provider="openai-compatible",
            model=openai_model or "未指定",
            available=bool(os.environ.get("OPENAI_API_KEY") and openai_model),
            local=False,
            note="需同时配置 OPENAI_API_KEY 与 DIGITAL_MIRROR_OPENAI_MODEL。",
        )
    )
    compatible_model = os.environ.get("DIGITAL_MIRROR_COMPATIBLE_MODEL", "")
    compatible_url = os.environ.get("DIGITAL_MIRROR_COMPATIBLE_BASE_URL", "")
    specs.append(
        ModelSpec(
            model_id="compatible",
            label="OpenAI 兼容模型",
            provider="openai-compatible",
            model=compatible_model or "未指定",
            available=bool(
                os.environ.get("DIGITAL_MIRROR_COMPATIBLE_API_KEY")
                and compatible_model
                and compatible_url
            ),
            local=False,
            note="可接入通义、Kimi 等提供 OpenAI 兼容接口的模型。",
        )
    )
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    gemini_model = os.environ.get(
        "DIGITAL_MIRROR_GEMINI_MODEL", "gemini-2.5-flash"
    )
    specs.append(
        ModelSpec(
            model_id="gemini",
            label="Google Gemini",
            provider="google",
            model=gemini_model,
            available=bool(gemini_key),
            local=False,
            note="通过 Google Gemini generateContent 接口生成时期化回答。",
        )
    )
    ollama_model = os.environ.get("DIGITAL_MIRROR_OLLAMA_MODEL", "")
    specs.append(
        ModelSpec(
            model_id="ollama",
            label="Ollama 本地模型",
            provider="ollama",
            model=ollama_model or "未指定",
            available=bool(ollama_model),
            local=True,
            note="默认连接本机 Ollama；是否已下载模型在调用时确认。",
        )
    )
    return specs


def model_spec(model_id: str) -> ModelSpec | None:
    return next((item for item in model_specs() if item.model_id == model_id), None)


def build_historical_prompt(
    profile: dict[str, Any],
    period: dict[str, Any],
    question: str,
    mode: AnswerMode,
    history: list[dict[str, str]],
) -> str:
    context = {
        "person": {
            "name": profile["name"],
            "life": profile["life"],
            "field": profile["field"],
            "method_note": profile.get("method_note"),
        },
        "selected_period": period,
        "conversation": history,
        "question": question,
        "mode": mode,
    }
    return (
        "你是历史研究界面中的证据约束回答器，不是通灵者，也不是真人复活。\n"
        "只以所给人物和所选时期作答，不得借用人物后来才知道的结局。\n"
        "可以回答天马行空的反事实问题，但必须把史料支持与创造性推演分开。\n"
        "不要模仿口音，不要声称知道未留下记录的真实内心。\n"
        "输出一个 JSON 对象，字段为 answer、supported_claims、speculative_claims、"
        "unknowns、evidence_ids、follow_up_questions。每个列表最多 5 项。\n\n"
        + json.dumps(context, ensure_ascii=False, sort_keys=True)
    )


def _json_request(
    url: str,
    payload: dict[str, Any],
    api_key: str | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        if "generativelanguage.googleapis.com" in url:
            headers["x-goog-api-key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError("model request failed") from exc


def _parse_json_content(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response must be an object")
    return value


def _builtin_answer(
    profile: dict[str, Any], period: dict[str, Any], question: str, mode: AnswerMode
) -> dict[str, Any]:
    anchors = list(period.get("anchors") or [])[:3]
    evidence = list(period.get("evidence") or [])
    answer = (
        f"你问的是：{question}。在{profile['name']}的“{period['label']}”时期，"
        f"现有材料首先支持这样的背景：{period.get('context', '')}"
    )
    if mode == "counterfactual":
        answer += " 这是反事实提问；本地模式先搭出证据边界，接入大模型后再生成带标注的创造性推演。"
    else:
        answer += " 本地模式不补写人物未曾留下的内心独白。"
    return {
        "answer": answer,
        "supported_claims": anchors,
        "speculative_claims": [],
        "unknowns": list(period.get("unknowns") or [])[:4]
        + ["仅凭当前时期材料，无法确认人物会如何回答这一全新问题。"],
        "evidence_ids": [
            item.get("evidence_id", "") for item in evidence[:4] if item.get("evidence_id")
        ],
        "follow_up_questions": [
            "要把问题限制在这个时期的哪一年？",
            "要更保守的史料回答，还是明确标注的反事实推演？",
        ],
    }


def answer_historical_question(
    profile: dict[str, Any],
    period: dict[str, Any],
    question: str,
    model_id: str,
    mode: AnswerMode,
    history: list[dict[str, str]],
    request: Callable[[str, dict[str, Any], str | None, int], dict[str, Any]] = _json_request,
) -> dict[str, Any]:
    spec = model_spec(model_id)
    if not spec or not spec.available:
        raise ModelUnavailableError("model is not available")
    if model_id == "evidence-synthesis":
        result = _builtin_answer(profile, period, question, mode)
    else:
        prompt = build_historical_prompt(profile, period, question, mode, history)
        if model_id == "deepseek":
            response = request(
                "https://api.deepseek.com/chat/completions",
                {
                    "model": spec.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2 if mode == "grounded" else 0.75,
                },
                os.environ.get("DEEPSEEK_API_KEY"),
                120,
            )
            result = _parse_json_content(response["choices"][0]["message"]["content"])
        elif model_id in {"openai", "compatible"}:
            if model_id == "openai":
                base_url = os.environ.get("DIGITAL_MIRROR_OPENAI_BASE_URL", "https://api.openai.com/v1")
                api_key = os.environ.get("OPENAI_API_KEY")
            else:
                base_url = os.environ["DIGITAL_MIRROR_COMPATIBLE_BASE_URL"]
                api_key = os.environ.get("DIGITAL_MIRROR_COMPATIBLE_API_KEY")
            response = request(
                base_url.rstrip("/") + "/chat/completions",
                {
                    "model": spec.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2 if mode == "grounded" else 0.75,
                },
                api_key,
                120,
            )
            result = _parse_json_content(response["choices"][0]["message"]["content"])
        elif model_id == "gemini":
            gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
                "GOOGLE_API_KEY"
            )
            response = request(
                "https://generativelanguage.googleapis.com/v1beta/models/"
                + spec.model
                + ":generateContent",
                {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.2 if mode == "grounded" else 0.75,
                        "responseMimeType": "application/json",
                    },
                },
                gemini_key,
                120,
            )
            result = _parse_json_content(
                response["candidates"][0]["content"]["parts"][0]["text"]
            )
        else:
            response = request(
                os.environ.get("DIGITAL_MIRROR_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
                + "/api/chat",
                {
                    "model": spec.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.2 if mode == "grounded" else 0.75},
                },
                None,
                180,
            )
            result = _parse_json_content(response["message"]["content"])

    period_anchors = [str(item) for item in period.get("anchors") or []][:5]
    model_supported = [str(item) for item in result.get("supported_claims") or []][:5]
    model_speculative = [str(item) for item in result.get("speculative_claims") or []][:5]
    speculative_claims: list[str] = []
    for item in model_supported + model_speculative:
        if item not in period_anchors and item not in speculative_claims:
            speculative_claims.append(item)
    allowed_evidence_ids = {
        str(item.get("evidence_id"))
        for item in period.get("evidence") or []
        if item.get("evidence_id")
    }
    evidence_ids = [
        str(item)
        for item in result.get("evidence_ids") or []
        if str(item) in allowed_evidence_ids
    ]
    if not evidence_ids:
        evidence_ids = sorted(allowed_evidence_ids)[:8]

    return {
        "figure_id": profile["figure_id"],
        "figure_name": profile["name"],
        "period_id": period["period_id"],
        "period_label": period["label"],
        "model": spec.public(),
        "mode": mode,
        "answer": str(result.get("answer") or ""),
        "supported_claims": period_anchors,
        "speculative_claims": speculative_claims[:5],
        "unknowns": [str(item) for item in result.get("unknowns") or []][:5],
        "evidence_ids": evidence_ids,
        "follow_up_questions": [str(item) for item in result.get("follow_up_questions") or []][:5],
        "boundary_note": "回答属于时期化证据推演，不代表人物真实说过这些话，也不等于其内心真值。",
    }
