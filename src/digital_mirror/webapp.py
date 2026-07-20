"""Authenticated web playground for historical digital-mirror replays."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .cloud_baseline import cloud_safety_errors, deepseek_generator
from .historical_catalog import FIGURE_ID_RE, load_catalog, public_summary
from .local_baseline import run_with_retries
from .replay import compile_replay_case, load_json, score_prediction


LOGGER = logging.getLogger(__name__)
EPISODE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")
SESSION_COOKIE = "digital_mirror_session"
PredictionRunner = Callable[
    [dict[str, Any], dict[str, Any], Literal["enabled", "disabled"]],
    tuple[dict[str, Any], int],
]


@dataclass(frozen=True)
class Settings:
    episode_root: Path
    historical_root: Path
    schema_path: Path
    access_password: str
    session_secret: str
    cookie_secure: bool = True
    session_ttl_seconds: int = 12 * 60 * 60
    prediction_limit_per_hour: int = 12

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            episode_root=Path(
                os.environ.get(
                    "DIGITAL_MIRROR_EPISODE_ROOT", "data/episodes/private"
                )
            ),
            historical_root=Path(
                os.environ.get(
                    "DIGITAL_MIRROR_HISTORICAL_ROOT",
                    str(Path(__file__).with_name("historical_figures")),
                )
            ),
            schema_path=Path(
                os.environ.get(
                    "DIGITAL_MIRROR_SCHEMA",
                    "schemas/replay_prediction.schema.json",
                )
            ),
            access_password=os.environ.get("DIGITAL_MIRROR_ACCESS_PASSWORD", ""),
            session_secret=os.environ.get("DIGITAL_MIRROR_SESSION_SECRET", ""),
            cookie_secure=os.environ.get(
                "DIGITAL_MIRROR_COOKIE_SECURE", "true"
            ).lower()
            not in {"0", "false", "no"},
            session_ttl_seconds=int(
                os.environ.get("DIGITAL_MIRROR_SESSION_TTL_SECONDS", str(12 * 60 * 60))
            ),
            prediction_limit_per_hour=int(
                os.environ.get("DIGITAL_MIRROR_PREDICTION_LIMIT_PER_HOUR", "12")
            ),
        )

    @property
    def configured(self) -> bool:
        return bool(self.access_password and self.session_secret)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    password: str = Field(min_length=1, max_length=256)


class AuthStatus(BaseModel):
    authenticated: bool
    configured: bool = True


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    configured: bool
    event_count: int
    figure_count: int
    model_available: bool


class PublicFigureListResponse(BaseModel):
    figures: list[dict[str, Any]]


class OptionView(BaseModel):
    option_id: str
    description: str


class EvidenceView(BaseModel):
    evidence_id: str
    observed_at: str
    author_role: str
    summary: str


class PersonaStateView(BaseModel):
    era: str | None = None
    machine_mode_version: str | None = None
    relationship_state: str | None = None
    stress_state: str | None = None
    resource_state: str | None = None


class EventSummary(BaseModel):
    episode_id: str
    domain: str
    cutoff_at: str
    question: str
    era: str | None = None
    machine_mode_version: str | None = None


class EventDetail(EventSummary):
    task_mode: str
    options: list[OptionView]
    persona_state: PersonaStateView
    evidence: list[EvidenceView]


class EventListResponse(BaseModel):
    events: list[EventSummary]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episode_id: str = Field(min_length=1, max_length=160, pattern=EPISODE_ID_RE.pattern)
    thinking: Literal["enabled", "disabled"] = "enabled"


class PredictionView(BaseModel):
    episode_id: str
    predicted_judgement: str
    predicted_action: str
    option_probabilities: dict[str, float]
    considered_option_ids: list[str]
    tensions: list[str]
    unknowns: list[str]
    deliberation_intensity: float
    confidence: float
    evidence_ids: list[str]


class ActualView(BaseModel):
    judgement: str | None
    action: str | None
    deliberation_intensity: float | None
    considered_option_ids: list[str]
    tensions: list[str]
    unknowns: list[str]


class ScoreView(BaseModel):
    judgement_correct: bool | None
    action_correct: bool | None
    judgement_brier_score: float | None
    deliberation_intensity_absolute_error: float | None


class PredictionResponse(BaseModel):
    model: str
    thinking: Literal["enabled", "disabled"]
    generated_at: str
    prediction: PredictionView
    actual: ActualView
    score: ScoreView


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self._entries: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            entries = self._entries[key]
            while entries and now - entries[0] > self.window_seconds:
                entries.popleft()
            if len(entries) >= self.limit:
                return False
            entries.append(now)
            return True


class SessionSigner:
    def __init__(self, secret: str, ttl_seconds: int):
        self._secret = secret.encode("utf-8")
        self._ttl_seconds = ttl_seconds

    def issue(self) -> str:
        payload = f"{int(time.time())}:{secrets.token_urlsafe(18)}".encode("utf-8")
        encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
        signature = hmac.new(self._secret, encoded.encode("ascii"), hashlib.sha256).hexdigest()
        return f"{encoded}.{signature}"

    def valid(self, token: str | None) -> bool:
        if not token or "." not in token or not self._secret:
            return False
        encoded, signature = token.rsplit(".", 1)
        expected = hmac.new(
            self._secret, encoded.encode("ascii"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return False
        try:
            padded = encoded + "=" * (-len(encoded) % 4)
            issued_at = int(
                base64.urlsafe_b64decode(padded.encode("ascii"))
                .decode("utf-8")
                .split(":", 1)[0]
            )
        except (ValueError, UnicodeDecodeError):
            return False
        age = int(time.time()) - issued_at
        return 0 <= age <= self._ttl_seconds


def client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def episode_map(root: Path) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    events: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    if not root.is_dir():
        return events
    for path in sorted(root.glob("*.json")):
        try:
            episode = load_json(path)
            case = compile_replay_case(episode)
        except (OSError, ValueError, KeyError, TypeError):
            LOGGER.warning("Skipping invalid episode file: %s", path.name)
            continue
        episode_id = case["episode_id"]
        if EPISODE_ID_RE.fullmatch(episode_id):
            events[episode_id] = (episode, case)
    return events


def summary_from_case(case: dict[str, Any]) -> EventSummary:
    persona = case.get("persona_state") or {}
    return EventSummary(
        episode_id=case["episode_id"],
        domain=case["domain"],
        cutoff_at=case["cutoff_at"],
        question=case["question"],
        era=persona.get("era"),
        machine_mode_version=persona.get("machine_mode_version"),
    )


def detail_from_case(case: dict[str, Any]) -> EventDetail:
    summary = summary_from_case(case)
    return EventDetail(
        **summary.model_dump(),
        task_mode=case["task_mode"],
        options=[OptionView.model_validate(item) for item in case["options"]],
        persona_state=PersonaStateView.model_validate(case.get("persona_state") or {}),
        evidence=[EvidenceView.model_validate(item) for item in case["evidence"]],
    )


def default_prediction_runner(
    case: dict[str, Any],
    schema: dict[str, Any],
    thinking: Literal["enabled", "disabled"],
) -> tuple[dict[str, Any], int]:
    return run_with_retries(
        case,
        deepseek_generator(
            model="deepseek-v4-pro",
            thinking=thinking,
            reasoning_effort="high",
            temperature=0.2,
            timeout_seconds=600,
        ),
        schema,
        2,
    )


def create_app(
    settings: Settings | None = None,
    prediction_runner: PredictionRunner | None = None,
    static_root: Path | None = None,
) -> FastAPI:
    settings = settings or Settings.from_environment()
    prediction_runner = prediction_runner or default_prediction_runner
    static_root = static_root or Path(__file__).with_name("web_static")
    signer = SessionSigner(settings.session_secret, settings.session_ttl_seconds)
    login_limiter = SlidingWindowLimiter(limit=10, window_seconds=10 * 60)
    prediction_limiter = SlidingWindowLimiter(
        limit=settings.prediction_limit_per_hour, window_seconds=60 * 60
    )
    prediction_lock = threading.Lock()

    app = FastAPI(
        title="Mirror Atlas",
        version="0.3.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.mount("/assets", StaticFiles(directory=static_root), name="assets")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
            "base-uri 'self'; form-action 'self'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = (
                "public, max-age=300"
                if request.url.path.startswith("/api/public/")
                else "no-store"
            )
        return response

    def require_session(request: Request) -> bool:
        if not settings.configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="服务尚未配置访问凭据",
            )
        if not signer.valid(request.cookies.get(SESSION_COOKIE)):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="需要登录",
            )
        return True

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            configured=settings.configured,
            event_count=len(episode_map(settings.episode_root)),
            figure_count=len(load_catalog(settings.historical_root)),
            model_available=bool(os.environ.get("DEEPSEEK_API_KEY")),
        )

    @app.get("/api/public/figures", response_model=PublicFigureListResponse)
    def list_public_figures() -> PublicFigureListResponse:
        figures = [
            public_summary(profile)
            for profile in load_catalog(settings.historical_root).values()
        ]
        return PublicFigureListResponse(figures=figures)

    @app.get("/api/public/figures/{figure_id}", response_model=dict[str, Any])
    def get_public_figure(figure_id: str) -> dict[str, Any]:
        if not FIGURE_ID_RE.fullmatch(figure_id):
            raise HTTPException(status_code=404, detail="人物不存在")
        profile = load_catalog(settings.historical_root).get(figure_id)
        if not profile:
            raise HTTPException(status_code=404, detail="人物不存在")
        return profile

    @app.get("/api/session", response_model=AuthStatus)
    def session_status(request: Request) -> AuthStatus:
        return AuthStatus(
            authenticated=settings.configured
            and signer.valid(request.cookies.get(SESSION_COOKIE)),
            configured=settings.configured,
        )

    @app.post("/api/login", response_model=AuthStatus)
    def login(payload: LoginRequest, request: Request, response: Response) -> AuthStatus:
        if not settings.configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="服务尚未配置访问凭据",
            )
        if not login_limiter.allow(client_key(request)):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="尝试次数过多，请稍后再试",
            )
        if not hmac.compare_digest(payload.password, settings.access_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="访问密码错误",
            )
        response.set_cookie(
            SESSION_COOKIE,
            signer.issue(),
            max_age=settings.session_ttl_seconds,
            httponly=True,
            secure=settings.cookie_secure,
            samesite="strict",
            path="/",
        )
        return AuthStatus(authenticated=True)

    @app.post(
        "/api/logout",
        response_model=AuthStatus,
        dependencies=[Depends(require_session)],
    )
    def logout(response: Response) -> AuthStatus:
        response.delete_cookie(SESSION_COOKIE, path="/")
        return AuthStatus(authenticated=False)

    @app.get(
        "/api/events",
        response_model=EventListResponse,
        dependencies=[Depends(require_session)],
    )
    def list_events() -> EventListResponse:
        events = [summary_from_case(case) for _, case in episode_map(settings.episode_root).values()]
        events.sort(key=lambda item: item.cutoff_at, reverse=True)
        return EventListResponse(events=events)

    @app.get(
        "/api/events/{episode_id}",
        response_model=EventDetail,
        dependencies=[Depends(require_session)],
    )
    def get_event(episode_id: str) -> EventDetail:
        if not EPISODE_ID_RE.fullmatch(episode_id):
            raise HTTPException(status_code=404, detail="事件不存在")
        pair = episode_map(settings.episode_root).get(episode_id)
        if not pair:
            raise HTTPException(status_code=404, detail="事件不存在")
        return detail_from_case(pair[1])

    @app.post(
        "/api/predict",
        response_model=PredictionResponse,
        dependencies=[Depends(require_session)],
    )
    def predict(
        payload: PredictionRequest,
        request: Request,
    ) -> PredictionResponse:
        pair = episode_map(settings.episode_root).get(payload.episode_id)
        if not pair:
            raise HTTPException(status_code=404, detail="事件不存在")
        episode, case = pair
        safety_errors = cloud_safety_errors(case)
        if safety_errors:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="该事件没有通过云端脱敏检查",
            )
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="模型服务尚未配置",
            )
        if not prediction_limiter.allow(client_key(request)):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="本小时预测次数已达上限",
            )
        if not prediction_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="已有预测正在运行，请稍后再试",
            )
        try:
            schema = load_json(settings.schema_path)
            prediction, _ = prediction_runner(case, schema, payload.thinking)
            score = score_prediction(episode, prediction, case)
        except Exception as exc:  # Provider details must not reach the browser.
            LOGGER.error("Prediction failed: %s", type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="模型调用失败，请稍后重试",
            ) from exc
        finally:
            prediction_lock.release()

        labels = episode["labels"]
        deliberation = labels["contemporaneous_deliberation"]
        return PredictionResponse(
            model="deepseek-v4-pro",
            thinking=payload.thinking,
            generated_at=datetime.now(timezone.utc).isoformat(),
            prediction=PredictionView.model_validate(prediction),
            actual=ActualView(
                judgement=labels["actual_judgement"]["option_id"],
                action=labels["actual_action"]["option_id"],
                deliberation_intensity=deliberation.get("deliberation_intensity"),
                considered_option_ids=deliberation.get("considered_option_ids", []),
                tensions=deliberation.get("tensions", []),
                unknowns=deliberation.get("unknowns", []),
            ),
            score=ScoreView(
                judgement_correct=score["judgement_correct"],
                action_correct=score["action_correct"],
                judgement_brier_score=score["judgement_brier_score"],
                deliberation_intensity_absolute_error=score[
                    "deliberation_intensity_absolute_error"
                ],
            ),
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_root / "index.html")

    @app.get("/{path:path}", include_in_schema=False)
    def spa_fallback(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="接口不存在")
        return FileResponse(static_root / "index.html")

    return app


app = create_app()
