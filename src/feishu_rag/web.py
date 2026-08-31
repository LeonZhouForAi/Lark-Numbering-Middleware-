"""FastAPI Webhook：接收飞书机器人消息并返回 RAG 答案。"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

from .config import ConfigError, Settings
from .faq import FaqService
from .feishu_client import FeishuClient, FeishuReplyNotSentError
from .llm import DeepSeekClient
from .rag import RagService
from .retry import RetryPolicy
from .store import IndexStore

try:
    from fastapi import FastAPI, HTTPException, Request
except ImportError:  # pragma: no cover - 本地无 FastAPI 时核心逻辑仍可测试
    FastAPI = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    Request = Any  # type: ignore[misc,assignment]


RATE_LIMIT_ANSWER = "请求过于频繁，请稍后再试。"


def verify_signature(timestamp: str, nonce: str, body: str, encrypt_key: str, signature: str) -> bool:
    if not encrypt_key:
        return False
    expected = hashlib.sha256((timestamp + nonce + encrypt_key + body).encode("utf-8")).hexdigest()
    return hmac.compare_digest(expected, signature)


def _message_text(message: dict[str, Any]) -> str | None:
    if message.get("message_type") != "text":
        return None
    try:
        content = json.loads(message.get("content", "{}"))
    except (TypeError, ValueError):
        return None
    text = content.get("text")
    return text.strip() if isinstance(text, str) and text.strip() else None


def _actor_id(event: dict[str, Any]) -> str | None:
    sender = event.get("sender")
    if not isinstance(sender, dict):
        return None
    sender_id = sender.get("sender_id")
    if not isinstance(sender_id, dict):
        return None
    for key in ("open_id", "union_id", "user_id"):
        value = sender_id.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def handle_event(
    payload: dict[str, Any],
    rag: RagService,
    feishu: FeishuClient,
    verification_token: str,
) -> dict[str, str]:
    if payload.get("type") == "url_verification":
        if payload.get("token") != verification_token:
            raise PermissionError("飞书 URL verification token 不匹配")
        challenge = payload.get("challenge")
        if not isinstance(challenge, str) or not challenge:
            raise ValueError("缺少 challenge")
        return {"challenge": challenge}

    if payload.get("header", {}).get("event_type") != "im.message.receive_v1":
        return {"status": "ignored"}
    event = payload.get("event", {})
    if event.get("sender", {}).get("sender_type") == "app":
        return {"status": "ignored"}
    message = event.get("message", {})
    question = _message_text(message)
    if not question:
        return {"status": "ignored"}
    message_id = message.get("message_id")
    if not isinstance(message_id, str) or not message_id:
        raise ValueError("飞书消息缺少 message_id")
    store = getattr(rag, "store", None)
    claim_message_lease = getattr(store, "claim_message_lease", None)
    claim_message_state = getattr(store, "claim_message_state", None)
    claim_message = getattr(store, "claim_message", None)
    complete_message = getattr(store, "complete_message", None)
    release_message = getattr(store, "release_message", None)
    begin_message_reply = getattr(store, "begin_message_reply", None)
    claimed = False
    owner_token: str | None = None
    token_fenced = False
    if callable(claim_message_lease):
        if not all(
            callable(operation)
            for operation in (
                begin_message_reply,
                complete_message,
                release_message,
            )
        ):
            raise RuntimeError("message lease requires ownership and finalization operations")
        claim_state, owner_token = claim_message_lease(message_id)
        if claim_state == "completed":
            return {"status": "duplicate"}
        if claim_state == "in_progress":
            return {"status": "in_progress"}
        if claim_state != "claimed" or owner_token is None:
            raise RuntimeError("invalid message claim lease")
        claimed = True
        token_fenced = True
    elif callable(claim_message_state):
        claim_state = claim_message_state(message_id)
        if claim_state == "completed":
            return {"status": "duplicate"}
        if claim_state == "in_progress":
            return {"status": "in_progress"}
        if claim_state != "claimed":
            raise RuntimeError("invalid message claim state")
        claimed = True
    elif callable(claim_message):
        if not claim_message(message_id):
            return {"status": "duplicate"}
        claimed = True

    def release_owned_claim() -> None:
        if not claimed or not callable(release_message):
            return
        if token_fenced:
            release_message(message_id, token=owner_token)
        else:
            release_message(message_id)

    def complete_owned_claim() -> None:
        if not claimed or not callable(complete_message):
            return
        if token_fenced:
            complete_message(message_id, token=owner_token)
        else:
            complete_message(message_id)

    per_minute = getattr(rag, "rate_limit_per_minute", 0)
    per_day = getattr(rag, "rate_limit_per_day", 0)
    status = "ok"
    try:
        if per_minute or per_day:
            actor_id = _actor_id(event)
            if actor_id is None:
                raise ValueError("missing actor_id")
            claim_rate_limit = getattr(store, "claim_rate_limit", None)
            if not callable(claim_rate_limit):
                raise RuntimeError("rate limiting requires a persistent store")
            if claim_rate_limit(actor_id, per_minute, per_day):
                answer_text = rag.answer(question).text
            else:
                answer_text = RATE_LIMIT_ANSWER
                status = "rate_limited"
        else:
            answer_text = rag.answer(question).text
    except Exception:
        release_owned_claim()
        raise
    # 回复前用单条 CAS 将租约原子封口；replying 不再允许过期接管。
    # 进程若在封口后崩溃会少回复，但不会让另一工作器重复发送。
    if token_fenced and not begin_message_reply(message_id, owner_token):
        return {"status": "superseded"}
    try:
        feishu.reply_text(message_id, answer_text)
    except FeishuReplyNotSentError:
        release_owned_claim()
        raise
    except Exception:
        complete_owned_claim()
        raise
    complete_owned_claim()
    return {"status": status}


def create_app(
    settings: Settings | None = None,
    rag: RagService | None = None,
    feishu: FeishuClient | None = None,
):
    if FastAPI is None:
        raise RuntimeError("运行 Web 服务需要安装 fastapi")
    app = FastAPI(title="Feishu RAG Bot")
    configured_error: ConfigError | None = None
    if rag is None or feishu is None:
        try:
            settings = settings or Settings.from_env()
            store = IndexStore(settings.rag_db_path)
            retry_policy = RetryPolicy(
                max_attempts=settings.api_retry_max_attempts,
                base_delay=settings.api_retry_base_delay,
            )
            llm = DeepSeekClient(
                settings.deepseek_api_key,
                settings.deepseek_base_url,
                settings.deepseek_model,
                retry_policy=retry_policy,
                usage_sink=store,
            )
            faq_service = FaqService(
                store,
                settings.rag_faq_enabled,
                settings.rag_faq_promotion_count,
                settings.rag_faq_window_days,
                settings.rag_faq_min_text_similarity,
                settings.rag_faq_min_source_overlap,
            )
            rag = rag or RagService(
                store,
                llm,
                top_k=settings.rag_top_k,
                min_relevance=settings.rag_min_relevance,
                question_max_chars=settings.rag_question_max_chars,
                rate_limit_per_minute=settings.rag_rate_limit_per_minute,
                rate_limit_per_day=settings.rag_rate_limit_per_day,
                faq_service=faq_service,
            )
            feishu = feishu or FeishuClient(
                settings.feishu_app_id,
                settings.feishu_app_secret,
                retry_policy=retry_policy,
            )
        except ConfigError as exc:
            configured_error = exc
    verification_token = settings.feishu_verification_token if settings else ""
    encrypt_key = settings.feishu_encrypt_key if settings else ""

    @app.get("/healthz")
    async def healthz():
        if configured_error:
            raise HTTPException(status_code=503, detail="服务配置不完整")
        return {"status": "ok"}

    @app.post("/webhook/feishu")
    async def feishu_webhook(request: Request):
        body = await request.body()
        timestamp = request.headers.get("x-lark-request-timestamp", "")
        nonce = request.headers.get("x-lark-request-nonce", "")
        signature = request.headers.get("x-lark-signature", "")
        if not verify_signature(timestamp, nonce, body.decode("utf-8"), encrypt_key, signature):
            raise HTTPException(status_code=403, detail="签名校验失败")
        try:
            payload = json.loads(body.decode("utf-8"))
            result = handle_event(payload, rag, feishu, verification_token)  # type: ignore[arg-type]
            if result.get("status") == "in_progress":
                raise HTTPException(status_code=503, detail="消息仍在处理中")
            return result
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="请求内容无效") from exc

    return app


if FastAPI is not None:
    try:
        app = create_app()
    except Exception:  # 配置缺失时仍让容器能启动并由 /healthz 报 503
        app = FastAPI(title="Feishu RAG Bot")

        @app.get("/healthz")
        async def unavailable_healthz():
            raise HTTPException(status_code=503, detail="服务配置不完整")
else:  # pragma: no cover
    app = None
