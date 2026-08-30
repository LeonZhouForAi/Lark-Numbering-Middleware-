"""FastAPI Webhook：接收飞书机器人消息并返回 RAG 答案。"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

from .config import ConfigError, Settings
from .feishu_client import FeishuClient
from .llm import DeepSeekClient
from .rag import RagService
from .store import IndexStore

try:
    from fastapi import FastAPI, HTTPException, Request
except ImportError:  # pragma: no cover - 本地无 FastAPI 时核心逻辑仍可测试
    FastAPI = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    Request = Any  # type: ignore[misc,assignment]


def verify_signature(timestamp: str, nonce: str, body: str, encrypt_key: str, signature: str) -> bool:
    if not encrypt_key:
        return True
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
    claim_message = getattr(store, "claim_message", None)
    release_message = getattr(store, "release_message", None)
    claimed = False
    if callable(claim_message):
        if not claim_message(message_id):
            return {"status": "duplicate"}
        claimed = True
    try:
        answer = rag.answer(question)
        feishu.reply_text(message_id, answer.text)
    except Exception:
        if claimed and callable(release_message):
            release_message(message_id)
        raise
    return {"status": "ok"}


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
            llm = DeepSeekClient(settings.deepseek_api_key, settings.deepseek_base_url, settings.deepseek_model)
            rag = rag or RagService(store, llm, settings.rag_top_k)
            feishu = feishu or FeishuClient(settings.feishu_app_id, settings.feishu_app_secret)
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
            return handle_event(payload, rag, feishu, verification_token)  # type: ignore[arg-type]
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
