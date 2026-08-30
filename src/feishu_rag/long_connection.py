"""通过飞书长连接接收机器人消息，无需公网 Webhook。"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from typing import Any

from .config import Settings
from .feishu_client import FeishuClient
from .llm import DeepSeekClient
from .logging_utils import configure_logging
from .rag import RagService
from .store import IndexStore
from .web import handle_event


logger = logging.getLogger(__name__)


def handle_message_event(event: Mapping[str, Any], rag: RagService, feishu: FeishuClient) -> dict[str, str]:
    """适配 SDK 长连接事件到既有的消息处理逻辑。"""
    return handle_event(
        {"header": {"event_type": "im.message.receive_v1"}, "event": dict(event)},
        rag,
        feishu,
        verification_token="",
    )


def _safe_handle_message(event: Mapping[str, Any], rag: RagService, feishu: FeishuClient) -> dict[str, str]:
    try:
        return handle_message_event(event, rag, feishu)
    except Exception as exc:
        logger.warning("message_handler_failed error_type=%s", type(exc).__name__)
        return {"status": "error"}


def _safe_handle_raw_message(
    data: Any,
    rag: RagService,
    feishu: FeishuClient,
    marshal: Callable[[Any], str],
) -> dict[str, str]:
    try:
        raw = json.loads(marshal(data))
        event = raw.get("event", raw) if isinstance(raw, dict) else {}
        return _safe_handle_message(event, rag, feishu)
    except Exception as exc:
        logger.warning("message_handler_failed error_type=%s", type(exc).__name__)
        return {"status": "error"}


def run() -> None:
    """建立并保持飞书长连接。"""
    import lark_oapi as lark

    settings = Settings.from_env()
    configure_logging(settings.log_level)
    store = IndexStore(settings.rag_db_path)
    rag = RagService(
        store,
        DeepSeekClient(settings.deepseek_api_key, settings.deepseek_base_url, settings.deepseek_model),
        settings.rag_top_k,
    )
    feishu = FeishuClient(settings.feishu_app_id, settings.feishu_app_secret)

    def on_message(data: Any) -> None:
        _safe_handle_raw_message(data, rag, feishu, lark.JSON.marshal)

    event_handler = lark.EventDispatcherHandler.builder("", "").register_p2_im_message_receive_v1(on_message).build()
    lark.ws.Client(settings.feishu_app_id, settings.feishu_app_secret, event_handler=event_handler).start()


if __name__ == "__main__":
    run()
