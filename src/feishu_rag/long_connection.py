"""通过飞书长连接接收机器人消息，无需公网 Webhook。"""

from __future__ import annotations

import asyncio
import http
import json
import logging
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
from typing import Any

from lark_oapi.core.const import UTF_8
from lark_oapi.core.json import JSON as LarkJSON
from lark_oapi.ws.client import Client as LarkWsClient
from lark_oapi.ws.client import _get_by_key
from lark_oapi.ws.const import (
    HEADER_BIZ_RT,
    HEADER_MESSAGE_ID,
    HEADER_SEQ,
    HEADER_SUM,
    HEADER_TRACE_ID,
    HEADER_TYPE,
)
from lark_oapi.ws.enum import MessageType
from lark_oapi.ws.model import Response
from lark_oapi.ws.pb.pbbp2_pb2 import Frame

from .config import Settings
from .faq import FaqService
from .feishu_client import FeishuClient
from .llm import DeepSeekClient
from .logging_utils import configure_logging
from .rag import RagService
from .retry import RetryPolicy
from .store import IndexStore
from .web import handle_event


logger = logging.getLogger(__name__)

ResourceFactory = Callable[[], tuple[IndexStore, RagService]]


class MessageProcessingError(RuntimeError):
    """消息处理失败；异常文本不得包含员工问题或凭据。"""


class AckAfterProcessingClient(LarkWsClient):
    """在工作线程完成事件处理后才向飞书确认消息。

    此适配器集中依赖 ``lark-oapi==1.7.3`` 的私有长连接契约。升级 SDK
    前必须重新核对 ``Client._handle_data_frame``；项目不会修改 site-packages。
    """

    def __init__(
        self,
        *args: Any,
        worker_threads: int,
        max_pending_messages: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._message_executor = ThreadPoolExecutor(max_workers=worker_threads)
        self._pending_slots = BoundedSemaphore(max_pending_messages)

    async def _handle_data_frame(self, frame: Frame) -> None:
        headers = frame.headers
        message_type = MessageType(_get_by_key(headers, HEADER_TYPE))
        if message_type != MessageType.EVENT:
            await super()._handle_data_frame(frame)
            return

        message_id = _get_by_key(headers, HEADER_MESSAGE_ID)
        _get_by_key(headers, HEADER_TRACE_ID)
        part_count = int(_get_by_key(headers, HEADER_SUM))
        sequence = int(_get_by_key(headers, HEADER_SEQ))

        payload = frame.payload
        if part_count > 1:
            payload = self._combine(message_id, part_count, sequence, payload)
            if payload is None:
                return

        started_at = int(round(time.time() * 1000))
        response = Response(code=http.HTTPStatus.OK)
        if not self._pending_slots.acquire(blocking=False):
            logger.warning("message_queue_full")
            response = Response(code=http.HTTPStatus.SERVICE_UNAVAILABLE)
        else:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    self._message_executor,
                    self._event_handler._do_without_validation,
                    payload,
                )
            except Exception as exc:
                logger.warning(
                    "message_handler_failed error_type=%s", type(exc).__name__
                )
                response = Response(code=http.HTTPStatus.INTERNAL_SERVER_ERROR)
            finally:
                self._pending_slots.release()

        completed_at = int(round(time.time() * 1000))
        biz_runtime = headers.add()
        biz_runtime.key = HEADER_BIZ_RT
        biz_runtime.value = str(completed_at - started_at)

        # Event ACKs intentionally contain only the status code. Handler return values
        # can include employee input and must never be reflected into the wire payload.
        frame.payload = LarkJSON.marshal(response).encode(UTF_8)
        await self._write_message(frame.SerializeToString())

    def shutdown(self) -> None:
        """等待所有已接收事件结束并关闭适配器线程池。"""
        self._message_executor.shutdown(wait=True, cancel_futures=False)


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


def _create_message_resources(
    settings: Settings,
) -> tuple[IndexStore, RagService]:
    store = IndexStore(settings.rag_db_path)
    try:
        retry_policy = RetryPolicy(
            max_attempts=settings.api_retry_max_attempts,
            base_delay=settings.api_retry_base_delay,
        )
        rag = RagService(
            store,
            DeepSeekClient(
                settings.deepseek_api_key,
                settings.deepseek_base_url,
                settings.deepseek_model,
                retry_policy=retry_policy,
                usage_sink=store,
            ),
            top_k=settings.rag_top_k,
            min_relevance=settings.rag_min_relevance,
            question_max_chars=settings.rag_question_max_chars,
            rate_limit_per_minute=settings.rag_rate_limit_per_minute,
            rate_limit_per_day=settings.rag_rate_limit_per_day,
            faq_service=FaqService(
                store,
                settings.rag_faq_enabled,
                settings.rag_faq_promotion_count,
                settings.rag_faq_window_days,
                settings.rag_faq_min_text_similarity,
                settings.rag_faq_min_source_overlap,
            ),
        )
        return store, rag
    except Exception:
        store.close()
        raise


def _handle_marshaled_message(
    raw_message: str,
    resource_factory: ResourceFactory,
    feishu: FeishuClient,
) -> None:
    store: IndexStore | None = None
    try:
        raw = json.loads(raw_message)
        event = raw.get("event", raw) if isinstance(raw, dict) else {}
        store, rag = resource_factory()
        result = _safe_handle_message(event, rag, feishu)
        if result.get("status") in {"error", "in_progress"}:
            raise MessageProcessingError("message processing failed")
    except MessageProcessingError:
        raise
    except Exception as exc:
        logger.warning("message_handler_failed error_type=%s", type(exc).__name__)
        raise MessageProcessingError("message processing failed") from None
    finally:
        if store is not None:
            try:
                store.close()
            except Exception as exc:
                logger.warning(
                    "message_store_close_failed error_type=%s", type(exc).__name__
                )


def run() -> None:
    """建立并保持飞书长连接。"""
    import lark_oapi as lark

    settings = Settings.from_env()
    configure_logging(settings.log_level)
    feishu = FeishuClient(
        settings.feishu_app_id,
        settings.feishu_app_secret,
        retry_policy=RetryPolicy(
            max_attempts=settings.api_retry_max_attempts,
            base_delay=settings.api_retry_base_delay,
        ),
    )

    def resource_factory() -> tuple[IndexStore, RagService]:
        return _create_message_resources(settings)

    def on_message(data: Any) -> None:
        try:
            raw_message = lark.JSON.marshal(data)
        except Exception as exc:
            logger.warning(
                "message_handler_failed phase=marshal error_type=%s",
                type(exc).__name__,
            )
            raise MessageProcessingError("message processing failed") from None
        _handle_marshaled_message(raw_message, resource_factory, feishu)

    event_handler = lark.EventDispatcherHandler.builder("", "").register_p2_im_message_receive_v1(on_message).build()
    client = AckAfterProcessingClient(
        settings.feishu_app_id,
        settings.feishu_app_secret,
        event_handler=event_handler,
        worker_threads=settings.rag_worker_threads,
        max_pending_messages=settings.rag_max_pending_messages,
    )
    try:
        client.start()
    finally:
        client.shutdown()


if __name__ == "__main__":
    run()
