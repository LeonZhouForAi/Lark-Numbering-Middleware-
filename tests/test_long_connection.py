from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from lark_oapi.ws.const import (
    HEADER_BIZ_RT,
    HEADER_MESSAGE_ID,
    HEADER_SEQ,
    HEADER_SUM,
    HEADER_TRACE_ID,
    HEADER_TYPE,
)
from lark_oapi.ws.enum import FrameType, MessageType
from lark_oapi.ws.pb.pbbp2_pb2 import Frame

from feishu_rag import long_connection
from feishu_rag.config import Settings
from feishu_rag.feishu_client import FeishuClient
from feishu_rag.long_connection import _safe_handle_message, handle_message_event
from feishu_rag.rag import RagAnswer
from feishu_rag.retry import RetryPolicy


class FakeRag:
    def __init__(self) -> None:
        self.question = ""

    def answer(self, question: str) -> RagAnswer:
        self.question = question
        return RagAnswer(text="请按现行流程提交", citations=[])


class FakeFeishu:
    def __init__(self) -> None:
        self.reply: tuple[str, str] | None = None

    def reply_text(self, message_id: str, text: str) -> None:
        self.reply = (message_id, text)


class FailingRag:
    def answer(self, question: str) -> RagAnswer:
        raise RuntimeError("model unavailable")


class ClosingStore:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _raw_event(message_id: str) -> dict:
    return {
        "event": {
            "message": {
                "message_id": message_id,
                "message_type": "text",
                "content": json.dumps({"text": "报销怎么走"}, ensure_ascii=False),
            },
            "sender": {"sender_type": "user"},
        }
    }


def _event_frame(payload: bytes = b'{"sensitive":"question body"}') -> Frame:
    frame = Frame()
    frame.SeqID = 0
    frame.LogID = 0
    frame.service = 1
    frame.method = FrameType.DATA.value
    frame.payload = payload
    for key, value in (
        (HEADER_MESSAGE_ID, "message-id"),
        (HEADER_TRACE_ID, "trace-id"),
        (HEADER_SUM, "1"),
        (HEADER_SEQ, "0"),
        (HEADER_TYPE, MessageType.EVENT.value),
    ):
        header = frame.headers.add()
        header.key = key
        header.value = value
    return frame


def _ack_code(serialized_frame: bytes) -> tuple[int, bytes, Frame]:
    response_frame = Frame()
    response_frame.ParseFromString(serialized_frame)
    response = json.loads(response_frame.payload.decode("utf-8"))
    return response["code"], response_frame.payload, response_frame


class BlockingEventHandler:
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self.started = started
        self.release = release

    def _do_without_validation(self, payload: bytes) -> None:
        self.started.set()
        assert self.release.wait(timeout=2)


def _ack_client(
    handler: object,
    *,
    worker_threads: int = 1,
    max_pending_messages: int = 1,
) -> long_connection.AckAfterProcessingClient:
    """Create the adapter without initializing the SDK's process-global loop."""
    with patch.object(long_connection.LarkWsClient, "__init__", return_value=None):
        client = long_connection.AckAfterProcessingClient(
            "app",
            "secret",
            event_handler=handler,
            worker_threads=worker_threads,
            max_pending_messages=max_pending_messages,
        )
    client._event_handler = handler
    return client


@pytest.mark.asyncio
async def test_ack_waits_for_processing_without_blocking_ws_event_loop() -> None:
    started = threading.Event()
    release = threading.Event()
    handler = BlockingEventHandler(started, release)
    client = _ack_client(handler)
    client._write_message = AsyncMock()

    try:
        task = asyncio.create_task(client._handle_data_frame(_event_frame()))
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.001)
        assert started.is_set()
        assert not task.done()
        client._write_message.assert_not_awaited()

        # If RAG ran on the websocket loop, this sleep could not complete.
        await asyncio.sleep(0.01)
        assert not task.done()

        release.set()
        await asyncio.wait_for(task, timeout=1)
    finally:
        release.set()
        client.shutdown()

    client._write_message.assert_awaited_once()
    code, payload, response_frame = _ack_code(
        client._write_message.await_args.args[0]
    )
    assert code == 200
    assert b"question body" not in payload
    assert any(header.key == HEADER_BIZ_RT for header in response_frame.headers)


@pytest.mark.asyncio
async def test_handler_exception_is_acked_as_redacted_500() -> None:
    sensitive = "secret question text"

    class FailingEventHandler:
        def _do_without_validation(self, payload: bytes) -> None:
            raise RuntimeError(sensitive)

    client = _ack_client(FailingEventHandler())
    client._write_message = AsyncMock()
    try:
        await client._handle_data_frame(_event_frame(sensitive.encode()))
    finally:
        client.shutdown()

    code, payload, _ = _ack_code(client._write_message.await_args.args[0])
    assert code == 500
    assert sensitive.encode() not in payload


@pytest.mark.asyncio
async def test_in_progress_redelivery_is_acked_500() -> None:
    class InProgressStore(ClosingStore):
        @staticmethod
        def claim_message_state(message_id: str) -> str:
            return "in_progress"

    stores: list[InProgressStore] = []

    def resources():
        store = InProgressStore()
        stores.append(store)
        rag = FakeRag()
        rag.store = store
        return store, rag

    class EventHandler:
        def _do_without_validation(self, payload: bytes) -> None:
            long_connection._handle_marshaled_message(
                payload.decode("utf-8"), resources, FakeFeishu()
            )

    sensitive = "不应出现在确认帧里的员工问题"
    raw_event = _raw_event("om_in_progress")
    raw_event["event"]["message"]["content"] = json.dumps(
        {"text": sensitive}, ensure_ascii=False
    )
    client = _ack_client(EventHandler())
    client._write_message = AsyncMock()
    try:
        await client._handle_data_frame(
            _event_frame(json.dumps(raw_event, ensure_ascii=False).encode("utf-8"))
        )
    finally:
        client.shutdown()

    code, payload, _ = _ack_code(client._write_message.await_args.args[0])
    assert code == 500
    assert sensitive.encode("utf-8") not in payload
    assert len(stores) == 1
    assert stores[0].closed


@pytest.mark.asyncio
async def test_superseded_worker_is_acked_200_without_reply_or_token() -> None:
    claim_token = "00112233445566778899aabbccddeeff"

    class SupersededStore(ClosingStore):
        @staticmethod
        def claim_message_lease(message_id: str) -> tuple[str, str]:
            return "claimed", claim_token

        @staticmethod
        def begin_message_reply(message_id: str, token: str) -> bool:
            return False

        @staticmethod
        def complete_message(message_id: str, token: str | None = None) -> bool:
            raise AssertionError("superseded worker must not complete a claim")

        @staticmethod
        def release_message(message_id: str, token: str | None = None) -> bool:
            raise AssertionError("superseded worker must not release a claim")

    stores: list[SupersededStore] = []
    feishu = FakeFeishu()

    def resources():
        store = SupersededStore()
        stores.append(store)
        rag = FakeRag()
        rag.store = store
        return store, rag

    class EventHandler:
        def _do_without_validation(self, payload: bytes) -> None:
            long_connection._handle_marshaled_message(
                payload.decode("utf-8"), resources, feishu
            )

    client = _ack_client(EventHandler())
    client._write_message = AsyncMock()
    try:
        await client._handle_data_frame(
            _event_frame(json.dumps(_raw_event("om_superseded_ack")).encode("utf-8"))
        )
    finally:
        client.shutdown()

    code, payload, _ = _ack_code(client._write_message.await_args.args[0])
    assert code == 200
    assert feishu.reply is None
    assert claim_token.encode() not in payload
    assert len(stores) == 1
    assert stores[0].closed


@pytest.mark.asyncio
async def test_capacity_full_is_acked_503_and_recovers_after_completion() -> None:
    started = threading.Event()
    release = threading.Event()
    client = _ack_client(BlockingEventHandler(started, release))
    sent: list[bytes] = []

    async def capture(message: bytes) -> None:
        sent.append(message)

    client._write_message = capture
    try:
        first = asyncio.create_task(client._handle_data_frame(_event_frame(b"first")))
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.001)
        assert started.is_set()

        await client._handle_data_frame(_event_frame(b"second-sensitive"))
        assert len(sent) == 1
        assert _ack_code(sent[0])[0] == 503
        assert b"second-sensitive" not in _ack_code(sent[0])[1]

        release.set()
        await asyncio.wait_for(first, timeout=1)
        assert _ack_code(sent[1])[0] == 200

        # The slot is released only after the accepted handler has completed.
        await client._handle_data_frame(_event_frame(b"third"))
        assert _ack_code(sent[2])[0] == 200
    finally:
        release.set()
        client.shutdown()


def test_long_connection_message_is_answered() -> None:
    rag = FakeRag()
    feishu = FakeFeishu()

    result = handle_message_event(
        {
            "message": {
                "message_id": "om_test",
                "message_type": "text",
                "content": '{"text":"报销怎么走"}',
            },
            "sender": {"sender_type": "user"},
        },
        rag,
        feishu,
    )

    assert result == {"status": "ok"}
    assert rag.question == "报销怎么走"
    assert feishu.reply == ("om_test", "请按现行流程提交")


def test_safe_message_handler_returns_error_and_redacts_question(caplog) -> None:
    question = "这是一条不应写入日志的敏感问题"
    event = {
        "message": {"message_id": "om_test", "message_type": "text", "content": '{"text":"%s"}' % question},
        "sender": {"sender_type": "user"},
    }

    with caplog.at_level(logging.WARNING, logger="feishu_rag.long_connection"):
        result = _safe_handle_message(event, FailingRag(), FakeFeishu())

    assert result == {"status": "error"}
    assert "message_handler_failed" in caplog.text
    assert "RuntimeError" in caplog.text
    assert question not in caplog.text


def test_superseded_result_and_logs_never_expose_claim_token(caplog) -> None:
    claim_token = "fedcba9876543210fedcba9876543210"

    class SupersededStore:
        @staticmethod
        def claim_message_lease(message_id: str) -> tuple[str, str]:
            return "claimed", claim_token

        @staticmethod
        def begin_message_reply(message_id: str, token: str) -> bool:
            return False

        @staticmethod
        def complete_message(message_id: str, token: str | None = None) -> bool:
            raise AssertionError("superseded worker must not complete a claim")

        @staticmethod
        def release_message(message_id: str, token: str | None = None) -> bool:
            raise AssertionError("superseded worker must not release a claim")

    rag = FakeRag()
    rag.store = SupersededStore()
    event = _raw_event("om_superseded")["event"]

    with caplog.at_level(logging.WARNING, logger="feishu_rag.long_connection"):
        result = _safe_handle_message(event, rag, FakeFeishu())

    assert result == {"status": "superseded"}
    assert claim_token not in json.dumps(result)
    assert claim_token not in caplog.text


def test_concurrent_tasks_use_distinct_stores_and_close_each_one() -> None:
    barrier = threading.Barrier(2)
    stores = []
    token_requests = 0
    replies = []
    token_request_lock = threading.Lock()
    second_token_request = threading.Event()

    def transport(method, url, headers, payload, timeout):
        nonlocal token_requests
        if url.endswith("/tenant_access_token/internal"):
            with token_request_lock:
                token_requests += 1
                if token_requests == 2:
                    second_token_request.set()
            second_token_request.wait(timeout=0.2)
            return 200, b'{"code":0,"tenant_access_token":"shared-token","expire":7200}'
        replies.append(url)
        return 200, b'{"code":0}'

    shared_feishu = FeishuClient(
        "app",
        "secret",
        transport=transport,
        retry_policy=RetryPolicy(sleep=lambda _: None),
    )

    class ConcurrentRag:
        def __init__(self, store) -> None:
            self.store = store

        def answer(self, question: str) -> RagAnswer:
            barrier.wait(timeout=2)
            return RagAnswer(text="ok", citations=[])

    def resources():
        store = ClosingStore()
        stores.append(store)
        return store, ConcurrentRag(store)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                long_connection._handle_marshaled_message,
                json.dumps(_raw_event(f"om_{index}"), ensure_ascii=False),
                resources,
                shared_feishu,
            )
            for index in range(2)
        ]
        assert [future.result(timeout=2) for future in futures] == [None, None]

    assert len(stores) == 2
    assert stores[0] is not stores[1]
    assert all(store.closed for store in stores)
    assert token_requests == 1
    assert len(replies) == 2


def test_run_configures_and_shuts_down_ack_adapter() -> None:
    settings = Settings(
        deepseek_api_key="key",
        feishu_app_id="app",
        feishu_app_secret="secret",
        feishu_verification_token="",
        rag_worker_threads=7,
        rag_max_pending_messages=9,
    )
    builder = MagicMock()
    builder.register_p2_im_message_receive_v1.return_value = builder
    handler = object()
    builder.build.return_value = handler
    client = MagicMock()
    fake_lark = SimpleNamespace(
        JSON=SimpleNamespace(marshal=lambda data: "{}"),
        EventDispatcherHandler=SimpleNamespace(builder=MagicMock(return_value=builder)),
    )

    with (
        patch.dict(sys.modules, {"lark_oapi": fake_lark}),
        patch.object(long_connection.Settings, "from_env", return_value=settings),
        patch.object(
            long_connection, "AckAfterProcessingClient", return_value=client
        ) as adapter_type,
    ):
        long_connection.run()

    adapter_type.assert_called_once_with(
        "app",
        "secret",
        event_handler=handler,
        worker_threads=7,
        max_pending_messages=9,
    )
    client.start.assert_called_once_with()
    client.shutdown.assert_called_once_with()
