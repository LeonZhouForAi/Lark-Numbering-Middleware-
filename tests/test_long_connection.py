from __future__ import annotations

import logging

from feishu_rag.long_connection import _safe_handle_raw_message, _safe_handle_message, handle_message_event
from feishu_rag.rag import RagAnswer


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


def test_safe_raw_message_handler_redacts_marshal_failure(caplog) -> None:
    sensitive_value = "不应写入日志的消息正文"

    def failing_marshal(data: object) -> str:
        raise RuntimeError(sensitive_value)

    with caplog.at_level(logging.WARNING, logger="feishu_rag.long_connection"):
        result = _safe_handle_raw_message({"message": sensitive_value}, FakeRag(), FakeFeishu(), failing_marshal)

    assert result == {"status": "error"}
    assert "message_handler_failed" in caplog.text
    assert "RuntimeError" in caplog.text
    assert sensitive_value not in caplog.text
