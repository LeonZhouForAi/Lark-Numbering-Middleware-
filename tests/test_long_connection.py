from __future__ import annotations

from feishu_rag.long_connection import handle_message_event
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
