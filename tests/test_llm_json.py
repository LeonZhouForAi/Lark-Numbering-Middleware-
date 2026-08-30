from __future__ import annotations

import pytest

from feishu_rag.llm import DeepSeekClient, DeepSeekError


class FakeTransport:
    def __init__(self, body: bytes = b'{"choices":[{"message":{"content":"{\\"groups\\": []}"}}]}', status: int = 200) -> None:
        self.body = body
        self.status = status
        self.payload: dict[str, object] = {}

    def __call__(self, url, headers, payload, timeout):
        self.payload = payload
        return self.status, self.body


def test_complete_json_requests_json_object_and_parses_response() -> None:
    transport = FakeTransport()
    client = DeepSeekClient("secret", transport=transport)

    result = client.complete_json("system", "user")

    assert result == {"groups": []}
    assert transport.payload["response_format"] == {"type": "json_object"}
    assert transport.payload["temperature"] == 0
    assert transport.payload["thinking"] == {"type": "disabled"}


def test_complete_json_rejects_invalid_json() -> None:
    client = DeepSeekClient("secret", transport=FakeTransport(b"not-json"))

    with pytest.raises(DeepSeekError, match="无法解析"):
        client.complete_json("system", "user")


@pytest.mark.parametrize("status", [401, 429, 500])
def test_complete_json_preserves_http_error_categories(status: int) -> None:
    client = DeepSeekClient("secret", transport=FakeTransport(b"{}", status=status))

    with pytest.raises(DeepSeekError):
        client.complete_json("system", "user")
