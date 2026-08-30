from __future__ import annotations

import time
import json

import pytest

from feishu_rag import feishu_client as feishu_client_module
from feishu_rag.feishu_client import FeishuAPIError, FeishuClient
from feishu_rag.retry import RetryPolicy


class SequenceTransport:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, method, url, headers, payload, timeout):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _client(outcomes, delays):
    transport = SequenceTransport(outcomes)
    client = FeishuClient(
        "app",
        "secret",
        transport=transport,
        retry_policy=RetryPolicy(sleep=delays.append),
    )
    return client, transport


def _cache_token(client: FeishuClient) -> None:
    client._tenant_token = "cached-token"
    client._tenant_token_expires_at = time.time() + 3600


def test_tenant_token_retries_network_429_and_5xx() -> None:
    delays = []
    success = b'{"code":0,"tenant_access_token":"token","expire":7200}'
    client, transport = _client(
        [OSError("secret"), (429, b"{}"), (200, success)], delays
    )

    assert client.tenant_access_token() == "token"
    assert transport.calls == 3
    assert delays == [0.5, 1.0]


@pytest.mark.parametrize(
    "expire",
    [None, "not-an-integer", True, 0, 7 * 24 * 3600 + 1, 10**400],
)
def test_tenant_token_rejects_invalid_expire_as_feishu_api_error(expire) -> None:
    delays = []
    body = json.dumps(
        {"code": 0, "tenant_access_token": "token", "expire": expire}
    ).encode()
    client, transport = _client([(200, body)], delays)

    with pytest.raises(FeishuAPIError) as error:
        client.tenant_access_token()

    assert type(error.value) is FeishuAPIError
    assert client._tenant_token == ""
    assert transport.calls == 1


def test_tenant_token_accepts_normal_expire(monkeypatch) -> None:
    delays = []
    body = b'{"code":0,"tenant_access_token":"token","expire":7200}'
    client, transport = _client([(200, body)], delays)
    monkeypatch.setattr(feishu_client_module.time, "time", lambda: 1000.0)

    assert client.tenant_access_token() == "token"
    assert client._tenant_token_expires_at == 8200.0
    assert transport.calls == 1


@pytest.mark.parametrize(
    ("operation", "success"),
    [
        (
            lambda client: client.list_wiki_nodes("space"),
            b'{"code":0,"data":{"items":[],"has_more":false}}',
        ),
        (
            lambda client: client.get_wiki_node("space", "node"),
            b'{"code":0,"data":{}}',
        ),
        (
            lambda client: client.get_document_raw_content("doc"),
            b'{"code":0,"data":{"content":"ok"}}',
        ),
        (lambda client: client.download_file("file"), b"file-bytes"),
    ],
)
def test_idempotent_reads_retry_retryable_failures(operation, success: bytes) -> None:
    delays = []
    client, transport = _client([(503, b"{}"), (200, success)], delays)
    _cache_token(client)

    operation(client)

    assert transport.calls == 2
    assert delays == [0.5]


def test_reply_text_never_retries_even_on_retryable_status() -> None:
    delays = []
    client, transport = _client([(503, b"{}")], delays)
    _cache_token(client)

    with pytest.raises(FeishuAPIError):
        client.reply_text("message", "answer")

    assert transport.calls == 1
    assert delays == []


def test_reply_text_wraps_token_failure_as_definitely_not_sent() -> None:
    delays = []
    client, transport = _client([(400, b"{}")], delays)

    with pytest.raises(FeishuAPIError) as error:
        client.reply_text("message", "answer")

    assert type(error.value).__name__ == "FeishuReplyNotSentError"
    assert transport.calls == 1


def test_feishu_network_error_does_not_leak_transport_detail() -> None:
    delays = []
    client, transport = _client([OSError("secret payload")] * 3, delays)
    _cache_token(client)

    with pytest.raises(FeishuAPIError) as error:
        client.get_document_raw_content("doc")

    assert transport.calls == 3
    assert "secret payload" not in str(error.value)
