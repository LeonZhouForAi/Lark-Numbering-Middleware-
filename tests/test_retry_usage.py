from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from feishu_rag.llm import DeepSeekClient, DeepSeekError
from feishu_rag.retry import RetryPolicy
from feishu_rag.store import IndexStore


SUCCESS = b'{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}'


class SequenceTransport:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, url, headers, payload, timeout):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class RecordingUsageSink:
    def __init__(self):
        self.records = []

    def record_llm_usage(
        self, model, purpose, prompt_tokens, completion_tokens, total_tokens
    ):
        self.records.append(
            (model, purpose, prompt_tokens, completion_tokens, total_tokens)
        )


def test_retry_policy_defaults_and_exponential_delays() -> None:
    delays = []
    policy = RetryPolicy(sleep=delays.append)

    assert policy.max_attempts == 3
    assert policy.base_delay == 0.5
    assert policy.max_delay == 4.0
    policy.wait(1)
    policy.wait(2)
    policy.wait(5)
    assert delays == [0.5, 1.0, 4.0]


@pytest.mark.parametrize("status", [429, 500, 503])
def test_deepseek_generation_http_failures_are_single_attempt(
    status: int,
) -> None:
    transport = SequenceTransport([(status, b"{}"), (200, SUCCESS)])
    delays = []
    client = DeepSeekClient(
        "secret",
        transport=transport,
        retry_policy=RetryPolicy(sleep=delays.append),
    )

    with pytest.raises(DeepSeekError):
        client.complete("system", "question")

    assert transport.calls == 1
    assert delays == []


def test_deepseek_unknown_transport_result_is_not_retried_or_counted() -> None:
    transport = SequenceTransport([OSError("secret request body"), (200, SUCCESS)])
    delays = []
    sink = RecordingUsageSink()
    client = DeepSeekClient(
        "secret",
        transport=transport,
        retry_policy=RetryPolicy(sleep=delays.append),
        usage_sink=sink,
    )

    with pytest.raises(DeepSeekError, match="结果不明") as error:
        client.complete("system", "question")

    assert transport.calls == 1
    assert delays == []
    assert sink.records == []
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("status", [400, 401])
def test_deepseek_does_not_retry_permanent_http_errors(status: int) -> None:
    transport = SequenceTransport([(status, b"{}")])
    delays = []
    client = DeepSeekClient(
        "secret",
        transport=transport,
        retry_policy=RetryPolicy(sleep=delays.append),
    )

    with pytest.raises(DeepSeekError):
        client.complete("system", "question")

    assert transport.calls == 1
    assert delays == []


def test_complete_defaults_to_answer_and_records_sanitized_usage() -> None:
    sink = RecordingUsageSink()
    transport = SequenceTransport([(200, SUCCESS)])
    client = DeepSeekClient("secret", transport=transport, usage_sink=sink)

    assert client.complete("private system", "private body") == "ok"

    assert sink.records == [("deepseek-v4-flash", "answer", 7, 3, 10)]


def test_missing_usage_still_records_one_request_with_zero_tokens() -> None:
    sink = RecordingUsageSink()
    body = b'{"choices":[{"message":{"content":"ok"}}]}'
    client = DeepSeekClient("secret", transport=SequenceTransport([(200, body)]), usage_sink=sink)

    client.complete("private system", "private body")

    assert sink.records == [("deepseek-v4-flash", "answer", 0, 0, 0)]


def test_invalid_usage_values_are_never_recorded_as_negative_or_boolean() -> None:
    sink = RecordingUsageSink()
    body = (
        b'{"choices":[{"message":{"content":"ok"}}],'
        b'"usage":{"prompt_tokens":-1,"completion_tokens":true,"total_tokens":-4}}'
    )
    client = DeepSeekClient("secret", transport=SequenceTransport([(200, body)]), usage_sink=sink)

    client.complete("system", "question")

    assert sink.records == [("deepseek-v4-flash", "answer", 0, 0, 0)]


def test_usage_float_and_values_above_sqlite_integer_range_are_recorded_as_zero(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "rag.sqlite3"
    store = IndexStore(db_path)
    body = (
        b'{"choices":[{"message":{"content":"ok"}}],'
        b'"usage":{"prompt_tokens":1.5,"completion_tokens":9223372036854775808,'
        b'"total_tokens":9223372036854775808}}'
    )
    try:
        client = DeepSeekClient(
            "secret", transport=SequenceTransport([(200, body)]), usage_sink=store
        )

        assert client.complete("system", "question") == "ok"
        assert store.query_llm_usage() == [
            {
                "day": store.query_llm_usage()[0]["day"],
                "model": "deepseek-v4-flash",
                "purpose": "answer",
                "requests": 1,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        ]
    finally:
        store.close()


def test_missing_total_usage_saturates_derived_sum_without_failing_answer(
    tmp_path: Path,
) -> None:
    maximum = 2**63 - 1
    store = IndexStore(tmp_path / "rag.sqlite3")
    body = (
        b'{"choices":[{"message":{"content":"ok"}}],'
        b'"usage":{"prompt_tokens":9223372036854775807,'
        b'"completion_tokens":9223372036854775807}}'
    )
    try:
        client = DeepSeekClient(
            "secret", transport=SequenceTransport([(200, body)]), usage_sink=store
        )

        assert client.complete("system", "question") == "ok"
        row = store.query_llm_usage()[0]
        assert row["prompt_tokens"] == maximum
        assert row["completion_tokens"] == maximum
        assert row["total_tokens"] == maximum
    finally:
        store.close()


def test_store_migrates_and_atomically_aggregates_daily_llm_usage(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "rag.sqlite3")
    try:
        store.record_llm_usage("model-a", "answer", 7, 3, 10, day="2026-08-30")
        store.record_llm_usage("model-a", "answer", 5, 2, 7, day="2026-08-30")
        store.record_llm_usage("model-a", "chunking", 11, 4, 15, day="2026-08-30")

        rows = store.query_llm_usage(day="2026-08-30")

        assert rows == [
            {
                "day": "2026-08-30",
                "model": "model-a",
                "purpose": "answer",
                "requests": 2,
                "prompt_tokens": 12,
                "completion_tokens": 5,
                "total_tokens": 17,
            },
            {
                "day": "2026-08-30",
                "model": "model-a",
                "purpose": "chunking",
                "requests": 1,
                "prompt_tokens": 11,
                "completion_tokens": 4,
                "total_tokens": 15,
            },
        ]
        columns = [row[1] for row in store.connection.execute("PRAGMA table_info(llm_usage_daily)")]
        assert columns == [
            "day",
            "model",
            "purpose",
            "requests",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ]
    finally:
        store.close()


def test_store_rejects_token_counts_outside_sqlite_integer_range(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "rag.sqlite3")
    try:
        for invalid in (-1, 1.0, True, 2**63):
            with pytest.raises(ValueError):
                store.record_llm_usage(
                    "model-a", "answer", invalid, 0, 0, day="2026-08-30"
                )
    finally:
        store.close()


def test_usage_aggregation_saturates_at_sqlite_max_and_stays_integer(tmp_path: Path) -> None:
    maximum = 2**63 - 1
    store = IndexStore(tmp_path / "rag.sqlite3")
    try:
        store.record_llm_usage(
            "model-a",
            "answer",
            maximum - 1,
            maximum - 1,
            maximum - 1,
            day="2026-08-30",
        )
        store.connection.execute(
            "UPDATE llm_usage_daily SET requests = ?",
            (maximum,),
        )
        store.connection.commit()

        store.record_llm_usage(
            "model-a", "answer", 2, 2, 2, day="2026-08-30"
        )

        row = store.connection.execute(
            "SELECT requests,prompt_tokens,completion_tokens,total_tokens,"
            "typeof(requests),typeof(prompt_tokens),typeof(completion_tokens),"
            "typeof(total_tokens) FROM llm_usage_daily"
        ).fetchone()
        assert tuple(row[:4]) == (maximum, maximum, maximum, maximum)
        assert tuple(row[4:]) == ("integer", "integer", "integer", "integer")
    finally:
        store.close()


def test_two_store_connections_concurrently_aggregate_without_lost_updates_or_lock(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "rag.sqlite3"
    seed = IndexStore(db_path)
    seed.close()
    barrier = threading.Barrier(2)

    def record_many() -> None:
        store = IndexStore(db_path)
        try:
            barrier.wait()
            for _ in range(100):
                store.record_llm_usage(
                    "model-a", "answer", 2, 3, 5, day="2026-08-30"
                )
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(record_many) for _ in range(2)]
        for future in futures:
            future.result()

    store = IndexStore(db_path)
    try:
        assert store.query_llm_usage(day="2026-08-30") == [
            {
                "day": "2026-08-30",
                "model": "model-a",
                "purpose": "answer",
                "requests": 200,
                "prompt_tokens": 400,
                "completion_tokens": 600,
                "total_tokens": 1000,
            }
        ]
    finally:
        store.close()


def test_usage_database_never_stores_prompt_or_response_body(tmp_path: Path) -> None:
    db_path = tmp_path / "rag.sqlite3"
    store = IndexStore(db_path)
    private_prompt = "PRIVATE-PROMPT-6f981"
    private_answer = "PRIVATE-ANSWER-4c912"
    body = (
        '{"choices":[{"message":{"content":"%s"}}],'
        '"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}'
        % private_answer
    ).encode()
    try:
        client = DeepSeekClient(
            "secret", transport=SequenceTransport([(200, body)]), usage_sink=store
        )
        assert client.complete("system", private_prompt) == private_answer
    finally:
        store.close()

    connection = sqlite3.connect(db_path)
    try:
        values = connection.execute(
            "SELECT day,model,purpose,requests,prompt_tokens,completion_tokens,total_tokens "
            "FROM llm_usage_daily"
        ).fetchall()
    finally:
        connection.close()
    serialized = repr(values)
    assert private_prompt not in serialized
    assert private_answer not in serialized
