import hashlib
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from feishu_rag.config import Settings
from feishu_rag.feishu_client import (
    FeishuAPIError,
    FeishuClient,
    FeishuReplyNotSentError,
)
from feishu_rag.rag import RagAnswer
from feishu_rag.store import IndexStore
from feishu_rag.web import create_app, handle_event, verify_signature


RATE_LIMIT_ANSWER = "请求过于频繁，请稍后再试。"


class FakeRag:
    def __init__(self, store=None, per_minute=None, per_day=None):
        self.questions = []
        self.store = store
        if per_minute is not None:
            self.rate_limit_per_minute = per_minute
        if per_day is not None:
            self.rate_limit_per_day = per_day

    def answer(self, question):
        self.questions.append(question)
        return RagAnswer("请先提交申请。", [])


class FakeFeishu:
    def __init__(self):
        self.replies = []

    def reply_text(self, message_id, text):
        self.replies.append((message_id, text))


class AmbiguousFeishu(FakeFeishu):
    def reply_text(self, message_id, text):
        self.replies.append((message_id, text))
        raise ConnectionError("reply result unknown")


class NotSentOnceFeishu(FakeFeishu):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def reply_text(self, message_id, text):
        self.calls += 1
        if self.calls == 1:
            raise FeishuReplyNotSentError("not sent")
        super().reply_text(message_id, text)


class WebhookTests(unittest.TestCase):
    def test_create_app_builds_faq_service_from_settings(self):
        settings = Settings(
            deepseek_api_key="key",
            feishu_app_id="app",
            feishu_app_secret="secret",
            feishu_verification_token="verify",
            rag_faq_enabled=False,
            rag_faq_promotion_count=7,
            rag_faq_window_days=30,
            rag_faq_min_text_similarity=0.9,
            rag_faq_min_source_overlap=0.7,
        )
        with (
            patch("feishu_rag.web.IndexStore", return_value=MagicMock()) as store_type,
            patch("feishu_rag.web.FaqService") as faq_type,
            patch("feishu_rag.web.RagService") as rag_type,
            patch("feishu_rag.web.DeepSeekClient"),
            patch("feishu_rag.web.FeishuClient"),
        ):
            create_app(settings)

        faq_type.assert_called_once_with(
            store_type.return_value, False, 7, 30, 0.9, 0.7
        )
        self.assertIs(rag_type.call_args.kwargs["faq_service"], faq_type.return_value)

    @staticmethod
    def _payload(message_id, sender_id=None):
        event = {
            "message": {
                "message_id": message_id,
                "message_type": "text",
                "content": json.dumps({"text": "报销怎么走"}, ensure_ascii=False),
            }
        }
        if sender_id is not None:
            event["sender"] = {"sender_type": "user", "sender_id": sender_id}
        return {"header": {"event_type": "im.message.receive_v1"}, "event": event}

    def test_url_verification_returns_challenge(self):
        payload = {"type": "url_verification", "token": "verify", "challenge": "abc"}

        result = handle_event(payload, FakeRag(), FakeFeishu(), verification_token="verify")

        self.assertEqual(result, {"challenge": "abc"})

    def test_signature_matches_feishu_formula(self):
        timestamp = "1700000000"
        nonce = "nonce"
        body = '{"event":"test"}'
        encrypt_key = "encrypt-key"
        signature = hashlib.sha256((timestamp + nonce + encrypt_key + body).encode("utf-8")).hexdigest()

        self.assertTrue(verify_signature(timestamp, nonce, body, encrypt_key, signature))
        self.assertFalse(verify_signature(timestamp, nonce, body, encrypt_key, "bad"))

    def test_signature_rejects_when_encrypt_key_is_missing(self):
        timestamp = "1700000000"
        nonce = "nonce"
        body = '{}'
        signature = hashlib.sha256((timestamp + nonce + "" + body).encode("utf-8")).hexdigest()

        self.assertFalse(verify_signature(timestamp, nonce, body, "", signature))

    def test_text_message_is_answered_once(self):
        rag = FakeRag()
        feishu = FakeFeishu()
        payload = {
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "message": {
                    "message_id": "om_123",
                    "message_type": "text",
                    "content": json.dumps({"text": "报销怎么走"}, ensure_ascii=False),
                }
            },
        }

        result = handle_event(payload, rag, feishu, verification_token="verify")

        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(rag.questions, ["报销怎么走"])
        self.assertEqual(feishu.replies, [("om_123", "请先提交申请。")])

    def test_duplicate_message_id_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                rag = FakeRag(store)
                feishu = FakeFeishu()
                payload = {
                    "header": {"event_type": "im.message.receive_v1"},
                    "event": {
                        "message": {
                            "message_id": "om_duplicate",
                            "message_type": "text",
                            "content": json.dumps({"text": "报销怎么走"}, ensure_ascii=False),
                        }
                    },
                }

                first = handle_event(payload, rag, feishu, verification_token="verify")
                second = handle_event(payload, rag, feishu, verification_token="verify")

                self.assertEqual(first, {"status": "ok"})
                self.assertEqual(second, {"status": "duplicate"})
                self.assertEqual(rag.questions, ["报销怎么走"])
                self.assertEqual(feishu.replies, [("om_duplicate", "请先提交申请。")])
            finally:
                store.close()

    def test_reply_is_atomically_sealed_before_external_send(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")

            class InspectingFeishu(FakeFeishu):
                def reply_text(self, message_id, text):
                    row = store.connection.execute(
                        "SELECT state,claim_token FROM processed_messages "
                        "WHERE message_id = ?",
                        (message_id,),
                    ).fetchone()
                    self.state_during_reply = tuple(row)
                    super().reply_text(message_id, text)

            rag = FakeRag(store)
            feishu = InspectingFeishu()
            try:
                result = handle_event(
                    self._payload("om_sealed"),
                    rag,
                    feishu,
                    verification_token="verify",
                )

                self.assertEqual(result, {"status": "ok"})
                self.assertEqual(feishu.state_during_reply[0], "replying")
                self.assertRegex(feishu.state_during_reply[1], r"^[0-9a-f]{32}$")
                self.assertEqual(
                    tuple(
                        store.connection.execute(
                            "SELECT state,claim_token FROM processed_messages "
                            "WHERE message_id = ?",
                            ("om_sealed",),
                        ).fetchone()
                    ),
                    ("completed", ""),
                )
            finally:
                store.close()

    def test_in_progress_message_is_returned_for_retry_without_answering(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                self.assertEqual(store.claim_message_state("om_busy"), "claimed")
                rag = FakeRag(store)
                feishu = FakeFeishu()

                result = handle_event(
                    self._payload("om_busy"), rag, feishu, verification_token="verify"
                )

                self.assertEqual(result, {"status": "in_progress"})
                self.assertEqual(rag.questions, [])
                self.assertEqual(feishu.replies, [])
            finally:
                store.close()

    def test_failed_concurrent_processing_releases_claim_for_third_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            IndexStore(db_path).close()
            retry_store = IndexStore(db_path)
            started = threading.Event()
            release = threading.Event()

            class BlockingFailingRag(FakeRag):
                def answer(self, question):
                    self.questions.append(question)
                    started.set()
                    release.wait(timeout=2)
                    raise RuntimeError("model unavailable")

            retry_rag = FakeRag(retry_store)
            feishu = FakeFeishu()
            payload = self._payload("om_retry_after_failure")

            def first_call():
                first_store = IndexStore(db_path)
                try:
                    return handle_event(
                        payload,
                        BlockingFailingRag(first_store),
                        feishu,
                        "verify",
                    )
                finally:
                    first_store.close()

            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    first = executor.submit(first_call)
                    self.assertTrue(started.wait(timeout=2))
                    self.assertEqual(
                        handle_event(payload, retry_rag, feishu, "verify"),
                        {"status": "in_progress"},
                    )
                    release.set()
                    with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                        first.result(timeout=2)

                self.assertEqual(
                    handle_event(payload, retry_rag, feishu, "verify"),
                    {"status": "ok"},
                )
                self.assertEqual(retry_rag.questions, ["报销怎么走"])
                self.assertEqual(
                    feishu.replies,
                    [("om_retry_after_failure", "请先提交申请。")],
                )
            finally:
                release.set()
                retry_store.close()

    def test_stale_slow_worker_never_replies_or_changes_new_owner_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            first_store = IndexStore(db_path)
            takeover_store = IndexStore(db_path)
            feishu = FakeFeishu()
            payload = self._payload("om_stale_worker")

            class TakeoverDuringAnswerRag(FakeRag):
                def answer(self, question):
                    self.questions.append(question)
                    first_store.connection.execute(
                        "UPDATE processed_messages SET processed_at = ? "
                        "WHERE message_id = ?",
                        (time.time() - 601, "om_stale_worker"),
                    )
                    first_store.connection.commit()
                    self.takeover = takeover_store.claim_message_lease(
                        "om_stale_worker"
                    )
                    return RagAnswer("旧工作器的答案", [])

            rag = TakeoverDuringAnswerRag(first_store)
            try:
                result = handle_event(payload, rag, feishu, verification_token="verify")

                self.assertEqual(result, {"status": "superseded"})
                self.assertEqual(feishu.replies, [])
                self.assertEqual(rag.takeover[0], "claimed")
                new_token = rag.takeover[1]
                self.assertIsNotNone(new_token)
                row = takeover_store.connection.execute(
                    "SELECT state,claim_token FROM processed_messages WHERE message_id = ?",
                    ("om_stale_worker",),
                ).fetchone()
                self.assertEqual(tuple(row), ("in_progress", new_token))
                self.assertNotIn(new_token or "", json.dumps(result))
            finally:
                first_store.close()
                takeover_store.close()

    def test_rate_limit_reply_is_suppressed_after_lease_loss(self):
        class LostOwnerStore:
            token = "0123456789abcdef0123456789abcdef"

            @classmethod
            def claim_message_lease(cls, message_id):
                return "claimed", cls.token

            @staticmethod
            def claim_rate_limit(actor_id, per_minute, per_day):
                return False

            @staticmethod
            def begin_message_reply(message_id, token):
                return False

            @staticmethod
            def complete_message(message_id, token=None):
                raise AssertionError("superseded worker must not complete the new owner")

            @staticmethod
            def release_message(message_id, token=None):
                raise AssertionError("superseded worker must not release the new owner")

        store = LostOwnerStore()
        rag = FakeRag(store, per_minute=1, per_day=1)
        feishu = FakeFeishu()

        result = handle_event(
            self._payload("om_lost_rate_limit", {"open_id": "ou_limited"}),
            rag,
            feishu,
            verification_token="verify",
        )

        self.assertEqual(result, {"status": "superseded"})
        self.assertEqual(feishu.replies, [])
        self.assertNotIn(store.token, json.dumps(result))

    def test_incomplete_lease_store_is_rejected_before_claiming(self):
        class IncompleteLeaseStore:
            claims = 0

            @classmethod
            def claim_message_lease(cls, message_id):
                cls.claims += 1
                return "claimed", "0123456789abcdef0123456789abcdef"

            @staticmethod
            def begin_message_reply(message_id, token):
                return True

        store = IncompleteLeaseStore()
        rag = FakeRag(store)

        with self.assertRaisesRegex(RuntimeError, "lease"):
            handle_event(
                self._payload("om_incomplete_lease"),
                rag,
                FakeFeishu(),
                verification_token="verify",
            )

        self.assertEqual(store.claims, 0)
        self.assertEqual(rag.questions, [])

    def test_webhook_returns_503_while_same_message_is_in_progress(self):
        class BusyStore:
            @staticmethod
            def claim_message_state(message_id):
                return "in_progress"

        settings = Settings(
            deepseek_api_key="key",
            feishu_app_id="app",
            feishu_app_secret="secret",
            feishu_verification_token="verify",
            feishu_encrypt_key="encrypt-key",
        )
        body = json.dumps(
            self._payload("om_http_busy"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        timestamp = "1700000000"
        nonce = "nonce"
        signature = hashlib.sha256(
            (timestamp + nonce + "encrypt-key" + body).encode("utf-8")
        ).hexdigest()
        client = TestClient(create_app(settings, FakeRag(BusyStore()), FakeFeishu()))

        response = client.post(
            "/webhook/feishu",
            content=body.encode("utf-8"),
            headers={
                "x-lark-request-timestamp": timestamp,
                "x-lark-request-nonce": nonce,
                "x-lark-signature": signature,
                "content-type": "application/json",
            },
        )

        self.assertEqual(response.status_code, 503)

    def test_ambiguous_reply_failure_keeps_claim_and_prevents_duplicate_reply(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                rag = FakeRag(store)
                feishu = AmbiguousFeishu()
                payload = {
                    "header": {"event_type": "im.message.receive_v1"},
                    "event": {
                        "message": {
                            "message_id": "om_ambiguous",
                            "message_type": "text",
                            "content": json.dumps(
                                {"text": "报销怎么走"}, ensure_ascii=False
                            ),
                        }
                    },
                }

                with self.assertRaises(ConnectionError):
                    handle_event(payload, rag, feishu, verification_token="verify")
                self.assertEqual(
                    store.connection.execute(
                        "SELECT state FROM processed_messages WHERE message_id = ?",
                        ("om_ambiguous",),
                    ).fetchone()[0],
                    "completed",
                )
                second = handle_event(
                    payload, rag, feishu, verification_token="verify"
                )

                self.assertEqual(second, {"status": "duplicate"})
                self.assertEqual(rag.questions, ["报销怎么走"])
                self.assertEqual(
                    feishu.replies, [("om_ambiguous", "请先提交申请。")]
                )
            finally:
                store.close()

    def test_reply_not_sent_releases_claim_and_allows_next_delivery_to_succeed(self):
        outcomes = [
            (400, b"{}"),
            (200, b'{"code":0,"tenant_access_token":"token","expire":7200}'),
            (200, b'{"code":0}'),
        ]
        urls = []

        def transport(method, url, headers, payload, timeout):
            urls.append(url)
            return outcomes.pop(0)

        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                rag = FakeRag(store)
                feishu = FeishuClient("app", "secret", transport=transport)
                payload = {
                    "header": {"event_type": "im.message.receive_v1"},
                    "event": {
                        "message": {
                            "message_id": "om_not_sent",
                            "message_type": "text",
                            "content": json.dumps(
                                {"text": "报销怎么走"}, ensure_ascii=False
                            ),
                        }
                    },
                }

                with self.assertRaises(FeishuAPIError):
                    handle_event(payload, rag, feishu, verification_token="verify")
                self.assertIsNone(
                    store.connection.execute(
                        "SELECT state FROM processed_messages WHERE message_id = ?",
                        ("om_not_sent",),
                    ).fetchone()
                )
                second = handle_event(
                    payload, rag, feishu, verification_token="verify"
                )

                self.assertEqual(second, {"status": "ok"})
                self.assertEqual(rag.questions, ["报销怎么走", "报销怎么走"])
                self.assertEqual(
                    sum(url.endswith("/reply") for url in urls),
                    1,
                )
            finally:
                store.close()

    def test_malformed_token_expire_releases_claim_and_retry_can_succeed(self):
        outcomes = [
            (
                200,
                b'{"code":0,"tenant_access_token":"token","expire":"invalid"}',
            ),
            (200, b'{"code":0,"tenant_access_token":"token","expire":7200}'),
            (200, b'{"code":0}'),
        ]

        def transport(method, url, headers, payload, timeout):
            return outcomes.pop(0)

        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                rag = FakeRag(store)
                feishu = FeishuClient("app", "secret", transport=transport)
                payload = {
                    "header": {"event_type": "im.message.receive_v1"},
                    "event": {
                        "message": {
                            "message_id": "om_invalid_expire",
                            "message_type": "text",
                            "content": json.dumps(
                                {"text": "报销怎么走"}, ensure_ascii=False
                            ),
                        }
                    },
                }

                with self.assertRaises(FeishuReplyNotSentError):
                    handle_event(payload, rag, feishu, verification_token="verify")
                self.assertIsNone(
                    store.connection.execute(
                        "SELECT state FROM processed_messages WHERE message_id = ?",
                        ("om_invalid_expire",),
                    ).fetchone()
                )
                second = handle_event(
                    payload, rag, feishu, verification_token="verify"
                )

                self.assertEqual(second, {"status": "ok"})
                self.assertEqual(rag.questions, ["报销怎么走", "报销怎么走"])
            finally:
                store.close()

    def test_rate_limit_uses_sender_id_priority_after_message_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                rag = FakeRag(store, per_minute=10, per_day=200)
                payload = self._payload(
                    "om_actor",
                    {
                        "open_id": "ou_preferred",
                        "union_id": "on_fallback",
                        "user_id": "legacy_fallback",
                    },
                )

                self.assertEqual(
                    handle_event(payload, rag, FakeFeishu(), verification_token="verify"),
                    {"status": "ok"},
                )
                self.assertEqual(
                    {
                        row[0]
                        for row in store.connection.execute(
                            "SELECT user_hash FROM rate_limit_buckets"
                        ).fetchall()
                    },
                    {sha256(b"ou_preferred").hexdigest()},
                )
                self.assertEqual(
                    handle_event(payload, rag, FakeFeishu(), verification_token="verify"),
                    {"status": "duplicate"},
                )
                self.assertEqual(
                    store.connection.execute(
                        "SELECT SUM(count) FROM rate_limit_buckets"
                    ).fetchone()[0],
                    2,
                )
            finally:
                store.close()

    def test_rate_limit_falls_back_to_union_then_user_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                rag = FakeRag(store, per_minute=10, per_day=0)
                feishu = FakeFeishu()
                handle_event(
                    self._payload("om_union", {"union_id": "on_union", "user_id": "u_old"}),
                    rag,
                    feishu,
                    verification_token="verify",
                )
                handle_event(
                    self._payload("om_user", {"user_id": "u_user"}),
                    rag,
                    feishu,
                    verification_token="verify",
                )
                self.assertEqual(
                    {
                        row[0]
                        for row in store.connection.execute(
                            "SELECT user_hash FROM rate_limit_buckets"
                        ).fetchall()
                    },
                    {
                        sha256(b"on_union").hexdigest(),
                        sha256(b"u_user").hexdigest(),
                    },
                )
            finally:
                store.close()

    def test_exceeded_rate_limit_sends_fixed_reply_without_calling_rag_and_keeps_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                rag = FakeRag(store, per_minute=1, per_day=0)
                feishu = FakeFeishu()
                first = self._payload("om_first", {"open_id": "ou_limited"})
                limited = self._payload("om_limited", {"open_id": "ou_limited"})

                self.assertEqual(
                    handle_event(first, rag, feishu, verification_token="verify"),
                    {"status": "ok"},
                )
                self.assertEqual(
                    handle_event(limited, rag, feishu, verification_token="verify"),
                    {"status": "rate_limited"},
                )
                self.assertEqual(rag.questions, ["报销怎么走"])
                self.assertEqual(feishu.replies[-1], ("om_limited", RATE_LIMIT_ANSWER))
                self.assertEqual(
                    handle_event(limited, rag, feishu, verification_token="verify"),
                    {"status": "duplicate"},
                )
            finally:
                store.close()

    def test_missing_actor_with_enabled_limit_releases_message_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                rag = FakeRag(store, per_minute=1, per_day=0)
                payload = self._payload("om_missing_actor")

                with self.assertRaisesRegex(ValueError, "actor"):
                    handle_event(payload, rag, FakeFeishu(), verification_token="verify")
                payload["event"]["sender"] = {
                    "sender_type": "user",
                    "sender_id": {"open_id": "ou_retry"},
                }
                self.assertEqual(
                    handle_event(payload, rag, FakeFeishu(), verification_token="verify"),
                    {"status": "ok"},
                )
            finally:
                store.close()

    def test_reply_not_sent_releases_claim_but_keeps_rate_limit_consumption(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                rag = FakeRag(store, per_minute=1, per_day=0)
                feishu = NotSentOnceFeishu()
                payload = self._payload("om_retry", {"open_id": "ou_retry"})

                with self.assertRaises(FeishuReplyNotSentError):
                    handle_event(payload, rag, feishu, verification_token="verify")
                self.assertEqual(
                    handle_event(payload, rag, feishu, verification_token="verify"),
                    {"status": "rate_limited"},
                )
                self.assertEqual(rag.questions, ["报销怎么走"])
                self.assertEqual(feishu.replies, [("om_retry", RATE_LIMIT_ANSWER)])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
