import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from hashlib import sha256

from feishu_rag.feishu_client import (
    FeishuAPIError,
    FeishuClient,
    FeishuReplyNotSentError,
)
from feishu_rag.rag import RagAnswer
from feishu_rag.store import IndexStore
from feishu_rag.web import handle_event, verify_signature


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
