import os
import unittest
from unittest.mock import patch

from feishu_rag.config import ConfigError, Settings


class SettingsTests(unittest.TestCase):
    def _base_env(self):
        return {
            "DEEPSEEK_API_KEY": "deepseek-secret-value",
            "FEISHU_APP_ID": "cli_test",
            "FEISHU_APP_SECRET": "secret_test",
            "FEISHU_VERIFICATION_TOKEN": "verify_test",
        }

    def test_requires_deepseek_api_key(self):
        env = self._base_env()
        env.pop("DEEPSEEK_API_KEY")
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ConfigError, "DEEPSEEK_API_KEY"):
                Settings.from_env()

    def test_repr_does_not_expose_api_key(self):
        with patch.dict(os.environ, self._base_env(), clear=True):
            settings = Settings.from_env()
        self.assertNotIn("deepseek-secret-value", repr(settings))

    def test_ocr_can_be_disabled_by_environment(self):
        env = self._base_env()
        env["RAG_ENABLE_OCR"] = "false"
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertFalse(settings.rag_enable_ocr)

    def test_long_connection_does_not_require_webhook_verification_token(self):
        env = self._base_env()
        env.pop("FEISHU_VERIFICATION_TOKEN")
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.feishu_verification_token, "")

    def test_semantic_chunking_defaults(self):
        with patch.dict(os.environ, self._base_env(), clear=True):
            settings = Settings.from_env()
        self.assertTrue(settings.rag_semantic_chunking)
        self.assertEqual(settings.deepseek_chunk_model, "deepseek-v4-flash")
        self.assertEqual(settings.deepseek_chunk_batch_chars, 12000)
        self.assertEqual(settings.rag_chunk_strategy_version, "hybrid-v4")

    def test_min_relevance_defaults_to_point_four_two(self):
        with patch.dict(os.environ, self._base_env(), clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.rag_min_relevance, 0.42)

    def test_min_relevance_accepts_zero_and_one(self):
        for value, expected in (("0", 0.0), ("1", 1.0)):
            env = self._base_env()
            env["RAG_MIN_RELEVANCE"] = value
            with self.subTest(value=value), patch.dict(os.environ, env, clear=True):
                self.assertEqual(Settings.from_env().rag_min_relevance, expected)

    def test_min_relevance_must_be_a_finite_probability(self):
        for value in ("-0.01", "1.01", "nan", "inf", "not-a-number"):
            env = self._base_env()
            env["RAG_MIN_RELEVANCE"] = value
            with self.subTest(value=value), patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ConfigError, "RAG_MIN_RELEVANCE"):
                    Settings.from_env()

    def test_faq_defaults(self):
        with patch.dict(os.environ, self._base_env(), clear=True):
            settings = Settings.from_env()
        self.assertTrue(settings.rag_faq_enabled)
        self.assertEqual(settings.rag_faq_promotion_count, 3)
        self.assertEqual(settings.rag_faq_window_days, 15)
        self.assertEqual(settings.rag_faq_min_text_similarity, 0.82)
        self.assertEqual(settings.rag_faq_min_source_overlap, 0.80)

    def test_faq_enabled_accepts_boolean_values(self):
        for value, expected in (("true", True), ("false", False)):
            env = self._base_env()
            env["RAG_FAQ_ENABLED"] = value
            with self.subTest(value=value), patch.dict(os.environ, env, clear=True):
                self.assertEqual(Settings.from_env().rag_faq_enabled, expected)

    def test_faq_rejects_invalid_environment_values(self):
        cases = [
            ("RAG_FAQ_ENABLED", "maybe"),
            ("RAG_FAQ_PROMOTION_COUNT", "0"),
            ("RAG_FAQ_PROMOTION_COUNT", "101"),
            ("RAG_FAQ_PROMOTION_COUNT", "not-an-integer"),
            ("RAG_FAQ_WINDOW_DAYS", "0"),
            ("RAG_FAQ_WINDOW_DAYS", "366"),
            ("RAG_FAQ_WINDOW_DAYS", "not-an-integer"),
            ("RAG_FAQ_MIN_TEXT_SIMILARITY", "-0.01"),
            ("RAG_FAQ_MIN_TEXT_SIMILARITY", "1.01"),
            ("RAG_FAQ_MIN_TEXT_SIMILARITY", "nan"),
            ("RAG_FAQ_MIN_TEXT_SIMILARITY", "inf"),
            ("RAG_FAQ_MIN_TEXT_SIMILARITY", "not-a-number"),
            ("RAG_FAQ_MIN_SOURCE_OVERLAP", "-0.01"),
            ("RAG_FAQ_MIN_SOURCE_OVERLAP", "1.01"),
            ("RAG_FAQ_MIN_SOURCE_OVERLAP", "nan"),
            ("RAG_FAQ_MIN_SOURCE_OVERLAP", "inf"),
            ("RAG_FAQ_MIN_SOURCE_OVERLAP", "not-a-number"),
        ]
        for name, value in cases:
            env = self._base_env()
            env[name] = value
            with self.subTest(name=name, value=value), patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ConfigError, name):
                    Settings.from_env()

    def test_faq_values_can_be_configured_and_repr_is_non_sensitive(self):
        env = self._base_env()
        env.update(
            RAG_FAQ_ENABLED="off",
            RAG_FAQ_PROMOTION_COUNT="7",
            RAG_FAQ_WINDOW_DAYS="30",
            RAG_FAQ_MIN_TEXT_SIMILARITY="0.9",
            RAG_FAQ_MIN_SOURCE_OVERLAP="0.7",
        )
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertFalse(settings.rag_faq_enabled)
        self.assertEqual(settings.rag_faq_promotion_count, 7)
        self.assertEqual(settings.rag_faq_window_days, 30)
        self.assertEqual(settings.rag_faq_min_text_similarity, 0.9)
        self.assertEqual(settings.rag_faq_min_source_overlap, 0.7)
        rendered = repr(settings)
        for name in (
            "rag_faq_enabled",
            "rag_faq_promotion_count",
            "rag_faq_window_days",
            "rag_faq_min_text_similarity",
            "rag_faq_min_source_overlap",
        ):
            self.assertIn(name, rendered)
        self.assertNotIn("deepseek-secret-value", rendered)

    def test_faq_preheat_defaults(self):
        with patch.dict(os.environ, self._base_env(), clear=True):
            settings = Settings.from_env()
        self.assertTrue(settings.rag_faq_preheat_enabled)
        self.assertEqual(settings.rag_faq_preheat_max_per_space, 10)
        self.assertEqual(settings.rag_faq_preheat_workers, 2)
        self.assertEqual(settings.rag_faq_preheat_max_retries, 1)

    def test_faq_preheat_values_are_configurable(self):
        env = self._base_env()
        env.update(
            RAG_FAQ_PREHEAT_ENABLED="false",
            RAG_FAQ_PREHEAT_MAX_PER_SPACE="25",
            RAG_FAQ_PREHEAT_WORKERS="4",
            RAG_FAQ_PREHEAT_MAX_RETRIES="2",
        )
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertFalse(settings.rag_faq_preheat_enabled)
        self.assertEqual(settings.rag_faq_preheat_max_per_space, 25)
        self.assertEqual(settings.rag_faq_preheat_workers, 4)
        self.assertEqual(settings.rag_faq_preheat_max_retries, 2)

    def test_faq_preheat_rejects_invalid_values(self):
        cases = [
            ("RAG_FAQ_PREHEAT_ENABLED", "maybe"),
            ("RAG_FAQ_PREHEAT_MAX_PER_SPACE", "0"),
            ("RAG_FAQ_PREHEAT_MAX_PER_SPACE", "51"),
            ("RAG_FAQ_PREHEAT_MAX_PER_SPACE", "x"),
            ("RAG_FAQ_PREHEAT_WORKERS", "0"),
            ("RAG_FAQ_PREHEAT_WORKERS", "9"),
            ("RAG_FAQ_PREHEAT_MAX_RETRIES", "-1"),
            ("RAG_FAQ_PREHEAT_MAX_RETRIES", "3"),
        ]
        for name, value in cases:
            env = self._base_env()
            env[name] = value
            with self.subTest(name=name), patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ConfigError, name):
                    Settings.from_env()

    def test_question_max_chars_defaults_to_500(self):
        with patch.dict(os.environ, self._base_env(), clear=True):
            self.assertEqual(Settings.from_env().rag_question_max_chars, 500)

    def test_question_max_chars_accepts_positive_integer(self):
        env = self._base_env()
        env["RAG_QUESTION_MAX_CHARS"] = "321"
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(Settings.from_env().rag_question_max_chars, 321)

    def test_question_max_chars_must_be_strictly_positive_integer(self):
        for value in ("0", "-1", "1.5", "not-an-integer", ""):
            env = self._base_env()
            env["RAG_QUESTION_MAX_CHARS"] = value
            with self.subTest(value=value), patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ConfigError, "RAG_QUESTION_MAX_CHARS"):
                    Settings.from_env()

    def test_api_retry_defaults(self):
        with patch.dict(os.environ, self._base_env(), clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.api_retry_max_attempts, 3)
        self.assertEqual(settings.api_retry_base_delay, 0.5)

    def test_api_retry_accepts_documented_bounds(self):
        for attempts in ("1", "5"):
            env = self._base_env()
            env["API_RETRY_MAX_ATTEMPTS"] = attempts
            env["API_RETRY_BASE_DELAY"] = "0"
            with self.subTest(attempts=attempts), patch.dict(os.environ, env, clear=True):
                settings = Settings.from_env()
            self.assertEqual(settings.api_retry_max_attempts, int(attempts))
            self.assertEqual(settings.api_retry_base_delay, 0.0)

    def test_api_retry_rejects_invalid_values(self):
        cases = [
            ("0", "0.5"),
            ("6", "0.5"),
            ("1.5", "0.5"),
            ("3", "-0.1"),
            ("3", "nan"),
            ("3", "inf"),
            ("3", "not-a-number"),
        ]
        for attempts, delay in cases:
            env = self._base_env()
            env["API_RETRY_MAX_ATTEMPTS"] = attempts
            env["API_RETRY_BASE_DELAY"] = delay
            with self.subTest(attempts=attempts, delay=delay), patch.dict(
                os.environ, env, clear=True
            ):
                with self.assertRaisesRegex(ConfigError, "API_RETRY"):
                    Settings.from_env()

    def test_rate_limit_defaults_and_accepts_zero_to_disable(self):
        with patch.dict(os.environ, self._base_env(), clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.rag_rate_limit_per_minute, 10)
        self.assertEqual(settings.rag_rate_limit_per_day, 200)

        env = self._base_env()
        env.update(
            RAG_RATE_LIMIT_PER_MINUTE="0",
            RAG_RATE_LIMIT_PER_DAY="0",
        )
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.rag_rate_limit_per_minute, 0)
        self.assertEqual(settings.rag_rate_limit_per_day, 0)

    def test_rate_limits_must_be_non_negative_integers(self):
        for name in ("RAG_RATE_LIMIT_PER_MINUTE", "RAG_RATE_LIMIT_PER_DAY"):
            for value in ("-1", "1.5", "not-an-integer", ""):
                env = self._base_env()
                env[name] = value
                with self.subTest(name=name, value=value), patch.dict(
                    os.environ, env, clear=True
                ):
                    with self.assertRaisesRegex(ConfigError, name):
                        Settings.from_env()

    def test_rate_limits_accept_sqlite_max_and_reject_larger_values(self):
        maximum = 2**63 - 1
        env = self._base_env()
        env.update(
            RAG_RATE_LIMIT_PER_MINUTE=str(maximum),
            RAG_RATE_LIMIT_PER_DAY=str(maximum),
        )
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.rag_rate_limit_per_minute, maximum)
        self.assertEqual(settings.rag_rate_limit_per_day, maximum)

        for name in ("RAG_RATE_LIMIT_PER_MINUTE", "RAG_RATE_LIMIT_PER_DAY"):
            env = self._base_env()
            env[name] = str(maximum + 1)
            with self.subTest(name=name), patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ConfigError, name):
                    Settings.from_env()

    def test_rag_worker_threads_defaults_to_four_and_accepts_bounds(self):
        with patch.dict(os.environ, self._base_env(), clear=True):
            self.assertEqual(Settings.from_env().rag_worker_threads, 4)

        for value in ("1", "32"):
            env = self._base_env()
            env["RAG_WORKER_THREADS"] = value
            with self.subTest(value=value), patch.dict(os.environ, env, clear=True):
                self.assertEqual(Settings.from_env().rag_worker_threads, int(value))

    def test_rag_worker_threads_must_be_an_integer_from_one_to_32(self):
        for value in ("0", "33", "-1", "1.5", "not-an-integer", ""):
            env = self._base_env()
            env["RAG_WORKER_THREADS"] = value
            with self.subTest(value=value), patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ConfigError, "RAG_WORKER_THREADS"):
                    Settings.from_env()

    def test_max_pending_messages_defaults_to_32_and_accepts_bounds(self):
        with patch.dict(os.environ, self._base_env(), clear=True):
            self.assertEqual(Settings.from_env().rag_max_pending_messages, 32)

        for workers, pending in (("1", "1"), ("32", "1000")):
            env = self._base_env()
            env.update(RAG_WORKER_THREADS=workers, RAG_MAX_PENDING_MESSAGES=pending)
            with self.subTest(workers=workers, pending=pending), patch.dict(
                os.environ, env, clear=True
            ):
                self.assertEqual(
                    Settings.from_env().rag_max_pending_messages, int(pending)
                )

    def test_max_pending_messages_must_cover_workers_and_be_one_to_1000(self):
        for workers, pending in (
            ("4", "0"),
            ("4", "1001"),
            ("4", "3"),
            ("4", "1.5"),
            ("4", "not-an-integer"),
            ("4", ""),
        ):
            env = self._base_env()
            env.update(RAG_WORKER_THREADS=workers, RAG_MAX_PENDING_MESSAGES=pending)
            with self.subTest(workers=workers, pending=pending), patch.dict(
                os.environ, env, clear=True
            ):
                with self.assertRaisesRegex(ConfigError, "RAG_MAX_PENDING_MESSAGES"):
                    Settings.from_env()


if __name__ == "__main__":
    unittest.main()
