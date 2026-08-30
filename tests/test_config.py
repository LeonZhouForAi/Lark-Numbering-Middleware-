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


if __name__ == "__main__":
    unittest.main()
