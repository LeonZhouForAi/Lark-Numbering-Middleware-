"""服务配置与凭据读取。"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Mapping


class ConfigError(ValueError):
    """配置缺失或格式不正确。"""


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ConfigError(f"缺少必填环境变量: {name}")
    return value


def _as_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} 必须是 true/false")


@dataclass(frozen=True, repr=False)
class Settings:
    """运行时配置；密钥字段不会出现在 repr 中。"""

    deepseek_api_key: str = field(repr=False)
    feishu_app_id: str
    feishu_app_secret: str = field(repr=False)
    feishu_verification_token: str = field(repr=False)
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    feishu_encrypt_key: str = field(default="", repr=False)
    feishu_space_id: str = ""
    rag_db_path: str = "./data/rag.sqlite3"
    rag_top_k: int = 6
    rag_min_relevance: float = 0.42
    rag_question_max_chars: int = 500
    rag_rate_limit_per_minute: int = 10
    rag_rate_limit_per_day: int = 200
    rag_worker_threads: int = 4
    rag_max_pending_messages: int = 32
    rag_max_chars: int = 900
    rag_enable_ocr: bool = True
    rag_semantic_chunking: bool = True
    deepseek_chunk_model: str = "deepseek-v4-flash"
    deepseek_chunk_batch_chars: int = 12000
    rag_chunk_strategy_version: str = "hybrid-v4"
    api_retry_max_attempts: int = 3
    api_retry_base_delay: float = 0.5
    log_level: str = "INFO"
    rag_faq_enabled: bool = True
    rag_faq_promotion_count: int = 3
    rag_faq_window_days: int = 15
    rag_faq_min_text_similarity: float = 0.82
    rag_faq_min_source_overlap: float = 0.80
    rag_faq_preheat_enabled: bool = True
    rag_faq_preheat_max_per_space: int = 10
    rag_faq_preheat_workers: int = 2
    rag_faq_preheat_max_retries: int = 1

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        if environ is None:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
        env = os.environ if environ is None else environ
        try:
            top_k = int(env.get("RAG_TOP_K", "6"))
            question_max_chars = int(env.get("RAG_QUESTION_MAX_CHARS", "500"))
            max_chars = int(env.get("RAG_MAX_CHARS", "900"))
            chunk_batch_chars = int(env.get("DEEPSEEK_CHUNK_BATCH_CHARS", "12000"))
        except ValueError as exc:
            raise ConfigError(
                "RAG_TOP_K、RAG_QUESTION_MAX_CHARS、RAG_MAX_CHARS 和 "
                "DEEPSEEK_CHUNK_BATCH_CHARS 必须是整数"
            ) from exc
        try:
            retry_max_attempts = int(env.get("API_RETRY_MAX_ATTEMPTS", "3"))
        except ValueError as exc:
            raise ConfigError("API_RETRY_MAX_ATTEMPTS 必须是 1 到 5 的整数") from exc
        if not 1 <= retry_max_attempts <= 5:
            raise ConfigError("API_RETRY_MAX_ATTEMPTS 必须在 1 到 5 之间")
        try:
            retry_base_delay = float(env.get("API_RETRY_BASE_DELAY", "0.5"))
        except ValueError as exc:
            raise ConfigError("API_RETRY_BASE_DELAY 必须是有限非负数") from exc
        if not math.isfinite(retry_base_delay) or retry_base_delay < 0:
            raise ConfigError("API_RETRY_BASE_DELAY 必须是有限非负数")
        if question_max_chars < 1:
            raise ConfigError("RAG_QUESTION_MAX_CHARS 必须是正整数")
        try:
            worker_threads = int(env.get("RAG_WORKER_THREADS", "4"))
        except ValueError as exc:
            raise ConfigError("RAG_WORKER_THREADS 必须是 1 到 32 的整数") from exc
        if not 1 <= worker_threads <= 32:
            raise ConfigError("RAG_WORKER_THREADS 必须在 1 到 32 之间")
        try:
            max_pending_messages = int(env.get("RAG_MAX_PENDING_MESSAGES", "32"))
        except ValueError as exc:
            raise ConfigError(
                "RAG_MAX_PENDING_MESSAGES 必须是 1 到 1000 的整数"
            ) from exc
        if not worker_threads <= max_pending_messages <= 1000:
            raise ConfigError(
                "RAG_MAX_PENDING_MESSAGES 必须在 RAG_WORKER_THREADS 到 1000 之间"
            )
        rate_limits: dict[str, int] = {}
        for name, default in (
            ("RAG_RATE_LIMIT_PER_MINUTE", "10"),
            ("RAG_RATE_LIMIT_PER_DAY", "200"),
        ):
            try:
                value = int(env.get(name, default))
            except ValueError as exc:
                raise ConfigError(f"{name} 必须是非负整数") from exc
            if not 0 <= value <= 2**63 - 1:
                raise ConfigError(f"{name} 必须是非负整数")
            rate_limits[name] = value
        if top_k < 1 or max_chars < 100:
            raise ConfigError("RAG_TOP_K 必须大于 0，RAG_MAX_CHARS 必须不小于 100")
        if chunk_batch_chars < 2000:
            raise ConfigError("DEEPSEEK_CHUNK_BATCH_CHARS 必须至少为 2000")
        try:
            min_relevance = float(env.get("RAG_MIN_RELEVANCE", "0.42"))
        except ValueError as exc:
            raise ConfigError("RAG_MIN_RELEVANCE 必须是 0 到 1 之间的数字") from exc
        if not math.isfinite(min_relevance) or not 0.0 <= min_relevance <= 1.0:
            raise ConfigError("RAG_MIN_RELEVANCE 必须在 0 到 1 之间")
        faq_enabled = _as_bool(env.get("RAG_FAQ_ENABLED", "true"), "RAG_FAQ_ENABLED")
        try:
            faq_promotion_count = int(env.get("RAG_FAQ_PROMOTION_COUNT", "3"))
        except ValueError as exc:
            raise ConfigError("RAG_FAQ_PROMOTION_COUNT 必须是 1 到 100 的整数") from exc
        if not 1 <= faq_promotion_count <= 100:
            raise ConfigError("RAG_FAQ_PROMOTION_COUNT 必须在 1 到 100 之间")
        try:
            faq_window_days = int(env.get("RAG_FAQ_WINDOW_DAYS", "15"))
        except ValueError as exc:
            raise ConfigError("RAG_FAQ_WINDOW_DAYS 必须是 1 到 365 的整数") from exc
        if not 1 <= faq_window_days <= 365:
            raise ConfigError("RAG_FAQ_WINDOW_DAYS 必须在 1 到 365 之间")
        faq_similarities: dict[str, float] = {}
        for name, default in (
            ("RAG_FAQ_MIN_TEXT_SIMILARITY", "0.82"),
            ("RAG_FAQ_MIN_SOURCE_OVERLAP", "0.80"),
        ):
            try:
                value = float(env.get(name, default))
            except ValueError as exc:
                raise ConfigError(f"{name} 必须是 0 到 1 之间的有限数字") from exc
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ConfigError(f"{name} 必须在 0 到 1 之间且为有限数字")
            faq_similarities[name] = value
        faq_preheat_enabled = _as_bool(
            env.get("RAG_FAQ_PREHEAT_ENABLED", "true"),
            "RAG_FAQ_PREHEAT_ENABLED",
        )
        faq_preheat_values: dict[str, int] = {}
        for name, default, minimum, maximum in (
            ("RAG_FAQ_PREHEAT_MAX_PER_SPACE", "10", 1, 50),
            ("RAG_FAQ_PREHEAT_WORKERS", "2", 1, 8),
            ("RAG_FAQ_PREHEAT_MAX_RETRIES", "1", 0, 2),
        ):
            try:
                value = int(env.get(name, default))
            except ValueError as exc:
                raise ConfigError(f"{name} 必须是 {minimum} 到 {maximum} 的整数") from exc
            if not minimum <= value <= maximum:
                raise ConfigError(f"{name} 必须在 {minimum} 到 {maximum} 之间")
            faq_preheat_values[name] = value
        strategy_version = env.get("RAG_CHUNK_STRATEGY_VERSION", "hybrid-v4").strip()
        if not strategy_version:
            raise ConfigError("RAG_CHUNK_STRATEGY_VERSION 不能为空")
        return cls(
            deepseek_api_key=_required(env, "DEEPSEEK_API_KEY"),
            feishu_app_id=_required(env, "FEISHU_APP_ID"),
            feishu_app_secret=_required(env, "FEISHU_APP_SECRET"),
            feishu_verification_token=env.get("FEISHU_VERIFICATION_TOKEN", "").strip(),
            deepseek_base_url=env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            deepseek_model=env.get("DEEPSEEK_MODEL", "deepseek-v4-flash").strip(),
            feishu_encrypt_key=env.get("FEISHU_ENCRYPT_KEY", "").strip(),
            feishu_space_id=env.get("FEISHU_SPACE_ID", "").strip(),
            rag_db_path=env.get("RAG_DB_PATH", "./data/rag.sqlite3").strip(),
            rag_top_k=top_k,
            rag_min_relevance=min_relevance,
            rag_question_max_chars=question_max_chars,
            rag_rate_limit_per_minute=rate_limits["RAG_RATE_LIMIT_PER_MINUTE"],
            rag_rate_limit_per_day=rate_limits["RAG_RATE_LIMIT_PER_DAY"],
            rag_worker_threads=worker_threads,
            rag_max_pending_messages=max_pending_messages,
            rag_max_chars=max_chars,
            rag_enable_ocr=_as_bool(env.get("RAG_ENABLE_OCR", "true"), "RAG_ENABLE_OCR"),
            rag_semantic_chunking=_as_bool(
                env.get("RAG_SEMANTIC_CHUNKING", "true"), "RAG_SEMANTIC_CHUNKING"
            ),
            deepseek_chunk_model=env.get("DEEPSEEK_CHUNK_MODEL", "deepseek-v4-flash").strip(),
            deepseek_chunk_batch_chars=chunk_batch_chars,
            rag_chunk_strategy_version=strategy_version,
            api_retry_max_attempts=retry_max_attempts,
            api_retry_base_delay=retry_base_delay,
            log_level=env.get("LOG_LEVEL", "INFO").upper(),
            rag_faq_enabled=faq_enabled,
            rag_faq_promotion_count=faq_promotion_count,
            rag_faq_window_days=faq_window_days,
            rag_faq_min_text_similarity=faq_similarities["RAG_FAQ_MIN_TEXT_SIMILARITY"],
            rag_faq_min_source_overlap=faq_similarities["RAG_FAQ_MIN_SOURCE_OVERLAP"],
            rag_faq_preheat_enabled=faq_preheat_enabled,
            rag_faq_preheat_max_per_space=faq_preheat_values[
                "RAG_FAQ_PREHEAT_MAX_PER_SPACE"
            ],
            rag_faq_preheat_workers=faq_preheat_values["RAG_FAQ_PREHEAT_WORKERS"],
            rag_faq_preheat_max_retries=faq_preheat_values[
                "RAG_FAQ_PREHEAT_MAX_RETRIES"
            ],
        )

    def __repr__(self) -> str:
        return (
            "Settings("
            f"deepseek_base_url={self.deepseek_base_url!r}, "
            f"deepseek_model={self.deepseek_model!r}, "
            f"feishu_app_id={self.feishu_app_id!r}, "
            f"feishu_space_id={self.feishu_space_id!r}, "
            f"rag_db_path={self.rag_db_path!r}, "
            f"rag_top_k={self.rag_top_k!r}, rag_min_relevance={self.rag_min_relevance!r}, "
            f"rag_question_max_chars={self.rag_question_max_chars!r}, "
            f"rag_rate_limit_per_minute={self.rag_rate_limit_per_minute!r}, "
            f"rag_rate_limit_per_day={self.rag_rate_limit_per_day!r}, "
            f"rag_worker_threads={self.rag_worker_threads!r}, "
            f"rag_max_pending_messages={self.rag_max_pending_messages!r}, "
            f"rag_max_chars={self.rag_max_chars!r}, "
            f"rag_semantic_chunking={self.rag_semantic_chunking!r}, "
            f"deepseek_chunk_model={self.deepseek_chunk_model!r}, "
            f"rag_chunk_strategy_version={self.rag_chunk_strategy_version!r}, "
            f"api_retry_max_attempts={self.api_retry_max_attempts!r}, "
            f"api_retry_base_delay={self.api_retry_base_delay!r}, "
            f"rag_faq_enabled={self.rag_faq_enabled!r}, "
            f"rag_faq_promotion_count={self.rag_faq_promotion_count!r}, "
            f"rag_faq_window_days={self.rag_faq_window_days!r}, "
            f"rag_faq_min_text_similarity={self.rag_faq_min_text_similarity!r}, "
            f"rag_faq_min_source_overlap={self.rag_faq_min_source_overlap!r}, "
            f"rag_faq_preheat_enabled={self.rag_faq_preheat_enabled!r}, "
            f"rag_faq_preheat_max_per_space={self.rag_faq_preheat_max_per_space!r}, "
            f"rag_faq_preheat_workers={self.rag_faq_preheat_workers!r}, "
            f"rag_faq_preheat_max_retries={self.rag_faq_preheat_max_retries!r}, "
            f"log_level={self.log_level!r})"
        )
