import re
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from feishu_rag.faq import FaqService
from feishu_rag.models import Chunk, FaqObservation, RetrievalScope
from feishu_rag.store import IndexStore, PreparedDocument, _pretokenize, _tokens


class StoreTests(unittest.TestCase):
    def test_faq_source_ids_are_persisted_as_sorted_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                observation = FaqObservation(
                    "intent", "global", "问题", "source-v1", 0,
                    source_ids=("source-b", "source-a", "source-b"),
                )
                match = store.record_faq_observation(
                    observation, answer="答案", day="2026-08-31", now=1.0,
                    promotion_count=1,
                )
                self.assertIsNotNone(match)
                self.assertEqual(
                    store.connection.execute(
                        "SELECT source_ids_json FROM faq_entries"
                    ).fetchone()[0],
                    '["source-a","source-b"]',
                )
                candidate = store.find_faq_candidates("global")[0]
                self.assertEqual(candidate["source_ids_json"], '["source-a","source-b"]')
            finally:
                store.close()

    def test_faq_promotion_count_is_configurable_and_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                observation = FaqObservation("intent", "global", "问题", "source-v1", 0)
                with self.assertRaises(ValueError):
                    store.record_faq_observation(
                        observation, answer="答案", day="2026-08-31", now=1.0,
                        promotion_count=0,
                    )
                with self.assertRaises(ValueError):
                    store.record_faq_observation(
                        observation, answer="答案", day="2026-08-31", now=1.0,
                        promotion_count=101,
                    )
                with self.assertRaises(TypeError):
                    store.record_faq_observation(
                        observation, answer="答案", day="2026-08-31", now=1.0,
                    )
                first = store.record_faq_observation(
                    observation, answer="答案", day="2026-08-31", now=1.0,
                    promotion_count=1,
                )
                self.assertIsNotNone(first)
                self.assertEqual(
                    store.connection.execute(
                        "SELECT source_signature FROM faq_entries WHERE id = ?",
                        (first.entry_id,),
                    ).fetchone()[0],
                    "source-v1",
                )
            finally:
                store.close()

    def test_refresh_current_enabled_entry_rejects_without_writing_alias_or_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                observation = FaqObservation("intent", "global", "问题", "source-v1", 0)
                for day in ("2026-08-29", "2026-08-30", "2026-08-31"):
                    match = store.record_faq_observation(
                        observation, answer="答案", day=day, now=1.0, promotion_count=3
                    )
                entry_id = match.entry_id
                before_aliases = [
                    tuple(row)
                    for row in store.connection.execute(
                        "SELECT faq_id,normalized_question,total_seen FROM faq_aliases ORDER BY normalized_question"
                    ).fetchall()
                ]
                before_metrics = [
                    tuple(row)
                    for row in store.connection.execute(
                        "SELECT scope_key,day,refreshes FROM faq_metrics_daily ORDER BY day"
                    ).fetchall()
                ]
                with self.assertRaisesRegex(ValueError, "already current"):
                    store.refresh_stale_faq(
                        entry_id, observation, answer="新答案", now=2.0
                    )
                self.assertEqual(
                    tuple(store.connection.execute(
                        "SELECT answer,state,knowledge_revision FROM faq_entries WHERE id = ?",
                        (entry_id,),
                    ).fetchone()),
                    ("答案", "enabled", 0),
                )
                self.assertEqual(
                    [
                        tuple(row)
                        for row in store.connection.execute(
                            "SELECT faq_id,normalized_question,total_seen FROM faq_aliases ORDER BY normalized_question"
                        ).fetchall()
                    ],
                    before_aliases,
                )
                self.assertEqual(
                    [
                        tuple(row)
                        for row in store.connection.execute(
                            "SELECT scope_key,day,refreshes FROM faq_metrics_daily ORDER BY day"
                        ).fetchall()
                    ],
                    before_metrics,
                )
            finally:
                store.close()

    def test_same_day_distinct_questions_promote_and_keep_each_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                observations = [
                    FaqObservation("intent", "global", question, "source-v1", 0)
                    for question in ("问题一", "问题二", "问题三")
                ]
                matches = [
                    store.record_faq_observation(
                        observation,
                        answer="答案",
                        day="2026-08-31",
                        now=float(index),
                        promotion_count=3,
                    )
                    for index, observation in enumerate(observations, 1)
                ]
                self.assertIsNotNone(matches[-1])
                self.assertEqual(
                    [
                        tuple(row)
                        for row in store.connection.execute(
                            "SELECT count FROM faq_observation_daily ORDER BY normalized_question"
                        ).fetchall()
                    ],
                    [(1,), (1,), (1,)],
                )
                for observation in observations:
                    self.assertEqual(
                        len(store.find_faq_candidates("global")),
                        3,
                    )
            finally:
                store.close()

    def test_legacy_observation_key_without_question_is_rebuilt_without_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            seed = sqlite3.connect(db_path)
            seed.executescript(
                """
                CREATE TABLE faq_observation_daily (
                    intent_key TEXT NOT NULL,
                    scope_key TEXT NOT NULL,
                    day TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 0,
                    normalized_question TEXT NOT NULL,
                    source_signature TEXT NOT NULL,
                    knowledge_revision INTEGER NOT NULL,
                    latest_safe_answer TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(scope_key, intent_key, source_signature, knowledge_revision, day)
                );
                INSERT INTO faq_observation_daily VALUES(
                    'intent', 'global', '2026-08-31', 2, '问题', 'source-v1', 0, '答案'
                );
                """
            )
            seed.commit()
            seed.close()
            store = IndexStore(db_path)
            try:
                primary_key = [
                    row[1]
                    for row in sorted(
                        (
                            row
                            for row in store.connection.execute(
                                "PRAGMA table_info(faq_observation_daily)"
                            ).fetchall()
                            if row[5]
                        ),
                        key=lambda row: row[5],
                    )
                ]
                self.assertEqual(
                    primary_key,
                    [
                        "scope_key",
                        "intent_key",
                        "normalized_question",
                        "source_signature",
                        "knowledge_revision",
                        "day",
                    ],
                )
                self.assertEqual(
                    tuple(store.connection.execute(
                        "SELECT count,normalized_question FROM faq_observation_daily"
                    ).fetchone()),
                    (2, "问题"),
                )
            finally:
                store.close()

    def test_mixed_observation_schema_rebuilds_old_primary_key_despite_new_unique_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            seed = sqlite3.connect(db_path)
            seed.executescript(
                """
                CREATE TABLE faq_observation_daily (
                    intent_key TEXT NOT NULL,
                    scope_key TEXT NOT NULL,
                    day TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 0,
                    normalized_question TEXT NOT NULL,
                    source_signature TEXT NOT NULL,
                    knowledge_revision INTEGER NOT NULL,
                    latest_safe_answer TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(scope_key, intent_key, source_signature, knowledge_revision, day)
                );
                CREATE UNIQUE INDEX mixed_new_observation_key
                    ON faq_observation_daily(
                        scope_key, intent_key, normalized_question,
                        source_signature, knowledge_revision, day
                    );
                INSERT INTO faq_observation_daily VALUES(
                    'intent', 'global', '2026-08-31', 2, '问题一', 'source-v1', 0, '答案'
                );
                """
            )
            seed.commit()
            seed.close()
            store = IndexStore(db_path)
            try:
                primary_key = [
                    row[1]
                    for row in sorted(
                        (
                            row
                            for row in store.connection.execute(
                                "PRAGMA table_info(faq_observation_daily)"
                            ).fetchall()
                            if row[5]
                        ),
                        key=lambda row: row[5],
                    )
                ]
                self.assertEqual(
                    primary_key,
                    [
                        "scope_key",
                        "intent_key",
                        "normalized_question",
                        "source_signature",
                        "knowledge_revision",
                        "day",
                    ],
                )
                self.assertIsNone(
                    store.record_faq_observation(
                        FaqObservation("intent", "global", "问题二", "source-v1", 0),
                        answer="答案",
                        day="2026-08-31",
                        now=2.0,
                        promotion_count=100,
                    )
                )
                self.assertEqual(
                    [tuple(row) for row in store.connection.execute(
                        "SELECT normalized_question,count FROM faq_observation_daily "
                        "ORDER BY normalized_question"
                    ).fetchall()],
                    [("问题一", 2), ("问题二", 1)],
                )
            finally:
                store.close()

        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                observation = FaqObservation("intent", "global", "问题", "source-v1", 0)
                for count in (1, 2, 3):
                    self.assertIsNone(
                        store.record_faq_observation(
                            observation, answer="答案", day="2026-08-31", now=float(count),
                            promotion_count=4,
                        )
                    )
                self.assertIsNotNone(
                    store.record_faq_observation(
                        observation, answer="答案", day="2026-08-31", now=4.0,
                        promotion_count=4,
                    )
                )
            finally:
                store.close()

    def test_faq_legacy_table_with_checks_but_without_scope_intent_unique_is_migrated(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            seed = sqlite3.connect(db_path)
            seed.executescript(
                """
                CREATE TABLE faq_entries (
                    id TEXT PRIMARY KEY,
                    intent_key TEXT NOT NULL,
                    scope_key TEXT NOT NULL,
                    canonical_question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    source_signature TEXT NOT NULL,
                    knowledge_revision INTEGER NOT NULL CHECK(knowledge_revision >= 0),
                    state TEXT NOT NULL CHECK(state IN ('enabled', 'stale')),
                    direct_hits INTEGER NOT NULL DEFAULT 0 CHECK(direct_hits >= 0),
                    created_at REAL NOT NULL CHECK(created_at >= 0),
                    updated_at REAL NOT NULL CHECK(updated_at >= 0),
                    last_hit_at REAL CHECK(last_hit_at IS NULL OR last_hit_at >= 0)
                );
                CREATE TABLE faq_aliases (
                    faq_id TEXT NOT NULL REFERENCES faq_entries(id) ON DELETE CASCADE,
                    normalized_question TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    first_seen_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    total_seen INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(faq_id, normalized_question)
                );
                INSERT INTO faq_entries VALUES(
                    'faq-legacy', 'intent', 'global', '问题', '答案', 'source',
                    0, 'enabled', 2, 1.0, 2.0, NULL
                );
                INSERT INTO faq_aliases VALUES(
                    'faq-legacy', '问题', '问题', 1.0, 2.0, 2
                );
                """
            )
            seed.commit()
            seed.close()
            store = IndexStore(db_path)
            try:
                with self.assertRaises(sqlite3.IntegrityError):
                    store.connection.execute(
                        "INSERT INTO faq_entries("
                        "id,intent_key,scope_key,canonical_question,answer,source_signature,"
                        "knowledge_revision,state,direct_hits,created_at,updated_at,last_hit_at"
                        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            "faq-duplicate", "intent", "global", "问题2", "答案2", "source2",
                            0, "enabled", 0, 1.0, 1.0, None,
                        ),
                    )
                self.assertEqual(
                    store.connection.execute("SELECT answer FROM faq_entries WHERE id='faq-legacy'").fetchone()[0],
                    "答案",
                )
            finally:
                store.close()

    def test_faq_legacy_alias_schema_is_rebuilt_and_cascades(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            seed = sqlite3.connect(db_path)
            seed.executescript(
                """
                CREATE TABLE faq_entries (
                    id TEXT PRIMARY KEY,
                    intent_key TEXT NOT NULL,
                    scope_key TEXT NOT NULL,
                    canonical_question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    source_signature TEXT NOT NULL,
                    knowledge_revision INTEGER NOT NULL CHECK(knowledge_revision >= 0),
                    state TEXT NOT NULL CHECK(state IN ('enabled', 'stale')),
                    direct_hits INTEGER NOT NULL DEFAULT 0 CHECK(direct_hits >= 0),
                    created_at REAL NOT NULL CHECK(created_at >= 0),
                    updated_at REAL NOT NULL CHECK(updated_at >= 0),
                    last_hit_at REAL CHECK(last_hit_at IS NULL OR last_hit_at >= 0),
                    UNIQUE(scope_key, intent_key)
                );
                CREATE TABLE faq_aliases (
                    faq_entry_id TEXT NOT NULL,
                    normalized_question TEXT NOT NULL,
                    alias_question TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                INSERT INTO faq_entries VALUES(
                    'faq-legacy', 'intent', 'global', '问题', '答案', 'source',
                    0, 'enabled', 0, 1.0, 1.0, NULL
                );
                INSERT INTO faq_aliases VALUES(
                    'faq-legacy', '问题', '问题', 1.0
                );
                """
            )
            seed.commit()
            seed.close()

            store = IndexStore(db_path)
            try:
                columns = {
                    row[1]
                    for row in store.connection.execute("PRAGMA table_info(faq_aliases)").fetchall()
                }
                self.assertTrue(
                    {
                        "faq_id",
                        "normalized_question",
                        "search_text",
                        "first_seen_at",
                        "last_seen_at",
                        "total_seen",
                    }.issubset(columns)
                )
                self.assertEqual(len(store.find_faq_candidates("global")), 1)
                foreign_key = next(
                    row
                    for row in store.connection.execute("PRAGMA foreign_key_list(faq_aliases)").fetchall()
                    if row[3] == "faq_id"
                )
                self.assertEqual(tuple(foreign_key[2:6]), ("faq_entries", "faq_id", "id", "NO ACTION"))
                self.assertEqual(foreign_key[6], "CASCADE")
                store.connection.execute("DELETE FROM faq_entries WHERE id = 'faq-legacy'")
                self.assertEqual(
                    store.connection.execute("SELECT COUNT(*) FROM faq_aliases").fetchone()[0],
                    0,
                )
            finally:
                store.close()

    def test_orphan_legacy_alias_aborts_alias_migration_and_rolls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            seed = sqlite3.connect(db_path)
            seed.executescript(
                """
                CREATE TABLE faq_entries (
                    id TEXT PRIMARY KEY,
                    intent_key TEXT NOT NULL,
                    scope_key TEXT NOT NULL,
                    canonical_question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    source_signature TEXT NOT NULL,
                    knowledge_revision INTEGER NOT NULL CHECK(knowledge_revision >= 0),
                    state TEXT NOT NULL CHECK(state IN ('enabled', 'stale')),
                    direct_hits INTEGER NOT NULL DEFAULT 0 CHECK(direct_hits >= 0),
                    created_at REAL NOT NULL CHECK(created_at >= 0),
                    updated_at REAL NOT NULL CHECK(updated_at >= 0),
                    last_hit_at REAL CHECK(last_hit_at IS NULL OR last_hit_at >= 0),
                    UNIQUE(scope_key, intent_key)
                );
                CREATE TABLE faq_aliases (
                    faq_entry_id TEXT NOT NULL,
                    normalized_question TEXT NOT NULL,
                    alias_question TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                INSERT INTO faq_entries VALUES(
                    'faq-real', 'intent', 'global', '问题', '答案', 'source',
                    0, 'enabled', 0, 1.0, 1.0, NULL
                );
                INSERT INTO faq_aliases VALUES(
                    'missing', '问题', '问题', 1.0
                );
                """
            )
            seed.commit()
            seed.close()

            with self.assertRaises(sqlite3.IntegrityError):
                IndexStore(db_path)
            check = sqlite3.connect(db_path)
            try:
                self.assertEqual(
                    {row[1] for row in check.execute("PRAGMA table_info(faq_aliases)").fetchall()},
                    {"faq_entry_id", "normalized_question", "alias_question", "created_at"},
                )
                self.assertEqual(
                    tuple(check.execute("SELECT faq_entry_id, normalized_question FROM faq_aliases").fetchone()),
                    ("missing", "问题"),
                )
            finally:
                check.close()

    def test_partial_scope_intent_unique_index_is_rebuilt_as_full_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            seed = sqlite3.connect(db_path)
            seed.executescript(
                """
                CREATE TABLE faq_entries (
                    id TEXT PRIMARY KEY,
                    intent_key TEXT NOT NULL,
                    scope_key TEXT NOT NULL,
                    canonical_question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    source_signature TEXT NOT NULL,
                    knowledge_revision INTEGER NOT NULL CHECK(knowledge_revision >= 0),
                    state TEXT NOT NULL CHECK(state IN ('enabled', 'stale')),
                    direct_hits INTEGER NOT NULL DEFAULT 0 CHECK(direct_hits >= 0),
                    created_at REAL NOT NULL CHECK(created_at >= 0),
                    updated_at REAL NOT NULL CHECK(updated_at >= 0),
                    last_hit_at REAL CHECK(last_hit_at IS NULL OR last_hit_at >= 0)
                );
                CREATE UNIQUE INDEX faq_enabled_scope_intent
                    ON faq_entries(scope_key, intent_key) WHERE state = 'enabled';
                INSERT INTO faq_entries VALUES(
                    'faq-one', 'intent', 'global', '问题', '答案', 'source',
                    0, 'enabled', 0, 1.0, 1.0, NULL
                );
                """
            )
            seed.commit()
            seed.close()

            store = IndexStore(db_path)
            try:
                with self.assertRaises(sqlite3.IntegrityError):
                    store.connection.execute(
                        "INSERT INTO faq_entries("
                        "id,intent_key,scope_key,canonical_question,answer,source_signature,"
                        "knowledge_revision,state,direct_hits,created_at,updated_at,last_hit_at"
                        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            "faq-two", "intent", "global", "问题2", "答案2", "source2",
                            0, "stale", 0, 1.0, 1.0, None,
                        ),
                    )
            finally:
                store.close()

    def test_faq_observation_promotes_on_third_recent_hit_and_returns_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                observation = FaqObservation("supplier-intake", "global", "供应商开发流程", "source-v1", 0)
                self.assertIsNone(
                    store.record_faq_observation(
                        observation, answer="安全答案", day="2026-08-17", now=1.0,
                        promotion_count=3,
                    )
                )
                self.assertIsNone(
                    store.record_faq_observation(
                        observation, answer="安全答案", day="2026-08-18", now=2.0,
                        promotion_count=3,
                    )
                )
                match = store.record_faq_observation(
                    observation, answer="安全答案", day="2026-08-31", now=3.0,
                    promotion_count=3,
                )
                self.assertIsNotNone(match)
                self.assertEqual(match.answer, "安全答案")
                self.assertEqual(
                    store.connection.execute("SELECT COUNT(*) FROM faq_entries").fetchone()[0],
                    1,
                )
                self.assertEqual(
                    store.connection.execute("SELECT total_seen FROM faq_aliases").fetchone()[0],
                    3,
                )
            finally:
                store.close()

    def test_faq_observation_promotes_after_three_observations_on_one_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                observation = FaqObservation("intent", "global", "真实问题", "source-v1", 0)
                matches = [
                    store.record_faq_observation(
                        observation, answer="答案", day="2026-08-31", now=float(index),
                        promotion_count=3,
                    )
                    for index in (1, 2, 3)
                ]
                self.assertEqual(sum(match is not None for match in matches), 1)
                self.assertEqual(
                    store.connection.execute(
                        "SELECT count FROM faq_observation_daily"
                    ).fetchone()[0],
                    3,
                )
            finally:
                store.close()

    def test_faq_observation_uses_real_question_for_alias_and_canonical_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                observation = FaqObservation(
                    "intent-hash", "global", "供应商开发流程是什么", "source-v1", 0
                )
                for day in ("2026-08-29", "2026-08-30", "2026-08-31"):
                    match = store.record_faq_observation(observation, answer="答案", day=day, now=1.0, promotion_count=3)
                self.assertIsNotNone(match)
                self.assertEqual(
                    store.connection.execute(
                        "SELECT canonical_question FROM faq_entries"
                    ).fetchone()[0],
                    "供应商开发流程是什么",
                )
                self.assertEqual(
                    len(store.find_faq_candidates("global")),
                    1,
                )
                self.assertEqual(len(store.find_faq_candidates("global")), 1)
            finally:
                store.close()

    def test_record_observation_refreshes_existing_entry_after_revision_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                old = FaqObservation("intent", "global", "问题", "source-v1", 0)
                for day in ("2026-08-29", "2026-08-30", "2026-08-31"):
                    match = store.record_faq_observation(old, answer="旧答案", day=day, now=1.0, promotion_count=3)
                entry_id = match.entry_id
                store.connection.execute(
                    "UPDATE faq_entries SET direct_hits = 7 WHERE id = ?", (entry_id,)
                )
                store.bump_knowledge_revision(now=1.5)
                refreshed = store.record_faq_observation(
                    FaqObservation("intent", "global", "问题", "source-v2", 1),
                    answer="新答案",
                    day="2026-09-01",
                    now=2.0,
                    promotion_count=3,
                )
                self.assertIsNotNone(refreshed)
                self.assertEqual(refreshed.answer, "新答案")
                self.assertEqual(
                    tuple(store.connection.execute(
                        "SELECT answer,source_signature,knowledge_revision,state,direct_hits "
                        "FROM faq_entries WHERE id = ?", (entry_id,)
                    ).fetchone()),
                    ("新答案", "source-v2", 1, "enabled", 7),
                )
            finally:
                store.close()

    def test_record_observation_rejects_late_revision_without_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                old = FaqObservation("intent", "global", "问题", "source-v1", 0)
                for day in ("2026-08-29", "2026-08-30", "2026-08-31"):
                    match = store.record_faq_observation(old, answer="旧答案", day=day, now=1.0, promotion_count=3)
                entry_id = match.entry_id
                store.connection.execute(
                    "UPDATE faq_entries SET direct_hits = 7 WHERE id = ?", (entry_id,)
                )
                store.bump_knowledge_revision(now=2.0)
                self.assertEqual(store.mark_faq_stale_before_revision(1), 1)
                before_entry = tuple(store.connection.execute(
                    "SELECT answer,state,knowledge_revision,direct_hits FROM faq_entries WHERE id = ?",
                    (entry_id,),
                ).fetchone())
                before_counts = tuple(store.connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(count), 0) FROM faq_observation_daily"
                ).fetchone())
                before_metrics = tuple(store.connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(refreshes), 0) FROM faq_metrics_daily"
                ).fetchone())
                self.assertIsNone(
                    store.record_faq_observation(
                        old, answer="迟到答案", day="2026-09-01", now=3.0, promotion_count=3
                    )
                )
                self.assertEqual(tuple(store.connection.execute(
                    "SELECT answer,state,knowledge_revision,direct_hits FROM faq_entries WHERE id = ?",
                    (entry_id,),
                ).fetchone()), before_entry)
                self.assertEqual(tuple(store.connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(count), 0) FROM faq_observation_daily"
                ).fetchone()), before_counts)
                self.assertEqual(tuple(store.connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(refreshes), 0) FROM faq_metrics_daily"
                ).fetchone()), before_metrics)
                refreshed = store.record_faq_observation(
                    FaqObservation("intent", "global", "问题", "source-v2", 1),
                    answer="新答案", day="2026-09-01", now=4.0,
                    promotion_count=3,
                )
                self.assertIsNotNone(refreshed)
                self.assertEqual(
                    tuple(store.connection.execute(
                        "SELECT answer,state,knowledge_revision,direct_hits FROM faq_entries WHERE id = ?",
                        (entry_id,),
                    ).fetchone()),
                    ("新答案", "enabled", 1, 7),
                )
            finally:
                store.close()

    def test_manual_faq_refresh_rejects_late_revision_and_accepts_current_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                old = FaqObservation("intent", "global", "问题", "source-v1", 0)
                for day in ("2026-08-29", "2026-08-30", "2026-08-31"):
                    match = store.record_faq_observation(old, answer="旧答案", day=day, now=1.0, promotion_count=3)
                entry_id = match.entry_id
                store.connection.execute(
                    "UPDATE faq_entries SET direct_hits = 7 WHERE id = ?", (entry_id,)
                )
                store.bump_knowledge_revision(now=2.0)
                self.assertEqual(store.mark_faq_stale_before_revision(1), 1)
                with self.assertRaisesRegex(ValueError, "revision"):
                    store.refresh_stale_faq(
                        entry_id,
                        old,
                        answer="迟到答案",
                        now=3.0,
                    )
                self.assertEqual(tuple(store.connection.execute(
                    "SELECT answer,state,knowledge_revision,direct_hits FROM faq_entries WHERE id = ?",
                    (entry_id,),
                ).fetchone()), ("旧答案", "stale", 0, 7))
                refreshed = store.refresh_stale_faq(
                    entry_id,
                    FaqObservation("intent", "global", "问题", "source-v2", 1),
                    answer="新答案",
                    now=4.0,
                )
                self.assertEqual(refreshed.answer, "新答案")
                self.assertEqual(tuple(store.connection.execute(
                    "SELECT answer,state,knowledge_revision,direct_hits FROM faq_entries WHERE id = ?",
                    (entry_id,),
                ).fetchone()), ("新答案", "enabled", 1, 7))
            finally:
                store.close()

    def test_faq_dates_require_iso_calendar_date_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                observation = FaqObservation("intent", "global", "问题", "source-v1", 0)
                for invalid_day in ("20260831", "2026-W35-7"):
                    with self.subTest(invalid_day=invalid_day):
                        with self.assertRaises(ValueError):
                            store.record_faq_observation(
                                observation, answer="答案", day=invalid_day, now=1.0,
                                promotion_count=3,
                            )
                with self.assertRaises(ValueError):
                    store.cleanup_faq(cutoff_day="20260831", stale_cutoff=0.0)
                with self.assertRaises(ValueError):
                    store.query_faq_metrics(since_day="2026-W35-7")
            finally:
                store.close()

    def test_faq_observation_does_not_mix_sources_or_expired_days(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                first = FaqObservation("intent", "global", "问题", "source-v1", 0)
                second = FaqObservation("intent", "global", "问题", "source-v2", 0)
                store.record_faq_observation(first, answer="a", day="2026-08-01", now=1.0, promotion_count=3)
                store.record_faq_observation(first, answer="a", day="2026-08-17", now=2.0, promotion_count=3)
                self.assertIsNone(
                    store.record_faq_observation(second, answer="b", day="2026-08-31", now=3.0, promotion_count=3)
                )
                self.assertEqual(
                    store.connection.execute("SELECT COUNT(*) FROM faq_entries").fetchone()[0],
                    0,
                )
            finally:
                store.close()

    def test_faq_observation_concurrent_third_hit_creates_one_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            seed = IndexStore(db_path)
            observation = FaqObservation("intent", "global", "问题", "source-v1", 0)
            seed.record_faq_observation(observation, answer="a", day="2026-08-29", now=1.0, promotion_count=3)
            seed.record_faq_observation(observation, answer="a", day="2026-08-30", now=2.0, promotion_count=3)
            seed.close()
            barrier = threading.Barrier(2)

            def observe():
                store = IndexStore(db_path)
                try:
                    barrier.wait(timeout=2)
                    return store.record_faq_observation(
                        observation, answer="a", day="2026-08-31", now=3.0,
                        promotion_count=3,
                    )
                finally:
                    store.close()

            with ThreadPoolExecutor(max_workers=2) as executor:
                matches = list(executor.map(lambda _: observe(), range(2)))
            self.assertEqual(sum(match is not None for match in matches), 1)
            store = IndexStore(db_path)
            try:
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM faq_entries").fetchone()[0], 1)
            finally:
                store.close()

    def test_faq_refresh_replaces_stale_content_without_clearing_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                observation = FaqObservation("intent", "global", "问题", "source-v1", 0)
                for day in ("2026-08-29", "2026-08-30", "2026-08-31"):
                    match = store.record_faq_observation(
                        observation, answer="旧答案", day=day, now=1.0,
                        promotion_count=3,
                    )
                entry_id = match.entry_id
                store.connection.execute("UPDATE faq_entries SET direct_hits=4,state='stale' WHERE id=?", (entry_id,))
                store.bump_knowledge_revision(now=2.0)
                refreshed = store.refresh_stale_faq(
                    entry_id,
                    FaqObservation("intent", "global", "问题", "source-v2", 1),
                    answer="新答案",
                    now=5.0,
                )
                self.assertEqual(refreshed.answer, "新答案")
                self.assertEqual(
                    tuple(store.connection.execute("SELECT source_signature,knowledge_revision,state,direct_hits FROM faq_entries WHERE id=?", (entry_id,)).fetchone()),
                    ("source-v2", 1, "enabled", 4),
                )
            finally:
                store.close()

    def test_faq_metrics_validate_fields_aggregate_and_cleanup_preserves_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.record_faq_metric("2026-08-31", "direct_hits")
                store.record_faq_metric("2026-08-31", "direct_hits")
                with self.assertRaises(ValueError):
                    store.record_faq_metric("2026-08-31", "scope_key")
                metrics = store.query_faq_metrics(since_day="2026-08-31")
                self.assertEqual(metrics[0]["direct_hits"], 2)
                observation = FaqObservation("intent", "global", "问题", "source-v1", 0)
                for day in ("2026-08-29", "2026-08-30", "2026-08-31"):
                    store.record_faq_observation(observation, answer="a", day=day, now=1.0, promotion_count=3)
                removed = store.cleanup_faq(cutoff_day="2026-08-30", stale_cutoff=0.0)
                self.assertGreaterEqual(removed["observations"], 1)
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM faq_entries").fetchone()[0], 1)
            finally:
                store.close()

    def test_record_faq_direct_hit_updates_entry_and_daily_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                observation = FaqObservation("intent", "global", "问题", "source-v1", 0)
                for day in ("2026-08-29", "2026-08-30", "2026-08-31"):
                    match = store.record_faq_observation(
                        observation, answer="答案", day=day, now=1.0, promotion_count=3
                    )
                store.record_faq_direct_hit(match.entry_id, now=1725148800.0)
                self.assertEqual(
                    store.connection.execute(
                        "SELECT direct_hits FROM faq_entries WHERE id = ?", (match.entry_id,)
                    ).fetchone()[0],
                    1,
                )
                metrics = store.query_faq_metrics(since_day="2024-01-01")
                self.assertEqual(sum(int(row["direct_hits"]) for row in metrics), 1)
            finally:
                store.close()

    def test_record_faq_direct_hit_aggregates_global_scope_with_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                observation = FaqObservation("intent", "global", "问题", "source-v1", 0)
                for day in ("2026-08-29", "2026-08-30", "2026-08-31"):
                    match = store.record_faq_observation(
                        observation, answer="答案", day=day, now=1.0, promotion_count=3
                    )
                store.record_faq_metric(
                    "2026-08-31", "eligible_questions", scope_key="global"
                )
                store.record_faq_direct_hit(match.entry_id, now=1788134400.0)
                row = next(
                    row for row in store.query_faq_metrics(since_day="2026-08-31")
                    if row["scope_key"] == "global"
                )
                self.assertEqual(row["eligible_questions"], 1)
                self.assertEqual(row["direct_hits"], 1)
            finally:
                store.close()

    def test_record_faq_direct_hit_aggregates_restricted_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                scope = RetrievalScope(frozenset({"space-a"}))
                scope_key = FaqService._scope_key(scope)
                observation = FaqObservation("intent", scope_key, "问题", "source-v1", 0)
                for day in ("2026-08-29", "2026-08-30", "2026-08-31"):
                    match = store.record_faq_observation(
                        observation, answer="答案", day=day, now=1.0, promotion_count=3
                    )
                store.record_faq_metric(
                    "2026-08-31", "eligible_questions", scope_key=scope_key
                )
                store.record_faq_direct_hit(match.entry_id, now=1788134400.0)
                row = next(
                    row for row in store.query_faq_metrics(since_day="2026-08-31")
                    if row["scope_key"] == scope_key
                )
                self.assertEqual(row["eligible_questions"], 1)
                self.assertEqual(row["direct_hits"], 1)
            finally:
                store.close()

    def test_record_faq_direct_hit_unknown_entry_does_not_create_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                self.assertFalse(
                    store.record_faq_direct_hit("missing", now=1788134400.0)
                )
                self.assertEqual(store.query_faq_metrics(), [])
            finally:
                store.close()

    def test_record_faq_direct_hit_rejects_stale_expected_revision_without_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                observation = FaqObservation("intent", "global", "问题", "source-v1", 0)
                for day in ("2026-08-29", "2026-08-30", "2026-08-31"):
                    match = store.record_faq_observation(
                        observation, answer="答案", day=day, now=1.0, promotion_count=3
                    )
                store.bump_knowledge_revision(now=2.0)
                self.assertFalse(
                    store.record_faq_direct_hit(
                        match.entry_id, expected_revision=0,
                        day="2026-08-31", now=1788134400.0,
                    )
                )
                self.assertEqual(
                    store.connection.execute(
                        "SELECT direct_hits FROM faq_entries WHERE id = ?", (match.entry_id,)
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(
                    sum(row["direct_hits"] for row in store.query_faq_metrics()), 0
                )
            finally:
                store.close()

    def test_empty_database_creates_faq_schema_and_initial_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                table_names = {
                    row[0]
                    for row in store.connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                self.assertTrue(
                    {
                        "knowledge_state",
                        "faq_entries",
                        "faq_aliases",
                        "faq_observation_daily",
                        "faq_metrics_daily",
                    }.issubset(table_names)
                )
                self.assertEqual(store.knowledge_revision(), 0)
                self.assertEqual(
                    tuple(
                        store.connection.execute(
                            "SELECT singleton_id, revision, updated_at FROM knowledge_state"
                        ).fetchone()
                    ),
                    (1, 0, 0.0),
                )
            finally:
                store.close()

    def test_legacy_v023_database_preserves_content_and_starts_at_revision_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            connection = sqlite3.connect(db_path)
            connection.executescript(
                """
                CREATE TABLE documents (
                    source_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE chunks (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES documents(source_id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    page INTEGER,
                    section TEXT
                );
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    chunk_id UNINDEXED, title, content
                );
                CREATE TABLE processed_messages (
                    message_id TEXT PRIMARY KEY,
                    processed_at REAL NOT NULL
                );
                INSERT INTO documents VALUES('legacy.txt', '旧文档', 'legacy.txt', 'v1', 0);
                INSERT INTO chunks VALUES('legacy-chunk', 'legacy.txt', '旧文档', '报销内容', NULL, NULL);
                INSERT INTO chunks_fts VALUES('legacy-chunk', '旧文档', '报销内容');
                INSERT INTO processed_messages VALUES('om_legacy', 1);
                """
            )
            connection.commit()
            connection.close()

            store = IndexStore(db_path)
            try:
                self.assertEqual(store.knowledge_revision(), 0)
                self.assertEqual(store.count_documents(), 1)
                self.assertEqual(store.count_chunks(), 1)
                self.assertEqual(store.search("报销内容")[0].chunk.id, "legacy-chunk")
                self.assertEqual(store.claim_message_state("om_legacy", retention_seconds=10**12), "completed")
            finally:
                store.close()

    def test_faq_migration_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            first = IndexStore(db_path)
            try:
                self.assertEqual(first.bump_knowledge_revision(now=10), 1)
            finally:
                first.close()

            second = IndexStore(db_path)
            try:
                self.assertEqual(second.knowledge_revision(), 1)
                self.assertEqual(
                    second.connection.execute("SELECT COUNT(*) FROM knowledge_state").fetchone()[0],
                    1,
                )
            finally:
                second.close()

    def test_knowledge_revision_is_monotonic_and_upsert_does_not_bump_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                self.assertEqual(store.knowledge_revision(), 0)
                self.assertEqual(store.bump_knowledge_revision(now=12.5), 1)
                self.assertEqual(store.bump_knowledge_revision(now=13.5), 2)
                self.assertEqual(
                    tuple(
                        store.connection.execute(
                            "SELECT revision, updated_at FROM knowledge_state"
                        ).fetchone()
                    ),
                    (2, 13.5),
                )
                store.upsert_document(
                    "policy.txt",
                    "报销制度",
                    "policy.txt",
                    "v1",
                    [Chunk("policy", "policy.txt", "报销制度", "报销内容")],
                )
                self.assertEqual(store.knowledge_revision(), 2)
            finally:
                store.close()

    def test_apply_document_snapshot_commits_updates_and_revision_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                updates = [
                    PreparedDocument(
                        "one.txt",
                        "第一份",
                        "one.txt",
                        "checksum-one",
                        (Chunk("one", "one.txt", "第一份", "内容一"),),
                    ),
                    PreparedDocument(
                        "two.txt",
                        "第二份",
                        "two.txt",
                        "checksum-two",
                        (Chunk("two", "two.txt", "第二份", "内容二"),),
                    ),
                ]

                self.assertEqual(store.apply_document_snapshot(updates), (2, 0))
                self.assertEqual(store.count_documents(), 2)
                self.assertEqual(store.knowledge_revision(), 1)
            finally:
                store.close()

    def test_apply_document_snapshot_invalidates_enabled_faqs_and_reports_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "feishu:space:old", "旧", "old", "v1",
                    [Chunk("old-chunk", "feishu:space:old", "旧", "旧内容")],
                )
                with store.connection:
                    store.connection.execute(
                        "INSERT INTO faq_entries(id,intent_key,scope_key,canonical_question,answer,"
                        "source_signature,source_ids_json,knowledge_revision,state,created_at,updated_at) "
                        "VALUES ('faq','intent','scope','问题','答案','sig','[]',0,'enabled',1,1)"
                    )
                updated, deleted = store.apply_document_snapshot(
                    [PreparedDocument(
                        "feishu:space:new", "新", "new", "v1",
                        (Chunk("new-chunk", "feishu:space:new", "新", "新内容"),),
                    )],
                    prune_prefix="feishu:space:",
                    retained={"feishu:space:new"},
                )
                self.assertEqual((updated, deleted), (1, 1))
                self.assertEqual(store.knowledge_revision(), 1)
                self.assertEqual(
                    store.connection.execute("SELECT state FROM faq_entries WHERE id='faq'").fetchone()[0],
                    "stale",
                )
                self.assertEqual(
                    store.query_faq_metrics()[0]["invalidations"],
                    1,
                )
                self.assertEqual(store.record_faq_metric("2026-09-01", "invalidations", scope_key="x"), None)
            finally:
                store.close()

    def test_apply_document_snapshot_rolls_back_all_updates_on_commit_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                updates = [
                    PreparedDocument(
                        "one.txt",
                        "第一份",
                        "one.txt",
                        "checksum-one",
                        (Chunk("duplicate", "one.txt", "第一份", "内容一"),),
                    ),
                    PreparedDocument(
                        "two.txt",
                        "第二份",
                        "two.txt",
                        "checksum-two",
                        (Chunk("duplicate", "two.txt", "第二份", "内容二"),),
                    ),
                ]

                with self.assertRaises(sqlite3.IntegrityError):
                    store.apply_document_snapshot(updates)
                self.assertEqual(store.count_documents(), 0)
                self.assertEqual(store.knowledge_revision(), 0)
            finally:
                store.close()

    def test_first_local_snapshot_failure_rolls_back_legacy_cleanup_and_root_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "legacy.txt", "遗留", "legacy.txt", "v1",
                    [Chunk("legacy-chunk", "legacy.txt", "遗留", "遗留内容")],
                )
                root = Path(tmp) / "root"
                root_hash = sha256(str(root.resolve()).encode("utf-8")).hexdigest()
                updates = [
                    PreparedDocument(
                        f"local:{root_hash}:one.txt", "一", "one.txt", "v1",
                        (Chunk("duplicate", f"local:{root_hash}:one.txt", "一", "内容一"),),
                    ),
                    PreparedDocument(
                        f"local:{root_hash}:two.txt", "二", "two.txt", "v1",
                        (Chunk("duplicate", f"local:{root_hash}:two.txt", "二", "内容二"),),
                    ),
                ]
                with self.assertRaises(sqlite3.IntegrityError):
                    store.apply_document_snapshot(updates, local_root=root, retained={item.source_id for item in updates})
                self.assertIsNotNone(store.document_checksum("legacy.txt"))
                self.assertEqual(
                    store.connection.execute("SELECT COUNT(*) FROM local_index_roots").fetchone()[0],
                    0,
                )
                self.assertEqual(store.knowledge_revision(), 0)
            finally:
                store.close()

    def test_first_local_snapshot_commit_failure_rolls_back_legacy_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            real_connection = store.connection
            try:
                store.upsert_document(
                    "legacy.txt", "遗留", "legacy.txt", "v1",
                    [Chunk("legacy-chunk", "legacy.txt", "遗留", "遗留内容")],
                )

                class FailingCommitConnection:
                    def execute(self, statement, parameters=()):
                        return real_connection.execute(statement, parameters)

                    def commit(self):
                        raise sqlite3.OperationalError("simulated commit failure")

                    def rollback(self):
                        return real_connection.rollback()

                    def __getattr__(self, name):
                        return getattr(real_connection, name)

                store.connection = FailingCommitConnection()
                root = Path(tmp) / "root"
                root_hash = sha256(str(root.resolve()).encode("utf-8")).hexdigest()
                source_id = f"local:{root_hash}:one.txt"
                with self.assertRaises(sqlite3.OperationalError):
                    store.apply_document_snapshot(
                        [PreparedDocument(
                            source_id, "一", "one.txt", "v1",
                            (Chunk("one-chunk", source_id, "一", "内容一"),),
                        )],
                        local_root=root,
                        retained={source_id},
                    )
                store.connection = real_connection
                self.assertIsNotNone(store.document_checksum("legacy.txt"))
                self.assertIsNone(store.document_checksum(source_id))
                self.assertEqual(
                    store.connection.execute("SELECT COUNT(*) FROM local_index_roots").fetchone()[0],
                    0,
                )
                self.assertEqual(store.knowledge_revision(), 0)
            finally:
                store.connection = real_connection
                store.close()

    def test_knowledge_revision_bump_rolls_back_when_commit_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            real_connection = store.connection

            class FailingCommitConnection:
                def __init__(self, connection: sqlite3.Connection):
                    self._connection = connection
                    self.rollback_called = False

                def execute(self, statement, parameters=()):
                    return self._connection.execute(statement, parameters)

                def commit(self):
                    raise sqlite3.OperationalError("simulated commit failure")

                def rollback(self):
                    self.rollback_called = True
                    return self._connection.rollback()

                def __getattr__(self, name):
                    return getattr(self._connection, name)

            failing_connection = FailingCommitConnection(real_connection)
            store.connection = failing_connection
            try:
                with self.assertRaises(sqlite3.OperationalError):
                    store.bump_knowledge_revision(now=10)
                self.assertTrue(failing_connection.rollback_called)
                store.connection = real_connection
                self.assertEqual(store.knowledge_revision(), 0)
                self.assertEqual(store.bump_knowledge_revision(now=11), 1)
            finally:
                store.connection = real_connection
                store.close()

    def test_faq_schema_enforces_entry_and_observation_uniqueness(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                entry = (
                    "faq-1", "intent", "scope", "问题", "答案", "source", 0,
                    "enabled", 0, 1.0, 1.0, None,
                )
                store.connection.execute(
                    "INSERT INTO faq_entries("
                    "id,intent_key,scope_key,canonical_question,answer,source_signature,"
                    "knowledge_revision,state,direct_hits,created_at,updated_at,last_hit_at"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    entry,
                )
                with self.assertRaises(sqlite3.IntegrityError):
                    store.connection.execute(
                        "INSERT INTO faq_entries("
                        "id,intent_key,scope_key,canonical_question,answer,source_signature,"
                        "knowledge_revision,state,direct_hits,created_at,updated_at,last_hit_at"
                        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        ("faq-2",) + entry[1:],
                    )
                observation = (
                    "intent",
                    "scope",
                    "2026-08-31",
                    1,
                    "问题",
                    "source",
                    0,
                    "答案",
                )
                store.connection.execute(
                    "INSERT INTO faq_observation_daily("
                    "intent_key,scope_key,day,count,normalized_question,source_signature,"
                    "knowledge_revision,latest_safe_answer"
                    ") VALUES(?,?,?,?,?,?,?,?)",
                    observation,
                )
                with self.assertRaises(sqlite3.IntegrityError):
                    store.connection.execute(
                        "INSERT INTO faq_observation_daily("
                        "intent_key,scope_key,day,count,normalized_question,source_signature,"
                        "knowledge_revision,latest_safe_answer"
                        ") VALUES(?,?,?,?,?,?,?,?)",
                        observation,
                    )
            finally:
                store.close()

    def test_faq_entries_reject_invalid_state_and_negative_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                def insert_entry(
                    entry_id: str,
                    state: str = "enabled",
                    knowledge_revision: int = 0,
                    direct_hits: int = 0,
                    created_at: float = 1.0,
                    updated_at: float = 1.0,
                    last_hit_at: float | None = None,
                ) -> None:
                    store.connection.execute(
                        "INSERT INTO faq_entries("
                        "id,intent_key,scope_key,canonical_question,answer,source_signature,"
                        "knowledge_revision,state,direct_hits,created_at,updated_at,last_hit_at"
                        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            entry_id,
                            f"intent-{entry_id}",
                            f"scope-{entry_id}",
                            "问题",
                            "答案",
                            "source",
                            knowledge_revision,
                            state,
                            direct_hits,
                            created_at,
                            updated_at,
                            last_hit_at,
                        ),
                    )

                with self.assertRaises(sqlite3.IntegrityError):
                    insert_entry("invalid-state", state="active")
                for field, value in (
                    ("knowledge_revision", -1),
                    ("direct_hits", -1),
                    ("created_at", -1.0),
                    ("updated_at", -1.0),
                    ("last_hit_at", -1.0),
                ):
                    with self.subTest(field=field):
                        kwargs = {field: value}
                        with self.assertRaises(sqlite3.IntegrityError):
                            insert_entry(f"invalid-{field}", **kwargs)
            finally:
                store.close()

    def test_existing_faq_entries_without_checks_are_migrated_and_keep_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            seed = sqlite3.connect(db_path)
            seed.executescript(
                """
                CREATE TABLE faq_entries (
                    id TEXT PRIMARY KEY,
                    intent_key TEXT NOT NULL,
                    scope_key TEXT NOT NULL,
                    canonical_question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    source_signature TEXT NOT NULL,
                    knowledge_revision INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    direct_hits INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_hit_at REAL,
                    UNIQUE(scope_key, intent_key)
                );
                CREATE TABLE faq_aliases (
                    faq_id TEXT NOT NULL REFERENCES faq_entries(id) ON DELETE CASCADE,
                    normalized_question TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    first_seen_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    total_seen INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(faq_id, normalized_question)
                );
                INSERT INTO faq_entries VALUES(
                    'faq-legacy', 'intent', 'scope', '问题', '答案', 'source',
                    0, 'enabled', 2, 1.0, 2.0, 2.0
                );
                INSERT INTO faq_aliases VALUES(
                    'faq-legacy', '问题', '问题', 1.0, 2.0, 2
                );
                """
            )
            seed.commit()
            seed.close()

            store = IndexStore(db_path)
            try:
                self.assertEqual(
                    tuple(
                        store.connection.execute(
                            "SELECT id,state,direct_hits FROM faq_entries"
                        ).fetchone()
                    ),
                    ("faq-legacy", "enabled", 2),
                )
                self.assertEqual(
                    store.connection.execute(
                        "SELECT faq_id FROM faq_aliases"
                    ).fetchone()[0],
                    "faq-legacy",
                )
                self.assertEqual(
                    store.connection.execute(
                        "SELECT source_ids_json FROM faq_entries"
                    ).fetchone()[0],
                    "[]",
                )
                with self.assertRaises(sqlite3.IntegrityError):
                    store.connection.execute(
                        "INSERT INTO faq_entries VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            "faq-invalid-state", "intent-2", "scope-2", "问题",
                            "答案", "source", "[]", 0, "active", 0, 1.0, 1.0, None,
                        ),
                    )
                with self.assertRaises(sqlite3.IntegrityError):
                    store.connection.execute(
                        "INSERT INTO faq_entries VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            "faq-invalid-revision", "intent-3", "scope-3", "问题",
                            "答案", "source", "[]", -1, "enabled", 0, 1.0, 1.0, None,
                        ),
                    )
            finally:
                store.close()

    def test_invalid_existing_faq_entry_aborts_migration_and_preserves_original_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            seed = sqlite3.connect(db_path)
            seed.executescript(
                """
                CREATE TABLE faq_entries (
                    id TEXT PRIMARY KEY,
                    intent_key TEXT NOT NULL,
                    scope_key TEXT NOT NULL,
                    canonical_question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    source_signature TEXT NOT NULL,
                    knowledge_revision INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    direct_hits INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_hit_at REAL,
                    UNIQUE(scope_key, intent_key)
                );
                INSERT INTO faq_entries VALUES(
                    'faq-invalid', 'intent', 'scope', '问题', '答案', 'source',
                    -1, 'enabled', 0, 1.0, 1.0, NULL
                );
                """
            )
            seed.commit()
            seed.close()

            with self.assertRaisesRegex(RuntimeError, "faq_entries"):
                IndexStore(db_path)

            check = sqlite3.connect(db_path)
            try:
                self.assertEqual(
                    tuple(
                        check.execute(
                            "SELECT id,knowledge_revision,state FROM faq_entries"
                        ).fetchone()
                    ),
                    ("faq-invalid", -1, "enabled"),
                )
                self.assertIsNotNone(
                    check.execute(
                        "SELECT sql FROM sqlite_master WHERE type='table' AND name='faq_entries'"
                    ).fetchone()[0]
                )
            finally:
                check.close()

    def test_concurrent_knowledge_revision_bumps_are_contiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            IndexStore(db_path).close()
            barrier = threading.Barrier(2)

            def bump() -> int:
                store = IndexStore(db_path)
                try:
                    barrier.wait(timeout=2)
                    return store.bump_knowledge_revision(now=100)
                finally:
                    store.close()

            with ThreadPoolExecutor(max_workers=2) as executor:
                revisions = list(executor.map(lambda _: bump(), range(2)))

            self.assertEqual(sorted(revisions), [1, 2])
            store = IndexStore(db_path)
            try:
                self.assertEqual(store.knowledge_revision(), 2)
            finally:
                store.close()

    def test_fts_migration_failure_preserves_documents_and_faq_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            seed = sqlite3.connect(db_path)
            seed.executescript(
                """
                CREATE TABLE documents (
                    source_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE chunks (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES documents(source_id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    page INTEGER,
                    section TEXT
                );
                INSERT INTO documents VALUES('legacy.txt', '旧文档', 'legacy.txt', 'v1', 0);
                INSERT INTO chunks VALUES('legacy-chunk', 'legacy.txt', '旧文档', '旧内容', NULL, NULL);
                """
            )
            seed.commit()
            seed.close()

            real_connect = sqlite3.connect(db_path, timeout=30.0)

            class FailingFtsConnection:
                def __init__(self, connection: sqlite3.Connection):
                    self._connection = connection

                def execute(self, statement, parameters=()):
                    if "CREATE VIRTUAL TABLE" in statement:
                        raise sqlite3.OperationalError("simulated FTS DDL failure")
                    return self._connection.execute(statement, parameters)

                @property
                def row_factory(self):
                    return self._connection.row_factory

                @row_factory.setter
                def row_factory(self, value):
                    self._connection.row_factory = value

                def __getattr__(self, name):
                    return getattr(self._connection, name)

            failing_connection = FailingFtsConnection(real_connect)
            with patch(
                "feishu_rag.store.sqlite3.connect",
                return_value=failing_connection,
            ):
                store = IndexStore(db_path)
            try:
                self.assertFalse(store._fts_available)
                self.assertEqual(store.count_documents(), 1)
                self.assertEqual(store.count_chunks(), 1)
                self.assertEqual(store.search("旧内容")[0].chunk.id, "legacy-chunk")
                tables = {
                    row[0]
                    for row in store.connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                self.assertTrue(
                    {"knowledge_state", "faq_entries", "faq_aliases"}.issubset(tables)
                )
            finally:
                store.close()

    def test_faq_schema_contains_alias_observation_and_metrics_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                expected_columns = {
                    "faq_aliases": {
                        "faq_id",
                        "normalized_question",
                        "search_text",
                        "first_seen_at",
                        "last_seen_at",
                        "total_seen",
                    },
                    "faq_observation_daily": {
                        "intent_key",
                        "scope_key",
                        "day",
                        "count",
                        "normalized_question",
                        "source_signature",
                        "knowledge_revision",
                        "latest_safe_answer",
                    },
                    "faq_metrics_daily": {
                        "day",
                        "eligible_questions",
                        "rag_answers",
                        "direct_hits",
                        "promotions",
                        "refreshes",
                        "rejected_answers",
                    },
                }
                for table, columns in expected_columns.items():
                    actual = {
                        row[1]
                        for row in store.connection.execute(
                            f"PRAGMA table_info({table})"
                        ).fetchall()
                    }
                    self.assertTrue(columns.issubset(actual), table)
            finally:
                store.close()

    def test_set_document_space_updates_only_an_existing_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "legacy.txt",
                    "旧文档",
                    "legacy.txt",
                    "v1",
                    [Chunk("legacy", "legacy.txt", "旧文档", "报销内容")],
                )

                self.assertTrue(store.set_document_space("legacy.txt", "space-a"))
                self.assertFalse(store.set_document_space("missing.txt", "space-b"))
                self.assertEqual(store.count_documents(), 1)
                self.assertEqual(
                    store.connection.execute(
                        "SELECT space_id FROM documents WHERE source_id = 'legacy.txt'"
                    ).fetchone()[0],
                    "space-a",
                )
            finally:
                store.close()

    def test_initialization_migrates_documents_space_id_and_creates_rate_limit_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            connection = sqlite3.connect(db_path)
            connection.executescript(
                """
                CREATE TABLE documents (
                    source_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                INSERT INTO documents VALUES('legacy.txt', '旧文档', 'legacy.txt', 'v1', 0);
                """
            )
            connection.commit()
            connection.close()

            store = IndexStore(db_path)
            try:
                space_column = next(
                    row
                    for row in store.connection.execute(
                        "PRAGMA table_info(documents)"
                    ).fetchall()
                    if row[1] == "space_id"
                )
                self.assertEqual(space_column[2], "TEXT")
                self.assertEqual(space_column[3], 1)
                self.assertEqual(space_column[4], "''")
                self.assertEqual(
                    store.connection.execute(
                        "SELECT space_id FROM documents WHERE source_id = 'legacy.txt'"
                    ).fetchone()[0],
                    "",
                )
                self.assertEqual(
                    [
                        row[1]
                        for row in store.connection.execute(
                            "PRAGMA table_info(rate_limit_buckets)"
                        ).fetchall()
                    ],
                    ["user_hash", "window_seconds", "bucket_start", "count"],
                )
            finally:
                store.close()

    def test_legacy_feishu_documents_backfill_space_id_for_scoped_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            connection = sqlite3.connect(db_path)
            connection.executescript(
                """
                CREATE TABLE documents (
                    source_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE chunks (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES documents(source_id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    page INTEGER,
                    section TEXT
                );
                INSERT INTO documents VALUES(
                    'feishu:space-a:node-1', '电子发票', 'wiki/node-1', 'v1', 0
                );
                INSERT INTO documents VALUES('local.txt', '本地文档', 'local.txt', 'v1', 0);
                INSERT INTO documents VALUES(
                    'feishu::malformed', '异常文档', 'malformed', 'v1', 0
                );
                INSERT INTO chunks VALUES(
                    'feishu-chunk', 'feishu:space-a:node-1', '电子发票',
                    '电子发票需要核验。', NULL, NULL
                );
                """
            )
            connection.commit()
            connection.close()

            store = IndexStore(db_path)
            try:
                spaces = dict(
                    store.connection.execute(
                        "SELECT source_id,space_id FROM documents ORDER BY source_id"
                    ).fetchall()
                )
                self.assertEqual(spaces["feishu:space-a:node-1"], "space-a")
                self.assertEqual(spaces["local.txt"], "")
                self.assertEqual(spaces["feishu::malformed"], "")
                self.assertEqual(
                    store.search(
                        "电子发票",
                        scope=RetrievalScope(frozenset({""})),
                    ),
                    [],
                )
                self.assertEqual(
                    store.search(
                        "电子发票",
                        scope=RetrievalScope(frozenset({"space-a"})),
                    )[0].chunk.id,
                    "feishu-chunk",
                )
            finally:
                store.close()

    def test_busy_timeout_is_configured_before_journal_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            real_connection = sqlite3.connect(Path(tmp) / "rag.sqlite3")
            statements = []

            class RecordingConnection:
                def __getattr__(self, name):
                    return getattr(real_connection, name)

                def __setattr__(self, name, value):
                    if name in {"row_factory"}:
                        setattr(real_connection, name, value)
                    else:
                        object.__setattr__(self, name, value)

                def execute(self, sql, parameters=()):
                    statements.append(sql)
                    return real_connection.execute(sql, parameters)

            with patch("feishu_rag.store.sqlite3.connect", return_value=RecordingConnection()):
                store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                busy_index = statements.index("PRAGMA busy_timeout = 30000")
                journal_index = statements.index("PRAGMA journal_mode = WAL")
                self.assertLess(busy_index, journal_index)
            finally:
                store.close()

    def test_concurrent_initialization_of_same_legacy_database_is_consistent(self):
        for iteration in range(3):
            with self.subTest(iteration=iteration), tempfile.TemporaryDirectory() as tmp:
                db_path = Path(tmp) / "rag.sqlite3"
                connection = sqlite3.connect(db_path)
                connection.executescript(
                    """
                    CREATE TABLE documents (
                        source_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        path TEXT NOT NULL,
                        checksum TEXT NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    CREATE TABLE chunks (
                        id TEXT PRIMARY KEY,
                        source_id TEXT NOT NULL REFERENCES documents(source_id) ON DELETE CASCADE,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        page INTEGER,
                        section TEXT
                    );
                    INSERT INTO documents VALUES(
                        'feishu:space-a:node-1', '电子发票', 'wiki/node-1', 'v1', 0
                    );
                    INSERT INTO chunks VALUES(
                        'legacy', 'feishu:space-a:node-1', '电子发票',
                        '电子发票需要核验。', NULL, NULL
                    );
                    CREATE VIRTUAL TABLE chunks_fts USING fts5(
                        chunk_id UNINDEXED, title, content
                    );
                    INSERT INTO chunks_fts VALUES(
                        'legacy', '电子发票', '电子发票需要核验。'
                    );
                    """
                )
                connection.commit()
                connection.close()
                barrier = threading.Barrier(8)

                def initialize():
                    barrier.wait()
                    store = IndexStore(db_path)
                    try:
                        return (
                            [
                                row[1]
                                for row in store.connection.execute(
                                    "PRAGMA table_info(documents)"
                                ).fetchall()
                            ],
                            [
                                row[1]
                                for row in store.connection.execute(
                                    "PRAGMA table_info(chunks)"
                                ).fetchall()
                            ],
                            [
                                row[1]
                                for row in store.connection.execute(
                                    "PRAGMA table_info(chunks_fts)"
                                ).fetchall()
                            ],
                            store.connection.execute(
                                "SELECT space_id FROM documents "
                                "WHERE source_id = 'feishu:space-a:node-1'"
                            ).fetchone()[0],
                            store.connection.execute(
                                "SELECT COUNT(*) FROM chunks_fts "
                                "WHERE chunk_id = 'legacy'"
                            ).fetchone()[0],
                        )
                    finally:
                        store.close()

                with ThreadPoolExecutor(max_workers=8) as executor:
                    results = list(executor.map(lambda _: initialize(), range(8)))

                for document_columns, chunk_columns, fts_columns, space_id, fts_count in results:
                    self.assertIn("space_id", document_columns)
                    self.assertIn("search_text", chunk_columns)
                    self.assertEqual(
                        fts_columns,
                        ["chunk_id", "title_terms", "content_terms", "search_terms"],
                    )
                    self.assertEqual(space_id, "space-a")
                    self.assertEqual(fts_count, 1)

    def test_retrieval_scope_preserves_an_explicit_immutable_space_set(self):
        allowed = frozenset({"space-a", "space-b"})
        self.assertEqual(RetrievalScope(allowed).allowed_space_ids, allowed)

    def test_pretokenize_normalizes_nfkc_case_and_chinese_terms(self):
        terms = _pretokenize("ＡＢＣ１２ 报销")
        self.assertIn("abc12", terms.split())
        self.assertIn("报销", terms.split())
        self.assertIn("报", terms.split())
        self.assertIn("销", terms.split())

    def test_tokens_exclude_single_chinese_when_requested(self):
        self.assertEqual(_tokens("这", include_single_chinese=False), [])
        self.assertEqual(_tokens("电子", include_single_chinese=False), ["电子"])

    def test_initialization_migrates_old_fts_schema_and_backfills_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            connection = sqlite3.connect(db_path)
            connection.executescript(
                """
                CREATE TABLE documents (
                    source_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE chunks (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES documents(source_id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    page INTEGER,
                    section TEXT,
                    search_text TEXT NOT NULL DEFAULT ''
                );
                INSERT INTO documents VALUES('legacy.pdf', '报销指南', 'legacy.pdf', 'v1', 0);
                INSERT INTO chunks VALUES(
                    'legacy-chunk', 'legacy.pdf', '报销指南', '提交申请单。', NULL, NULL,
                    '电子发票 核验'
                );
                CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, title, content);
                INSERT INTO chunks_fts VALUES('legacy-chunk', '报销指南', '提交申请单。');
                """
            )
            connection.commit()
            connection.close()

            store = IndexStore(db_path)
            try:
                columns = [
                    row[1]
                    for row in store.connection.execute("PRAGMA table_info(chunks_fts)").fetchall()
                ]
                self.assertEqual(
                    columns,
                    ["chunk_id", "title_terms", "content_terms", "search_terms"],
                )
                fts_row = store.connection.execute(
                    "SELECT title_terms,content_terms,search_terms FROM chunks_fts "
                    "WHERE chunk_id = 'legacy-chunk'"
                ).fetchone()
                self.assertIn("报销", fts_row[0].split())
                self.assertIn("申请", fts_row[1].split())
                self.assertIn("发票", fts_row[2].split())
                self.assertEqual(store.search("电子发票")[0].chunk.id, "legacy-chunk")
            finally:
                store.close()

    def test_failed_fts_backfill_rolls_back_and_next_start_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            connection = sqlite3.connect(db_path)
            connection.executescript(
                """
                CREATE TABLE documents (
                    source_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE chunks (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES documents(source_id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    page INTEGER,
                    section TEXT,
                    search_text TEXT NOT NULL DEFAULT ''
                );
                INSERT INTO documents VALUES('one.pdf', '报销指南', 'one.pdf', 'v1', 0);
                INSERT INTO documents VALUES('two.pdf', '发票指南', 'two.pdf', 'v1', 0);
                INSERT INTO chunks VALUES(
                    'one', 'one.pdf', '报销指南', '提交申请。', NULL, NULL, ''
                );
                INSERT INTO chunks VALUES(
                    'two', 'two.pdf', '发票指南', '核验发票。', NULL, NULL, ''
                );
                CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, title, content);
                INSERT INTO chunks_fts VALUES('one', '报销指南', '提交申请。');
                INSERT INTO chunks_fts VALUES('two', '发票指南', '核验发票。');
                """
            )
            connection.commit()
            connection.close()

            call_count = 0

            def fail_during_second_row(text):
                nonlocal call_count
                call_count += 1
                if call_count == 5:
                    raise sqlite3.OperationalError("injected backfill failure")
                return _pretokenize(text)

            with patch("feishu_rag.store._pretokenize", side_effect=fail_during_second_row):
                failed_store = IndexStore(db_path)
            try:
                self.assertFalse(failed_store._fts_available)
                columns = [
                    row[1]
                    for row in failed_store.connection.execute(
                        "PRAGMA table_info(chunks_fts)"
                    ).fetchall()
                ]
                self.assertEqual(columns, ["chunk_id", "title", "content"])
                self.assertEqual(
                    failed_store.connection.execute(
                        "SELECT COUNT(*) FROM chunks_fts"
                    ).fetchone()[0],
                    2,
                )
            finally:
                failed_store.close()

            recovered_store = IndexStore(db_path)
            try:
                columns = [
                    row[1]
                    for row in recovered_store.connection.execute(
                        "PRAGMA table_info(chunks_fts)"
                    ).fetchall()
                ]
                self.assertEqual(
                    columns,
                    ["chunk_id", "title_terms", "content_terms", "search_terms"],
                )
                self.assertEqual(
                    {
                        row[0]
                        for row in recovered_store.connection.execute(
                            "SELECT chunk_id FROM chunks_fts"
                        ).fetchall()
                    },
                    {"one", "two"},
                )
            finally:
                recovered_store.close()

    def test_initialization_rebuilds_v2_fts_when_ids_do_not_match_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            store = IndexStore(db_path)
            try:
                store.upsert_document(
                    "one.pdf",
                    "报销指南",
                    "one.pdf",
                    "v1",
                    [Chunk("one", "one.pdf", "报销指南", "提交申请。")],
                )
                store.upsert_document(
                    "two.pdf",
                    "发票指南",
                    "two.pdf",
                    "v1",
                    [Chunk("two", "two.pdf", "发票指南", "核验发票。")],
                )
                store.connection.execute("DELETE FROM chunks_fts WHERE chunk_id = 'two'")
                store.connection.execute(
                    "INSERT INTO chunks_fts VALUES('orphan', '孤儿', '孤儿', '')"
                )
                store.connection.commit()
            finally:
                store.close()

            recovered_store = IndexStore(db_path)
            try:
                self.assertEqual(
                    [
                        row[0]
                        for row in recovered_store.connection.execute(
                            "SELECT chunk_id FROM chunks_fts ORDER BY chunk_id"
                        ).fetchall()
                    ],
                    ["one", "two"],
                )
            finally:
                recovered_store.close()

    def test_prune_documents_removes_only_stale_documents_in_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "feishu:space-a:keep", "保留", "keep.txt", "v1", [Chunk("keep-chunk", "feishu:space-a:keep", "保留", "保留内容")]
                )
                store.upsert_document(
                    "feishu:space-a:stale", "过期", "stale.txt", "v1", [Chunk("stale-chunk", "feishu:space-a:stale", "过期", "过期内容")]
                )
                store.upsert_document(
                    "feishu:space-b:other", "其他", "other.txt", "v1", [Chunk("other-chunk", "feishu:space-b:other", "其他", "其他内容")]
                )
                store.upsert_document(
                    "local.txt", "本地", "local.txt", "v1", [Chunk("local-chunk", "local.txt", "本地", "本地内容")]
                )

                self.assertEqual(
                    store.prune_documents("feishu:space-a:", {"feishu:space-a:keep"}),
                    1,
                )
                self.assertEqual(store.search("过期"), [])
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = 'stale-chunk'").fetchone()[0], 0)
                self.assertEqual(store.count_chunks("feishu:space-a:stale"), 0)
                self.assertEqual(store.document_checksum("feishu:space-a:keep"), "v1")
                self.assertEqual(store.count_chunks("feishu:space-a:keep"), 1)
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = 'keep-chunk'").fetchone()[0], 1)
                self.assertEqual(store.document_checksum("feishu:space-b:other"), "v1")
                self.assertEqual(store.count_chunks("feishu:space-b:other"), 1)
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = 'other-chunk'").fetchone()[0], 1)
                self.assertEqual(store.document_checksum("local.txt"), "v1")
                self.assertEqual(store.count_chunks("local.txt"), 1)
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = 'local-chunk'").fetchone()[0], 1)
                self.assertEqual(store.prune_documents("feishu:space-a:", {"feishu:space-a:keep"}), 0)
            finally:
                store.close()

    def test_prune_documents_validates_prefix_and_retained_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                with self.assertRaises(ValueError):
                    store.prune_documents("", set())
                with self.assertRaises(ValueError):
                    store.prune_documents("feishu:space-a:", {"feishu:space-b:other"})
            finally:
                store.close()

    def test_prune_documents_with_empty_retained_clears_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "feishu:space-a:one", "一", "one.txt", "v1", [Chunk("one-chunk", "feishu:space-a:one", "一", "内容一")]
                )
                store.upsert_document(
                    "feishu:space-b:two", "二", "two.txt", "v1", [Chunk("two-chunk", "feishu:space-b:two", "二", "内容二")]
                )

                self.assertEqual(store.prune_documents("feishu:space-a:", set()), 1)
                self.assertIsNone(store.document_checksum("feishu:space-a:one"))
                self.assertEqual(store.document_checksum("feishu:space-b:two"), "v1")
            finally:
                store.close()

    def test_prune_documents_requires_single_feishu_space_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                for prefix in ("", "local:", "feishu:", "feishu:space:a:", "FEISHU:space-a:"):
                    with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                        store.prune_documents(prefix, set())
            finally:
                store.close()

    def test_prune_documents_uses_case_sensitive_literal_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "FEISHU:space-a:upper", "大写", "upper.txt", "v1", [Chunk("upper-chunk", "FEISHU:space-a:upper", "大写", "大写内容")]
                )
                self.assertEqual(store.prune_documents("feishu:space-a:", set()), 0)
                self.assertEqual(store.document_checksum("FEISHU:space-a:upper"), "v1")
                self.assertEqual(store.count_chunks("FEISHU:space-a:upper"), 1)
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = 'upper-chunk'").fetchone()[0], 1)
            finally:
                store.close()

    def test_prune_documents_handles_more_than_sqlite_variable_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                for index in range(1200):
                    source_id = f"feishu:space-a:stale-{index}"
                    store.upsert_document(
                        source_id, "过期", f"{index}.txt", "v1", [Chunk(f"chunk-{index}", source_id, "过期", "过期内容")]
                    )
                self.assertEqual(store.prune_documents("feishu:space-a:", set()), 1200)
                self.assertEqual(store.count_documents(), 0)
                self.assertEqual(store.count_chunks(), 0)
            finally:
                store.close()

    def test_prune_documents_rolls_back_fts_when_document_delete_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "feishu:space-a:stale", "过期", "stale.txt", "v1", [Chunk("rollback-chunk", "feishu:space-a:stale", "过期", "过期内容")]
                )
                store.connection.execute(
                    """
                    CREATE TRIGGER fail_document_delete
                    BEFORE DELETE ON documents
                    BEGIN
                        SELECT RAISE(ABORT, 'blocked');
                    END
                    """
                )
                with self.assertRaisesRegex(Exception, "blocked"):
                    store.prune_documents("feishu:space-a:", set())
                self.assertEqual(store.document_checksum("feishu:space-a:stale"), "v1")
                self.assertEqual(store.count_chunks("feishu:space-a:stale"), 1)
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = 'rollback-chunk'").fetchone()[0], 1)
            finally:
                store.close()

    def test_sqlite_connection_uses_concurrency_safe_pragmas(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                self.assertEqual(
                    store.connection.execute("PRAGMA journal_mode").fetchone()[0],
                    "wal",
                )
                self.assertGreaterEqual(
                    store.connection.execute("PRAGMA busy_timeout").fetchone()[0],
                    30000,
                )
                self.assertEqual(
                    store.connection.execute("PRAGMA synchronous").fetchone()[0],
                    1,
                )
            finally:
                store.close()

    def test_search_text_is_searchable_but_original_content_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "finance.pdf",
                    "财务制度",
                    "finance.pdf",
                    "v1",
                    [
                        Chunk(
                            "c-meta",
                            "finance.pdf",
                            "财务制度",
                            "提交费用申请单。",
                            search_text="报销 付款 审批流程",
                        )
                    ],
                )
                result = store.search("报销审批", 3)[0].chunk
                self.assertEqual(result.content, "提交费用申请单。")
                self.assertEqual(result.search_text, "报销 付款 审批流程")
            finally:
                store.close()

    def test_search_ranks_the_specific_chinese_document_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "travel.pdf",
                    "差旅报销办法",
                    "travel.pdf",
                    "v1",
                    [Chunk("travel", "travel.pdf", "差旅报销办法", "差旅发票须经部门审批。")],
                )
                store.upsert_document(
                    "invoice.pdf",
                    "发票开具说明",
                    "invoice.pdf",
                    "v1",
                    [Chunk("invoice", "invoice.pdf", "发票开具说明", "电子发票可以下载。")],
                )

                results = store.search("请问差旅发票怎么报销", top_k=2)

                self.assertEqual(results[0].chunk.id, "travel")
                self.assertGreaterEqual(results[0].score, 0.42)
                self.assertLessEqual(results[0].score, 1.0)
            finally:
                store.close()

    def test_search_returns_no_result_for_generic_or_unrelated_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "finance.pdf",
                    "财务制度",
                    "finance.pdf",
                    "v1",
                    [Chunk("finance", "finance.pdf", "财务制度", "报销流程按规定办理。")],
                )
                for query in (
                    "请问是什么制度",
                    "这是什么制度",
                    "那都有哪些办法呢",
                    "流程规定办法",
                    "量子芯片温度",
                ):
                    with self.subTest(query=query):
                        self.assertEqual(store.search(query), [])
            finally:
                store.close()

    def test_search_does_not_strip_residual_characters_inside_real_entities(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                documents = (
                    ("ownership", "所有权说明", "本制度说明资产所有权。"),
                    ("purpose", "文件目的", "文件目的用于说明适用范围。"),
                    ("summary", "内容摘要", "摘要概括主要内容。"),
                    ("key-points", "操作要点", "要点包括核验和审批。"),
                )
                for chunk_id, title, content in documents:
                    source_id = f"{chunk_id}.pdf"
                    store.upsert_document(
                        source_id,
                        title,
                        source_id,
                        "v1",
                        [Chunk(chunk_id, source_id, title, content)],
                    )

                for query, expected_id in (
                    ("所有权", "ownership"),
                    ("所有权规定", "ownership"),
                    ("目的", "purpose"),
                    ("摘要", "summary"),
                    ("要点", "key-points"),
                ):
                    with self.subTest(query=query):
                        self.assertEqual(store.search(query)[0].chunk.id, expected_id)
                self.assertEqual(store.search("这是什么制度"), [])
            finally:
                store.close()

    def test_search_strips_common_which_question_shells(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "invoice.pdf",
                    "电子发票管理规定",
                    "invoice.pdf",
                    "v1",
                    [Chunk("invoice", "invoice.pdf", "电子发票管理规定", "电子发票需要核验。")],
                )
                for query in ("电子发票有什么规定", "请问电子发票有哪些规定"):
                    with self.subTest(query=query):
                        self.assertEqual(store.search(query)[0].chunk.id, "invoice")
            finally:
                store.close()

    def test_search_treats_match_syntax_as_plain_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "finance.pdf",
                    "报销指引",
                    "finance.pdf",
                    "v1",
                    [Chunk("finance", "finance.pdf", "报销指引", "提交电子发票。")],
                )
                results = store.search('电子发票" OR * NOT (')
                self.assertEqual(results[0].chunk.id, "finance")
            finally:
                store.close()

    def test_search_falls_back_to_literal_scan_when_fts_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "finance.pdf",
                    "报销指引",
                    "finance.pdf",
                    "v1",
                    [Chunk("finance", "finance.pdf", "报销指引", "提交电子发票。")],
                )
                store._fts_available = False
                self.assertEqual(store.search("电子发票")[0].chunk.id, "finance")
            finally:
                store.close()

    def test_search_applies_configurable_confidence_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "finance.pdf",
                    "报销指引",
                    "finance.pdf",
                    "v1",
                    [Chunk("finance", "finance.pdf", "报销指引", "提交电子发票。")],
                )
                result = store.search("电子发票", min_relevance=0.0)[0]
                self.assertGreaterEqual(result.score, 0.0)
                self.assertLessEqual(result.score, 1.0)
                self.assertEqual(store.search("电子发票", min_relevance=1.0), [])
            finally:
                store.close()

    def test_search_scope_filters_all_candidates_and_none_preserves_full_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "space-a.pdf",
                    "A 空间报销指引",
                    "space-a.pdf",
                    "v1",
                    [Chunk("space-a", "space-a.pdf", "A 空间报销指引", "提交电子发票。")],
                    space_id="space-a",
                )
                store.upsert_document(
                    "space-b.pdf",
                    "B 空间报销指引",
                    "space-b.pdf",
                    "v1",
                    [Chunk("space-b", "space-b.pdf", "B 空间报销指引", "提交电子发票。")],
                    space_id="space-b",
                )
                store.upsert_document(
                    "local.pdf",
                    "本地报销指引",
                    "local.pdf",
                    "v1",
                    [Chunk("local", "local.pdf", "本地报销指引", "提交电子发票。")],
                )

                self.assertEqual(
                    {result.chunk.id for result in store.search("电子发票", top_k=6)},
                    {"space-a", "space-b", "local"},
                )
                self.assertEqual(
                    [
                        result.chunk.id
                        for result in store.search(
                            "电子发票",
                            top_k=6,
                            scope=RetrievalScope(frozenset({"space-a"})),
                        )
                    ],
                    ["space-a"],
                )
                self.assertEqual(
                    store.search(
                        "电子发票",
                        scope=RetrievalScope(frozenset()),
                    ),
                    [],
                )
            finally:
                store.close()

    def test_search_document_frequency_is_calculated_only_inside_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "allowed.pdf",
                    "允许文档",
                    "allowed.pdf",
                    "v1",
                    [Chunk("allowed", "allowed.pdf", "允许文档", "alpha")],
                    space_id="allowed",
                )
                scope = RetrievalScope(frozenset({"allowed"}))
                before = store.search("alpha beta", min_relevance=0.0, scope=scope)[0].score
                for index in range(12):
                    source_id = f"blocked-{index}.pdf"
                    store.upsert_document(
                        source_id,
                        "禁止文档",
                        source_id,
                        "v1",
                        [Chunk(f"blocked-{index}", source_id, "禁止文档", "alpha")],
                        space_id="blocked",
                    )

                after = store.search("alpha beta", min_relevance=0.0, scope=scope)[0].score

                self.assertEqual(after, before)
            finally:
                store.close()

    def test_claim_rate_limit_validates_actor_and_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                for actor in ("", "   ", None):
                    with self.subTest(actor=actor), self.assertRaises(ValueError):
                        store.claim_rate_limit(actor, 10, 200, now=0)
                for per_minute, per_day in ((-1, 1), (1, -1), (True, 1), (1, 1.5)):
                    with self.subTest(
                        per_minute=per_minute, per_day=per_day
                    ), self.assertRaises(ValueError):
                        store.claim_rate_limit("actor", per_minute, per_day, now=0)
                self.assertTrue(store.claim_rate_limit("actor", 0, 0, now=0))
                self.assertEqual(
                    store.connection.execute(
                        "SELECT COUNT(*) FROM rate_limit_buckets"
                    ).fetchone()[0],
                    0,
                )
                maximum = 2**63 - 1
                self.assertTrue(
                    store.claim_rate_limit("max-user", maximum, maximum, now=0)
                )
                with self.assertRaises(ValueError):
                    store.claim_rate_limit("too-large", maximum + 1, 0, now=0)
            finally:
                store.close()

    def test_claim_rate_limit_enforces_minute_and_day_windows_without_partial_increment(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                for _ in range(10):
                    self.assertTrue(store.claim_rate_limit("minute-user", 10, 0, now=59))
                self.assertFalse(store.claim_rate_limit("minute-user", 10, 0, now=59))
                self.assertTrue(store.claim_rate_limit("minute-user", 10, 0, now=60))

                for _ in range(200):
                    self.assertTrue(store.claim_rate_limit("day-user", 0, 200, now=86399))
                self.assertFalse(store.claim_rate_limit("day-user", 0, 200, now=86399))
                self.assertTrue(store.claim_rate_limit("day-user", 0, 200, now=86400))

                self.assertTrue(store.claim_rate_limit("both-user", 1, 2, now=0))
                self.assertFalse(store.claim_rate_limit("both-user", 1, 2, now=1))
                self.assertTrue(store.claim_rate_limit("both-user", 1, 2, now=60))
            finally:
                store.close()

    def test_claim_rate_limit_stores_only_sha256_actor_hash_and_prunes_old_buckets(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            actor = "ou_original_user_id"
            try:
                self.assertTrue(store.claim_rate_limit(actor, 10, 200, now=0))
                hashes = {
                    row[0]
                    for row in store.connection.execute(
                        "SELECT user_hash FROM rate_limit_buckets"
                    ).fetchall()
                }
                self.assertEqual(hashes, {sha256(actor.encode("utf-8")).hexdigest()})
                self.assertNotIn(actor.encode("utf-8"), Path(store.db_path).read_bytes())

                self.assertTrue(store.claim_rate_limit("another-user", 1, 1, now=86400))
                self.assertEqual(
                    store.connection.execute(
                        "SELECT COUNT(*) FROM rate_limit_buckets WHERE bucket_start = 0"
                    ).fetchone()[0],
                    0,
                )
            finally:
                store.close()

    def test_two_connections_concurrently_never_exceed_the_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            IndexStore(db_path).close()
            barrier = threading.Barrier(2)

            def claim_many() -> int:
                store = IndexStore(db_path)
                try:
                    barrier.wait()
                    return sum(
                        store.claim_rate_limit("shared-user", 10, 0, now=0)
                        for _ in range(10)
                    )
                finally:
                    store.close()

            with ThreadPoolExecutor(max_workers=2) as executor:
                accepted = list(executor.map(lambda _: claim_many(), range(2)))

            self.assertEqual(sum(accepted), 10)
            store = IndexStore(db_path)
            try:
                self.assertEqual(
                    store.connection.execute(
                        "SELECT count FROM rate_limit_buckets WHERE window_seconds = 60"
                    ).fetchone()[0],
                    10,
                )
            finally:
                store.close()

    def test_message_claim_prevents_duplicate_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                self.assertTrue(
                    store.claim_message("om_duplicate", retention_seconds=60)
                )
                self.assertFalse(
                    store.claim_message("om_duplicate", retention_seconds=60)
                )
                self.assertEqual(
                    tuple(
                        store.connection.execute(
                            "SELECT state,claim_token FROM processed_messages "
                            "WHERE message_id = ?",
                            ("om_duplicate",),
                        ).fetchone()
                    ),
                    ("completed", ""),
                )
                self.assertTrue(store.release_message("om_duplicate"))
                self.assertTrue(store.claim_message("om_duplicate"))
            finally:
                store.close()

    def test_legacy_message_claim_respects_short_retention_without_a_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                with patch("feishu_rag.store.time.time", return_value=100):
                    self.assertTrue(
                        store.claim_message("om_short", retention_seconds=60)
                    )
                with patch("feishu_rag.store.time.time", return_value=159):
                    self.assertFalse(
                        store.claim_message("om_short", retention_seconds=60)
                    )
                with patch("feishu_rag.store.time.time", return_value=161):
                    self.assertTrue(
                        store.claim_message("om_short", retention_seconds=60)
                    )
            finally:
                store.close()

    def test_message_claim_state_tracks_in_progress_and_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                state, token = store.claim_message_lease("om_state")
                self.assertEqual(state, "claimed")
                self.assertIsNotNone(token)
                self.assertEqual(
                    store.claim_message_lease("om_state"),
                    ("in_progress", None),
                )

                self.assertTrue(store.begin_message_reply("om_state", token))
                self.assertTrue(store.complete_message("om_state", token=token))

                self.assertEqual(store.claim_message_state("om_state"), "completed")
                self.assertTrue(store.release_message("om_state"))
                self.assertEqual(store.claim_message_state("om_state"), "claimed")
            finally:
                store.close()

    def test_initialization_migrates_legacy_processed_messages_as_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            connection = sqlite3.connect(db_path)
            connection.executescript(
                """
                CREATE TABLE processed_messages (
                    message_id TEXT PRIMARY KEY,
                    processed_at REAL NOT NULL
                );
                INSERT INTO processed_messages VALUES('om_legacy', 1);
                """
            )
            connection.commit()
            connection.close()

            store = IndexStore(db_path)
            try:
                state_column = next(
                    row
                    for row in store.connection.execute(
                        "PRAGMA table_info(processed_messages)"
                    ).fetchall()
                    if row[1] == "state"
                )
                self.assertEqual(state_column[2], "TEXT")
                self.assertEqual(state_column[3], 1)
                self.assertEqual(state_column[4], "'completed'")
                token_column = next(
                    row
                    for row in store.connection.execute(
                        "PRAGMA table_info(processed_messages)"
                    ).fetchall()
                    if row[1] == "claim_token"
                )
                self.assertEqual(token_column[2], "TEXT")
                self.assertEqual(token_column[3], 1)
                self.assertEqual(token_column[4], "''")
                self.assertEqual(
                    store.claim_message_state("om_legacy", retention_seconds=10**12),
                    "completed",
                )
                self.assertEqual(
                    store.claim_message_lease(
                        "om_legacy", retention_seconds=10**12
                    ),
                    ("completed", None),
                )
                self.assertTrue(store.complete_message("om_legacy"))
                self.assertTrue(store.release_message("om_legacy"))
            finally:
                store.close()

    def test_message_lease_rotates_token_and_fences_stale_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                state, first_token = store.claim_message_lease("om_fenced", now=100)
                self.assertEqual(state, "claimed")
                self.assertIsNotNone(first_token)
                self.assertRegex(first_token or "", r"^[0-9a-f]{32}$")
                self.assertEqual(
                    store.claim_message_lease("om_fenced", now=101),
                    ("in_progress", None),
                )

                state, second_token = store.claim_message_lease(
                    "om_fenced",
                    in_progress_timeout_seconds=600,
                    now=701,
                )
                self.assertEqual(state, "claimed")
                self.assertIsNotNone(second_token)
                self.assertNotEqual(second_token, first_token)
                self.assertTrue(store.is_message_claim_owner("om_fenced", second_token))
                self.assertFalse(store.is_message_claim_owner("om_fenced", first_token))

                self.assertFalse(store.begin_message_reply("om_fenced", first_token))
                self.assertFalse(store.complete_message("om_fenced", token=first_token))
                self.assertFalse(store.release_message("om_fenced", token=first_token))
                self.assertFalse(store.complete_message("om_fenced"))
                self.assertFalse(store.release_message("om_fenced"))
                self.assertFalse(store.complete_message("om_fenced", token=second_token))
                row = store.connection.execute(
                    "SELECT state,claim_token FROM processed_messages WHERE message_id = ?",
                    ("om_fenced",),
                ).fetchone()
                self.assertEqual(tuple(row), ("in_progress", second_token))

                self.assertTrue(store.begin_message_reply("om_fenced", second_token))
                self.assertFalse(store.begin_message_reply("om_fenced", second_token))
                self.assertEqual(
                    store.claim_message_lease(
                        "om_fenced",
                        retention_seconds=700,
                        in_progress_timeout_seconds=600,
                        now=10_000,
                    ),
                    ("in_progress", None),
                )
                self.assertEqual(
                    tuple(
                        store.connection.execute(
                            "SELECT state,claim_token FROM processed_messages "
                            "WHERE message_id = ?",
                            ("om_fenced",),
                        ).fetchone()
                    ),
                    ("replying", second_token),
                )
                self.assertTrue(store.complete_message("om_fenced", token=second_token))
                self.assertEqual(
                    store.connection.execute(
                        "SELECT claim_token FROM processed_messages WHERE message_id = ?",
                        ("om_fenced",),
                    ).fetchone()[0],
                    "",
                )
                self.assertEqual(
                    store.claim_message_lease("om_fenced", now=702),
                    ("completed", None),
                )
            finally:
                store.close()

    def test_two_connections_share_only_the_current_atomic_lease_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            IndexStore(db_path).close()
            barrier = threading.Barrier(2)

            def claim() -> tuple[str, str | None]:
                store = IndexStore(db_path)
                try:
                    barrier.wait(timeout=2)
                    return store.claim_message_lease("om_token_concurrent")
                finally:
                    store.close()

            with ThreadPoolExecutor(max_workers=2) as executor:
                leases = list(executor.map(lambda _: claim(), range(2)))

            self.assertEqual(sorted(state for state, _ in leases), ["claimed", "in_progress"])
            claimed_token = next(token for state, token in leases if state == "claimed")
            waiting_token = next(
                token for state, token in leases if state == "in_progress"
            )
            self.assertIsNotNone(claimed_token)
            self.assertTrue(re.fullmatch(r"[0-9a-f]{32}", claimed_token or ""))
            self.assertIsNone(waiting_token)

    def test_two_connections_atomically_seal_reply_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            seed = IndexStore(db_path)
            state, token = seed.claim_message_lease("om_seal_concurrent")
            self.assertEqual(state, "claimed")
            seed.close()
            barrier = threading.Barrier(2)

            def seal() -> bool:
                store = IndexStore(db_path)
                try:
                    barrier.wait(timeout=2)
                    return store.begin_message_reply("om_seal_concurrent", token)
                finally:
                    store.close()

            with ThreadPoolExecutor(max_workers=2) as executor:
                sealed = list(executor.map(lambda _: seal(), range(2)))

            self.assertEqual(sorted(sealed), [False, True])
            store = IndexStore(db_path)
            try:
                self.assertEqual(
                    tuple(
                        store.connection.execute(
                            "SELECT state,claim_token FROM processed_messages "
                            "WHERE message_id = ?",
                            ("om_seal_concurrent",),
                        ).fetchone()
                    ),
                    ("replying", token),
                )
            finally:
                store.close()

    def test_two_connections_atomically_claim_one_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            IndexStore(db_path).close()
            barrier = threading.Barrier(2)

            def claim() -> str:
                store = IndexStore(db_path)
                try:
                    barrier.wait(timeout=2)
                    return store.claim_message_state("om_concurrent")
                finally:
                    store.close()

            with ThreadPoolExecutor(max_workers=2) as executor:
                states = list(executor.map(lambda _: claim(), range(2)))

            self.assertEqual(sorted(states), ["claimed", "in_progress"])

    def test_expired_in_progress_message_can_be_reclaimed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                self.assertEqual(
                    store.claim_message_state("om_expired", now=100), "claimed"
                )
                self.assertEqual(
                    store.claim_message_state(
                        "om_expired", in_progress_timeout_seconds=600, now=701
                    ),
                    "claimed",
                )
                self.assertEqual(
                    store.connection.execute(
                        "SELECT state, processed_at FROM processed_messages "
                        "WHERE message_id = ?",
                        ("om_expired",),
                    ).fetchone()[0:2],
                    ("in_progress", 701),
                )
            finally:
                store.close()

    def test_non_expired_in_progress_message_is_not_reclaimed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                self.assertEqual(
                    store.claim_message_state("om_active", now=100), "claimed"
                )
                self.assertEqual(
                    store.claim_message_state(
                        "om_active", in_progress_timeout_seconds=600, now=699
                    ),
                    "in_progress",
                )
            finally:
                store.close()

    def test_completed_message_is_not_reclaimed_after_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                state, token = store.claim_message_lease("om_done", now=100)
                self.assertEqual(state, "claimed")
                self.assertTrue(store.begin_message_reply("om_done", token))
                self.assertTrue(store.complete_message("om_done", token=token))
                self.assertEqual(
                    store.claim_message_state(
                        "om_done", in_progress_timeout_seconds=1, now=1000
                    ),
                    "completed",
                )
            finally:
                store.close()

    def test_two_connections_expired_message_have_one_reclaimer(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            seed = IndexStore(db_path)
            try:
                self.assertEqual(seed.claim_message_state("om_expired_concurrent", now=100), "claimed")
            finally:
                seed.close()
            barrier = threading.Barrier(2)

            def reclaim() -> str:
                store = IndexStore(db_path)
                try:
                    barrier.wait(timeout=2)
                    return store.claim_message_state(
                        "om_expired_concurrent",
                        in_progress_timeout_seconds=600,
                        now=701,
                    )
                finally:
                    store.close()

            with ThreadPoolExecutor(max_workers=2) as executor:
                states = list(executor.map(lambda _: reclaim(), range(2)))

            self.assertEqual(sorted(states), ["claimed", "in_progress"])

    def test_message_claim_state_rejects_invalid_retention_and_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                for retention, lease in ((0, 1), (1, 0), (1, 1), (True, 1), (10, False)):
                    with self.subTest(retention=retention, lease=lease):
                        with self.assertRaises(ValueError):
                            store.claim_message_state(
                                "om_invalid",
                                retention_seconds=retention,
                                in_progress_timeout_seconds=lease,
                            )
            finally:
                store.close()

    def test_search_returns_chinese_keyword_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    source_id="finance/reimbursement.pdf",
                    title="财务报销制度",
                    path="finance/reimbursement.pdf",
                    checksum="v1",
                    chunks=[
                        Chunk("c1", "finance/reimbursement.pdf", "财务报销制度", "差旅费报销需要提供发票。", 2, "差旅")
                    ],
                )

                results = store.search("发票报销", top_k=3)

                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].chunk.title, "财务报销制度")
                self.assertEqual(results[0].chunk.page, 2)
            finally:
                store.close()

    def test_reindex_replaces_previous_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    source_id="policy.txt",
                    title="旧政策",
                    path="policy.txt",
                    checksum="v1",
                    chunks=[Chunk("old", "policy.txt", "旧政策", "旧内容")],
                )
                store.upsert_document(
                    source_id="policy.txt",
                    title="新制度",
                    path="policy.txt",
                    checksum="v2",
                    chunks=[Chunk("new", "policy.txt", "新制度", "新内容")],
                )

                self.assertEqual(store.count_documents(), 1)
                self.assertEqual(store.count_chunks("policy.txt"), 1)
                self.assertEqual(store.search("旧政策"), [])
                self.assertEqual(store.search("新内容")[0].chunk.title, "新制度")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
