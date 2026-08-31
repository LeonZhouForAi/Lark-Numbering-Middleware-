from __future__ import annotations

import pytest

from feishu_rag.store import IndexStore
from scripts.cleanup_faq import main as cleanup_main


def _seed(path):
    store = IndexStore(path)
    with store.connection:
        store.connection.execute(
            "INSERT INTO faq_entries(id, intent_key, scope_key, canonical_question, answer, source_signature, source_ids_json, knowledge_revision, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, 1, ?)",
            ("current", "intent", "scope", "q", "a", "sig", "[]", "enabled", 1_800_000_000),
        )
        store.connection.execute(
            "INSERT INTO faq_entries(id, intent_key, scope_key, canonical_question, answer, source_signature, source_ids_json, knowledge_revision, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, 1, ?)",
            ("stale", "intent-stale", "scope", "q", "a", "sig", "[]", "stale", 1),
        )
    return store


def test_cleanup_faq_keeps_current_enabled_entries(tmp_path, capsys):
    db = tmp_path / "rag.sqlite3"
    store = _seed(db)
    store.close()
    assert cleanup_main([str(db), "--today", "2026-08-31"]) == 0
    output = capsys.readouterr().out
    assert "observations_deleted" in output
    assert "stale_entries_deleted" in output
    check = IndexStore(db)
    assert check.connection.execute("SELECT state FROM faq_entries WHERE id='current'").fetchone()[0] == "enabled"
    check.close()


def test_cleanup_faq_rejects_invalid_today(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cleanup_main([str(tmp_path / "db"), "--today", "2026-02-30"])
    assert exc.value.code != 0
    assert "invalid date" in capsys.readouterr().err.lower()
