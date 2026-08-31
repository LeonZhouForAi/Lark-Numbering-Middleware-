from __future__ import annotations

import json

import pytest

from feishu_rag.store import IndexStore
from scripts.report_faq import main as report_main


def _seed(path):
    store = IndexStore(path)
    with store.connection:
        store.connection.execute(
            "INSERT INTO faq_metrics_daily(scope_key, day, eligible_questions, rag_answers, direct_hits, promotions, refreshes, rejected_answers) VALUES (?, ?, 4, 3, 2, 1, 1, 1)",
            ("space-hash", "2026-08-31"),
        )
        store.connection.execute(
            "INSERT INTO faq_metrics_daily(scope_key, day, eligible_questions, direct_hits, refreshes) VALUES (?, ?, 8, 4, 2)",
            ("space-hash", "2026-08-20"),
        )
        store.connection.execute(
            "INSERT INTO faq_entries(id, intent_key, scope_key, canonical_question, answer, source_signature, source_ids_json, knowledge_revision, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 'enabled', 1, 1)",
            ("id", "intent", "space-hash", "内部问题", "标准答案", "sig", "[]"),
        )
        store.connection.execute(
            "INSERT INTO faq_entries(id, intent_key, scope_key, canonical_question, answer, source_signature, source_ids_json, knowledge_revision, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 'stale', 1, 1)",
            ("stale", "old-intent", "space-hash", "另一个问题", "另一个答案", "sig", "[]"),
        )
        store.connection.execute(
            "INSERT INTO faq_observation_daily(intent_key, scope_key, day, count, normalized_question, source_signature, knowledge_revision) VALUES (?, ?, ?, 3, ?, ?, 0)",
            ("intent", "space-hash", "2026-08-31", "q", "sig"),
        )
        store.connection.execute(
            "INSERT INTO faq_observation_daily(intent_key, scope_key, day, count, normalized_question, source_signature, knowledge_revision) VALUES (?, ?, ?, 2, ?, ?, 0)",
            ("second-intent", "space-hash", "2026-08-20", "q2", "sig"),
        )
    store.close()


def test_report_faq_outputs_only_aggregate_metrics(tmp_path, capsys):
    db = tmp_path / "rag.sqlite3"
    _seed(db)
    assert report_main([str(db)]) == 0
    output = capsys.readouterr().out
    assert "direct_hits" in output
    assert "direct_hit_rate" in output
    assert "标准答案" not in output
    assert "内部问题" not in output
    assert "source" not in output.lower()
    daily = [json.loads(line) for line in output.splitlines()[1:] if not line.startswith("summary\t")]
    assert next(row for row in daily if row["day"] == "2026-08-31")["direct_hit_rate"] == pytest.approx(2 / 4)
    summary = json.loads(next(line.split("\t", 1)[1] for line in output.splitlines() if line.startswith("summary\t")))
    assert summary == {
        "hot_intents": 1,
        "enabled_faqs": 1,
        "stale_faqs": 1,
        "estimated_deepseek_requests_saved": 6,
        "faq_refreshes": 3,
        "current_invalid_faqs": 1,
        "direct_hit_rate": 0.5,
    }


def test_report_faq_promotion_count_option_changes_summary(tmp_path, capsys, monkeypatch):
    db = tmp_path / "rag.sqlite3"
    _seed(db)
    monkeypatch.setenv("RAG_FAQ_PROMOTION_COUNT", "2")
    assert report_main([str(db), "--promotion-count", "2"]) == 0
    output = capsys.readouterr().out
    summary = json.loads(next(line.split("\t", 1)[1] for line in output.splitlines() if line.startswith("summary\t")))
    assert summary["hot_intents"] == 2


def test_report_faq_rejects_invalid_since_date(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        report_main([str(tmp_path / "db"), "--since", "2026-02-30"])
    assert exc.value.code != 0
    assert "invalid date" in capsys.readouterr().err.lower()
