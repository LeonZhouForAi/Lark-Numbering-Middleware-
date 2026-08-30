from __future__ import annotations

from pathlib import Path

from feishu_rag.store import IndexStore
from scripts.report_usage import main


def test_report_usage_uses_cli_prices_and_groups_by_day_model_purpose(
    tmp_path: Path, capsys
) -> None:
    db_path = tmp_path / "rag.sqlite3"
    store = IndexStore(db_path)
    try:
        store.record_llm_usage(
            "model-a", "answer", 1_000_000, 500_000, 1_500_000, day="2026-08-30"
        )
    finally:
        store.close()

    result = main(
        [
            str(db_path),
            "--input-price",
            "2.5",
            "--output-price",
            "8",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "2026-08-30" in output
    assert "model-a" in output
    assert "answer" in output
    assert "1000000" in output
    assert "500000" in output
    assert "1500000" in output
    assert "6.500000" in output
