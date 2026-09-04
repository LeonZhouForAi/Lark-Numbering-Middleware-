from feishu_rag.store import IndexStore
from scripts import report_question_gaps


def test_question_gap_report_is_read_only_and_filters_count(tmp_path, capsys) -> None:
    database = tmp_path / "rag.sqlite3"
    store = IndexStore(database)
    try:
        store.record_question_gap("global", "missing", "宠物补贴", 1, now=1.0)
        store.record_question_gap("global", "missing", "宠物补贴", 1, now=2.0)
        store.record_question_gap("global", "ambiguous", "工时是多少", 1, now=3.0)
    finally:
        store.close()
    before = database.read_bytes()

    assert report_question_gaps.main(["--db", str(database)]) == 0

    output = capsys.readouterr().out
    assert "missing\t2\t宠物补贴" in output
    assert "工时是多少" not in output
    assert database.read_bytes() == before
