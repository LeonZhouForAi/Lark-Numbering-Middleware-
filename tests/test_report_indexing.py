from pathlib import Path

from feishu_rag.models import Chunk
from feishu_rag.store import IndexStore, PreparedDocument
from scripts import report_indexing


def _document(source_id: str, state: str, path: str) -> PreparedDocument:
    return PreparedDocument(
        source_id,
        source_id,
        path,
        f"checksum-{source_id}",
        (Chunk(f"chunk-{source_id}", source_id, source_id, f"{source_id}正文"),),
        document_code="HBW-OP-001" if state == "conflict" else "",
        document_version="B1" if state == "conflict" else "",
        lifecycle_state=state,
        decision_reason=(
            "uncomparable-revisions" if state == "conflict" else "unique-or-unversioned"
        ),
    )


def test_report_lists_lifecycle_counts_without_writing(tmp_path, capsys) -> None:
    database = tmp_path / "rag.sqlite3"
    store = IndexStore(database)
    try:
        store.apply_document_snapshot(
            [
                _document("current", "current", "current.docx"),
                _document("superseded", "superseded", "old.pdf"),
                _document("conflict", "conflict", "conflict.xlsx"),
            ]
        )
    finally:
        store.close()
    before = database.read_bytes()

    assert report_indexing.main(["--db", str(database)]) == 0

    output = capsys.readouterr().out
    assert "documents=3 current=1 superseded=1 conflict=1" in output
    assert "formats=.docx:1,.pdf:1,.xlsx:1" in output
    assert "conflict\tHBW-OP-001\tB1" in output
    assert "conflict正文" not in output
    assert database.read_bytes() == before


def test_report_rejects_missing_database_without_creating_it(tmp_path) -> None:
    database = tmp_path / "missing.sqlite3"

    assert report_indexing.main(["--db", str(database)]) == 2
    assert not database.exists()
