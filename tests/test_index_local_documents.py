import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts import index_local_documents


def test_chunk_strategy_version_defaults_when_missing():
    assert index_local_documents._chunk_strategy_version({}) == "hybrid-v4"


def test_chunk_strategy_version_trims_explicit_value():
    assert index_local_documents._chunk_strategy_version(
        {"RAG_CHUNK_STRATEGY_VERSION": "  custom-v4  "}
    ) == "custom-v4"


def test_chunk_strategy_version_rejects_empty_values():
    for value in ("", "   "):
        with patch.dict(os.environ, {"RAG_CHUNK_STRATEGY_VERSION": value}):
            try:
                index_local_documents._chunk_strategy_version(os.environ)
            except SystemExit as exc:
                assert str(exc) == "RAG_CHUNK_STRATEGY_VERSION 不能为空"
            else:
                raise AssertionError("expected empty strategy version to be rejected")


def test_parser_version_defaults_and_trims_explicit_value():
    assert index_local_documents._parser_version({}) == "parser-v2"
    assert index_local_documents._parser_version(
        {"RAG_PARSER_VERSION": "  parser-custom  "}
    ) == "parser-custom"


def test_parser_version_rejects_empty_values():
    for value in ("", "   "):
        try:
            index_local_documents._parser_version({"RAG_PARSER_VERSION": value})
        except SystemExit as exc:
            assert str(exc) == "RAG_PARSER_VERSION 不能为空"
        else:
            raise AssertionError("expected empty parser version to be rejected")


def test_main_passes_versions_to_index_directory_and_prints_lifecycle(capsys):
    store = MagicMock()
    store.count_chunks.return_value = 42
    store.document_lifecycle_counts.return_value = {
        "current": 3,
        "superseded": 1,
        "conflict": 2,
    }
    with (
        patch.object(sys, "argv", ["index_local_documents.py", "docs"]),
        patch.dict(
            os.environ,
            {
                "RAG_CHUNK_STRATEGY_VERSION": "hybrid-v4",
                "RAG_PARSER_VERSION": "parser-v2",
            },
        ),
        patch.object(index_local_documents, "IndexStore", return_value=store),
        patch.object(index_local_documents, "index_directory", return_value=0) as index,
    ):
        index_local_documents.main()

    index.assert_called_once_with(
        Path("docs"),
        store,
        max_chars=900,
        enable_ocr=True,
        chunk_strategy_version="hybrid-v4",
        parser_version="parser-v2",
    )
    assert (
        capsys.readouterr().out
        == "indexed_files=0 indexed_chunks=42 current=3 superseded=1 conflict=2\n"
    )
