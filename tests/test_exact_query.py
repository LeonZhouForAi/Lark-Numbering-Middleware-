from feishu_rag.exact_query import ExactQueryService
from feishu_rag.models import Chunk, StructuredFact
from feishu_rag.store import IndexStore


def _store_with_facts(tmp_path) -> IndexStore:
    store = IndexStore(tmp_path / "rag.sqlite3")
    facts = [
        StructuredFact(
            "ie",
            "UPPH",
            2,
            "upph",
            operation_name="Cell AOI2线开机",
            metric_name="UPPH",
            numeric_value=450,
        ),
        StructuredFact(
            "ie",
            "总表",
            5,
            "series_hours",
            series_name="氧化物系列",
            process_stage="绑定",
            metric_name="工时",
            numeric_value=101,
        ),
        StructuredFact(
            "ie",
            "By料号",
            3,
            "part_hours",
            part_number="055010D",
            series_name="常规系列",
            process_stage="合计",
            metric_name="工时",
            numeric_value=1026,
        ),
        StructuredFact(
            "ie",
            "总表",
            8,
            "series_hours",
            series_name="常规系列",
            process_stage="贴合",
            metric_name="工时",
            numeric_value=0,
        ),
    ]
    store.upsert_document(
        "ie",
        "IE数据库",
        "ie.xlsx",
        "v1",
        [Chunk("ie-chunk", "ie", "IE数据库", "数据正文")],
        structured_facts=facts,
    )
    return store


def test_exact_query_returns_upph_series_and_part_values(tmp_path) -> None:
    store = _store_with_facts(tmp_path)
    service = ExactQueryService(store)
    try:
        assert service.answer("Cell AOI2线开机的UPPH是多少").text == (
            "Cell AOI2线开机 UPPH：450"
        )
        assert service.answer("氧化物系列绑定工时").text == (
            "氧化物系列 绑定工时：101"
        )
        assert service.answer("055010D总工时").text == "055010D 合计工时：1026"
    finally:
        store.close()


def test_exact_query_preserves_zero_value(tmp_path) -> None:
    store = _store_with_facts(tmp_path)
    try:
        answer = ExactQueryService(store).answer("常规系列贴合工时")
        assert answer.text == "常规系列 贴合工时：0"
    finally:
        store.close()


def test_exact_query_asks_for_one_missing_dimension(tmp_path) -> None:
    store = _store_with_facts(tmp_path)
    service = ExactQueryService(store)
    try:
        assert service.answer("这个工序UPPH是多少").status == "ambiguous"
        assert service.answer("这个系列工时是多少").text == "请说明要查询的产品系列？"
        assert service.answer("氧化物系列工时是多少").text == "请说明要查询的工段？"
    finally:
        store.close()


def test_unknown_part_number_falls_back_to_rag(tmp_path) -> None:
    store = _store_with_facts(tmp_path)
    try:
        assert ExactQueryService(store).answer("UNKNOWN99总工时") is None
    finally:
        store.close()
