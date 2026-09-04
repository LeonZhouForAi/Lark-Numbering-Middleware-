from feishu_rag.document_metadata import (
    DocumentMetadata,
    extract_document_metadata,
    resolve_lifecycle_states,
    revision_key,
)


def test_extracts_company_code_and_revision_from_filename() -> None:
    metadata = extract_document_metadata(
        "HBW-PZ-LCM-WI-002 品质异常处理规范A5.docx"
    )

    assert metadata.document_code == "HBW-PZ-LCM-WI-002"
    assert metadata.document_version == "A5"
    assert metadata.effective_date is None


def test_extracts_revision_when_code_and_title_are_separated_by_punctuation() -> None:
    metadata = extract_document_metadata("HBW-OP-009-内部审核控制程序 B1.docx")

    assert metadata.document_code == "HBW-OP-009"
    assert metadata.document_version == "B1"


def test_extracts_compact_date_without_inventing_document_code() -> None:
    metadata = extract_document_metadata("提案改善管理办法20260615.docx")

    assert metadata.document_code is None
    assert metadata.document_version is None
    assert metadata.effective_date == "2026-06-15"


def test_does_not_treat_operation_or_equipment_tokens_as_revision() -> None:
    assert extract_document_metadata("IPQC作业指导书.docx").document_version is None
    assert extract_document_metadata("Cell AOI1线开机.xlsx").document_version is None


def test_revision_key_orders_letter_before_number() -> None:
    assert revision_key("A9") < revision_key("B0") < revision_key("B1")
    assert revision_key("版本一") is None


def test_resolver_marks_only_lower_revision_as_superseded() -> None:
    resolved = resolve_lifecycle_states(
        {
            "a.docx": DocumentMetadata("HBW-OP-022", "A1"),
            "b.docx": DocumentMetadata("HBW-OP-022", "B0"),
        }
    )

    assert resolved["a.docx"].lifecycle_state == "superseded"
    assert resolved["a.docx"].decision_reason == "lower-revision"
    assert resolved["b.docx"].lifecycle_state == "current"
    assert resolved["b.docx"].decision_reason == "highest-revision"


def test_resolver_keeps_uncomparable_versions_as_conflict() -> None:
    resolved = resolve_lifecycle_states(
        {
            "a.docx": DocumentMetadata("HBW-OP-022", None),
            "b.docx": DocumentMetadata("HBW-OP-022", "B0"),
        }
    )

    assert {item.lifecycle_state for item in resolved.values()} == {"conflict"}
    assert {item.decision_reason for item in resolved.values()} == {
        "uncomparable-revisions"
    }


def test_resolver_marks_duplicate_highest_revisions_as_conflict() -> None:
    resolved = resolve_lifecycle_states(
        {
            "a.docx": DocumentMetadata("HBW-OP-022", "B1"),
            "b.docx": DocumentMetadata("HBW-OP-022", "B1"),
        }
    )

    assert {item.lifecycle_state for item in resolved.values()} == {"conflict"}
    assert {item.decision_reason for item in resolved.values()} == {
        "duplicate-highest-revision"
    }


def test_resolver_leaves_unversioned_unique_files_current() -> None:
    resolved = resolve_lifecycle_states(
        {"policy.docx": DocumentMetadata(None, None, "2026-06-15")}
    )

    assert resolved["policy.docx"].lifecycle_state == "current"
    assert resolved["policy.docx"].decision_reason == "unique-or-unversioned"
