"""从文件名提取可解释的文档版本元数据。"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Mapping


_DOCUMENT_CODE_RE = re.compile(
    r"(?<![A-Z0-9])HBW(?:-[A-Z0-9]+)+-\d{3}(?=$|[^A-Z0-9])",
    re.IGNORECASE,
)
_VERSION_RE = re.compile(r"(?<![A-Z0-9])([A-Z]\d+)\s*$", re.IGNORECASE)
_COMPACT_DATE_RE = re.compile(r"(?<!\d)((?:19|20)\d{6})(?!\d)")


@dataclass(frozen=True)
class DocumentMetadata:
    document_code: str | None = None
    document_version: str | None = None
    effective_date: str | None = None
    lifecycle_state: str = "current"
    decision_reason: str = "unique-or-unversioned"


def _iso_date(raw_value: str) -> str | None:
    try:
        parsed = date(int(raw_value[:4]), int(raw_value[4:6]), int(raw_value[6:8]))
    except ValueError:
        return None
    return parsed.isoformat()


def extract_document_metadata(filename: str) -> DocumentMetadata:
    """Extract only metadata that can be read reliably from a filename."""

    stem = Path(filename).stem
    code_match = _DOCUMENT_CODE_RE.search(stem)
    version_match = _VERSION_RE.search(stem)
    date_match = _COMPACT_DATE_RE.search(stem)
    return DocumentMetadata(
        document_code=code_match.group(0).upper() if code_match else None,
        document_version=version_match.group(1).upper() if version_match else None,
        effective_date=_iso_date(date_match.group(1)) if date_match else None,
    )


def revision_key(value: str) -> tuple[int, int] | None:
    """Return a sortable key for revisions such as A9, B0 and B1."""

    match = re.fullmatch(r"([A-Z])(\d+)", value.strip().upper())
    if match is None:
        return None
    return ord(match.group(1)) - ord("A"), int(match.group(2))


def resolve_lifecycle_states(
    documents: Mapping[str, DocumentMetadata],
) -> dict[str, DocumentMetadata]:
    """Resolve lifecycle states without discarding ambiguous versions."""

    resolved: dict[str, DocumentMetadata] = {}
    grouped: dict[str, list[tuple[str, DocumentMetadata]]] = defaultdict(list)
    for path, metadata in documents.items():
        if metadata.document_code is None:
            resolved[path] = replace(
                metadata,
                lifecycle_state="current",
                decision_reason="unique-or-unversioned",
            )
        else:
            grouped[metadata.document_code].append((path, metadata))

    for group in grouped.values():
        if len(group) == 1:
            path, metadata = group[0]
            resolved[path] = replace(
                metadata,
                lifecycle_state="current",
                decision_reason="unique-or-unversioned",
            )
            continue

        keyed_versions = [
            revision_key(metadata.document_version)
            if metadata.document_version is not None
            else None
            for _, metadata in group
        ]
        if any(key is None for key in keyed_versions):
            for path, metadata in group:
                resolved[path] = replace(
                    metadata,
                    lifecycle_state="conflict",
                    decision_reason="uncomparable-revisions",
                )
            continue

        comparable_versions = [key for key in keyed_versions if key is not None]
        highest = max(comparable_versions)
        if comparable_versions.count(highest) > 1:
            for path, metadata in group:
                resolved[path] = replace(
                    metadata,
                    lifecycle_state="conflict",
                    decision_reason="duplicate-highest-revision",
                )
            continue

        for (path, metadata), key in zip(group, comparable_versions, strict=True):
            is_highest = key == highest
            resolved[path] = replace(
                metadata,
                lifecycle_state="current" if is_highest else "superseded",
                decision_reason="highest-revision" if is_highest else "lower-revision",
            )
    return resolved
