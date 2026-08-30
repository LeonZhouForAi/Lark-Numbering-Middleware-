"""应用日志配置。"""

from __future__ import annotations

import logging


def configure_logging(level: str) -> None:
    """配置标准库日志；未知等级安全回退到 INFO。"""
    normalized_level = str(level).upper()
    if normalized_level not in logging.getLevelNamesMapping():
        normalized_level = "INFO"
    logging.basicConfig(
        level=normalized_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
