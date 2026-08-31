from __future__ import annotations

import inspect
import json
import tomllib
from importlib.metadata import version
from pathlib import Path

from lark_oapi.core.json import JSON as LarkJSON
from lark_oapi.ws.client import Client as LarkWsClient
from lark_oapi.ws.client import _get_by_key
from lark_oapi.ws.model import Response
from lark_oapi.ws.pb.pbbp2_pb2 import Frame

from feishu_rag.long_connection import AckAfterProcessingClient


def test_lark_sdk_private_ack_contract_is_pinned_to_tested_version() -> None:
    assert version("lark-oapi") == "1.7.3"
    assert issubclass(AckAfterProcessingClient, LarkWsClient)
    assert inspect.iscoroutinefunction(AckAfterProcessingClient._handle_data_frame)
    assert callable(_get_by_key)
    assert all(
        hasattr(LarkWsClient, name)
        for name in ("_handle_data_frame", "_combine", "_write_message")
    )
    assert hasattr(Frame().headers, "add")
    assert json.loads(LarkJSON.marshal(Response(code=200))) == {"code": 200}

    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    lark_dependencies = [
        dependency
        for dependency in project["project"]["dependencies"]
        if dependency.startswith("lark-oapi")
    ]
    assert lark_dependencies == ["lark-oapi==1.7.3"]
