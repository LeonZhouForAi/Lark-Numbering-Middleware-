"""飞书开放平台客户端：机器人回复与知识库只读同步接口。"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any


class FeishuAPIError(RuntimeError):
    """飞书 API 调用失败。"""


Transport = Callable[[str, str, dict[str, str], dict[str, Any] | None, float], tuple[int, bytes]]


def _urllib_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None,
    timeout: float,
) -> tuple[int, bytes]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FeishuAPIError("无法连接飞书开放平台，请检查服务器网络") from exc


class FeishuClient:
    base_url = "https://open.feishu.cn"

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        transport: Transport | None = None,
        timeout: float = 20.0,
    ):
        self.app_id = app_id
        self._app_secret = app_secret
        self.timeout = timeout
        self._transport = transport or _urllib_transport
        self._tenant_token = ""
        self._tenant_token_expires_at = 0.0

    def __repr__(self) -> str:
        return f"FeishuClient(app_id={self.app_id!r}, base_url={self.base_url!r})"

    def _raw_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
        token: str | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        status, raw = self._transport(method, url, headers, payload, self.timeout)
        if status >= 400:
            raise FeishuAPIError(f"飞书 API 请求失败（HTTP {status}）")
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise FeishuAPIError("飞书 API 返回了无法解析的结果") from exc
        if data.get("code", 0) != 0:
            raise FeishuAPIError(f"飞书 API 返回错误（code {data.get('code')}）")
        return data

    def tenant_access_token(self) -> str:
        if self._tenant_token and time.time() < self._tenant_token_expires_at - 60:
            return self._tenant_token
        data = self._raw_request(
            "POST",
            "/open-apis/auth/v3/tenant_access_token/internal",
            {"app_id": self.app_id, "app_secret": self._app_secret},
        )
        token = data.get("tenant_access_token")
        if not isinstance(token, str) or not token:
            raise FeishuAPIError("飞书没有返回 tenant_access_token")
        self._tenant_token = token
        self._tenant_token_expires_at = time.time() + int(data.get("expire", 7200))
        return token

    def reply_text(self, message_id: str, text: str) -> None:
        self._raw_request(
            "POST",
            f"/open-apis/im/v1/messages/{message_id}/reply",
            {"content": json.dumps({"text": text}, ensure_ascii=False), "msg_type": "text"},
            token=self.tenant_access_token(),
        )

    def list_wiki_nodes(
        self,
        space_id: str,
        page_token: str | None = None,
        page_size: int = 50,
        parent_node_token: str | None = None,
    ) -> dict[str, Any]:
        query = {"page_size": str(page_size)}
        if page_token:
            query["page_token"] = page_token
        if parent_node_token:
            query["parent_node_token"] = parent_node_token
        return self._raw_request(
            "GET", f"/open-apis/wiki/v2/spaces/{space_id}/nodes", query=query, token=self.tenant_access_token()
        )

    def get_wiki_node(self, space_id: str, node_token: str) -> dict[str, Any]:
        return self._raw_request(
            "GET", f"/open-apis/wiki/v2/spaces/{space_id}/nodes/{node_token}", token=self.tenant_access_token()
        )

    def get_document_raw_content(self, document_id: str) -> dict[str, Any]:
        return self._raw_request(
            "GET", f"/open-apis/docx/v1/documents/{document_id}/raw_content", token=self.tenant_access_token()
        )

    def download_file(self, file_token: str) -> bytes:
        url = f"{self.base_url}/open-apis/drive/v1/files/{file_token}/download"
        headers = {"Authorization": f"Bearer {self.tenant_access_token()}"}
        status, raw = self._transport("GET", url, headers, None, self.timeout)
        if status >= 400:
            raise FeishuAPIError(f"飞书文件下载失败（HTTP {status}）")
        return raw
