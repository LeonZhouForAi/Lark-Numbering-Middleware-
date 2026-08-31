"""飞书开放平台客户端：机器人回复与知识库只读同步接口。"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from .retry import RetryPolicy, run_with_retry


class FeishuAPIError(RuntimeError):
    """飞书 API 调用失败。"""


class FeishuReplyNotSentError(FeishuAPIError):
    """回复请求尚未发送，可以安全重试。"""


Transport = Callable[[str, str, dict[str, str], dict[str, Any] | None, float], tuple[int, bytes]]


class _FeishuNetworkError(RuntimeError):
    pass


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
        raise _FeishuNetworkError from exc


class FeishuClient:
    base_url = "https://open.feishu.cn"

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        transport: Transport | None = None,
        timeout: float = 20.0,
        retry_policy: RetryPolicy | None = None,
    ):
        self.app_id = app_id
        self._app_secret = app_secret
        self.timeout = timeout
        self._transport = transport or _urllib_transport
        self.retry_policy = retry_policy or RetryPolicy()
        self._tenant_token = ""
        self._tenant_token_expires_at = 0.0
        self._tenant_token_lock = threading.Lock()

    def __repr__(self) -> str:
        return f"FeishuClient(app_id={self.app_id!r}, base_url={self.base_url!r})"

    def _raw_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
        token: str | None = None,
        retryable: bool | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        status, raw = self._request(
            method,
            url,
            headers,
            payload,
            retryable=method == "GET" if retryable is None else retryable,
        )
        if status >= 400:
            raise FeishuAPIError(f"飞书 API 请求失败（HTTP {status}）")
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise FeishuAPIError("飞书 API 返回了无法解析的结果") from exc
        if data.get("code", 0) != 0:
            raise FeishuAPIError(f"飞书 API 返回错误（code {data.get('code')}）")
        return data

    @staticmethod
    def _retryable_exception(exc: Exception) -> bool:
        return isinstance(
            exc,
            (_FeishuNetworkError, urllib.error.URLError, TimeoutError, OSError),
        )

    def _request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any] | None,
        *,
        retryable: bool,
    ) -> tuple[int, bytes]:
        def request() -> tuple[int, bytes]:
            return self._transport(method, url, headers, payload, self.timeout)

        try:
            if retryable:
                return run_with_retry(
                    request,
                    self.retry_policy,
                    retry_result=lambda result: result[0] == 429
                    or 500 <= result[0] < 600,
                    retry_exception=self._retryable_exception,
                )
            return request()
        except Exception as exc:
            if self._retryable_exception(exc):
                raise FeishuAPIError(
                    "无法连接飞书开放平台，请检查服务器网络"
                ) from exc
            raise

    def tenant_access_token(self) -> str:
        if self._tenant_token and time.time() < self._tenant_token_expires_at - 60:
            return self._tenant_token
        with self._tenant_token_lock:
            if self._tenant_token and time.time() < self._tenant_token_expires_at - 60:
                return self._tenant_token
            data = self._raw_request(
                "POST",
                "/open-apis/auth/v3/tenant_access_token/internal",
                {"app_id": self.app_id, "app_secret": self._app_secret},
                retryable=True,
            )
            token = data.get("tenant_access_token")
            if not isinstance(token, str) or not token:
                raise FeishuAPIError("飞书没有返回 tenant_access_token")
            expire = data.get("expire")
            if type(expire) is not int or not 0 < expire <= 7 * 24 * 3600:
                raise FeishuAPIError("飞书返回了无效的 token 有效期")
            self._tenant_token = token
            self._tenant_token_expires_at = time.time() + expire
            return token

    def reply_text(self, message_id: str, text: str) -> None:
        try:
            token = self.tenant_access_token()
        except FeishuAPIError as exc:
            raise FeishuReplyNotSentError(
                "飞书回复尚未发送：无法获取访问令牌"
            ) from exc
        self._raw_request(
            "POST",
            f"/open-apis/im/v1/messages/{message_id}/reply",
            {"content": json.dumps({"text": text}, ensure_ascii=False), "msg_type": "text"},
            token=token,
            retryable=False,
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
        status, raw = self._request("GET", url, headers, None, retryable=True)
        if status >= 400:
            raise FeishuAPIError(f"飞书文件下载失败（HTTP {status}）")
        return raw
