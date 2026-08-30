# DeepSeek 混合语义切片实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 Ubuntu RAG 中间件中加入 DeepSeek 语义切片优化、校验和回退，并删除员工可见的来源与引用编号。

**架构：** 服务器先把文档拆成带稳定编号的原子段落，DeepSeek 仅返回段落分组和检索元数据，程序使用原始段落重建切片。SQLite 用独立 `search_text` 字段检索模型生成的标题、关键词和摘要；最终回答只使用原始 `content`。模型异常时回退现有本地切片。

**技术栈：** Python 3.14、SQLite、FastAPI、DeepSeek Chat Completions JSON Output、pytest、Docker Compose、systemd timer。

**版本控制说明：** 当前目录不是 Git 仓库。每个任务结束时运行完整测试并生成带版本号的部署包；部署前备份 SQLite 数据库。不得把 `.env`、文档或数据库放入部署包。

---

## 文件结构

- 创建 `src/feishu_rag/semantic_chunker.py`：原子段落、DeepSeek 分组协议、输出校验、原文重组和本地回退。
- 创建 `tests/test_semantic_chunker.py`：合法分组、原文不变、非法编号、API 失败和批次限制测试。
- 创建 `scripts/evaluate_chunking.py`：对财务、采购、行政问题执行上线前后检索验收。
- 修改 `src/feishu_rag/models.py`：为 `Chunk` 增加 `search_text`。
- 修改 `src/feishu_rag/store.py`：数据库迁移、写入和检索 `search_text`。
- 修改 `src/feishu_rag/llm.py`：增加 JSON Output 调用，不改变现有回答接口。
- 修改 `src/feishu_rag/config.py`：增加混合切片配置。
- 修改 `src/feishu_rag/sync.py`：接入混合切片、策略签名缓存和失败回退。
- 修改 `src/feishu_rag/ingest.py`：让本地文件索引可复用同一切片器。
- 修改 `src/feishu_rag/rag.py`：隐藏来源区块和引用编号。
- 修改 `.env.example`、`README.md`：记录配置、隐私边界、回退和运维方式。
- 修改相关测试文件：覆盖迁移、同步缓存、回答隐藏来源和回归行为。

---

### 任务 1：增加检索元数据字段和数据库迁移

**文件：** `src/feishu_rag/models.py`、`src/feishu_rag/store.py`、`tests/test_store.py`

- [ ] **步骤 1：编写失败的迁移与检索测试**

```python
def test_search_text_is_searchable_but_original_content_is_preserved():
    store = IndexStore(db_path)
    store.upsert_document(
        "finance.pdf", "财务制度", "finance.pdf", "v1",
        [Chunk("c1", "finance.pdf", "财务制度", "提交费用申请单。", None, None,
               search_text="报销 付款 审批流程")],
    )
    result = store.search("报销审批", 3)[0].chunk
    assert result.content == "提交费用申请单。"
    assert result.search_text == "报销 付款 审批流程"
```

- [ ] **步骤 2：运行测试并确认失败**

运行：`.\.venv\Scripts\python.exe -m pytest tests\test_store.py -q`

预期：FAIL，指出 `Chunk` 不接受 `search_text` 或数据库没有该列。

- [ ] **步骤 3：实现模型字段与幂等迁移**

```python
@dataclass(frozen=True)
class Chunk:
    id: str
    source_id: str
    title: str
    content: str
    page: int | None = None
    section: str | None = None
    search_text: str = ""
```

在 `IndexStore._initialize()` 中使用 `PRAGMA table_info(chunks)` 检查列；缺少时执行：

```sql
ALTER TABLE chunks ADD COLUMN search_text TEXT NOT NULL DEFAULT '';
```

更新插入、读取和搜索逻辑：

```python
score += title.count(needle) * 3.0
score += content.count(needle)
score += search_text.count(needle) * 0.75
```

- [ ] **步骤 4：运行存储测试**

运行：`.\.venv\Scripts\python.exe -m pytest tests\test_store.py -q`

预期：全部 PASS。

- [ ] **步骤 5：运行完整测试并记录安全点**

运行：`.\.venv\Scripts\python.exe -m pytest -q`

预期：全部 PASS；当前目录无 Git，不执行 commit。

---

### 任务 2：为 DeepSeek 客户端增加受控 JSON 调用

**文件：** `src/feishu_rag/llm.py`、`tests/test_llm_json.py`

- [ ] **步骤 1：编写失败测试**

```python
def test_complete_json_requests_json_object_and_parses_response():
    transport = FakeTransport({"choices": [{"message": {"content": '{"groups": []}'}}]})
    client = DeepSeekClient("secret", transport=transport)
    result = client.complete_json("system", "user")
    assert result == {"groups": []}
    assert transport.payload["response_format"] == {"type": "json_object"}
    assert transport.payload["temperature"] == 0
```

- [ ] **步骤 2：运行测试验证接口不存在**

运行：`.\.venv\Scripts\python.exe -m pytest tests\test_llm_json.py -q`

预期：FAIL，`DeepSeekClient` 没有 `complete_json`。

- [ ] **步骤 3：实现最小 JSON 调用**

复用现有鉴权和错误分类，增加：

```python
def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
    payload = {
        "model": self.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    content = self._chat_completion(payload)
    result = json.loads(content)
    if not isinstance(result, dict):
        raise DeepSeekError("DeepSeek JSON 输出必须是对象")
    return result
```

现有 `complete()` 的网络代码提取为 `_chat_completion(payload)`，不得改变异常信息或密钥隐藏行为。

- [ ] **步骤 4：测试 JSON 与错误分类**

运行：`.\.venv\Scripts\python.exe -m pytest tests\test_llm_json.py tests\test_rag.py -q`

预期：合法 JSON、非法 JSON、401、429 和 5xx 测试全部 PASS，输出不包含 API Key。

---

### 任务 3：实现 DeepSeek 语义分组与原文重组

**文件：** `src/feishu_rag/semantic_chunker.py`、`tests/test_semantic_chunker.py`

- [ ] **步骤 1：编写合法分组失败测试**

```python
def test_semantic_groups_reassemble_original_units_without_rewriting():
    units = [
        AtomicUnit("u1", "第一条：提交申请。", 1, "申请"),
        AtomicUnit("u2", "第二条：主管审批。", 1, "审批"),
    ]
    planner = FakePlanner({
        "groups": [{
            "unit_ids": ["u1", "u2"],
            "title": "申请与审批",
            "keywords": ["申请", "主管审批"],
            "summary": "申请后由主管审批",
        }]
    })
    chunks = semantic_chunks(units, "finance", "报销制度", planner, 900)
    assert chunks[0].content == "第一条：提交申请。\n\n第二条：主管审批。"
    assert "申请与审批" in chunks[0].search_text
```

- [ ] **步骤 2：运行测试确认模块不存在**

运行：`.\.venv\Scripts\python.exe -m pytest tests\test_semantic_chunker.py -q`

预期：FAIL，无法导入 `semantic_chunker`。

- [ ] **步骤 3：实现原子段落和请求协议**

```python
@dataclass(frozen=True)
class AtomicUnit:
    unit_id: str
    text: str
    page: int | None
    section: str | None
```

DeepSeek 返回格式：

```json
{"groups":[{"unit_ids":["u1","u2"],"title":"申请与审批","keywords":["申请","审批"],"summary":"申请后进入审批"}]}
```

提示词明确禁止返回改写正文，只允许引用输入编号。

- [ ] **步骤 4：实现严格校验和重组**

校验每个输入编号恰好出现一次、组内连续、无未知编号、无空分组；标题、关键词和摘要必须为字符串。重组超过 `max_chars` 时按原子段落边界再次拆分，任一校验失败抛出 `SemanticChunkError`。

- [ ] **步骤 5：增加异常和批次测试**

覆盖编号遗漏、重复、越界、非连续、空 JSON、12,000 字符批次边界，以及输出正文逐字等于输入正文。

运行：`.\.venv\Scripts\python.exe -m pytest tests\test_semantic_chunker.py -q`

预期：全部 PASS。

---

### 任务 4：接入配置、缓存和本地回退

**文件：** `src/feishu_rag/config.py`、`src/feishu_rag/sync.py`、`src/feishu_rag/ingest.py`、`.env.example`、`tests/test_config.py`、`tests/test_feishu_sync.py`、`tests/test_ingest.py`

- [ ] **步骤 1：编写配置与回退失败测试**

```python
def test_semantic_chunking_defaults_to_enabled():
    settings = Settings.from_env(valid_env)
    assert settings.rag_semantic_chunking is True
    assert settings.deepseek_chunk_model == "deepseek-v4-flash"
    assert settings.deepseek_chunk_batch_chars == 12000
```

```python
def test_sync_falls_back_to_local_chunks_when_planner_fails():
    result = sync_wiki_space("space", client, store, semantic_planner=FailingPlanner())
    assert result.indexed == 1
    assert store.search("原始制度正文")
```

- [ ] **步骤 2：运行测试并确认失败**

运行：`.\.venv\Scripts\python.exe -m pytest tests\test_config.py tests\test_feishu_sync.py tests\test_ingest.py -q`

预期：FAIL，新配置和 `semantic_planner` 参数不存在。

- [ ] **步骤 3：增加配置**

```dotenv
RAG_SEMANTIC_CHUNKING=true
DEEPSEEK_CHUNK_MODEL=deepseek-v4-flash
DEEPSEEK_CHUNK_BATCH_CHARS=12000
RAG_CHUNK_STRATEGY_VERSION=hybrid-v1
```

批次字符数必须至少 2,000，切片策略版本必须非空。

- [ ] **步骤 4：实现索引签名缓存**

```python
effective_checksum = hashlib.sha256(
    f"{content_checksum}:{strategy_version}:{chunk_model}".encode("utf-8")
).hexdigest()
```

在线文档和附件都在调用 DeepSeek 前比较 `documents.checksum`；相同则跳过模型调用和重新切片。

- [ ] **步骤 5：接入混合切片和回退**

同步入口构造 DeepSeek 客户端和语义规划器。出现 `DeepSeekError`、`SemanticChunkError` 或超时时使用现有 `chunk_text()`；只记录错误类型和 `source_id`。本地文件索引接收可选规划器，未传入时保持现有行为。

- [ ] **步骤 6：运行集成测试**

运行：`.\.venv\Scripts\python.exe -m pytest tests\test_config.py tests\test_feishu_sync.py tests\test_ingest.py -q`

预期：全部 PASS；相同校验和测试断言规划器只调用一次。

---

### 任务 5：隐藏员工可见来源和引用编号

**文件：** `src/feishu_rag/rag.py`、`tests/test_rag.py`

- [ ] **步骤 1：编写失败测试**

```python
def test_answer_hides_sources_but_keeps_internal_citations():
    answer = service.answer("报销怎么走")
    assert "来源：" not in answer.text
    assert "[1]" not in answer.text
    assert len(answer.citations) == 1
```

假 LLM 保存 system prompt，并断言包含“不得输出资料编号或来源列表”。

- [ ] **步骤 2：运行测试验证现有回答仍附加来源**

运行：`.\.venv\Scripts\python.exe -m pytest tests\test_rag.py -q`

预期：FAIL，回答包含“来源：”。

- [ ] **步骤 3：修改回答输出**

系统提示词增加：

```text
直接回答问题，不得输出资料编号、引用编号、来源列表或“来源”区块。
```

返回值改为：

```python
return RagAnswer(generated.strip(), citations)
```

- [ ] **步骤 4：运行测试**

运行：`.\.venv\Scripts\python.exe -m pytest tests\test_rag.py tests\test_webhook.py -q`

预期：全部 PASS，内部 citations 仍存在。

---

### 任务 6：建立切片效果验收脚本

**文件：** `scripts/evaluate_chunking.py`、`tests/test_evaluate_chunking.py`

- [ ] **步骤 1：编写指标输出失败测试**

```python
def test_evaluation_reports_matches_and_insufficient_flag():
    report = evaluate_questions(store, rag, ["财务报销流程是什么"])
    assert report[0]["matches"] >= 1
    assert report[0]["insufficient"] is False
```

- [ ] **步骤 2：实现固定问题和 JSON 报告**

```python
QUESTIONS = [
    "财务报销流程是什么",
    "供应商管理程序是什么",
    "员工入职流程是什么",
]
```

输出问题、检索命中数、回答长度、是否包含“现有资料不足”、内部引用数量；不得输出正文或文档内容。

- [ ] **步骤 3：运行验收脚本测试**

运行：`.\.venv\Scripts\python.exe -m pytest tests\test_evaluate_chunking.py -q`

预期：PASS。

---

### 任务 7：完整验证、Ubuntu 部署和回滚保障

**文件：** `README.md`、`Dockerfile`、`docker-compose.yml`、`deploy/feishu-rag-sync.service`、`deploy/feishu-rag-sync.timer`

- [ ] **步骤 1：运行完整本地验证**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src
docker compose config --quiet
```

预期：全部退出码为 0。

- [ ] **步骤 2：创建无密钥部署包**

部署包排除 `.env`、`.venv`、`data`、`documents`、缓存目录和 SQLite 数据库；列出归档内容并断言没有这些路径。

- [ ] **步骤 3：备份生产数据库**

```bash
docker compose exec -T rag python -c \
  'import sqlite3,time; s=sqlite3.connect("/app/data/rag.sqlite3"); d=sqlite3.connect(f"/app/data/rag-before-hybrid-{int(time.time())}.sqlite3"); s.backup(d); d.close(); s.close()'
```

验证备份文件存在且大小大于 0。

- [ ] **步骤 4：部署并重建容器**

解压到 `/opt/data-assistant/feishu-rag`，保留 `.env`、`data` 和 `documents`，运行 `sudo docker compose up --build -d`。

预期：`rag` 为 healthy，`rag-events` 为 running，事件监听进程为 1。

- [ ] **步骤 5：生成混合索引并安全切换**

暂停 `rag-events`，运行一次 `feishu-rag-sync.service`，确认三个知识库同步成功后启动 `rag-events`。同步失败时恢复数据库备份并重新启动旧容器。

- [ ] **步骤 6：运行生产验收**

运行：`docker compose exec -T rag python scripts/evaluate_chunking.py --db /app/data/rag.sqlite3`

预期：三类问题命中数均大于 0、`insufficient=false`、回答无“来源：”和引用编号。

- [ ] **步骤 7：验证回退路径**

使用测试注入让规划器抛出异常，运行单份测试文档索引，断言仍产生本地切片；不得修改生产 `.env` 或使用真实员工消息。

- [ ] **步骤 8：验证定时同步与运行状态**

```bash
systemctl is-active feishu-rag-sync.timer
systemctl show feishu-rag-sync.service -p Result --value
curl -fsS http://127.0.0.1:8010/healthz
```

预期：`active`、`success`、健康接口 HTTP 200。

- [ ] **步骤 9：更新运维文档**

README 说明完整正文会发送给 DeepSeek、模型元数据不作为回答依据、缓存条件、回退行为、额度消耗和关闭开关 `RAG_SEMANTIC_CHUNKING=false`。

---

## 计划自检映射

- 原文不可改写：任务 3。
- DeepSeek JSON 协议与错误处理：任务 2、3。
- 独立检索元数据：任务 1。
- 校验和、模型和策略缓存：任务 4。
- DeepSeek 失败回退：任务 4、7。
- 隐藏员工来源：任务 5。
- 三部门效果验收：任务 6、7。
- 隐私、运维和回滚：任务 7。
