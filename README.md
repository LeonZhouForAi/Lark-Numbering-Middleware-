# 飞书 + DeepSeek RAG 机器人

这是一个面向公司 Ubuntu 服务器部署的轻量 RAG 服务：文档先在本地解析，再由 DeepSeek 可选地优化语义切片，飞书自建应用负责接收员工问题并回复。当前 v0.4.0 已完成代码与测试，尚未部署生产环境。

## 目录

```text
RAGcode/
├── pyproject.toml
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── README.md
├── src/feishu_rag/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── chunker.py
│   ├── store.py
│   ├── ingest.py
│   ├── llm.py
│   ├── rag.py
│   ├── semantic_chunker.py
│   ├── feishu_client.py
│   ├── sync.py
│   └── web.py
├── tests/
│   ├── test_config.py
│   ├── test_chunker.py
│   ├── test_store.py
│   ├── test_ingest.py
│   ├── test_rag.py
│   ├── test_webhook.py
│   ├── test_feishu_sync.py
│   ├── test_evaluation.py
│   ├── test_evaluate_chunking.py
│   ├── test_index_local_documents.py
│   ├── test_llm_json.py
│   ├── test_long_connection.py
│   └── test_semantic_chunker.py
├── scripts/
│   ├── index_local_documents.py
│   ├── evaluate_chunking.py
│   └── smoke_test.py
├── data/                         # 运行时 SQLite，已加入 gitignore
└── documents/                    # Ubuntu 上的待索引文档，已加入 gitignore
```

## 运行方式

### 本地测试

需要 Python 3.14+。真实密钥只放在本机或服务器的 `.env`，不要提交 Git。

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[test]"
cp .env.example .env
python -m pytest -q
```

### 准备 DeepSeek

在 `.env` 设置：

```dotenv
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_CHUNK_MODEL=deepseek-v4-flash
DEEPSEEK_CHUNK_BATCH_CHARS=12000
RAG_SEMANTIC_CHUNKING=true
RAG_CHUNK_STRATEGY_VERSION=hybrid-v4
RAG_MIN_RELEVANCE=0.42
RAG_QUESTION_MAX_CHARS=500
RAG_RATE_LIMIT_PER_MINUTE=10
RAG_RATE_LIMIT_PER_DAY=200
RAG_WORKER_THREADS=4
RAG_MAX_PENDING_MESSAGES=32
API_RETRY_MAX_ATTEMPTS=3
API_RETRY_BASE_DELAY=0.5
```

DeepSeek 使用 OpenAI 兼容的 `/chat/completions` 接口，模型和价格以官方文档为准：
<https://api-docs.deepseek.com/guides/function_calling/>

### 准备飞书自建应用

1. 在飞书开放平台创建企业自建应用并启用机器人能力。
2. 订阅事件 `im.message.receive_v1`。
3. 服务器有公网 HTTPS 地址时，可将事件请求地址配置为：`https://你的域名/webhook/feishu`；没有公网入口时，推荐改用飞书长连接模式。
4. 把飞书后台生成的 Verification Token 和 Encrypt Key 写入 `.env`。
5. 为应用申请机器人发消息、知识库只读、文档只读和云盘只读权限。常用知识读取权限包括 `wiki:wiki:readonly`、`docx:document:readonly`、`drive:drive:readonly`，最终以开放平台权限列表为准。
6. 如知识库不是企业公开范围，把应用或机器人加入对应知识库成员/管理员，否则 Wiki API 可能返回无权限。
7. 发布应用并让管理员审批权限。

服务会自动处理 URL verification challenge，并校验 `X-Lark-Signature`。机器人只处理文本消息，忽略机器人自己发送的消息，避免循环回复。

### 索引本地文档

把需要检索的 PDF、DOCX、Markdown、TXT 文件复制到 Ubuntu 的 `documents/`。`.doc` 和 `.wps` 请先转换为 `.docx`。扫描型 PDF 需要中文 OCR，Docker 镜像会安装 Poppler 和 Tesseract 中文语言包。

```bash
python scripts/index_local_documents.py documents --db data/rag.sqlite3
```

索引过程在服务器本地解析和 OCR，先按结构初切，再将文档正文按批次发送给 DeepSeek 做语义分组。模型只返回段落编号和检索元数据，程序用原文重组切片。DOCX 会按文档顺序提取父页正文、表格和嵌套可见文本；混合 PDF 仅对没有原生文本的页面逐页 OCR。重复运行同一个文件会按内容、模型和策略签名跳过，不会产生重复片段；模型失败时自动退回本地切片。

飞书同步只有在整个空间成功完成完整快照后，才会清理本次快照中已失效的索引；分页异常或同步失败不会触发删除。质量可用 `scripts/evaluate_chunking.py` 配合不含制度正文的金标 JSON/报告脱敏 CLI 验收（支持 `--cases`、`--retrieval-only` 和阈值参数）。

### 启动服务

```bash
uvicorn feishu_rag.web:app --host 0.0.0.0 --port 8000
python scripts/smoke_test.py
```

Docker 宿主机端口发布绑定地址由 `RAG_BIND_HOST` 控制，默认仅发布到 `127.0.0.1:8010`。手工运行 Uvicorn 仍监听 `0.0.0.0:8000`。生产环境使用 Docker：

```bash
cp .env.example .env
chmod 600 .env
# 编辑 .env，填入 DeepSeek 和飞书配置
docker compose up -d --build
docker compose run --rm rag python scripts/index_local_documents.py /app/documents --db /app/data/rag.sqlite3
docker compose ps
```

生产环境可启用 `deploy/feishu-rag-sync.timer`，每小时递归同步三个知识库并刷新索引。

飞书 Webhook 必须通过 HTTPS 暴露。建议在服务前放置公司网关、Caddy 或 Nginx，只开放 443。容器内部使用 8000，Docker 默认映射为服务器的 8010 端口（可通过 `RAG_HOST_PORT` 修改）。

长连接模式由 `rag-events` 服务运行：在飞书后台选择“使用长连接接收事件”，订阅 `im.message.receive_v1` 后，该服务会主动连接飞书，不需要公网域名或开放 443 端口，也无需将 8010 端口暴露到公网。事件处理通过有界工作线程执行，但只有在完整 RAG 处理结束后才向飞书返回成功 ACK；异常返回失败状态以便重投。并发线程数和最大在途消息数分别由 `RAG_WORKER_THREADS`、`RAG_MAX_PENDING_MESSAGES` 控制。

### 从 v0.4 回滚到 v0.3

v0.4 的 `chunks_fts` 列结构与 v0.3 不兼容，不能直接用 v0.3 容器打开并继续写入。切换代码前必须停止问答、长连接和同步定时器，并使用 SQLite Backup API 生成一致性备份：

```bash
sudo systemctl stop feishu-rag-sync.timer
sudo docker compose stop rag rag-events
mkdir -p data/backups

# 第一次只读预检，不修改数据库
python scripts/prepare_v03_rollback.py data/rag.sqlite3 \
  --backup data/backups/rag-before-v03.sqlite3

# 核对路径后显式执行：先备份，再只删除 v2 chunks_fts
python scripts/prepare_v03_rollback.py data/rag.sqlite3 \
  --backup data/backups/rag-before-v03.sqlite3 --execute
```

备份路径必须位于已经存在的目录且不能已有同名文件，脚本永不覆盖备份。执行成功后再切换到 v0.3；v0.3 首次启动会创建旧 FTS 列，现有 `documents/chunks` 不会被删除。v0.3 的字面检索仍可使用；若需要完整重建旧 FTS，应在启动后执行一次全量同步。详细步骤见 [`docs/operations/v0.4-to-v0.3-rollback.md`](docs/operations/v0.4-to-v0.3-rollback.md)。

### 从飞书知识库同步

在 `.env` 设置 `FEISHU_SPACE_ID` 后，可手动同步知识库节点和附件：

```bash
python -m feishu_rag.sync --db data/rag.sqlite3
```

查看按日期、模型和用途聚合的 Token 用量，并从命令行传入输入/输出单价：

```bash
python scripts/report_usage.py data/rag.sqlite3 --input-price 1 --output-price 2
```

FAQ 运维指标仅输出按日期和匿名范围聚合的数量（不包含问题、答案、来源或员工身份）：

```bash
python scripts/report_faq.py data/rag.sqlite3
python scripts/report_faq.py data/rag.sqlite3 --since 2026-08-01
python scripts/cleanup_faq.py data/rag.sqlite3 --today 2026-08-31
```

FAQ 在最近 15 天内同一问题第 3 次安全回答后晋级，第 4 次起可直接回复；知识库资料更新后，首个安全回答会刷新旧条目。可将 `RAG_FAQ_ENABLED=false` 关闭 FAQ 功能，其余阈值由 `.env` 中的 `RAG_FAQ_PROMOTION_COUNT`、`RAG_FAQ_WINDOW_DAYS`、`RAG_FAQ_MIN_TEXT_SIMILARITY` 和 `RAG_FAQ_MIN_SOURCE_OVERLAP` 控制。

该报表只汇总 DeepSeek 成功响应中返回的 `usage`，属于本地观测值而非服务商账单。网络中断、超时、429 或 5xx 等未返回可用 `usage` 的调用可能已经产生费用，但本地无法取得其 Token 数，因此不会进入报表；成本核对应以 DeepSeek 账单为准。

你当前的三个知识库 ID 如下，可分别执行同步：

```bash
python -m feishu_rag.sync --space-id 7678686555778583752 --db data/rag.sqlite3  # 财务内控库
python -m feishu_rag.sync --space-id 7678686754827685162 --db data/rag.sqlite3  # 采购与供应商管理库
python -m feishu_rag.sync --space-id 7678687286343273653 --db data/rag.sqlite3  # 行政人事内部库
```

同步使用飞书应用的只读权限；PDF 和支持的附件会自动下载到内存临时文件并在服务器本地解析，不保留额外副本。

## 当前 RAG 行为

- 先使用 SQLite FTS5/BM25 和字面关键词检索生成候选，再以 RRF 混排；confidence 低于 `RAG_MIN_RELEVANCE`（默认 0.42）的泛词或无关命中会被过滤。
- 当前不使用向量数据库或外部 Embedding 服务；针对中文制度文档，检索完全在本地 SQLite 完成。
- 回答模型只接收匿名结构化 JSON 中的原始正文，不接收标题、来源、页码或引用编号；员工不会看到来源列表或 `[1]`、`[2]` 引用编号。
- 回答严格校验 `answer` 与 `evidence_sufficient` 字段；证据不足、来源/链接/密钥泄漏或其他危险输出均 fail-closed，返回固定提示。
- 无命中时不会调用 DeepSeek，直接返回“知识库中暂无依据”。
- 提示词要求模型只依据召回资料回答，不补造金额、日期、审批人或制度条款。
- `RAG_SEMANTIC_CHUNKING=true` 时使用本地结构切片加 DeepSeek 语义分组；失败自动回退本地切片。
- 模型生成的标题、关键词和摘要只参与检索，最终回答上下文只包含原始正文。
- DeepSeek Chat Completions 属于可能计费的生成 POST，429、5xx、网络中断和超时均只尝试一次，不自动重试；飞书幂等读取请求仍按有限次数重试，发送回复等非幂等写入也不重试。Token usage 可用上方 `report_usage.py` 命令查看。
- 消息去重后按用户哈希执行固定分钟/日限流（默认每分钟 10 次、每天 200 次，设为 0 可禁用）；数据库不保存原始用户 ID。
- `RetrievalScope` 对检索提供空间过滤 seam：未指定时检索全库，指定空间集合时严格限制候选范围；后续可在此接入更细粒度 ACL。

## 密钥和数据安全

- `.env`、SQLite 数据库和 `documents/` 均不提交 Git。
- API Key 和 App Secret 只从环境变量读取，日志和对象 repr 不包含密钥。
- 员工可见回答会拦截密码、口令、密钥、API 密钥、访问令牌等中英文凭据标签和值，并返回固定安全提示。
- Webhook 开启 Encrypt Key 后强制校验签名。
- 语义切片开启时，完整文档会在索引阶段按批次发送给 DeepSeek；回答阶段只发送命中的原始片段。
- 切片策略版本固定为 `hybrid-v4`；修改策略版本后应重新索引现有文档。
- 当前飞书知识库按你的要求设置为企业全员可读；如果以后改为分部门权限，Webhook 需要增加按用户过滤召回结果的逻辑。

## 验收清单

- `python -m pytest -q` 全部通过。
- `/healthz` 返回 `{"status":"ok"}`。
- 飞书 URL verification 返回 challenge。
- 错误签名返回 HTTP 403。
- 员工发送“报销怎么走”能收到不带来源区块的回答。
- 缺少 API Key 时服务健康检查报配置不完整，且不会发起外部请求。
- 回滚预检不修改数据库；带 `--execute` 才会创建独占备份并移除 v2 FTS。

v0.4.0 发布状态：代码与测试已完成，尚未部署生产；仍使用 SQLite（含 FTS5/BM25）和 `hybrid-v4`，没有引入向量数据库。

长连接适配器依赖 `lark-oapi==1.7.3` 的私有 ACK 契约。升级 SDK 必须显式修改锁定版本，并通过 `tests/test_lark_sdk_contract.py` 的真实 SDK 合约测试后才能发布。
