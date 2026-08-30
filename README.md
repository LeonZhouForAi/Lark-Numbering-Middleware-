# 飞书 + DeepSeek RAG 机器人

这是一个部署在公司 Ubuntu 服务器上的轻量 RAG 服务：文档先在本地解析，再由 DeepSeek 可选地优化语义切片，飞书自建应用负责接收员工问题并回复。

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
│   └── test_feishu_sync.py
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
python -m unittest discover -s tests -v
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
RAG_CHUNK_STRATEGY_VERSION=hybrid-v1
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

索引过程在服务器本地解析和 OCR，先按结构初切，再将文档正文按批次发送给 DeepSeek 做语义分组。模型只返回段落编号和检索元数据，程序用原文重组切片。重复运行同一个文件会按内容、模型和策略签名跳过，不会产生重复片段；模型失败时自动退回本地切片。

### 启动服务

```bash
uvicorn feishu_rag.web:app --host 0.0.0.0 --port 8000
python scripts/smoke_test.py
```

生产环境使用 Docker：

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

长连接模式由 `rag-events` 服务运行：在飞书后台选择“使用长连接接收事件”，订阅 `im.message.receive_v1` 后，该服务会主动连接飞书，不需要公网域名或开放 443 端口。

### 从飞书知识库同步

在 `.env` 设置 `FEISHU_SPACE_ID` 后，可手动同步知识库节点和附件：

```bash
python -m feishu_rag.sync --db data/rag.sqlite3
```

你当前的三个知识库 ID 如下，可分别执行同步：

```bash
python -m feishu_rag.sync --space-id 7678686555778583752 --db data/rag.sqlite3  # 财务内控库
python -m feishu_rag.sync --space-id 7678686754827685162 --db data/rag.sqlite3  # 采购与供应商管理库
python -m feishu_rag.sync --space-id 7678687286343273653 --db data/rag.sqlite3  # 行政人事内部库
```

同步使用飞书应用的只读权限；PDF 和支持的附件会自动下载到内存临时文件并在服务器本地解析，不保留额外副本。

## 当前 RAG 行为

- 先使用 SQLite 本地关键词检索，针对中文制度文档无需额外嵌入 API。
- 命中片段会用于生成回答，但员工不会看到来源列表或 `[1]`、`[2]` 引用编号。
- 无命中时不会调用 DeepSeek，直接返回“知识库中暂无依据”。
- 提示词要求模型只依据召回资料回答，不补造金额、日期、审批人或制度条款。
- `RAG_SEMANTIC_CHUNKING=true` 时使用本地结构切片加 DeepSeek 语义分组；失败自动回退本地切片。
- 模型生成的标题、关键词和摘要只参与检索，最终回答上下文只包含原始正文。

## 密钥和数据安全

- `.env`、SQLite 数据库和 `documents/` 均不提交 Git。
- API Key 和 App Secret 只从环境变量读取，日志和对象 repr 不包含密钥。
- Webhook 开启 Encrypt Key 后强制校验签名。
- 语义切片开启时，完整文档会在索引阶段按批次发送给 DeepSeek；回答阶段只发送命中的原始片段。
- 当前飞书知识库按你的要求设置为企业全员可读；如果以后改为分部门权限，Webhook 需要增加按用户过滤召回结果的逻辑。

## 验收清单

- `python -m unittest discover -s tests -v` 全部通过。
- `/healthz` 返回 `{"status":"ok"}`。
- 飞书 URL verification 返回 challenge。
- 错误签名返回 HTTP 403。
- 员工发送“报销怎么走”能收到不带来源区块的回答。
- 缺少 API Key 时服务健康检查报配置不完整，且不会发起外部请求。
