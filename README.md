# 飞书 + DeepSeek RAG 机器人

这是一个部署在公司 Ubuntu 服务器上的轻量 RAG MVP：文档在本地切片和检索，DeepSeek 负责生成答案，飞书自建应用负责接收员工问题并回复。

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
│   └── smoke_test.py
├── data/                         # 运行时 SQLite，已加入 gitignore
└── documents/                    # Ubuntu 上的待索引文档，已加入 gitignore
```

## 运行方式

### 本地测试

需要 Python 3.11+。真实密钥只放在本机或服务器的 `.env`，不要提交 Git。

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

索引过程只在服务器本地解析、OCR 和切片，不打印文档正文。重复运行同一个文件会按校验和更新，不会产生重复片段。若服务器暂时未安装 OCR，可加 `--no-ocr`，但扫描型 PDF 会被跳过。

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

飞书 Webhook 必须通过 HTTPS 暴露。建议在服务前放置公司网关、Caddy 或 Nginx，只开放 443。容器内部使用 8000，Docker 默认映射为服务器的 8010 端口（可通过 `RAG_HOST_PORT` 修改）。

长连接模式由 `rag-events` 服务运行：在飞书后台选择“使用长连接接收事件”，订阅 `im.message.receive_v1` 后，该服务会主动连接飞书，不需要公网域名或开放 443 端口。

### 从飞书知识库同步

在 `.env` 设置 `FEISHU_SPACE_ID` 后，可按计划任务周期性同步 DOCX 类型节点：

```bash
python -m feishu_rag.sync --db data/rag.sqlite3
```

你当前的三个知识库 ID 如下，可分别执行同步：

```bash
python -m feishu_rag.sync --space-id 7678686555778583752 --db data/rag.sqlite3  # 财务内控库
python -m feishu_rag.sync --space-id 7678686754827685162 --db data/rag.sqlite3  # 采购与供应商管理库
python -m feishu_rag.sync --space-id 7678687286343273653 --db data/rag.sqlite3  # 行政人事内部库
```

同步使用飞书应用的只读权限；扫描型 PDF/附件节点若没有可读文本，会在统计中标记为 `skipped`，可继续使用本地目录索引或 OCR。

## 当前 RAG 行为

- 先使用 SQLite 本地关键词检索，针对中文制度文档无需额外嵌入 API。
- 命中片段会以 `[1]`、`[2]` 形式附上文档标题和页码/章节。
- 无命中时不会调用 DeepSeek，直接返回“知识库中暂无依据”。
- 提示词要求模型只依据召回资料回答，不补造金额、日期、审批人或制度条款。
- 目前支持服务器本地目录索引和飞书 Wiki DOCX 节点同步；PDF/附件仍建议走本地 OCR 索引。

## 密钥和数据安全

- `.env`、SQLite 数据库和 `documents/` 均不提交 Git。
- API Key 和 App Secret 只从环境变量读取，日志和对象 repr 不包含密钥。
- Webhook 开启 Encrypt Key 后强制校验签名。
- 发送给 DeepSeek 的只有检索到的必要片段，不发送整个数据库。
- 当前飞书知识库按你的要求设置为企业全员可读；如果以后改为分部门权限，Webhook 需要增加按用户过滤召回结果的逻辑。

## 验收清单

- `python -m unittest discover -s tests -v` 全部通过。
- `/healthz` 返回 `{"status":"ok"}`。
- 飞书 URL verification 返回 challenge。
- 错误签名返回 HTTP 403。
- 员工发送“报销怎么走”能收到带来源的回答。
- 缺少 API Key 时服务健康检查报配置不完整，且不会发起外部请求。
