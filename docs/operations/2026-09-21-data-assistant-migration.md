# 2026-09-21 瀚邦为问数助手迁移

旧机 192.168.1.73，新机 10.2.13.173，SSH 用户 hbw。密码和服务密钥不写入仓库。

## 迁移对象

独立项目 `/opt/feishu-e10-data-assistant`，与知识库 RAG 是不同应用。
四个容器：data-assistant-api、data-assistant-worker、data-assistant-postgres、data-assistant-redis。
API 实际运行目录为 releases/815af84；Worker 为 releases/0bdccb5。迁移保留实际运行镜像、环境和挂载，不强行统一版本。
数据库为 PostgreSQL 16 Alpine，队列为 Redis 7 Alpine，外部 E10 数据库为 MSSQL。

## 执行与验证

- 导出实际运行的四个镜像，打为 `hbw-data-migration:<api|worker|postgres|redis>-20260921`。
- 停 API/Worker，Redis SAVE，随后停止 PostgreSQL/Redis，复制完整项目及两个数据卷。
- 新机原有代码目录未运行且无对应卷，已保留为 `/opt/feishu-e10-data-assistant.before-migration-20260921`。
- 数据卷名称沿用 `feishu-e10-data-assistant_postgres_data` 和 `feishu-e10-data-assistant_redis_data`。
- 以旧实例实际配置生成新机 `/opt/feishu-e10-data-assistant/compose.migration.json`，root 权限 600。该文件含运行环境凭据，禁止提交 Git。
- 新机四个容器正常启动，PostgreSQL/Redis healthy。
- `/health` HTTP 200，`/health/ready` 确认 PostgreSQL 和 Redis ready。
- API 日志确认飞书 ws client ready；Worker 日志确认 BullMQ Worker initialized。
- E10 TCP 可达，使用现有配置成功执行只读 `SELECT 1`，结果为 1。
- 旧机四个容器停止，restart policy=no；旧机其他 Leantime/MySQL 服务保持运行。
- 知识库助手两个容器保持正常，未受问数助手迁移影响。
- 尚未通过员工飞书消息发起实际业务问数和附件导出验收；未主动向员工发送测试消息。

## 新机维护

使用 `cd /opt/feishu-e10-data-assistant` 后执行 `sudo docker compose -f compose.migration.json ps` 或相应 up/stop 命令。
当前迁移使用具体镜像，无需重新执行数据库迁移；后续升级应按问数助手自身发布流程操作。
API 端口沿用 3000；飞书采用长连接。浏览器或外部调用旧机 IP 的客户端需要改用新 IP。

## 备份与回滚

旧机备份：`/opt/data-assistant/migration-backups/wenshu-20260921`，包含 compose.json、images.tar、payload.tar.gz。
新机备份：`/opt/data-assistant/migration-backups/20260921-wenshu`，包含 compose.json、payload.tar.gz。
两端 payload 校验相同：`d27608b44626e4b90cb0cc512c81e0dd0407c85db860981289d4614d5c96adac`。
回滚先停止新机 API/Worker 和数据库容器，保存新机新增数据，再恢复旧机 restart policy 并启动旧机对应容器。禁止双机同时处理飞书消息。
