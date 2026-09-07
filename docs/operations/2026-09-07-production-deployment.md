# 2026-09-07 生产部署记录

- 部署代码：445ba4c，基于 v0.9.0，包含能力介绍和预热运行修复。
- 环境：Ubuntu，Docker，Python 3.14.7。
- 服务：rag、rag-events；健康检查正常，飞书长连接已连接。
- 索引：49 份文档、347 个片段、2605 条结构化事实。
- 预热：首轮 10 个候选，9 个成功，1 个失败。失败候选不启用。
- 定时器：feishu-rag-sync.timer、feishu-rag-preheat.timer 均 active。

## 验证

服务器 Python 3.14 镜像内全量测试完成，未发现失败项。最终自动入队修复另通过 9 项预热测试。
数据库副本先完成迁移和索引；能力介绍、三个 IE 精确查询通过。
费用报销、供应商开发、品质异常三项真实 DeepSeek 问答均返回 answerable，且无来源区块。
生产切换后健康检查、长连接和两个定时器正常。
尚未通过员工手机发起端到端验收消息。

## 回滚材料

- 备份目录：`/opt/data-assistant/rag-backups/20260907-7451257`。
- 旧代码及配置：`code-config.tar.gz`。
- SQLite 在线一致性备份：`rag.sqlite3`。
- 停机后数据库副本：`final-rag.sqlite3`。
- 旧镜像：`hbw-rag:rollback-20260907`、`hbw-rag-events:rollback-20260907`。
- 当前镜像：`hbw-rag:445ba4c`。

恢复旧版时先停止定时器和两个服务，恢复对应代码、配置和一致性数据库备份，再使用旧镜像启动。
不得直接让 v0.2.3 操作迁移后的生产数据库。
