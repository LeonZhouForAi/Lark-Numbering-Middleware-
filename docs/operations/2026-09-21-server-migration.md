# 2026-09-21 Linux 服务器迁移

旧服务器：192.168.1.73。新服务器：10.2.13.173。SSH 用户：hbw。凭据通过受控渠道保存，不写入 Git。

## 已执行

- 新机 Ubuntu 26.04.1 安装并启用 Docker、Compose。
- 从旧机导出实际运行的 rag 和 rag-events 镜像；停止旧机定时器、活动同步/预热任务以及两个容器后，打包代码、配置、documents 和 data。
- 迁移包两端 SHA256 一致：9975614e3c66ce24981fb8cf5b603f52c667f013d3e63c60c99cf037c5dcdde7。
- 新机应用路径保持 `/opt/data-assistant/feishu-rag`，数据目录所有权设置为容器 UID/GID 10001。
- 迁移旧机实际使用的四个 systemd 单元，并启用 sync/preheat 两个 timer。
- 旧机两个容器保持停止，restart policy 设置为 no，两个 timer 禁用。旧代码和数据保留，不删除。

## 验证结果

- SQLite integrity_check：ok。
- 迁移后：58 份文档、359 个片段、2605 条结构化事实。
- 能力介绍原句“请问你会做什么”通过。
- 氧化物系列绑定工时精确查询为 101。
- DeepSeek 报销问答复测 answerable，198 字。
- `/healthz` 返回 ok，rag 容器 healthy，rag-events 日志确认飞书长连接 connected。
- feishu-rag-sync.timer 和 feishu-rag-preheat.timer 均 active。
- 尚未通过员工手机发送实际消息进行本次迁移的端到端验收。

## 迁移中修复

真实问答两次发现模型返回 status 与 answer，但省略空的 clarifying_question，原校验拒绝该结果。
现在仅对 answerable/insufficient 状态补齐缺失的空澄清字段；ambiguous 仍必须提供有效澄清问题。
相关 RAG 测试通过。新机修复镜像为 `hbw-rag:migrated-20260921`，两个 Compose 镜像标签均指向该镜像。
源代码修复也写回新机应用目录，后续标准 Dockerfile 重建可以保留此修复。

## 回滚和注意事项

- 两台机器均保留迁移包：`/opt/data-assistant/rag-backups/migration-20260921/hbw-rag-migrate-20260921-data.tar.gz`，权限 600。
- 回滚时先停新机两个 timer/service 和容器，再启动旧机；不能让两个机器人长连接同时处理消息。
- 若新机已产生新问答或同步数据，回滚前备份新机数据库，不能直接丢弃增量。
- 新机截图显示地址为 DHCP 动态分配。运维应在路由器保留地址或按公司网络规范固定地址；本次未修改网络配置。
- 新机未安装 buildx，`docker build --progress=plain` 不支持，改用普通 `docker build`。
