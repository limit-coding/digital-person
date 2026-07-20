# Digital Mirror

一个强调隐私、时间隔离和可审计评测的数字镜像实验框架。它同时支持两类严格隔离的镜像：个人历史回放，以及基于公开史料的历史人物数字切片。系统只在限定时间截面上重建选项、冲突和不确定性，不把模型生成的独白冒充人物的真实内心。

网站目标地址：[`mirror.learnpath.tech`](https://mirror.learnpath.tech)。当前人物馆包含 13 位人物，并新增贝多芬与牛顿的三时期研究原型。

## 核心设计

- `actual_self`：预测现实行为，包括疲劳、犹豫和判断—行动落差。
- `constitutional_self`：依据明确规则给出制度化决策。
- 时间隔离：回放题只包含决策截止时间之前的证据。
- 证据溯源：每项预测引用实际使用的证据编号。
- 隐私分层：原始资料、派生语料、逐事件数据和模型输出全部留在本地。
- 云端同意门：云端调用必须通过敏感字段检查并显式启用上传参数。
- 人物隔离：每位历史人物使用独立目录；公开人物 API 与私人本人事件 API 使用不同命名空间和访问控制。
- 竞争假设：历史人物切片展示概率、证据与反证入口，不声称完成“读心”。
- 时期化提问：选择人物的特定时期后提出开放问题，回答分离史料支持、创造性推演与仍然未知。
- 多模型路由：支持 Google Gemini、DeepSeek、OpenAI、OpenAI 兼容接口和本地 Ollama；密钥只从环境变量读取。
- 私人总体画像：聚合跨事件的领域覆盖、判断—行动一致度、权衡强度与临场改道，并显示样本置信度。

架构和评测原则分别见 [docs/architecture.md](docs/architecture.md) 与 [docs/evaluation.md](docs/evaluation.md)。

## 公开仓库边界

本仓库只包含通用代码、Schema、公开历史资料的短摘要与来源链接、无个人事实的模板和合成测试，不包含：

- 原始文章、传记、笔记或聊天记录；
- 私人真实人物、私人时间线、本人事件标签或本人历史回放题；
- 检索索引、派生语料、逐题预测或实验报告；
- API 密钥、账户信息、本地绝对路径或来源哈希。

`data/`、`artifacts/`、`iCloud/` 和 `texts/` 均被 Git 忽略。请勿通过强制添加绕过这些边界。

公开历史人物资料位于 `src/digital_mirror/historical_figures/<figure_id>/`。每个人物目录独立加载，不允许引用私人 `data/`，也不收录受版权保护的传记全文。

## 安装与测试

项目需要 Python 3.11 或更高版本：

```bash
python3 -m pip install -e .
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

测试完全使用模板和合成数据；公开克隆不需要任何私人文件。

## 基本工作流

本地启动公开人物馆与受保护的私人镜像入口：

```bash
export DIGITAL_MIRROR_ACCESS_PASSWORD='设置独立访问密码'
export DIGITAL_MIRROR_SESSION_SECRET='设置高强度随机会话密钥'
fastapi dev
```

复制 `.env.example` 到 `.env` 后可配置模型。Google Gemini 至少需要：

```bash
GEMINI_API_KEY=replace-with-new-key
DIGITAL_MIRROR_GEMINI_MODEL=gemini-2.5-flash
```

不要把真实密钥提交到 Git；`.env` 已被忽略。时期提问 API 为 `POST /api/ask`，需要先登录私人会话，以限制模型调用成本。

审计本地数据源，只生成不含正文的清单和摘要：

```bash
python3 -m digital_mirror.audit \
  --workspace . \
  --policy config/data_policy.json \
  --output artifacts/data_audit
```

将本地私有事件编译成不含答案与未来证据的盲测输入：

```bash
PYTHONPATH=src python3 -m digital_mirror.replay compile \
  data/episodes/private/example_event.json \
  --output artifacts/replay_cases/example_event.json
```

构建本地检索索引：

```bash
PYTHONPATH=src python3 -m digital_mirror.ingest \
  --workspace . \
  --manifest artifacts/data_audit/manifest.jsonl \
  --output artifacts/private/corpus

PYTHONPATH=src python3 -m digital_mirror.retrieval build \
  artifacts/private/corpus/records.jsonl \
  --database artifacts/private/index/digital_mirror.sqlite
```

## 可选云端基线

DeepSeek 基线从 `DEEPSEEK_API_KEY` 环境变量读取密钥。程序不会把密钥写入源码或结果，并要求显式提供 `--allow-cloud-upload`：

```bash
PYTHONPATH=src python3 -m digital_mirror.cloud_baseline \
  artifacts/replay_cases/example_event.json \
  --model deepseek-v4-pro \
  --thinking enabled \
  --reasoning-effort high \
  --allow-cloud-upload \
  --output artifacts/predictions/deepseek/example_event.json
```

即使经过脱敏，也应在上传前人工确认内容边界。默认情况下，任何个人历史都不应发送到云端。
