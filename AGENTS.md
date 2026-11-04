# MulitiModal：工程规范与项目知识

本文件供参与本仓库工作的 Agent 使用。开始改动前读对应 Issue 的完整内容；若 Issue 最新冻结小节与本文件冲突，以该冻结小节为准，不自行改写研究目标。

## 工程规范

- 实现保持简单、职责清晰；先检查现有依赖和成熟方案。已废弃的路径直接删除，不增加兼容层。
- AutoResearch 由 `skills/autoresearch/` 驱动；研究简报指定唯一可修改文件（证候任务为 `src/research/train.py`）。固定数据、标签、划分、评估器、指标、预算和 holdout 不得修改。每轮读取已有证据并提出一个假设，按预先确定的规则记录 KEEP/REVERT/CRASH。真实科研评分必须来自可信评估器；当前本地记录未经独立认证，只可用于合成 smoke。
- 可信评估器必须运行在 Agent 不可写的隔离环境；不可信代码必须有 OS/容器级隔离，单独进程或审查 diff 不算隔离。目前端到端隔离状态为 `AGENT_ISOLATION_UNVERIFIED`。
- 同一受试者全部记录属于同一 partition；预处理、特征选择和权重只在训练折内拟合。选择和比较不得访问 holdout。体质线 2/2、证候线 1/1 的 holdout 解锁预算已用尽；再评估须有新数据里程碑或用户明确授权。
- 生理线推理只使用按摩前字段；按摩中和按摩后数据仅可用于训练期标签构造，不可通过别名或衍生特征进入推理。
- RULE、PUBLISHED、LLM_SILVER 标签必须保留来源且不可互换；LLM_SILVER 不作参考真值。规则缺项时 ABSTAIN，不补零或猜测。
- 原始数据只读，导入前后核对哈希；真实数据、检查点、工件、日志、`runs/` 和 `artifacts/` 不入 Git。合成数据必须显式标记。
- 记录数据哈希、标准和标签版本、模型配置、种子、依赖与代码 commit。未知信息用 `NOT_AVAILABLE`、`UNVERIFIED`、`SMOKE_ONLY`、`NOT_VALIDATED` 或 `AGENT_NOT_VERIFIED` 等状态表达。
- `GO_ENGINEERING`、`GO_REAL_DATA_TRAINING`、`GO_RESEARCH_CLAIMS` 独立判定，不能由合成或 dev/CV 分数推导科研结论。不得宣称临床疗效。
- Conventional Commits，一个 purpose 一个提交；改动保持最小且可验证。schema 改动带契约测试，adapter 带形状、mask 和负例测试，bug 修复带回归测试。用户明确要求在开发机安装依赖、跑测试或启动验证时使用 `remote-dev`。

## 项目知识

- 项目面向按摩椅场景的中医健康参考，不是多模态 LLM、硬件控制或诊断系统。证候、体质、生理三条线的输入和标签不可互换。
- 当前证候线使用 TCM-SD 中文症状主诉，54,152 病例、148 类证候，标签为 PUBLISHED。数据许可为 CC BY-NC-SA 4.0，未经授权不得商业部署。同一主诉对应不同证型是数据属性；训练集内重复文本保留，折内对 dev 做单向精确文本去重。
- #57 已完成的 protocol-v2 dev/CV 对比中，字符 1–3 gram TF-IDF+LR 的 macro-F1 为 0.3199，优于已完成的神经方案；它是交付候选。新神经研究需要更多信息（新特征、标签或数据），不要把工程探索表述为临床验证。
- 体质线历史交付 v11，Mendeley 数据为 274 人、8 维公开评分。生理线缺少同人配对真实数据，状态 BLOCKED，不得伪造标签。
- 外部验证覆盖不足，`GO_RESEARCH_CLAIMS` 未放行。#35 CI、#46 端到端与 #36 签收状态须以 Issue 为准。
