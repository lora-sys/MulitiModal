# MulitiModal — Phase 1 治理文档

本文件是当前仓库的唯一治理文档，规则提炼自 #22–#51 的冻结小节。旧仓库的 AGENTS.md / DECISIONS.md / PROJECT_STATUS.md 已随历史删除（仅见 GitHub 历史 commit `9fc5087d`），不构成任何约束。单条 issue 与本文件冲突时，以该 issue 最新的冻结小节为准；Agent 不得自行选择研究问题或改写任务目标。

## 项目是什么 / 不是什么

- **是**：按摩前（Pre-only）生理状态评估与预期疲劳缓解预测（BPSI/EFRI）+ 公开数据中医体质九维模型；双模型工程闭环 + 受限 AutoResearch/RSI V1。
- **不是**：多模态 LLM；不是按摩硬件控制；不产出临床疗效声明；不恢复旧 OPLRI/TCM checkpoint、Gate A/B、M0/M1/M2（#32/#33 已 Deferred）；不训练 LLM 本身，不做无界自我修改。

## 主干与优先级

- **P0**：#48 双模型 Neural Core 与契约骨架；#49 无真实数据的 Trainer / Evaluator / Checkpoint / CLI。
- **TCM 线**：#37 → #38 → #39 → #40；先行真实训练入口 #50。
- **生理线**：#23 → #24 → #25 → #26 → #27 → #28 → #29 → #30 → #31；研究指标 #34、规则标签 #42、双输出网络 #43。
- **QA / 集成**：#35 CI → #46 端到端 → #47 Science Gate → #36 Go/No-Go；AutoResearch：#44 → #45。
- **Legacy**：#32 / #33 仅历史参考，不阻塞任何任务。

## 硬规则（每条可追溯到 issue）

1. **Pre-only 推理**（#23/#31/#42）：预测输入只允许按摩前身体与生理字段；按摩中 EEG 与按摩后测量仅用于训练期标签构造，禁止经特征别名或衍生计算混入 Pre-only 接口。
2. **Subject-level 划分**（#27）：同一受试者的全部 Session / Window 只能属于一个 partition；锁定测试不得在候选比较中反复使用。
3. **折内拟合**（#29/#42）：scaler、特征筛选、BPSI/EFRI 权重只能在训练折内 fit，冻结折内状态后应用于验证/测试；禁止全量预拟合标签。
4. **数据边界**（#26/#35/#37）：真实数据、检查点、工件不入 Git；synthetic fixture 必须显式标记；原始文件只读，导入前后哈希一致。
5. **标签分层**（#38/#50）：RULE / PUBLISHED / LLM_SILVER 全链路溯源、不可互换；LLM_SILVER 永不作参考真值；规则缺项拒判（ABSTAIN），不得补零或猜标签。
6. **指标语义**（#41/#42/#43）：EFRI 是主要预测目标（实验性构造复合指标）；BPSI 是 Pre 直接计算的确定性基线 + 可选辅助头；两者都不是独立临床金标准；`EFRI_JOINT_EQUAL_BLOCK_V0` 的 0.5/0.5 是项目设计选择，不是 PDF 或医学系数。
7. **证据状态显式化**（#22/#47）：未知事实用状态表达（`NOT_AVAILABLE` / `UNVERIFIED` / `ABSTAIN` / `SMOKE_ONLY` / `NOT_VALIDATED` / `AGENT_NOT_VERIFIED` 等），Agent 不得猜测或补全。
8. **三层放行**（#47）：`GO_ENGINEERING`（代码/合成测试）、`GO_REAL_DATA_TRAINING`（真实标签过 QC）、`GO_RESEARCH_CLAIMS`（独立参考数据 + 未动用最终测试）独立判定，不得由 smoke 分数推导。
9. **AutoResearch 边界**（#45）：Agent 只能修改白名单内 train / model / config；数据、标签、规则、评估器、split、holdout 冻结；Trusted Evaluator 在 Agent 不可写的隔离环境运行；不可信代码必须 OS / 容器级隔离，仅审查 diff 不算隔离。
10. **Provenance**（#37/#40/#49）：数据 DOI / 哈希、规则与标准版本、标签 tier、模型配置、种子、依赖、代码 commit 全部入档，可独立重算。

## 提交与协作

- Conventional Commits，一个 purpose 一个提交；不使用 `update` / `fix` / `！` 类标题。
- 不提交：真实数据、`runs/`、`artifacts/`、检查点、日志。
- 开工前读对应 issue 全文（含追加冻结小节）；工程改动保持最小端到端可验证：schema 改动带契约测试，adapter 带形状 / mask / 负例测试，bug 修复带回归测试。
- 训练与推理预处理必须一致；synthetic 结果只表述工程有效，不表述科研 / 临床有效。

## 词汇表（节选）

| 术语 | 含义 |
| --- | --- |
| BPSI | 按摩前生理状态基线指数（Pre 特征方向对齐加权，PDF Eq.2） |
| EFRI | 预期疲劳缓解指数（During EEG + Post−Pre 构造的实验性复合目标，PDF Eq.3） |
| Pre / During / Post | 按摩前 / 按摩中 / 按摩后测量阶段 |
| RULE / PUBLISHED / LLM_SILVER | 规则计分 / 作者已发布 / LLM 弱标签三级标签来源 |
| SMOKE_ONLY | 仅合成数据工程验证，无科学结论 |
| GO_* | #47 定义的三层放行判定 |
