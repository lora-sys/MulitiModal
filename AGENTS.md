# MulitiModal — 工程规范与项目知识

本文件只讲两件事：**必须遵守的规范**，和**这个项目的关键事实**。规则与单条 issue 冻结小节冲突时，以该 issue 最新冻结小节为准；Agent 不得自行改写任务目标或放宽规范。

## 一、项目是什么

**按摩椅场景下的中医健康参考**：用户可能不填写症状问卷，设备只有生理传感器；系统要给出有依据、可追溯、不冒充诊断的中医参考。

三条独立、标签不可互换的线：

| 线 | 输入 → 输出 | 数据 | 状态 |
| --- | --- | --- | --- |
| **证候线（当前主线）** | 中文症状主诉 → 148 类证候倾向 | TCM-SD 公开病历 54,152 例 / 148 证型，PUBLISHED 标签，CC BY-NC-SA 4.0（非商业） | 阶段 1 已交付（TF-IDF+LR 为交付候选）；AutoResearch 在 dev/CV 上的进一步探索仅作工程闭环验证 |
| **体质线（历史）** | Body/Vitals → 九维体质评分 | Mendeley 274 人，8 维公开评分 | 已交付 v11；holdout 预算用尽 |
| **生理线** | Pre 生理 → BPSI/EFRI | 无同人配对真实数据 | **BLOCKED**，不得伪造标签 |

**不是**：多模态 LLM；不是按摩硬件控制；不产出临床疗效声明。

## 二、硬规范

1. **Pre-only**：推理输入只允许按摩前字段。按摩中/后数据只能用于构造训练标签，禁止经任何别名或衍生计算进入推理接口。
2. **Subject-level 划分**：同一受试者的全部记录只能属于一个 partition；锁定测试不得在候选比较中反复使用。
3. **折内拟合**：scaler、特征筛选、权重只能在训练折内 fit，冻结折内状态后应用于验证/测试；禁止全量预拟合。
4. **数据边界**：真实数据、检查点、工件不入 Git；synthetic fixture 必须显式标记；原始文件只读，导入前后哈希一致。
5. **标签分层**：RULE / PUBLISHED / LLM_SILVER 全链路溯源、不可互换；LLM_SILVER 永不作参考真值；规则缺项拒判（ABSTAIN），不得补零或猜标签。
6. **证据状态显式化**：未知就用 `NOT_AVAILABLE` / `UNVERIFIED` / `SMOKE_ONLY` / `NOT_VALIDATED` / `AGENT_NOT_VERIFIED` 表达，不得猜测或补全。
7. **三层放行独立判定**：`GO_ENGINEERING`、`GO_REAL_DATA_TRAINING`、`GO_RESEARCH_CLAIMS` 分别判定，不得由 smoke 或 dev/CV 分数推导后两层。
8. **AutoResearch 边界**：Agent 可改任意建模/训练/特征源码（`src/**/*.py`，冻结清单见 `research.guard.FROZEN_PATHS`）；数据、标签、split、评估器、指标、holdout 冻结。**Trusted Evaluator 必须在 Agent 不可写的隔离环境运行；不可信代码必须 OS/容器级隔离——仅审查 diff 不算隔离。**
9. **Provenance**：数据哈希、规则与标准版本、标签 tier、模型配置、种子、依赖、代码 commit 全部入档，可独立重算。
10. **Holdout 预算已用尽**：体质线 2/2、证候线 1/1 均已解锁。任何选择/比较轮次不得触碰 holdout；再评估必须换新数据里程碑或经用户明确授权。

## 三、AutoResearch 怎么用

Skill 驱动，Karpathy 形状：一份只读的冻结输入 + 一个可改的训练文件 + 固定墙钟预算 + 每轮 git KEEP/REVERT。

```bash
export PYTHONPATH=src
python -m research.cli prepare --task syndrome --out runs/R --repo . \
  --manifest <绝对路径>/selection_manifest.json --dataset <绝对路径>/selection_records.json \
  --folds 2 --repeats 1 --budget 60 --max-rounds 6 --seed 20261104
python -m research.cli state      --run runs/R    # 读上一轮结果：提案的唯一依据
# 写一个 p.json（见 skills/autoresearch/program.md），一次只提一个假设：
python -m research.cli step       --run runs/R --proposal-file p.json --proposer "<id>"
python -m research.cli dashboard  --run runs/R --port 8766    # 只读监控
```

- `--manifest/--dataset` 必须**绝对路径**：runner 在 git worktree 内执行，相对路径会解析到别处。
- 证候任务的 selection 输入由 SHA-256 钉住（`research.cli.SELECTION_DATASET_SHA256`），错配直接拒绝——旧中间产物与 #56/#57 冻结口径不可比。
- 每轮必须**先读当前文件再写 patch**：KEEP 后文件已变，复用旧 hunk 会被 `git apply` 拒绝并记 `INVALID_PROTOCOL`。
- 分数只来自可信 evaluator，Agent 不得自报。

## 四、项目知识

- **证候数据**：TCM-SD，148 类，以短主诉为主。同一主诉对应不同证型是**数据属性，不是标注错误**（如「突发晕厥1次」→ class 1/43）；训练集内重复文本按「真实复诊」保留；折内对 dev 做单向精确文本去重。
- **#57 冻结结论**：在 protocol-v2 冻结输入上，字符 1–3 gram TF-IDF+LR（CV macro-F1 **0.3199**）胜过全部神经方案；AutoResearch 9 轮里最大的真实改进也是「换线性头」。**要再评估神经方案必须换信息量（新特征/新标签/新数据），不是继续在同一搜索空间调参。**
- **外部验证缺口**：TCM-Lung 无名称解码（映射覆盖 0/148）、TCMEval-SDT 可对齐覆盖率仅 18.9% 且多标签语义不匹配 → `GO_RESEARCH_CLAIMS` NOT_GRANTED。
- **验收门状态**：#35 CI / #46 端到端 / #36 签收仍 OPEN；OS/容器隔离的端到端验收为 `AGENT_ISOLATION_UNVERIFIED`。
- **数据许可**：TCM-SD 为 CC BY-NC-SA 4.0，仅限非商业；不得用于未授权商业部署。

## 五、提交与协作

- Conventional Commits，一个 purpose 一个提交；不使用 `update` / `fix` / `！` 类标题。
- 不提交：真实数据、`runs/`、`artifacts/`、检查点、日志。
- 改动保持最小端到端可验证：schema 改动带契约测试，adapter 带形状 / mask / 负例测试，bug 修复带回归测试。
- synthetic 结果只表述工程有效，不表述科研 / 临床有效。

## 词汇表

| 术语 | 含义 |
| --- | --- |
| Pre / During / Post | 按摩前 / 中 / 后测量阶段 |
| RULE / PUBLISHED / LLM_SILVER | 规则计分 / 作者已发布 / LLM 弱标签 |
| selection 输入 | 冻结的 train+dev 选择池（holdout 为空），AutoResearch 唯一可用的数据 |
| SMOKE_ONLY | 仅合成数据工程验证，无科学结论 |
| GO_* | 工程准入 / 真实数据训练准入 / 科研声明，三层独立判定 |
