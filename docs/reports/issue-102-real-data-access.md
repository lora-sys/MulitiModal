# 真实 94 字段 Session 接入核对与数据不足下的工程预实验（#102）

**结论级别**：`LOCAL_EXPERIMENTAL` / `AGENT_ISOLATION_UNVERIFIED` / `SMOKE_ONLY`
**审计状态**：`VERIFIED_WITH_EXCEPTIONS`（0 项值级 `mismatch`，3 项 `ABSTAIN` 开放项）
**评估状态**：`INSUFFICIENT_EVIDENCE`

本文只报告**可复现的核对事实**与**工程路径是否跑通**。未宣称任何预测性能、
疲劳改善、因果效应或临床有效性。`GO_REAL_DATA_TRAINING(physio)` 与
`GO_RESEARCH_CLAIMS(physio)` 仍为 `NOT_GRANTED`，不因本次接入放宽。

## 1. 输入与隔离

| 文件 | 行数 | sha256 | 隔离 |
|---|---|---|---|
| `data/local/physio-legacy/raw_records_5sessions.csv` | 41 条记录 | `4200b467…b5f1f` | `data/local/`，不入 Git |
| `…/reference_train94_5sessions.csv` | 5 行 × 94 列 | `a1cee8a4…f8b9` | 同上 |
| `…/reference_qc_5sessions.csv` | 5 行 × 11 列 | `990947d3…1a690` | 同上 |
| `…/reprocessed_5sessions.csv` | 5 行 × 82 列 | `34686b28…b8544` | 同上（另一版中文表头明细，未纳入核对） |

5 条 Session、6 类 `source_type`、1 个 `member_id_hash` 对应 5 个 `subject_key`。

## 2. 字段映射核对（94/94 列全部有判定）

从 41 条长表原始记录重建 Pre/Post 端点、`y_delta_*` 标签与静态特征，
再与参考训练集逐列比对：

| 判定 | 数量 | 含义 |
|---|---|---|
| `match` | **190** | 重建值与参考值在 1e-6 相对容差内一致 |
| `explained` | **22** | 差异由已声明规则解释（参考补值 / 契约外可导出标签 / `ABSTAIN` 标签） |
| `not_recoverable` | **196** | 本脱敏抽取集不含该列，不假设、不补造 |
| `mismatch` | **0** | 无值级不一致 |

参考质量报告有 **8 项可比对交叉核对**：`健康完成态数`（含 `0(仅快照)` 的前导整数）、
`窗口`、`缺失字段` 四项（bodyfat / tongue / brain / rr）、`异常`、`label_masks`，
共 **40 项 `match` / 0 项 `mismatch`**。`qc_status` 属于训练集与质量报告之间的
**跨文档一致性核对**（用训练集的 `qc_status` 对照质量报告自身的 `缺失字段` 列），
不计入上述 8 项。
另有 5 项重建观察（共用账号身份、快照端点、点级流排除、跨模块心率、模块族覆盖）
参考报告没有对应列，记为 `observation`，**不计入一致数**。

**完全一致的字段**：`source_record_ids`（5/5 条 Session 的记录集合与原始记录完全相等，
无多无缺）、`sbp/dbp/hr/spo2/micro/rr/fatigue/vendor_score` 的 pre/post 端点、
`y_delta_*`、`gender`（原始 `sex` 码逐字符相等）、`constitution_names`、
`date`/`batch_date`、`protocol_version`/`pipeline_version`/`score_version`/`label_version`
常量、`y_pri` 恒空、`qc_status`、`label_masks`（5/5 条）。

**`y_delta_*` 重建结论**：审计对 35 个 `y_delta_*` 单元格（5 条 Session × 7 个契约键）
逐值比对，**35/35 `match`**，含 `ΔSBP = -7 / -17 / +3 / -52 / -1`、
`Δvendor_health = -2.50 / +6.76 / +2.93 / +0.74 / +2.02`。
规则（端点 = 首末测量组、组内按时间取第一个可用值、`spo2` 按 0.01 缩放）见
[docs/design/physio-real-data-v1.md](../design/physio-real-data-v1.md)。

## 3. 数据质量核对

| QC 码 | 命中 | 说明 |
|---|---|---|
| `snapshot_endpoint` | 10 | 所有 Pre/Post 端点取自 `is_complete=否` 记录；完成态语义未公开 |
| `hr_cross_module_mismatch` | 7 | 血压 `pulse` 与健康 `hr` 心率不一致（最大 18 bpm，Session 3 pre） |
| `missing_module` | 7 | 模块族缺失（`体脂`/`舌面`/`脑波`），在 Session 级登记而非只留到下游 |
| `shared_account_identity` | 5 | 1 个 member hash → 5 个 subject key；受试者级泛化声明被拒 |
| `invalid_value` | 6 | `rr=0` 在快照记录中表示"无信号"，该字段无可用值 |
| `naive_timestamp` | 5 | 文件为裸本地时间，按 UTC 读取，仅用相对顺序 |
| `coded_field_semantics_unverified` | 5 | `sex` 原始码 0/1 语义未验证，不派生特征 |
| `declared_unusable_raw_column` | 5 | `fatigue_level` 已声明但本契约无可用语义，明确排除而非静默丢弃 |
| `point_stream_excluded` | 3 | `脑波逐点` 为点级流，永不作为特征或标签 |
| `brain_record_selection_not_recoverable` | 2 | 多 `脑波` 记录时的选择规则无法还原；**不输出任何脑波值** |
| `bp_delta_anomaly` | 1 | Session 4 `ΔSBP=-52/ΔDBP=-18` 超出声明护栏，血压标签被排除 |

`呼吸率补均值 15.41` 的来源样本集超出本 5 条抽取集，登记 `not_recoverable`。

## 4. 训练准入与预实验

- **身份**：`SHARED_ACCOUNT_UNKNOWN`，`GO_REAL_DATA_TRAINING` 未放行，
  训练以 `SMOKE_ONLY` 证据级运行，`allow_untrusted_pipeline=True`。
- **标签**：`y_pri` 恒不可用；补值端点、同 record 冒充、间隔超限、零值语义均拒绝。
  契约准入视图对 3 个"原始记录能产出但参考契约排除"的标签判 `ABSTAIN`（见第 5 节）。
- **特征契约**：冻结的 `physio-input-pre-only-v0` 在本抽取集实现
  `body_pre` 2/12（`weight_kg`、`bmi`）+ `physio_pre` 7/7 = **9/19**；
  `respiration_rate` 3/5 条不可用，`weight_kg`/`bmi` 4/5 条不可用（无 `体脂` 记录）。
- **折内拟合**：中位数插补与标准化都只吃训练折；标准化**只统计观测值**
  （`observed_mask`），与 `WindowScaler.fit` 的约定一致，插补位置不进入统计量。
- **评估**：`evaluate_regression` 是唯一状态权威 → `INSUFFICIENT_EVIDENCE`
  （labeled samples 4 < required 30；第 3 折被拒绝，未进入汇总）。

留一受试者折（5 折，每折 4 训练 / 1 测试）：

| 目标 | 训练折均值基线 | 岭回归（sklearn `Ridge`） | realtime MLP（torch 2.15） |
|---|---|---|---|
| `post_state`（厂商疲劳指数/100） | MAE 0.08563 | 0.14851 | 44.78747（末损 1946–2001） |
| `delta_vendor_fatigue`（厂商分） | MAE 6.8750 | 3.50529 | 52.62228（末损 2457–2941） |

上表基于 **4/5 折**（`refused_folds = (3,)`）。第 3 折被**拒绝**而非降级运行：
该折的留出受试者是唯一有实测体成分的 Session，训练折从未观测 `weight_kg`/`bmi`，
`_scaled_row` 按 `WindowScaler.transform` 的同一策略拒绝伪造标准化，而不是把
`65.6 / 23.2` 当"已标准化"值放行。

上表**不排序、不可比**。每折只有 1 个测试行，三个基线之间不存在有意义的先后；
加粗最小值已移除。MLP 的末损与 MAE 只证明链路跑通（掩码损失、有限梯度、
checkpoint 保存/重载、每折独立目录、预测），不证明它更好或更差。

**这些数字是 pipeline regression，不是模型对比结论。** 每折只有 1 个测试行，
三个基线不可能被排序；MLP 的末损与 MAE 只证明链路跑通（掩码损失、有限梯度、
checkpoint 保存/重载、每折独立目录、预测），不证明它更好或更差。
要得到可比较的结论，需要受试者级样本量 ≥30 且身份可信。

**目标近似自回归**：`post_state` 是 Post 端点厂商疲劳指数，Pre 特征已含同名指标的
Pre 端点值；`delta_vendor_fatigue` 是两个厂商分之差。该回归在厂商分上近似自回归，
不能说明"能否从独立生理量预测疲劳"。

**MLP 的输出域与两个目标都不匹配（结构性错配，不只是"没训练好"）**：
`RealtimeFatigueModel` 的疲劳状态头被限制在 `[0, 100]`，而 `post_state` 在
`0.25–0.48` 附近、`delta_vendor_fatigue` 跨 `-18..0`。对 delta 目标而言，
每个非正标签都落在声明输出域之外，模型无法表达该目标域的大部分。
因此表内 MLP 的 MAE 与末损**是模型/目标结构性错配的结果**，
不得读成关于任一目标的证据。合成 fixture 上把缩放退化为恒等反而让 post_state
的 MLP MAE 从 43.23 降到 13.49——"更差的缩放看起来更好"，只因为该输出头倾向产生
大正值，这正说明这些数字没有模型意义。

**折是按 `subject_key` 分的，不是按已验证的生理受试者分的**：
导出只给出 1 个共用 `member_id_hash` 对应 5 个 `subject_key`，且没有身份字典，
这 5 个键可能对应少于 5 个真人。因此这是**留一 `subject_key` 折**，
`RealTrainingReport.subject_isolation = PHYSICAL_SUBJECT_ISOLATION_UNVERIFIED`。
若两个键属于同一人，折划分就会把该人泄漏到多个折里。

**设计矩阵里的 0 是掩码占位符，不是测量值**：折内插补要求训练折至少
`FoldImputer.min_observed=2` 个观测，否则不建中位数、该列保持无效，进入
`ridge`/MLP 设计矩阵的就是 0.0 占位符。按列细分（真实抽取集）：

| 列 | 有实测的 Session | 可插补的折 | 说明 |
|---|---|---|---|
| `weight_kg` | 1/5 | **0/5** | 留出那一条时训练折 0 观测，否则 1 观测，始终不足 2 |
| `bmi` | 1/5 | **0/5** | 同上 |
| `respiration_rate` | 2/5 | **3/5** | 留出两条有实测 rr 的 Session 时训练折只剩 1 观测，无法插补 |

`RealTrainingReport.masked_placeholder_features` 报的是**至少在一折里是占位符**
的列的并集（真实数据为三者全部），不携带折级粒度；MAE 无法区分占位零与真实零值。

## 5. 开放项（必须带回厂商字段字典）

1. **3 个标签判 `ABSTAIN`**：
   - Session 4 的 `y_delta_vendor_health`（`Δ=0.74`）：原始记录可产出，参考训练集与
     参考质量报告**一致**只列 4 个标签不含它；排除规则未说明。
   - Session 1、2 的 `y_delta_rr`：可导出（17→17、15→15）但不在参考标签契约内。
   按 `规则缺项一律 ABSTAIN` 一律不判准入，派生值保留供审计，不按猜测修改重建。
2. `脑波` 记录选择规则（多记录时选哪条）不可还原。
3. `is_complete` 完成态语义未公开。
4. 参考行 `response_hash` 的构造方式未声明 → 参考无法被端到端独立验证。
5. 呼吸率补值 15.41 的来源样本集、`age`/`height_cm`/体成分其余 11 列
   （`body_fat_pct`、`muscle_kg`、`protein_kg`、`bone_mineral_kg`、`skeletal_muscle_kg`、
   `water_pct`、`visceral_fat`、`impedance`、`body_bmr_kcal`、`body_age`、`body_score`）
   不在本脱敏抽取集内。

## 6. 自动化验收

| 检查 | 结果 |
|---|---|
| `pytest tests`（本地，torch 2.15 已装） | **829 passed / 9 skipped** |
| `pytest tests/physio` | 369 passed / 0 skipped |
| `ruff check src/physio tests/physio scripts/physio_real_data_check.py` | PASS |
| `run_acceptance()` | **16/16 PASS**（新增 `real_export_mapping`、`real_data_audit`、`real_training_pipeline` 三项 #102 检查）。报告另含独立字段 `real_data_mapping_status = MAPPING_NOT_VERIFIED`（合成 fixture 上为 10 项 blocker，真实导出上为 9 项——合成 fixture 多出的 1 项是 `SYNTH-C` 上复现真实 Session 4 情形的 `ABSTAIN`），**工程 PASS 不等于 94 字段映射已核实** |
| `scripts/physio_verify.py` | **OVERALL PASS** |
| `scripts/physio_real_data_check.py`（私有导出） | 工程 OVERALL PASS + `mapping status: MAPPING_NOT_VERIFIED (9 blockers)`，报告落盘 `runs/physio-real-data/`（`audit.json` 含 `mapping_status` / `mapping_blockers` / `mapping_verified` / `passed`） |

**两类结论必须分开读**：

| 结论 | 状态 | 含义 |
|---|---|---|
| 工程路径 | **PASS** | 导入→端点→标签→折内拟合→训练→checkpoint→预测全链路跑通，且与参考无矛盾 |
| 94 字段映射 | **NOT_VERIFIED** | 9 项 blocker：3 个 `ABSTAIN` 标签、5 个 `response_hash` 不可独立验证、1 条汇总项列出 50 个抽取集未含列 |

`mapping_blockers` 的 9 项里，`response_hash` 同时出现在 5 个逐 Session 条目和
50 列汇总项中——汇总是列级事实，逐 Session 条目是行级事实，不是重复计数。
50 列 = `EXTRACT_ABSENT_REFERENCE_COLUMNS` 的 48 列 + `brain_record_id`
（多 `脑波` 记录时选择规则不可还原）+ `response_hash`（构造方式未声明）。

新增测试 87 项（`test_real_export.py` 25、`test_real_data_audit.py` 26、
`test_real_training.py` 29、`test_real_training_torch.py` 7），
另有 `tests/physio/physio_reference_columns.py`（94 列期望顺序常量，提交在库内，
使列序断言在无私有数据的主机上也有效），
全部只跑 `tests/fixtures/physio-legacy-*-synthetic.csv`（合成 fixture，带 synthetic 标记）。
合成 fixture 覆盖：快照/完成健康记录对、`rr=0` 回落到完成记录、血压 delta 异常、
共用账号身份、静态画像碰撞、`脑波逐点`、单/多 `脑波` 记录、缺失模块族、
`fatigue_level` 无声明语义、未知 `source_type` 拒判、缺列拒判、空文件拒判、
`ABSTAIN` 调和（含复现真实 Session 4 情形的 `SYNTH-C`）、
折内不可插补的掩码占位符（`SYNTH-E` 无 `体脂` 记录）。
合成参考 fixture 为**完整 94 列**，列序与真实参考一致（有私有数据时逐列断言，
无私有数据时跳过而非默认通过）。

折内拟合与缩放的负例：
- 折内统计量必须与全集统计量不同；
- 插补位置不得进入统计量（期望中位数独立手算，不重建 imputer 后自我比较）；
- 折内拟合的岭回归必须与全集拟合结果不同；
- **端到端钉住**：报告的 pooled MAE 必须等于用显式缩放设计矩阵独立重算的值，
  且必须不等于用未缩放矩阵重算的值——单独调用 `_scaled_row` 的单测无法发现
  "调用点传了未缩放行"这类回归；
- 被拒绝的折必须带理由记录， predictions / targets 清空，MAE 全 `None`；
- 每折必须有自己的 checkpoint 目录。

## 7. 本地环境变更（需知悉）

为运行神经网络路径，在本机 `.venv` 安装了 `torch==2.15.0`（CPU wheel）。
`.venv/` 已被 gitignore，**未**写入任何 `requirements*.txt`，项目依赖集合不变。
原先 `torch` 缺失时有 18 个用例 skip，装后 9 个 skip（其余为缺私有工件的合理跳过）。

## 8. 未解除的边界（不随本次工作改变）

- 真实 94 字段完整导出与厂商字段字典：`BLOCKED_BY_DATA_OR_DEVICE`
- 独立配对疲劳标签：`INSUFFICIENT_EVIDENCE`；`GO_REAL_DATA_TRAINING(physio)` /
  `GO_RESEARCH_CLAIMS(physio)` 仍 `NOT_GRANTED`，推理固定 `NOT_AVAILABLE`
- 生理受试者隔离：`PHYSICAL_SUBJECT_ISOLATION_UNVERIFIED`（1 个 member hash 对应
  5 个 `subject_key`，无身份字典）
- 真实设备控制：`NOT_CONNECTED`
- 结论级别 `LOCAL_EXPERIMENTAL` / `AGENT_ISOLATION_UNVERIFIED` / `SMOKE_ONLY`

## 9. 评审

### 9.1 多 provider 评审团：19 次尝试，取得 8 份独立评审

用户要求"开个评审团，跟 claude code 好好讨论"。实际尝试 10 个 provider/模型组合：

| provider | 模型 | 结果 |
|---|---|---|
| codex | gpt-6.1-sol (reasoningEffort=high) | usage limit 已达上限（恢复时间 Nov 11 16:38） |
| claudeAgent | claude-opus-5-5 (effort=high) | fetch failed（早期两次）→ **后续已完成，取得评审报告** |
| opencode | opencode/claude-opus-5-5 (variant=high) | Insufficient account funds |
| antigravity | gemini-3.8-flash-high | fetch failed |
| antigravity | claude-opus-4-6-thinking | Antigravity quota for this period 已达上限 |
| pi | magpie/auto/glm-5.3 | 429 rate_limit_error：OpenCode Go usage limit exceeded（恢复时间 18:45 / 22:26 UTC+8） |
| pi | magpie/auto/gpt-6.1-sol | 401 authentication_error |
| pi | default | **已完成，取得评审报告** |

失败原因是账号额度/配额/限流/传输错误，**不是代码或环境问题**；重开评审团需要等对应
provider 的额度恢复。

**共取得 8 份来自 2 个不同 provider（`pi`、`claudeAgent`）的独立评审，仍未满足三个不同
provider 的 council 要求**，因此 council 不记为"通过"。

- **第 1 份**（`pi` / default，task `2031a6d0-…`）：**8 / 10**，结论
  "sound engineering work with honest, reproducible verification; no P0"。
  处置见 9.2。
- **第 2 份**（`claudeAgent` / claude-opus-5-5，task `ee042c4c-…`）：**6.5 / 10**，
  审 `36ea482`。结论原文：*"The importer is unusually disciplined about provenance, masks,
unknowns, and NOT_RECOVERABLE outcomes, and the report avoids clinical claims.
But the pre-experiment's fold-internal scaler is never actually applied; the
module frames the resulting three-model MAE table as a comparison; and acceptance
passes despite both an unresolved reference mismatch and a large number of
unrecoverable fields. Those issues make the claimed 工程验证 less trustworthy
than it first appears."*

评审者明确说明只审 `36ea482`，未把 `090a64b`/`dfba7f0` 的后续修复计入。
9.3 的表格逐项区分**当时已修**与**本轮新修**。

### 9.2 第 1 轮评审（pi / default，8/10）发现的问题及修复情况

| ID | 严重度 | 问题 | 处置 |
|---|---|---|---|
| P1-1 | P1 | 标签准入夸大参考契约：Session 4 的 `y_delta_vendor_health` 被判可准入，而参考契约排除它且无规则说明；违反 `规则缺项一律 ABSTAIN` | **已修复**：新增 `reconcile_reference_contract`，派生视图与契约准入视图分离；3 个标签判 `ABSTAIN`，派生值保留供审计。合成 fixture 增加 `SYNTH-C` 复现该情形 |
| P2-1 | P2 | 每折 checkpoint 落到同一个 cwd 相对路径 `runs/physio-smoke/model_state.pt`，后折重载会读到前折权重 | **已修复**：`_fold_checkpoint_dir` 保证每折独立目录（默认路径也按折分目录），并加测试 |
| P2-2 | P2 | `eeg_record_id`/`eeg_values` 用未声明的"取最后一条"启发式填充，同时又称选择规则不可还原 | **已修复**：多 `脑波` 记录时**不输出任何脑波值**；单记录时才输出 |
| P2-3 | P2 | `model MAE: train_fold_mean=… ridge=… mlp=…` 这一行 stdout 容易被读成模型排名 | **已修复**：改为 `pipeline MAE … <- pipeline regression only, NOT a model comparison`，并显式打印"model ranking: not reported"；`real_training` 模块 docstring 同步改写 |
| P2-4 | P2 | 缺 `体脂` 模块只在 Session 级静默，直到下游逐列 `not_recoverable` | **已修复**：`missing_module` 覆盖全部 5 个模块族，在 Session 级登记 |
| P3-1 | P3 | `fatigue_level` 被静默丢弃 | **已修复**：登记为 `declared_unusable_raw_column`；同时新增 `unmapped_raw_column` 机制，任何未声明映射的原始列都会被发现 |
| P3-2 | P3 | 文档数字错误（148→193、10/19、测试数、通过数） | **已修复**：全部更正为实测值 |
| P3-3 | P3 | 同 P3-2 | 同上 |
| P3-4 | P3 | "both reference documents exclude" 硬编码假设了两个参考文件互相一致 | **已修复**：改为由 `abstained_labels` 判定动态生成的方向中性表述 |
| P3-5 | P3 | `_static_profile_key` 按标称特征表投影，`0.0` 与"真实测得 0"混淆，暗示不存在的覆盖 | **已修复**：改为只对实际持有的静态值投影 |
| — | P2 | 16 个 acceptance 门的 `except Exception` 把瞬时环境错误吞成检查失败，难以诊断（评审者观察到一次无法复现的失败） | **已修复**：`_failure_detail` 输出限定名 + 最内层 traceback 位置；`run_acceptance` 的 `repo_root` 已确认与 cwd 无关 |

### 9.3 第二轮评审（Claude Opus）逐项处置

| ID | 严重度 | 问题 | 状态 |
|---|---|---|---|
| C1 | **P1** | `_scaled_row()` 收到拟合好的 `WindowScaler` 却原样返回值，只把无效位置换成 0。岭回归与 MLP 实际在**未标准化**的原始值上训练/预测，所谓"折内标准化"是空操作，报告的 MAE 全部失真 | **本轮新修**：`_scaled_row` 现在按 `(v - mean) / std` 变换；新增 3 个回归测试（缩放必须作用于值、变换不得是恒等、统计量必须来自训练折）。修复后 MAE 从 ridge 0.1049→0.1320、MLP 18.97→50.42，已在报告更正 |
| C2 | **P1** | 三模型 MAE 表被框定为对比 | **当时已修**（`dfba7f0` 起脚本与模块 docstring 已改为 "pipeline regression only, NOT a model comparison"）；**本轮补强**：报告表格去掉加粗最小值，新增"目标近似自回归"与"0 是掩码占位符"两段显式声明 |
| C3 | **P1** | 即使存在未解 mismatch 与大量 not_recoverable，acceptance 仍显示 PASS / "mapping verified" | **本轮新修**：审计新增独立的 `mapping_status` / `mapping_blockers`；`run_acceptance()` 顶层新增 `real_data_mapping_status` 字段；`real_data_audit` 检查的 detail 明写"the 94-field mapping itself is MAPPING_NOT_VERIFIED"；报告 §6 拆成"工程路径 PASS"与"94 字段映射 NOT_VERIFIED"两行 |
| C4 | **P1** | `_dense()` 把无效特征静默换成 0.0，而 MAE 与报告都无法区分真实零值与掩码占位符 | **本轮新修**：`RealTrainingReport.masked_placeholder_features` 显式列出折内插补后仍无效的列（真实数据为 `weight_kg`、`bmi`、`respiration_rate`；其中前两者在 0/5 折可插补，`respiration_rate` 在 3/5 折可插补），报告 §4 按列给出可插补折数 |
| C5 | P2 | `_fit_scaler` 复制 `WindowScaler.fit`；岭回归在已有 sklearn 的情况下手写 | 岭回归**当时已修**（`090a64b` 改用 `sklearn.linear_model.Ridge`）；`_fit_scaler` **本轮新修**：改为 6 行适配器委托给 `WindowScaler.fit`，全库只剩一套零方差/掩码策略，并加测试证明委托而非复制 |
| C6 | P2 | 端点规则与 BP 阈值是从 5 条 Session 反推的，却被称为"冻结"，且护栏可能静默丢弃未来合法标签 | **本轮新修**：代码常量注释改为"EXTRACT-FITTED HYPOTHESIS, NOT A VENDOR RULE / NOT A CLINICAL RULE"；设计文档第 3 节加警告框；报告同步 |
| C7 | P2 | `post_state` 目标近似自回归（Pre 特征已含同名厂商分） | **本轮新修**：`real_training` 模块 docstring 第 2 条 + 设计文档 §5.1 + 报告 §4 三处显式声明 |
| C8 | P2 | 测试不断言缩放已生效；`final_loss == final_loss` 对 `inf` 也通过；一个测试重建 imputer 后与自身比较（恒真） | **本轮新修**：改用 `math.isfinite`；imputer 测试改为独立手算期望中位数；新增 3 个缩放测试 + 1 个掩码占位符机制测试 + 1 个委托测试 |
| C9 | P2 | 合成参考 fixture 只有 53 列而非 94，"every reference column accounted for" 只证明覆盖了缩减后的 fixture；缺 mismatch / response_hash / not_recoverable 分支 | **本轮新修**：合成参考 fixture 扩到**完整 94 列**（列序与真实一致），新增 `test_synthetic_fixture_carries_the_full_94_column_shape`，以及 `mapping_blockers` / `response_hash` 分支测试 |
| C10 | P3 | `real_export.py` ~950 行做了太多事 | **接受为已知维护性风险**，本轮未拆分。理由：拆分是纯重构、无功能收益，且 C1/C5 已消除它造成的实际危害（缩放统计被复制）；按奥卡姆原则不增加无需求的结构。已在设计文档记录该风险 |
| C11 | P3 | 报告 §6 "All green"紧邻§8 "BLOCKED_BY_DATA_OR_DEVICE"，未解 mismatch 与 PASS 并列 | **本轮新修**：§6 拆成两张表（工程路径 / 映射状态），不再用单一 "All green" |

### 9.3b 第三轮复核（`57c37f6`）：5 份评审

| 轮次 | provider / 模型 | 评分 | 审的提交 |
|---|---|---|---|
| 3a | `pi` / default | 8.5/10 | `57c37f6` |
| 3b | `antigravity` / gemini-3.8-flash-high | 9.6/10 | `57c37f6` |
| 3c | `antigravity` / gemini-3.8-flash-high | 9.0/10 | `57c37f6` |
| 3d | `antigravity` / gemini-3.8-flash-high | 9.2/10 | `57c37f6` |
| 3e | `claudeAgent` / claude-opus-5-5 | no score | `57c37f6`（被循环检测停止，未发出评审） |

`claudeAgent` 在第三轮连续三次因传输错误 `fetch failed` 未返回，一次因 Agent 循环调用
同一命令被平台循环检测停止；其余 4 次重试中 3 次取得 Gemini 评审、1 次取得 `pi` 评审。

#### 第三轮发现的问题及处置

| ID | 严重度 | 问题 | 处置 |
|---|---|---|---|
| R3-F1 | **P2** | `_scaled_row` 比 `WindowScaler.transform` 宽松：当 `observed_counts[i] == 0` 时共享策略**拒绝**（"refusing to fabricate a normalization"），而 `_scaled_row` 计算 `(value - 0.0)/1.0`，把原始未标准化值当"已标准化"值放行。真实数据第 3 折实测泄漏 `weight_kg=65.6`、`bmi=23.2` | **已修**：`_scaled_row` 在该位置改为抛 `RealTrainingError`，与共享策略一致；调用方把该折记为**已拒绝**（带理由、predictions/targets 清空、MAE 全 `None`）而非降级运行。真实数据因此为 **4/5 折**（`refused_folds=(3,)`），MAE 已更正 |
| R3-F2 | P2 | `RealDataAudit.to_dict()` 不含 `mapping_status`/`mapping_blockers`/`mapping_verified`/`passed`，持久化的 `audit.json` 读起来像无限定的一致通过 | **已修**：四个字段全部进入 `to_dict()` |
| R3-F3 | P2 | 报告 §6 "51 列"错误，实际 50 列（48 + `brain_record_id` + `response_hash`），且 `response_hash` 在 5 个逐 Session 条目与 1 条汇总项中重复出现 | **已修**：改为 50 列并给出构成；说明汇总项是列级事实、逐 Session 条目是行级事实，不是重复计数 |
| R3-F4 | P2 | 报告 §2 "38 个标签值"错误，实际审计 35 个 `y_delta_*` 单元格（5×7） | **已修**：改为 35/35 `match` |
| R3-F5 | P2 | 报告 §2 "7 项可比对交叉核对"错误，实际 8 项，且 `qc_status` 不属于这 8 项 | **已修**：改为 8 项并说明 `qc_status` 是跨文档一致性核对 |
| R3-F6 | P2 | C1 修复未被端到端钉住——把调用点改回 `scaled = imputed` 时全部测试仍通过 | **已修**：新增 `test_pooled_mae_uses_the_scaled_design_matrix`，断言 pooled MAE 等于显式缩放矩阵独立重算值、且不等于未缩放矩阵重算值 |
| R3-F7 | P2 | `neural_status` 汇总把被拒绝折的默认状态算进去，4 折完成却报 `torch_unavailable` | **已修**：`_neural_status` 只统计实际跑过的折；报告新增 `refused_folds` |
| R3-G1 | P3 | `test_realtime_model.py:62` 对 `requires_grad=True` 的张量做 `float()`，触发 PyTorch UserWarning | **已修**：加 `.detach()` |
| R3-G2 | P3 | `masked_placeholder_features` 的断言在合成 fixture 上空转（该 fixture 全部特征可插补） | **已修**：`SYNTH-E` 去掉 `体脂` 记录，使 `weight_kg`/`bmi` 只在 2/5 条被观测，折内确实不可插补；断言改为与独立计算比对 |
| R3-G3 | P3 | fixture 列序无断言 | **已修**：有私有参考时逐列断言列序，无私有参考时跳过而非默认通过 |
| R3-G4 | P3 | "体成分其余 12 列"错误，实际 11 列 | **已修** |
| R3-G5 | P3 | 合成 gate 报 10 项 blocker、真实 9 项，差异未说明 | **已修**：报告说明多出的 1 项是 `SYNTH-C` 复现真实 Session 4 的 `ABSTAIN` |
| R3-G6 | P3 | 心率容差缺"抽取集反推假设"标注 | **已修**：设计文档 §4 补齐 |
| R3-G7 | P3 | `masked_placeholder_features` 是跨折并集，无折级粒度 | **接受为已知限制**：并集已回答"哪些列在任一折里是占位符"；折级粒度需要改 `FoldResult` 结构，收益不足以抵偿重构风险 |
| R3-G8 | P3 | `OVERALL: PASS` 是最后一行，读者容易停在绿色上 | **已修**：脚本把 `mapping verified: MAPPING_NOT_VERIFIED` 移到最后一行 |
| R3-G9 | P3 | `real_export.py` ~1100 行 | **维持第三轮判断**：纯重构无功能收益，R3-F1/F2 已消除它造成的实际危害 |

#### 第四轮复核（`claudeAgent` / claude-opus-5-5，两份：8/10 与 7.5/10，均审 `57c37f6`）

| ID | 严重度 | 问题 | 处置 |
|---|---|---|---|
| R4-F4 | **P2** | MLP 输出域与两个目标都不匹配：`RealtimeFatigueModel` 的状态头限制在 `[0, 100]`，而 `post_state` 在 `0.25–0.48`、`delta_vendor_fatigue` 跨 `-18..0`；对 delta 目标每个非正标签都在声明输出域之外。MLP 的误差是**结构性错配**，不只是"没训练好"。评审者在合成 fixture 上把缩放退化为恒等，反而让 post_state 的 MLP MAE 从 43.23 降到 13.49——"更差的缩放看起来更好"，正说明这些数字没有模型意义 | **已修**：`real_training` 模块 docstring 增第 4 条显式声明；报告 §4 增加专段，含上述 43.23→13.49 的反直觉实证；结论级别明确"不得读成关于任一目标的证据" |
| R4-F5 | **P2** | "leave-one-subject-out" 未验证物理受试者隔离：1 个 `member_id_hash` 对应 5 个 `subject_key`，无身份字典，5 个键可能对应少于 5 个真人；若两键属同一人，折划分就把该人泄漏到多折 | **已修**：新增 `SUBJECT_ISOLATION_UNVERIFIED = "PHYSICAL_SUBJECT_ISOLATION_UNVERIFIED"` 并进入 `RealTrainingReport.subject_isolation` 与 `to_dict()`；脚本、报告 §4/§8、设计文档统一改称"留一 `subject_key` 折"；报告 §8 增列该边界 |
| R4-F6 | P3 | `real_export.py` 模块 docstring 与设计文档开头仍用 "freezes the mapping"，与 C6 的"抽取集反推假设"标注不完全一致 | **已修**：两处改为"**声明实现假设**，不是经厂商字段字典验证过的映射"，并说明没有字典时重建必然对 94 个字段施加语义解读，由两类 blocker 限定可信范围 |
| R4-F1 | P2 | 报告 `real_data_mapping_status` 同时出现 10 项（合成 fixture）与 9 项（私有导出）blocker，读者无法分辨各自输入 | **在 `4830f68` 已修**：报告明确标注合成 fixture 10 项 / 私有导出 9 项，并说明多出的 1 项是 `SYNTH-C` 复现真实 Session 4 的 `ABSTAIN` |
| R4-F2 | P2 | 报告的 MAE 表在 `57c37f6` 上看起来是 C1 修复前的旧值 | **在 `4830f68` 已修**：该提交引入第 3 折拒绝后表已改为 4/5 折的 `0.08563 / 0.14851 / 44.78747` 与 `6.8750 / 3.50529 / 52.62228` |
| R4-F3 | P2 | `to_dict()` 不含 `mapping_status`/`mapping_blockers`/`mapping_verified`，持久化 `audit.json` 读不出独立映射结论 | **在 `4830f68` 已修**：四字段（含 `passed`）全部进入 `to_dict()` |

评审者另外披露：其验证过程用默认输出目录跑了一次 `run_real_pre_experiment()`，
可能覆盖了 `runs/physio-real-data/fold-*/model_state.pt` 与 `manifest.json`。
这些是 gitignore 产物，`git status` 确认**无任何被跟踪文件改动**；已重新生成
`runs/` 产物。

#### 第五轮复核（`claudeAgent` / claude-opus-5-5，7.5/10，审 `57c37f6`）

该轮同样确认 C1–C9 在 `57c37f6` 上基本成立、无 P0，并用两个实验独立证明：
（a）改动留出行值**不影响**训练折的 scaler / imputer / ridge 拟合 / 折划分
（即 R4-1 是伪造标准化，不是统计泄漏）；（b）`_fit_scaler` 委托给
`WindowScaler.fit` 的均值、标准差、计数、最小标准差策略与掩码语义逐折一致。
它另外发现三项本轮新问题：

| ID | 严重度 | 问题 | 处置 |
|---|---|---|---|
| R5-1 | **P2** | 报告引用的 `reference_train94_5sessions.csv` sha256 后缀写作 `d8b9`，**实际是 `f8b9`** —— 属转写错误，使报告的溯源声明不可验证 | **已修**：更正为 `f8b9`；新增 `test_report_hash_fragments_match_the_private_inputs`，有私有数据时逐文件核对前 8 位与后 4 位，无则跳过 |
| R5-2 | P2 | 报告仍写 `labeled samples 5 < required 30`，而第 3 折被拒绝后实际为 **4** | **已修**：改为 4 并注明"第 3 折被拒绝，未进入汇总" |
| R5-3 | P2 | C4 的"全部折内观测数不足 2"以偏概全：`respiration_rate` 在 3/5 折内**可以**插补 | **已修**：报告 §4 改为按列给出"有实测的 Session 数 / 可插补的折数"表格（`weight_kg` 1/5→0/5、`bmi` 1/5→0/5、`respiration_rate` 2/5→3/5）；评审表条目同步更正 |
| R5-4 | P3 | 三列 `pipeline MAE` 行虽带"NOT a model comparison"标签，表格形态仍易被扫读成排名 | **已修**：改为每项附 `(4/5 folds)` 覆盖度，明确三者覆盖同一批折且不做排序 |
| R5-5 | P3 | `to_dict()` 缺映射字段、MAE 表旧值、"freezes the mapping" 措辞 | **`4830f68` / `8dd932d` 已修**（该评审针对 `57c37f6`） |

评审者未执行 pytest / ruff / acceptance，因此报告中的测试与验收总数在其审阅时点是
源码声明而非独立复跑结果；其本地环境缺 `sklearn`，仅用内存桩做折与缩放检查。

#### 第六轮复核（`claudeAgent` / claude-opus-5-5，7.5/10，审 `57c37f6`）

该轮的 P1/P2 发现（scaler 策略缺口、MLP 目标域、`to_dict()` 缺映射字段、
报告数字陈旧）**均已在 `4830f68` / `4e9dc16` / `8dd932d` 修复**，评审者本人也逐条注明了
对应后续提交。它真正的新贡献是列出 **5 个"回归可存活于测试套件"的缺口**——即早先缺陷
即使被修好，也没有测试能在它复发时失败。这 5 个缺口本轮全部补齐，并逐个用突变验证：

| ID | 缺口 | 新增测试 | 突变验证 |
|---|---|---|---|
| M1 | `to_dict()` 丢映射字段（R4-F3 复发无测试可捕） | `test_to_dict_carries_the_independent_mapping_verdict`：断言 8 个键存在、值与对象一致，且 `passed=True` 与 `mapping_verified=False` 可同时读出 | 删除 `to_dict()` 的 4 个映射字段 → **该测试失败** |
| M2 | 零观测特征未被拒绝（R4-F1 复发无测试可捕） | `test_a_refused_fold_never_reaches_the_models`：构造"两条训练行不观测特征 0、留出行观测"的真实形状，断言 `WindowScaler.transform` 与 `_scaled_row` **都**拒绝、且 `FoldImputer` 不伪造中位数 | 删除 `_scaled_row` 的拒绝守卫 → **该测试失败** |
| M3 | 双目标有限损失只断言 `is not None`（R4-F5） | `test_neural_loss_is_finite_for_both_target_modes` 改为逐折 `math.isfinite`，失败信息带目标与折号 | — |
| M4 | 94 列顺序用 `set` 断言，重排/重复列可通过（R4-F6） | 新增**提交在库内**的 `tests/physio/physio_reference_columns.py`（94 列期望顺序），`test_synthetic_fixture_column_order_matches_the_committed_expectation` 直接比对 tuple；有私有数据时再用私有参考核对常量本身 | 交换 fixture 前两列 → **两个顺序测试都失败**；临时移走私有数据 → 顺序测试仍运行（不跳过） |
| M5 | MLP 输出域与目标域不匹配无测试钉住（R4-F2 复发会使报告声明变陈旧） | `test_declared_targets_are_outside_the_model_output_scale`：钉住目标侧——`post_state` 全部落在 `[0,100]` 内但 `max < 1.0`，`delta_vendor_fatigue` 全部为负即完全在声明输出域外 | — |

该轮另确认（独立复算，非读代码）：改动留出行值不影响训练折的 scaler / imputer /
ridge 系数 / 折指派；`_fit_scaler` 委托的均值、标准差、计数、最小标准差策略与掩码语义
逐折一致；190/22/196/0 字段判定、40/0/25 QC 判定、3 个开放项、9 项 blocker、
41 条记录、6 类 source_type、1→5 的 member/subject 关系、`ΔSBP`/`Δvendor_health` 示例
全部复现正确。

该轮评审者披露：一次诊断调用用了默认输出目录，写入了 gitignore 的
`runs/physio-real-data/`；被跟踪工作树保持干净，未再重跑该命令。

### 9.4 六轮独立评审确认正确的部分

第一轮（`pi`）：独立重算 5 条真实 Session 的 `y_delta` 值与参考完全一致；
未找到另一条同样简洁且能复现的规则；折内拟合经 observed counts 验证无泄漏；
新测试含负例与篡改→FAILED 路径。

第二轮（Claude Opus）：provenance、掩码、未知项与 `NOT_RECOVERABLE` 结果的处理
"unusually disciplined"，报告避免了临床声明。

第三轮（`pi` 8.5 + 三份 Gemini 9.6/9.0/9.2）：一致确认 C1–C9 全部修复到位，
第四、五、六轮（Claude 8/10 与两份 7.5/10）独立复核同样确认九项修复成立、无 P0，
并用"改动留出行值"实验证明折内拟合无泄漏、用"缩放退化为恒等"实验证明
C1 修复可被 mutation 检测；
受试者级折边界、折内插补、折作用域 checkpoint 均保持；
三份 Gemini 均独立重跑真实数据并确认报告数字与代码一致（在其审阅的时点）；
`pi` 确认无 P0/P1，剩余问题集中在持久化工件与文档数字。

### 9.5 仍需第三方复核的事项

- **两轮评审来自两个不同 provider（`pi`、`claudeAgent`），仍未满足三个不同 provider 的
  council 要求**，因此 council 仍不记为通过。其余 provider 的失败原因是账号额度 /
  配额 / 限流 / 传输错误，与代码无关。
- 两轮评审均无法验证 `fatigue_level`、`脑波` 记录选择规则、`is_complete` 语义
  （需厂商字段字典）。
- 第三轮 `claudeAgent` 评审未实际发出（循环检测 + 传输错误），该 provider 的独立意见
  在本轮缺失。
- 第四、五、六轮评审者均**未执行** pytest / ruff / acceptance（第六轮只跑聚焦测试与
  `run_acceptance()`），因此报告中的测试与验收总数在其审阅时点是源码与产物声明，
  不是独立复跑结果；第五轮评审者本地缺 `sklearn`。
- 五轮评审均无法验证 `fatigue_level`、`脑波` 记录选择规则、`is_complete` 语义、
  以及 5 个 `subject_key` 是否为 5 个生理受试者（需厂商字段字典）。
