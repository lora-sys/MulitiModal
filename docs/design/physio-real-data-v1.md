# 真实 94 字段 Session 导入与核对（#102）

**状态**：`VERIFIED_WITH_EXCEPTIONS` / `INSUFFICIENT_EVIDENCE` / `LOCAL_EXPERIMENTAL`。
本文冻结"真实导出 → 规范契约"的映射规则与判定口径，**不是**结果报告。
未宣称任何预测性能、疲劳改善、因果效应或临床有效性。

## 1. 为什么需要这一层

`src/physio` 已统一 `PhysioEventV1` / `PhysioSessionV1`（设备 / 厂商服务 / 离线导入共用），
但真实厂商导出是**长表**格式：一行一个设备记录，没有 phase、没有规范单位、
没有标签。`legacy_export` 的合成 94 列注册表无法表达它。
本模块 `src/physio/real_export.py` 冻结真实长表 → 规范事件的映射，并**从原始记录重建**
Pre/Post 端点与 `y_delta_*` 标签；参考训练集只用于核对，不作为标签来源。

原始导出私有，位于 `data/local/physio-legacy/`，**不入 Git**。

## 2. 冻结的模块映射

| `source_type` | 模块 | 端点对 | 说明 |
|---|---|---|---|
| `血压` | `bp` | 是 | `sys`/`dia`/`pulse` |
| `健康` | `health` | 是 | `hr`/`spo2`/`micro_hrv`/`rr`/`fatigue_index`/`vendor_score` |
| `体脂` | `body` | 否 | `weight_kg`/`bmi`，静态基线 |
| `舌面` | `tcm` | 否 | `constitution`/`health_index`，仅中医侧车，非模型特征 |
| `脑波` | `eeg` | 否 | `EXCLUDED_FROM_PRE`，非 Pre 特征 |
| `脑波逐点` | `eeg_points` | 否 | 点级流，**永不**作为特征或标签 |

`source_type` 不在表内 → `RealExportError`，拒绝猜测模块。

## 3. 端点规则（抽取集反推假设，非厂商规则）

> **警告**：本节常量是从给定 5 条 Session 的参考训练集**反推**出来的假设，
> 不是厂商提供的契约。它们在该抽取集上复现了全部参考值，但换一个测量时序不同的
> 导出就可能失效；血压护栏若被当作通用契约，会静默丢弃合法标签。
> 它们是常量以便测试与修改，**不是冻结规范**。

`ENDPOINT_RULE = "first_and_last_measurement_group_by_measured_at"`

1. 端点模块记录按 `measured_at` 排序；
2. Pre 组 = 距最早记录 `ENDPOINT_GROUP_WINDOW_S`(60 s) 内的记录，Post 组 = 距最晚记录 60 s 内的记录；
3. 组内每个指标取**按时间序第一个可用值**：`rr=0` 在快照记录中表示"无信号"
   （`respiration_rate` 零值策略 `ZERO_MISSING`），因此回落到同组的完成记录，
   而 `hr`/`spo2`/`micro_hrv`/`fatigue_index`/`vendor_score` 仍取快照记录的值；
4. Pre/Post 各合成一个 `PhysioEventV1`，每指标至多一个 `MeasurementV1`，
   `provenance.source_record` 为组锚记录，保证端点独立；
5. 时间戳为文件内裸本地时间：按 UTC 读取，**只用相对顺序与差值**，因此偏移不可能改变任何标签；
   裸时间戳仍登记为 `naive_timestamp` QC。

该规则由 5 条真实 Session 的参考训练集逐字段反推并验证（190 项 `match`、0 项值级 `mismatch`）。

## 4. 标签契约

| 规范指标 | 原始列 | 参考标签键 | 规模 |
|---|---|---|---|
| `systolic_bp` | `sys` | `y_delta_sbp` | 1 |
| `diastolic_bp` | `dia` | `y_delta_dbp` | 1 |
| `heart_rate` | `hr`（health 优先）/`pulse`（bp 交叉核对） | `y_delta_hr` | 1 |
| `spo2` | `spo2` | `y_delta_spo2` | 0.01（百分数→分数） |
| `microcirculation_index` | `micro_hrv` | `y_delta_micro` | 1 |
| `respiration_rate` | `rr` | `y_delta_rr`（**不在**参考契约内） | 1 |
| `fatigue_index` | `fatigue_index` | `y_delta_vendor_fatigue` | 1 |
| `vendor_health_index` | `vendor_score` | `y_delta_vendor_health` | 1 |

- `REFERENCE_LABEL_KEYS` 是 7 个键的固定集合，参考质量报告与训练集一致列出。
- `session_labels(...).available` 只回答"原始记录能产出什么"；
  **契约准入**视图由 `reconcile_reference_contract` 产生：参考契约排除、
  且未声明排除规则的标签一律 `ABSTAIN`（`规则缺项一律 ABSTAIN`），
  派生值保留供审计，只改准入。
- `y_pri` 恒为 `None` / 不可用，任何路径都不得训练它。
- 新增规范指标 `microcirculation_index`、`vendor_health_index` 均为**厂商代理分**，
  语义与既有的 `fatigue_index` / `stress_index` 同级，不是独立或临床标签。
- 血压 delta 护栏 `|ΔSBP|>30 mmHg` 或 `|ΔDBP|>20 mmHg` 排除血压标签；
  该阈值在给定 5 条 Session 上复现参考结论，**不是**临床规则。
- 跨模块心率差 > 5 bpm 登记 QC，永不修复。
- 折内标准化只统计**观测值**（`observed_mask`），与 `WindowScaler.fit` 一致；插补位置不进入统计量。

## 5. 身份与特征

### 5.1 目标与特征的近似自回归性质（必读）

`post_state` 目标是 Post 端点的厂商疲劳指数，而 Pre 特征里已经含同名指标的
Pre 端点值；`delta_vendor_fatigue` 是两个厂商分之差。因此该预实验的回归在
**厂商分上近似自回归**，完全不能说明"能否从独立生理量预测疲劳"。
任何由 `real_training` 产出的数字都不得读成此类证据。

- 1 个 `member_id_hash` 对应 5 个 `subject_key` → `SHARED_ACCOUNT_UNKNOWN`；
  任何受试者级泛化声明都被拒绝。
- `sex` 原始码 0/1 与规范 `sex` 指标的语义类别不匹配：原样保留用于核对，
  `coded_field_semantics_unverified`，**不派生任何特征**。
- Pre 特征契约 = 冻结的 `physio-input-pre-only-v0`：本抽取集实现
  `body_pre` 2/12（`weight_kg`、`bmi`）+ `physio_pre` 7/7 = **9/19**。
- 多 `脑波` 记录且无选择规则时**不输出任何脑波值**（不使用未声明的"取最后一条"启发式）。
- 体成分其余字段、年龄、身高不在本脱敏抽取集内，一律 `not_recoverable`。

## 6. 不可从抽取集还原的决策

| 项 | 状态 |
|---|---|
| `脑波` 记录选择规则（多记录时选哪条） | `NOT_RECOVERABLE`，需厂商字段字典 |
| `is_complete` 完成态语义 | `UNKNOWN`，仅按字面登记 |
| 参考 Session 4 排除 `y_delta_vendor_health` 的规则 | `OPEN`，两份参考文件一致排除但未说明原因 |
| 呼吸率补值 15.41 的来源样本集 | `NOT_RECOVERABLE`（超出本 5 条抽取集） |
| `age` / `height_cm` / 体成分其余列 | `NOT_RECOVERABLE`（抽取集未含） |

## 7. 判定口径

- `evaluate_regression` 是唯一状态权威；`min_samples=30` 未达即 `INSUFFICIENT_EVIDENCE`。
- `GO_REAL_DATA_TRAINING(physio)`、`GO_RESEARCH_CLAIMS(physio)` 仍为 `NOT_GRANTED`，
  不因本次接入放宽。
- 结论级别 `LOCAL_EXPERIMENTAL` / `AGENT_ISOLATION_UNVERIFIED` / `SMOKE_ONLY`。
