# 真实 94 字段 Session 接入核对与数据不足下的工程预实验（#102）

**结论级别**：`LOCAL_EXPERIMENTAL` / `AGENT_ISOLATION_UNVERIFIED` / `SMOKE_ONLY`
**审计状态**：`VERIFIED_WITH_EXCEPTIONS`
**评估状态**：`INSUFFICIENT_EVIDENCE`

本文只报告**可复现的核对事实**与**工程路径是否跑通**。未宣称任何预测性能、
疲劳改善、因果效应或临床有效性。`GO_REAL_DATA_TRAINING(physio)` 与
`GO_RESEARCH_CLAIMS(physio)` 仍为 `NOT_GRANTED`，不因本次接入放宽。

## 1. 输入与隔离

| 文件 | 行数 | sha256 | 隔离 |
|---|---|---|---|
| `data/local/physio-legacy/raw_records_5sessions.csv` | 41 条记录 | `4200b467…b5f1f` | `data/local/`，不入 Git |
| `…/reference_train94_5sessions.csv` | 5 行 × 94 列 | `a1cee8a4…d8b9` | 同上 |
| `…/reference_qc_5sessions.csv` | 5 行 × 11 列 | `990947d3…1a690` | 同上 |
| `…/reprocessed_5sessions.csv` | 5 行 × 82 列 | `34686b28…b8544` | 同上（另一版中文表头明细，未纳入核对） |

5 条 Session、6 类 `source_type`、1 个 `member_id_hash` 对应 5 个 `subject_key`。

## 2. 字段映射核对（94/94 列全部有判定）

从 41 条长表原始记录重建 Pre/Post 端点、`y_delta_*` 标签与静态特征，
再与参考训练集逐列比对：

| 判定 | 数量 | 含义 |
|---|---|---|
| `match` | **193** | 重建值与参考值在 1e-6 相对容差内一致 |
| `explained` | **19** | 差异由已声明规则解释（参考补值 / 契约外可导出标签） |
| `not_recoverable` | **195** | 本脱敏抽取集不含该列，不假设、不补造 |
| `mismatch` | **1** | 见第 5 节开放项 |

**完全一致的字段**（举例）：`source_record_ids`（5/5 条 Session 的记录集合与原始记录
完全相等，无多无缺）、`sbp/dbp/hr/spo2/micro/rr/fatigue/vendor_score` 的 pre/post 端点、
7 个 `y_delta_*` 值、`gender`（原始 `sex` 码逐字符相等）、`constitution_names`、
`date`/`batch_date`、`protocol_version`/`pipeline_version`/`score_version`/`label_version`
常量、`y_pri` 恒空、`qc_status`、`label_masks`（4/5 条）。

**`y_delta_*` 重建结论**：5 条 Session 共 38 个标签值，全部与参考一致，含
`ΔSBP = -7 / -17 / +3 / -52 / -1`、`Δvendor_health = -2.50 / +6.76 / +2.93 / +0.74 / +2.02`。
规则（端点 = 首末测量组、组内按时间取第一个可用值、`spo2` 按 0.01 缩放）
见 [docs/design/physio-real-data-v1.md](../design/physio-real-data-v1.md)。

## 3. 数据质量核对

| QC 码 | 命中 | 说明 |
|---|---|---|
| `shared_account_identity` | 5 | 1 个 member hash → 5 个 subject key；受试者级泛化声明被拒 |
| `snapshot_endpoint` | 10 | 所有 Pre/Post 端点取自 `is_complete=否` 记录；完成态语义未公开 |
| `invalid_value` | 6 | `rr=0` 在快照记录中表示"无信号"，该字段无可用值 |
| `hr_cross_module_mismatch` | 7 | 血压 `pulse` 与健康 `hr` 心率不一致（最大 18 bpm，Session 3 pre） |
| `point_stream_excluded` | 3 | `脑波逐点` 为点级流，永不作为特征或标签 |
| `brain_record_selection_not_recoverable` | 2 | 多 `脑波` 记录时的选择规则无法还原 |
| `coded_field_semantics_unverified` | 5 | `sex` 原始码 0/1 语义未验证，不派生特征 |
| `bp_delta_anomaly` | 1 | Session 4 `ΔSBP=-52/ΔDBP=-18` 超出声明护栏，血压标签被排除 |
| `naive_timestamp` | 5 | 文件为裸本地时间，按 UTC 读取，仅用相对顺序 |

参考质量报告有 8 项可比对交叉核对：`健康完成态数`（含 `0(仅快照)` 的前导整数）、
`窗口`、`缺失字段`（bodyfat/tongue/brain/rr 四项）、`异常`、`label_masks`，
其中 **39 项 `match` / 1 项 `mismatch`**（即第 5 节开放项）。另有 4 项重建观察
（共用账号身份、快照端点、点级流排除、跨模块心率）参考报告没有对应列，
记为 `observation`，**不计入一致数**。
`呼吸率补均值 15.41` 的来源样本集超出本 5 条抽取集，登记 `not_recoverable`。

## 4. 训练准入与预实验

- **身份**：`SHARED_ACCOUNT_UNKNOWN`，`GO_REAL_DATA_TRAINING` 未放行，
  训练以 `SMOKE_ONLY` 证据级运行，`allow_untrusted_pipeline=True`。
- **标签**：`y_pri` 恒不可用；补值端点、同 record 冒充、间隔超限、零值语义均拒绝。
- **特征契约**：冻结的 `physio-input-pre-only-v0` 在本抽取集实现
  `body_pre` 2/12 + `physio_pre` 7/7 = **10/19**；
  `respiration_rate` 3/5 条不可用，`weight_kg`/`bmi` 4/5 条不可用（无 `体脂` 记录）。
- **评估**：`evaluate_regression` 是唯一状态权威 → `INSUFFICIENT_EVIDENCE`
  （labeled samples 5 < required 30）。

留一受试者折（5 折，每折 4 训练 / 1 测试），折内中位数插补 + 折内标准化：

| 目标 | 训练折均值基线 | 岭回归（sklearn `Ridge`） | realtime MLP（torch 2.15） |
|---|---|---|---|
| `post_state`（厂商疲劳指数/100） | MAE **0.0790** | 0.1049 | 18.9675（末损 459–625） |
| `delta_vendor_fatigue`（厂商分） | MAE **5.8000** | 2.5362 | 26.3544（末损 853–1197） |

**这些数字不是模型对比结论。** 5 个样本下任何排序都不可泛化；MLP 的末损与 MAE
只证明链路跑通（掩码损失、有限梯度、checkpoint 保存/重载、预测），
不证明它更好或更差。要得到可比较的结论，需要受试者级样本量 ≥30 且身份可信。

## 5. 开放项（必须带回厂商字段字典）

1. **Session 4 的 `label_masks`**：重建得到 5 个可准入标签
   （`hr, micro, spo2, vendor_fatigue, vendor_health`），参考训练集与参考质量报告
   **一致**只列 4 个（不含 `y_delta_vendor_health`，其 `Δ=0.74`）。
   排除规则未在质量报告中说明，无法还原 → 记为 `mismatch` 开放项，
   不按猜测修改重建。
2. `脑波` 记录选择规则（多记录时选哪条）不可还原。
3. `is_complete` 完成态语义未公开。
4. 参考行 `response_hash` 的构造方式未声明 → 参考无法被端到端独立验证。
5. 呼吸率补值 15.41 的来源样本集、`age`/`height_cm`/体成分其余 12 列
   不在本脱敏抽取集内。

## 6. 自动化验收

| 检查 | 结果 |
|---|---|
| `pytest tests`（本地，torch 2.15 已装） | **795 passed / 9 skipped** |
| `pytest tests/physio` | 335 passed / 2 skipped |
| `ruff check src/physio tests/physio scripts/physio_real_data_check.py` | PASS |
| `run_acceptance()` | **16/16 PASS**（新增 `real_export_mapping`、`real_data_audit`、`real_training_pipeline` 三项 #102 检查） |
| `scripts/physio_verify.py` | **OVERALL PASS** |
| `scripts/physio_real_data_check.py`（私有导出） | OVERALL PASS，报告落盘 `runs/physio-real-data/` |

新增测试 61 项（`test_real_export.py` 20、`test_real_data_audit.py` 15、
`test_real_training.py` 16、`test_real_training_torch.py` 5、acceptance 3 项扩展），
全部只跑 `tests/fixtures/physio-legacy-*-synthetic.csv`（合成 fixture，带 synthetic 标记）。
新增合成 fixture 覆盖：快照/完成健康记录对、`rr=0` 回落到完成记录、血压 delta 异常、
共用账号身份、静态画像碰撞、`脑波逐点`、多 `脑波` 记录、缺失模块、
未知 `source_type` 拒判、缺列拒判、空文件拒判。

## 7. 本地环境变更（需知悉）

为运行神经网络路径，在本机 `.venv` 安装了 `torch==2.15.0`（CPU wheel）。
`.venv/` 已被 gitignore，**未**写入任何 `requirements*.txt`，项目依赖集合不变。
原先 `torch` 缺失时有 18 个用例 skip，装后 9 个 skip（其余为缺私有工件的合理跳过）。

## 8. 未解除的边界（不随本次工作改变）

- 真实 94 字段完整导出与厂商字段字典：`BLOCKED_BY_DATA_OR_DEVICE`
- 独立配对疲劳标签：`INSUFFICIENT_EVIDENCE`；`GO_REAL_DATA_TRAINING(physio)` /
  `GO_RESEARCH_CLAIMS(physio)` 仍 `NOT_GRANTED`，推理固定 `NOT_AVAILABLE`
- 真实设备控制：`NOT_CONNECTED`
- 结论级别 `LOCAL_EXPERIMENTAL` / `AGENT_ISOLATION_UNVERIFIED` / `SMOKE_ONLY`

## 9. 评审

### 9.1 多 provider 评审团未能召开

用户要求"开个评审团，跟 claude code 好好讨论"。实际尝试了 5 个 provider / 6 个模型，
**全部失败**，因此没有取得任何第三方独立评审意见，也没有虚构评分：

| provider | 模型 | 结果 |
|---|---|---|
| codex | gpt-6.1-sol (reasoningEffort=high) | usage limit 已达上限 |
| claudeAgent | claude-opus-5-5 (effort=high) | fetch failed（两次） |
| opencode | opencode/claude-opus-5-5 (variant=high) | Insufficient account funds |
| antigravity | gemini-3.8-flash-high | fetch failed |
| pi | magpie/auto/glm-5.3 / default | fetch failed |

按 council 规则，"无法访问指定 provider、评审无回复或任务失败都不算通过"，
因此本次评审结论为**未通过**，`no score`。需要 provider 恢复后重开评审团。

### 9.2 主 Agent 自审发现并已修复的问题

评审团不可用，改为对本次 diff 做逐项自审，发现并修复 4 个真实缺陷：

| ID | 严重度 | 问题 | 修复 |
|---|---|---|---|
| S1 | P1 | `_fit_scaler` 用 `valid_mask` 统计，插补值会进入折内均值/方差，违背 `WindowScaler.fit` "未观测项不进入统计量"的既定约定 | 改用 `observed_mask`（`FoldImputer.transform` 保留该掩码），并加 3 个负例测试 |
| S2 | P1 | 12 项 QC 交叉核对中有 5 项硬编码 `VERDICT_MATCH`，实际不做任何比较，使"交叉核对全部一致"的结论失真 | 新增 `observation` 判定；只有参考报告有对应列的才算 `match`/`mismatch`；`health_completion_count` 改为真实比较前导整数（含 `0(仅快照)`） |
| S3 | P2 | 岭回归基线手写中心化 + `lstsq`，而 sklearn 已是 `src/` 级既有依赖（`src/research/train.py`、`src/syndrome/baselines.py`） | 改用 `sklearn.linear_model.Ridge`，删除手写线性代数 |
| S4 | P2 | 报告把 4 项无参考对应的观察写成"一致"，夸大核对范围 | 报告改为"8 项可比对：39 match / 1 mismatch；4 项观察不计入一致数" |

### 9.3 自审确认为正确的部分

- 端点规则（首末测量组 + 组内第一个可用值）在 5 条真实 Session 上复现全部 38 个 `y_delta` 值；
  未找到另一条同样简洁且能复现的规则。`spo2` 0.01 缩放、`y_delta_hr` 归 health 模块、
  `y_delta_rr` 因两端点 `rr=0` 被拒，均有参考证据支撑。
- 无数据泄漏：`FoldImputer.fit` 与 `_fit_scaler` 都只吃训练折；`admit_delta_targets` 要求
  Pre/Post 来自不同 record；参考行只用于比对，不作为特征或标签来源。
- 无伪造：不可用值一律 `value=None` + `valid=False` 并带拒绝原因；
  未知 `source_type`、缺列、空文件直接拒判。
- 报告无性能/疗效/临床表述；MLP 的 MAE 明确标注为链路跑通证据而非模型对比结论。

### 9.4 自审无法验证、需带回的事项

与第 5 节开放项相同（Session 4 的 `label_masks` 排除规则、`脑波` 记录选择规则、
`is_complete` 语义、`response_hash` 构造、呼吸率补值来源样本集、抽取集缺失的 12 列）。
