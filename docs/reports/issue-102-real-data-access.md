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
| `…/reference_train94_5sessions.csv` | 5 行 × 94 列 | `a1cee8a4…d8b9` | 同上 |
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

参考质量报告有 7 项可比对交叉核对：`健康完成态数`（含 `0(仅快照)` 的前导整数）、
`窗口`、`缺失字段`（bodyfat / tongue / brain / rr 四项）、`异常`、`label_masks`、
`qc_status`（与参考自身的 `缺失字段` 列对照），共 **40 项 `match` / 0 项 `mismatch`**。
另有 5 项重建观察（共用账号身份、快照端点、点级流排除、跨模块心率、模块族覆盖）
参考报告没有对应列，记为 `observation`，**不计入一致数**。

**完全一致的字段**：`source_record_ids`（5/5 条 Session 的记录集合与原始记录完全相等，
无多无缺）、`sbp/dbp/hr/spo2/micro/rr/fatigue/vendor_score` 的 pre/post 端点、
`y_delta_*`、`gender`（原始 `sex` 码逐字符相等）、`constitution_names`、
`date`/`batch_date`、`protocol_version`/`pipeline_version`/`score_version`/`label_version`
常量、`y_pri` 恒空、`qc_status`、`label_masks`（5/5 条）。

**`y_delta_*` 重建结论**：5 条 Session 共 38 个标签值，全部与参考一致，含
`ΔSBP = -7 / -17 / +3 / -52 / -1`、`Δvendor_health = -2.50 / +6.76 / +2.93 / +0.74 / +2.02`。
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
  （labeled samples 5 < required 30）。

留一受试者折（5 折，每折 4 训练 / 1 测试）：

| 目标 | 训练折均值基线 | 岭回归（sklearn `Ridge`） | realtime MLP（torch 2.15） |
|---|---|---|---|
| `post_state`（厂商疲劳指数/100） | MAE 0.0790 | 0.1049 | 18.9675（末损 459–625） |
| `delta_vendor_fatigue`（厂商分） | MAE 5.8000 | 2.5362 | 26.3544（末损 853–1197） |

**这些数字是 pipeline regression，不是模型对比结论。** 每折只有 1 个测试行，
三个基线不可能被排序；MLP 的末损与 MAE 只证明链路跑通（掩码损失、有限梯度、
checkpoint 保存/重载、每折独立目录、预测），不证明它更好或更差。
要得到可比较的结论，需要受试者级样本量 ≥30 且身份可信。

## 5. 开放项（必须带回厂商字段字典）

1. **3 个标签判 `ABSTAIN`**：
   - Session 4 的 `y_delta_vendor_health`（`Δ=0.74`）：原始记录可产出，参考训练集与
     参考质量报告**一致**只列 4 个标签不含它；排除规则未说明。
   - Session 1、2 的 `y_delta_rr`：可导出（17→17、15→15）但不在参考标签契约内。
   按 `规则缺项一律 ABSTAIN` 一律不判准入，派生值保留供审计，不按猜测修改重建。
2. `脑波` 记录选择规则（多记录时选哪条）不可还原。
3. `is_complete` 完成态语义未公开。
4. 参考行 `response_hash` 的构造方式未声明 → 参考无法被端到端独立验证。
5. 呼吸率补值 15.41 的来源样本集、`age`/`height_cm`/体成分其余 12 列
   不在本脱敏抽取集内。

## 6. 自动化验收

| 检查 | 结果 |
|---|---|
| `pytest tests`（本地，torch 2.15 已装） | **808 passed / 9 skipped** |
| `pytest tests/physio` | 348 passed / 0 skipped |
| `ruff check src/physio tests/physio scripts/physio_real_data_check.py` | PASS |
| `run_acceptance()` | **16/16 PASS**（新增 `real_export_mapping`、`real_data_audit`、`real_training_pipeline` 三项 #102 检查） |
| `scripts/physio_verify.py` | **OVERALL PASS** |
| `scripts/physio_real_data_check.py`（私有导出） | OVERALL PASS，报告落盘 `runs/physio-real-data/` |

新增测试 66 项（`test_real_export.py` 25、`test_real_data_audit.py` 17、
`test_real_training.py` 18、`test_real_training_torch.py` 6），
全部只跑 `tests/fixtures/physio-legacy-*-synthetic.csv`（合成 fixture，带 synthetic 标记）。
合成 fixture 覆盖：快照/完成健康记录对、`rr=0` 回落到完成记录、血压 delta 异常、
共用账号身份、静态画像碰撞、`脑波逐点`、单/多 `脑波` 记录、缺失模块族、
`fatigue_level` 无声明语义、未知 `source_type` 拒判、缺列拒判、空文件拒判、
`ABSTAIN` 调和（含复现真实 Session 4 情形的 `SYNTH-C`）。
折内拟合有专门负例：折内统计量必须与全集统计量不同、插补位置不得进入统计量、
折内拟合的岭回归必须与全集拟合结果不同、每折必须有自己的 checkpoint 目录。

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

### 9.1 多 provider 评审团：部分失败，取得 1 份独立评审

用户要求"开个评审团，跟 claude code 好好讨论"。实际尝试 8 个 provider/模型组合：

| provider | 模型 | 结果 |
|---|---|---|
| codex | gpt-6.1-sol (reasoningEffort=high) | usage limit 已达上限 |
| claudeAgent | claude-opus-5-5 (effort=high) | fetch failed（两次） |
| opencode | opencode/claude-opus-5-5 (variant=high) | Insufficient account funds |
| antigravity | gemini-3.8-flash-high | fetch failed |
| antigravity | claude-opus-4-6-thinking | quota 已达上限 |
| pi | magpie/auto/glm-5.3 | fetch failed |
| pi | magpie/auto/gpt-6.1-sol | 401 authentication_error |
| pi | default | **已完成，取得评审报告** |

**未取得三个不同 provider 的独立意见**，因此不按 council 规则记为"通过"。
已取得的 1 份评审（`pi` / default，thread `d49f1e07…`，task
`2031a6d0-6c0d-4aa4-9a26-91452e624530`）给出 **8 / 10**，结论：
"sound engineering work with honest, reproducible verification; no P0"。

### 9.2 独立评审发现的问题及修复情况

| ID | 严重度 | 问题 | 处置 |
|---|---|---|---|
| P1-1 | P1 | 标签准入夸大参考契约：Session 4 的 `y_delta_vendor_health` 被判可准入，而参考契约排除它且无规则说明；违反 `规则缺项一律 ABSTAIN` | **已修复**：新增 `reconcile_reference_contract`，派生视图与契约准入视图分离；3 个标签判 `ABSTAIN`，派生值保留供审计。合成 fixture 增加 `SYNTH-C` 复现该情形 |
| P2-1 | P2 | 每折 checkpoint 落到同一个 cwd 相对路径 `runs/physio-smoke/model_state.pt`，后折重载会读到前折权重 | **已修复**：`_fold_checkpoint_dir` 保证每折独立目录（默认路径也按折分目录），并加测试 |
| P2-2 | P2 | `eeg_record_id`/`eeg_values` 用未声明的"取最后一条"启发式填充，同时又称选择规则不可还原 | **已修复**：多 `脑波` 记录时**不输出任何脑波值**；单记录时才输出 |
| P2-3 | P2 | `model MAE: train_fold_mean=… ridge=… mlp=…` 这一行 stdout 容易被读成模型排名 | **已修复**：改为 `pipeline MAE … <- pipeline regression only, NOT a model comparison`，并显式打印"model ranking: not reported"；`real_training` 模块 docstring 同步改写 |
| P2-4 | P2 | 缺 `体脂` 模块只在 Session 级静默，直到下游逐列 `not_recoverable` | **已修复**：`missing_module` 覆盖全部 5 个模块族，在 Session 级登记 |
| P3-1 | P3 | `fatigue_level` 被静默丢弃 | **已修复**：登记为 `declared_unusable_raw_column`；同时新增 `unmapped_raw_column` 机制，任何未声明映射的原始列都会被发现 |
| P3-2 | P3 | 文档数字错误（148→193、10/19、测试数、通过数） | **已修复**：全部更正为实测值（190 match、9/19、66 项新测试、808 passed、348 passed/0 skipped） |
| P3-3 | P3 | 同 P3-2 | 同上 |
| P3-4 | P3 | "both reference documents exclude" 硬编码假设了两个参考文件互相一致 | **已修复**：改为由 `abstained_labels` 判定动态生成的方向中性表述 |
| P3-5 | P3 | `_static_profile_key` 按标称特征表投影，`0.0` 与"真实测得 0"混淆，暗示不存在的覆盖 | **已修复**：改为只对实际持有的静态值投影 |
| — | P2 | 16 个 acceptance 门的 `except Exception` 把瞬时环境错误吞成检查失败，难以诊断（评审者观察到一次无法复现的失败） | **已修复**：`_failure_detail` 输出限定名 + 最内层 traceback 位置；`run_acceptance` 的 `repo_root` 已确认与 cwd 无关 |

### 9.3 独立评审确认正确的部分

评审者独立重算并确认：5 条真实 Session 的 38 个 `y_delta` 值与参考完全一致；
未找到另一条同样简洁且能复现的规则；折内拟合经 observed counts 验证无泄漏；
59 个新测试含负例与篡改→FAILED 路径；状态权威得到尊重。

### 9.4 仍需第三方复核的事项

- 单 provider 评审不构成 council 通过。provider 恢复后需重开三个不同 provider 的评审团。
- 评审者无法验证 `fatigue_level` 是否承载可还原语义（需厂商字段字典）。
- 评审者无法验证 `脑波` 记录选择规则与 `is_complete` 语义（同上）。
