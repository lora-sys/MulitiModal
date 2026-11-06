# 真实用户验证数据字典

适用范围：[real-user-syndrome-protocol.md](real-user-syndrome-protocol.md) 的采集、
冻结、推理与评估数据。**当前尚无真实患者数据**；本文件只定义将来采集的字段，
所有演练数据一律 `SMOKE_ONLY`。

## 1. `external-records.json`（采集 + 专家标注）

顶层为记录数组，每条记录代表**一名受试者的一次采集**。`subject_id` 必须唯一。

| 字段 | 类型 | 必填 | 允许值 / 约束 | 来源与说明 |
| --- | --- | --- | --- | --- |
| `subject_id` | string | 是 | 非空、唯一、研究伪名，如 `sub-0001` | 数据管理员生成；不得含姓名、电话、身份证等可识别信息 |
| `symptom_text` | string | 是 | 非空自由文本；日常语言描述 | 受试者自填/访谈记录；入库前人工脱敏；**不得含证候名、诊断结论或中医术语引导** |
| `recruitment_stratum` | string | 是 | `fatigue` / `sleep_problem` / `shoulder_neck_discomfort` / `other` | 仅用于招募分层与分层报告；**不得映射到任何证候** |
| `consent` | boolean | 是 | `true` / `false` | 同意书回执；`false` 单独计数并从主指标排除 |
| `form_version` | string | 是 | 如 `symptom-form-v1` | 采集表版本，冻结后不得中途改版 |
| `collected_at` | string | 否 | ISO-8601 日期 | 采集日期，仅用于审计时间线 |
| `expert_status` | string | 是 | `DETERMINED` / `UNDETERMINABLE` / `DISAGREEMENT` | 专家标注状态，见 [annotation spec](real-user-syndrome-annotation-spec.md) |
| `expert_labels` | array[string] | 是 | 冻结 148 类词表中的名称，可为空数组 | 两名专家各自给出的候选证候，可多个；不自动合并、不自动转单标签 |
| `primary_label` | string | `DETERMINED` 时必填 | 冻结 148 类词表中的**单个**名称 | 裁定后的参考标签，用于 Top-1 / Macro-F1；其他状态可为空或省略 |
| `annotator_count` | integer | 是 | ≥2 | 独立标注专家人数；`DETERMINED` 时脚本要求 ≥2 |
| `reference_source` | string | 是 | 仅 `TCM_PROFESSIONAL` | 参考标注来源；`MODEL` / `LLM_SILVER` / `RULE` / `PUBLISHED` 一律拒收 |
| `adjudication_id` | string | 否 | 非空 | 分歧裁定记录编号，便于回溯 |
| `notes` | string | 否 | 自由文本 | 标注说明，不含可识别个人信息 |

**禁止字段**：`syndrome`、`syndrome_label`、`diagnosis`。出现任一字段，评估脚本直接
抛错，以保证采集阶段不把症状直接映射为证候。

**排除语义**（每条记录只归入一类）：

| 情形 | 计数字段 | 是否进入主指标 |
| --- | --- | --- |
| `consent != true` | `n_no_consent` | 否 |
| `expert_status = UNDETERMINABLE` | `n_undeterminable` | 否 |
| `expert_status = DISAGREEMENT` | `n_disagreement` | 否 |
| `DETERMINED` 但 `primary_label` 缺失/不在词表 | `n_unlabeled` | 否 |
| `reference_source` 不合格 | `n_invalid_reference` | 否 |
| 有记录但无对应预测 | `n_missing_prediction` | 否 |

恒等式：`n_records = n_evaluated + n_excluded_total`。

## 2. `freeze-manifest.json`（执行前冻结）

由 `python -m syndrome.external_validation freeze` 生成，
schema 为 `tcm-sd-external-validation-v1`。

| 字段 | 说明 |
| --- | --- |
| `frozen.model_id` | `syndrome-tfidf-charwb-1-4-c10-ar1` |
| `frozen.taxonomy_version` | `tcm-sd-syndrome-vocab-v1` |
| `frozen.threshold` | `0.27808433274565425` |
| `frozen.n_classes` | `148` |
| `frozen.artifact_sha256` | `model.joblib` / `labels.json` / `registry_entry.json` / `syndrome_vocab.txt` 的 sha256 |
| `frozen.evaluation_code_sha256` | 冻结时 `src/syndrome/external_validation.py` 的 sha256；评估时自校验 |
| `frozen.dataset_sha256` | `--dataset` 指向的 records 文件 sha256；未提供时为 `NOT_AVAILABLE`，此时不得执行带数据的评估 |
| `split.policy` | `ALL_CONSENTED_EXTERNAL_SUBJECTS_FORM_ONE_EVALUATION_SET` |
| `split.train` / `dev` / `holdout` | 均为 0 |
| `split.rules` | 受试者唯一、无训练/调参、排除样本单独计数、测试后调参即失效 |
| `status.evidence` | `EXTERNAL_VALIDATION_NOT_STARTED` |
| `status.go_research_claims` | `NOT_GRANTED` |

## 3. `predictions.json`（冻结推理输出）

以 `subject_id` 为键的对象，值为：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `scores` | array[float]，长度 148 | 冻结模型输出的未校准分数，**不是概率** |
| `top1_index` | integer 0–147 | 分数最高类别的冻结词表索引；平分按索引升序 |
| `top3_indices` | array[integer]，长度 3 | 分数最高的 3 个类别索引 |
| `max_score` | float | 最高分数 |
| `status` | `OK` / `UNCERTAIN` / `ABSTAIN` / `OUT_OF_DOMAIN` | `max_score >= 冻结阈值` 为 `OK`，否则 `UNCERTAIN`；输入异常时为 ABSTAIN / OUT_OF_DOMAIN |

该文件只由 `predict` 子命令生成；不得人工编辑，不得用模型生成的标签回填参考标注。

## 4. `evaluation.json`（冻结评估报告）

schema 同为 `tcm-sd-external-validation-v1`。主要字段：

| 字段 | 说明 |
| --- | --- |
| `status.evidence` | `SMOKE_ONLY`（演练）或 `LOCAL_EXPERIMENTAL` |
| `status.external_validation` | `NOT_PERFORMED` / `RESEARCH_USE_ONLY` / `INDEPENDENT_TEST` |
| `status.use_conditions_met` | 冻结已验证、样本量达标、受试者独立、无测试后调参、参考来源合格，全部为真才为 `true` |
| `status.go_research_claims` | 恒为 `NOT_GRANTED` |
| `conditions` | 上述逐项布尔值 |
| `counts` | `n_records`、各排除计数、`n_evaluated`、状态分布 |
| `metrics` | `top1_accuracy`、`top3_accuracy`、`macro_f1`、`rejection_rate`、`decision_coverage`、`n_classes_with_effective_samples`、`n_classes_meeting_minimum` |
| `metric_definitions` | 各指标的计算口径 |
| `per_class[]` | 148 行：索引、名称、`effective_sample_size`、`tp`/`fp`/`fn`、`recall`、`recall_ci95`（Wilson）、`top3_hits`、`top3_hit_rate`、`rejected`、`rejection_rate`、`status_counts`、`evidence_status`、`meets_minimum` |
| `unverified_generalization` | 使用条件未满足时不得声称独立泛化测试等边界声明 |

`evidence_status`：`NO_SAMPLES`（0 样本）、`LOW_SAMPLE_INSUFFICIENT`（<30）、
`MEASURABLE`（≥30）；`meets_minimum` 只对 ≥30 为真。

## 5. 数据血统与许可

- 参考标注是**新采集的独立专家标注**，不复用 `RULE` / `PUBLISHED` /
  `LLM_SILVER` 等既有血统；不同血统不可互换，`LLM_SILVER` 绝不作为参考真值。
- 模型与标签词表来自 TCM-SD（CC BY-NC-SA 4.0，仅非商业科研）；新采集数据的
  使用范围以知情同意书为限。
- 原始自由文本、同意记录、伪名映射表、冻结清单、预测与评估报告分开存放；
  对外只发布聚合指标，不发布可复原的个体文本。
- `runs/`、`artifacts/`、真实数据与二进制工件均不提交 Git；演练产物标注
  `SMOKE_ONLY`，真实数据到位后另行标注 `LOCAL_EXPERIMENTAL`。
