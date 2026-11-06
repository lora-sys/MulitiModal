# 真实用户证候模型外部验证协议（任务 03 / 任务 04 前置）

状态：**协议已制定，尚未执行**。任务 04（独立泛化测试）当前为 `BLOCKED`：
还没有新的独立受试者数据，也没有独立中医师参考标注。合成症状或模型生成的标签
不得替代专家参考。

配套文件：

- 数据字典：[real-user-syndrome-data-dictionary.md](real-user-syndrome-data-dictionary.md)
- 专家标注规范：[real-user-syndrome-annotation-spec.md](real-user-syndrome-annotation-spec.md)
- 冻结与评估脚本：`src/syndrome/external_validation.py`
- 纯合成流程测试：`tests/syndrome/test_external_validation.py`
- 任务 02 依据：[../reports/2026-11-07-syndrome-ability-audit.md](../reports/2026-11-07-syndrome-ability-audit.md)

证据等级：本文件及所有演练结果均为 `SMOKE_ONLY` / `LOCAL_EXPERIMENTAL`；
`GO_RESEARCH_CLAIMS=NOT_GRANTED`。

## 1. 目标与不做什么

目标：将来在**未参与训练、模型选择、阈值选定的新受试者**上，用普通人日常语言描述的
症状，评估冻结证候模型在冻结 148 类标签上的 Top-1、Top-3、Macro-F1、逐类 Recall
与拒判率，并如实报告每个类别的有效样本量与置信区间。

明确不做：

- 不从「疲劳 / 睡眠问题 / 肩颈不适」等场景直接映射为任何证候；
- 不用中医术语引导用户填写，不让采集表出现证候名或诊断字段；
- 不用模型输出、LLM 伪标签或规则标签充当专家参考；
- 不在测试后调阈值、重训、换词表或改评估代码后仍宣称「独立测试」；
- 不解锁历史 holdout，不把旧 `test.json` 当外部数据。

## 2. 目标人群、招募与入排标准

**目标人群**：自愿参加研究的成年人，能用自己的话描述近期身体感受。按招募场景分层，
优先但不限于：

| 场景标签 | 用途 |
| --- | --- |
| `fatigue` | 近期容易疲倦、乏力相关的日常主诉 |
| `sleep_problem` | 入睡困难、易醒、睡不踏实等 |
| `shoulder_neck_discomfort` | 肩颈发紧、酸痛、活动不适等 |
| `other` | 其他日常不适，避免把样本限制在三类场景 |

场景标签**只用于招募分层和后续分层报告**，不是标签、不是特征、不得参与任何证候推断。

**纳入**：签署知情同意；提供非空、非中医术语模板化的自由文本症状；年龄 ≥18 岁；
同一受试者只采集一次主记录；`subject_id` 为研究伪名。

**排除**：未同意或撤回同意；文本为空、纯复制模板、含可识别个人身份信息且无法脱敏；
重复或近重复文本；文本直接给出诊断/证候结论；同一受试者重复入组；专家标注
`UNDETERMINABLE`、未解决 `DISAGREEMENT`、未标注或参考来源不合格（这些单独计数，
不静默丢弃，见 §6）。

**隐私与知情同意**：

- 研究目的、数据用途、保存期限、退出与删除权利、非诊断性质、CC BY-NC-SA 4.0
  非商业科研限制，均在同意书中写明；同意记录与症状数据分开保存。
- 采集表不收集姓名、电话、身份证号、住址、病历号等直接标识；`subject_id` 为
  随机伪名，映射表由数据管理员单独保管。
- 自由文本可能含隐含个人信息，入库前人工脱敏；分析只使用脱敏后的文本。
- 模型输出仅作研究估计，不向受试者反馈诊断或治疗建议；不用于临床决策。
- 数据访问限定为研究组；对外只发布聚合指标，不发布可复原的文本。

## 3. 采集表与字段

采集表只问日常语言，例如「最近哪里不舒服、持续多久、什么情况下加重、有没有影响
睡觉或吃饭」，不出现证候名、脏腑名、舌脉术语。字段定义与允许值见
[data dictionary](real-user-syndrome-data-dictionary.md)，核心字段：

`subject_id`、`symptom_text`、`recruitment_stratum`、`consent`、`form_version`、
`collected_at`、`expert_status`、`expert_labels`、`primary_label`、
`annotator_count`、`reference_source`。

任何记录若包含 `syndrome` / `syndrome_label` / `diagnosis` 字段，评估脚本直接拒收，
以强制「不得把症状直接映射为证候」。

## 4. 专家标注

见 [annotation spec](real-user-syndrome-annotation-spec.md)。要点：两名独立中医师
在**屏蔽模型输出**的条件下标注，允许多个候选证候、允许「不能确定」、允许并记录
分歧；分歧由第三名资深专家裁定，未解决的分歧单独计数。参考来源只能是
`TCM_PROFESSIONAL`；`RULE`、`PUBLISHED`、`LLM_SILVER` 与模型输出均不得作为参考真值。

## 5. 冻结清单（执行前必须完成）

冻结对象与值：

| 对象 | 冻结值 |
| --- | --- |
| 模型工件 | `artifacts/sd-v2-ar1-champion/model.joblib` `944b325d…e74f`；`labels.json` `65d797c4…285d`；`registry_entry.json` `a73830d8…246f` |
| 标签词表 | `data/local/tcm-sd/syndrome_vocab.txt` `63ee1087…5c83`，148 类、顺序固定 |
| 拒判阈值 | `0.27808433274565425`（交付工件选定值，外部数据上未验证） |
| 评估代码 | 执行 `freeze` 时写入 `evaluation_code_sha256`；执行 `evaluate` 时自校验 |
| 数据划分 | 全部同意入组的外部受试者构成**一个**评估集；train/dev/holdout 均为 0 |
| 外部数据集 | `freeze --dataset <records.json>` 把数据文件 sha256 写入清单 |

命令（生产机 `acam-lab`，`PYTHONPATH=src`）：

```bash
# 1) 执行前冻结：工件、词表、阈值、评估代码、数据与划分
python -m syndrome.external_validation freeze \
  --dataset runs/validation/external-records.json \
  --out runs/validation/freeze-manifest.json

# 2) 仅推理：用冻结工件产出分数、Top-1/Top-3 与状态；不训练、不调阈值
python -m syndrome.external_validation predict \
  --freeze runs/validation/freeze-manifest.json \
  --records runs/validation/external-records.json \
  --dataset runs/validation/external-records.json \
  --out runs/validation/predictions.json

# 3) 冻结评估：以独立专家标注为参考
python -m syndrome.external_validation evaluate \
  --freeze runs/validation/freeze-manifest.json \
  --records runs/validation/external-records.json \
  --predictions runs/validation/predictions.json \
  --dataset runs/validation/external-records.json \
  --out runs/validation/evaluation.json

# 4) 无真实患者数据的流程演练（SMOKE_ONLY）
python -m syndrome.external_validation smoke-test \
  --out runs/validation/smoke.json
```

`verify_freeze()` 会拒绝任何工件、词表、阈值、模型 id、评估代码或数据集漂移。
**测试后任何调参、重训、改阈值、换评估代码都会使独立性失效**，必须重新冻结、
重新执行，并在报告中显式声明。

## 6. 指标定义与排除统计

| 指标 | 定义 |
| --- | --- |
| Top-1 准确率 | 冻结预测 Top-1 标签 == 裁定后的 `primary_label` 的样本占比 |
| Top-3 命中率 | `primary_label` 落在冻结预测 Top-3 标签集合内的样本占比 |
| Macro-F1 | 全部 148 类逐类 `F1 = 2TP/(2TP+FP+FN)` 的算术平均；缺席类记 0，仍进入分母 |
| 逐类 Recall | `TP/(TP+FN)`，附有效样本量（dev 中该类样本数）与 Wilson 95% CI |
| 拒判率 | 评估集内模型状态非 `OK` 的样本占比；决策覆盖率 = 1 − 拒判率 |
| 有效样本量 | 每类进入主指标的真实样本数；`< 30` 标 `LOW_SAMPLE_INSUFFICIENT` |

**单独计数、不进入主指标**：无同意、专家分歧（`DISAGREEMENT`）、不能确定
（`UNDETERMINABLE`）、未标注、参考来源不合格、无预测。`n_records` 必须等于
`n_evaluated + n_excluded_total`；任何静默丢弃样本都视为协议违规。

`use_conditions_met` 需同时满足：冻结清单已验证、评估样本数达到预设下限、
受试者独立、无测试后调参、参考标注全部来自中医师。任一不满足时结果状态为
`RESEARCH_USE_ONLY`，不得称为独立泛化测试。

## 7. 样本量理由（依据任务 02 实测分布）

任务 02 在 2844 个 dev 样本上得到：148 类中逐类 support 中位数 **3**，
仅 **19** 类 ≥30、**7** 类 ≥100，**4** 类为 0；最大类 `气虚血瘀证` 543 样本
（占 19.09%）。按这一实际分布线性外推，在**新受试者人群构成相近**的假设下：

| 计划样本量 N | 预计能达到 ≥30 有效样本的类别数 |
| --- | --- |
| 100 | 0 |
| 200 | 1（仅 `气虚血瘀证`，约 38 人） |
| 500 | 3 |
| 1000 | 8 |

- 要让**当前最稀有但已观测到**的类别（1/2844）也拿到 ≥30 个样本，需要约
  **85,320** 名受试者；要让第 20 常见的类别拿到 ≥100 个样本，需要约
  **11,850** 名。
- 即使每类有 30 个样本，Recall≈0.5 的 Wilson 95% CI 仍约为 **[0.33, 0.67]**；
  100 个样本时才收窄到约 **[0.40, 0.60]**。因此 30 只是「可报告」下限，
  不是「可下定论」下限。

**结论**：100–200 人可以作为**初始可行性试验**规模，用来跑通采集、知情同意、
双标注一致性、冻结与评估流程，并只能对最高频的 1 个类别形成粗略区间；
它**不足以可靠验证全部 148 类**，不能支撑「模型对哪些证候有效」的总体结论。
建议分阶段：阶段一 100–200 人只做流程与一致性验证（不发布疗效/能力结论）；
阶段二 ≥1000 人覆盖约 8 个高频类别；阶段三对任务 02 标出的薄弱类别做
前瞻性分层富集，而不是靠总体样本量硬堆。

上述外推基于「新人群类别分布与 dev 相近」这一未经验证的假设，真实分布若不同，
所需样本量会变化；协议要求在执行时按实际入组分布重新报告有效样本量。

## 8. 专家标注成本估算（规划假设，非实测）

按每例 2 名独立中医师、每人 8 分钟阅读与标注；分歧按 20–30% 计，由 2 名资深专家
每人 10 分钟裁定：

| 规模 | 双标注 | 裁定 | 合计专家工时 |
| --- | --- | --- | --- |
| 100 人 | 200 次 × 8 min ≈ 26.7 h | 20–30 例 × 2 × 10 min ≈ 6.7–10 h | 约 33–37 h |
| 200 人 | 400 次 × 8 min ≈ 53.3 h | 40–60 例 × 2 × 10 min ≈ 13.3–20 h | 约 67–73 h |

按 **假设** 的 ¥300–500/专家小时预算，100 人约 ¥1.0–1.9 万，200 人约
¥2.0–3.7 万；另有招募、知情同意、数据脱敏与质控人力未计入。费率、阅读时长与
分歧率都是预算假设，执行前须用阶段一的实测值替换，不得当作已发生成本或研究结论。

## 9. 任务 04 启动条件与交付边界

任务 04 只有在以下条件全部满足时才启动：

1. 存在训练/模型选择/阈值选定期间**完全未使用**的新受试者数据；
2. 存在符合 [annotation spec](real-user-syndrome-annotation-spec.md) 的独立中医师
   参考标注（含分歧、不能确定、未标注的显式记录）；
3. 数据与规则在推理前已按 §5 冻结并通过 `verify_freeze()`；
4. 评估按 §6 执行，报告逐类有效样本量与 CI、单独列出排除样本；
5. 无测试后调参；若发现预定义使用条件未满足，如实保留 `RESEARCH_USE_ONLY` 状态。

在此之前，任何「独立泛化能力」「临床有效性」「148 类可靠/不可靠」的表述都
不成立；体质与证候冠军保持冻结，不产生没有新证据支持的模型升级声明。
