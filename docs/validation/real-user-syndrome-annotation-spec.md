# 独立中医师参考标注规范

适用于 [real-user-syndrome-protocol.md](real-user-syndrome-protocol.md) 的
任务 03 / 04。**当前无任何真实患者标注数据**；所有流程演练使用 `SMOKE_ONLY`
合成记录，不构成参考真值。

## 1. 角色与盲法

| 角色 | 人数 | 职责 |
| --- | --- | --- |
| 标注医师 A / B | 每例 2 名 | 独立阅读脱敏后的日常语言症状文本，从冻结 148 类词表中给出候选证候 |
| 资深裁定医师 C | 1 名 | 仅在 A/B 不一致时介入，给出裁定意见 |
| 数据管理员 | 1 名 | 脱敏、伪名、同意记录、数据版本与冻结清单管理，不参与标注 |

盲法要求：

- A、B 互相看不到对方标注，也**看不到模型输出、分数、Top-3 候选、阈值状态**；
- C 只看 A/B 的标注与原始症状文本，不看模型输出；
- 任何人都不得用模型输出、LLM 生成标签、规则标签或历史 `LLM_SILVER` 作为参考；
- 标注顺序与模型推理解耦：标注完成后才冻结并运行 `predict`。

## 2. 允许的标注结果

对每名受试者，专家必须能表达以下四类结果：

1. **多个候选证候**：`expert_labels` 为冻结 148 类词表中的名称数组，可含 1–N 个；
   不强制单选、不自动合并、不按模型分数排序。
2. **不能确定**：症状信息不足或彼此矛盾，标注为 `UNDETERMINABLE`；
   这是合法结果，不得强行补一个证候。
3. **专家分歧**：A/B 给出不同候选且 C 未能裁定一致，标注为 `DISAGREEMENT`；
   保留双方原始候选，不得静默取多数或取第一个。
4. **裁定后单一参考**：A/B 一致，或 C 完成裁定后，给出 `primary_label`
   （冻结词表中单个名称），`expert_status=DETERMINED`，`annotator_count>=2`。

`expert_labels` 可以为空数组（例如 `UNDETERMINABLE`）；`primary_label` 只在
`DETERMINED` 且完成裁定时填写。任何不在 148 类词表中的名称都视为未标注，
单独计入 `n_unlabeled`，不得临时扩词表。

## 3. 标注流程

1. **校准**：用合成样例（无真实患者数据）让 A/B/C 统一理解字段、允许值与
   「不能确定」的判定标准；不校准模型输出。
2. **双标注**：A、B 独立标注全部记录，各自记录阅读时长与不确定原因。
3. **一致性检查**：计算 A/B 一致率与 Cohen's κ（多候选情形先按「主候选是否一致」
   与「候选集合 Jaccard」两种口径分别报告）；κ 偏低时先复盘指南，不改标签迁就模型。
4. **裁定**：不一致记录交由 C，形成 `adjudication_id`、`primary_label` 与裁定理由。
5. **状态落库**：按 §2 写入 `expert_status`、`expert_labels`、`primary_label`、
   `annotator_count=2`、`reference_source=TCM_PROFESSIONAL`。
6. **冻结**：数据管理员对 records 文件取 sha256，写入
   `freeze-manifest.json` 后才允许运行推理与评估。

## 4. 与评估脚本的对应

| 标注结果 | `expert_status` | 主指标 | 计数字段 |
| --- | --- | --- | --- |
| 一致 / 已裁定单标签 | `DETERMINED` | 计入 | `n_evaluated` |
| 不能确定 | `UNDETERMINABLE` | 排除 | `n_undeterminable` |
| 分歧未解决 | `DISAGREEMENT` | 排除 | `n_disagreement` |
| 未标注 / 标签不在词表 | `DETERMINED` 但缺 `primary_label` | 排除 | `n_unlabeled` |
| 非医师来源 | — | 排除 | `n_invalid_reference` |

`src/syndrome/external_validation.py::_validate_records()` 强制上述约束：
拒绝重复 `subject_id`、拒绝 `reference_source` 不是 `TCM_PROFESSIONAL`、
拒绝 `DETERMINED` 但 `annotator_count < 2`，并拒绝记录中出现 `syndrome` /
`syndrome_label` / `diagnosis` 字段。

## 5. 质量控制与禁止事项

- 双标注覆盖率 100%；不得由单名专家事后补标另一人的结果。
- 分歧、不能确定、未标注样本必须**单独报告数量**，不得从分母中静默删除；
  评估脚本强制 `n_records = n_evaluated + n_excluded_total`。
- 不得用模型 Top-1/Top-3 结果反推或修正 `primary_label`。
- 不得把「疲劳 / 睡眠问题 / 肩颈不适」等招募场景标签当作证候，也不得据此
  预填 `expert_labels`。
- 不得把 `RULE`、`PUBLISHED`、`LLM_SILVER` 等既有血统与本次独立标注混用；
  不同血统不可互换，`LLM_SILVER` 绝不作为参考真值。
- 标注指南、词表版本、裁定记录与冻结清单一起归档；冻结后修改标注必须重新
  计算数据集哈希并重新冻结，原评估结果作废，不得沿用「独立测试」称谓。

## 6. 输出边界

专家标注是**研究用参考标签**，不是临床诊断；模型输出是研究估计，不是概率或诊断。
即使双标注一致性很高，若 [protocol](real-user-syndrome-protocol.md) §6 的
`use_conditions_met` 未满足，结果状态仍为 `RESEARCH_USE_ONLY`，
`GO_RESEARCH_CLAIMS` 保持 `NOT_GRANTED`。
