# TCM 真实训练报告（#50 交付，AutoResearch V1，v3）

> v3 为第二轮审查修复后的重新交付：第二轮零上下文对抗审查（Pi）的 12 项
> 发现冻结为 `evals/review-eval-50.md` 的 E26–E36，全部修复后在开发机以
> **全新冻结划分**（seed 20261101）全量重跑。旧划分（v1/v2 使用）的 holdout
> 已解锁两次（均已披露，v1 数字作废，v2 数字作废）；本次划分的 holdout 仅
> 在本报告解锁一次。训练与评估全部在开发机（acam-lab）完成；本仓库只保留
> 聚合指标与哈希，完整产物在 `artifacts/tcm-50-v3/`（git-ignored）。

## 冻结契约

- 数据：`mendeley-cxtph3tjsg-v1/Share data.xls`，SHA-256 `6206f432b683f8a7…`，
  274 名受试者；QC 异常行 [273] 四维计分指纹不符（判定与标签/模型/holdout 无关）。
- 标准：ZYYXH/T157-2009（`tcm-label-rules-historical-verified` = VERIFIED）。
- 标签：RULE 派生类别 `derived-multilabel-v1:RULE(ZYYXH/T157-2009):YES>=40 -> 1,
  TENDENCY/NO -> 0, balanced abstain`；平和维永远 ABSTAIN。CCMQ 分数粒度使
  有效切点随各维题数漂移（6 题维 ≈41.7、7 题维 ≈42.9、8 题维 ≈40.6）。
- 划分：subject-level 70/15/15 = train 191 / dev 41 / holdout 42（#27）；
  scaler 仅在 train 折内拟合（#29，逐列有限值，退化列置零；
  `scaler_fit_rows=191` 已入档）。
- 分类 manifest（v3）：`be457713e8b82e2f27eb8aad8bd62155e7aed2600c67ed7cbe8a73448d11cda1`。

## AutoResearch 轮次（REAL_TCM_PLAN，6 轮，primary = dev subject MAE）

| 轮 | 意图 | 决定 | dev MAE |
| --- | --- | --- | --- |
| tcm-0000 | baseline: 32-wide trunk, 40 epochs, lr 3e-3 | KEEP | 0.3949 |
| tcm-0001 | wider trunk (64/64) | REVERT | 0.3958 |
| tcm-0002 | longer schedule (80 epochs) | KEEP | 0.3690 |
| tcm-0003 | higher LR with grad clip | REVERT | 0.3814 |
| tcm-0004 | smaller batches (16), 64 epochs | REVERT | 0.3715 |
| tcm-0005 | wide + high LR + long | **KEEP（冠军）** | **0.3674** |

所有候选使用相同冻结种子与 epoch 级确定性 shuffle，仅可编辑超参不同。
冠军权重 SHA-256 `9cd37814b586f9ba…`；checkpoint 携带 `smoke_only=False` +
`manifest_sha256` + `weight_epoch="last"`（本轮审查明确：权重为末轮而非
best-eval-loss 轮——A/B 实测按 eval loss 恢复会选中欠拟合的基率轮，
dev F1 0.03 vs 0.27，故 restore 为 opt-in 且默认关闭）。

## 最终评估（holdout 本次划分仅解锁一次）

| 模型 | holdout macro-F1 | holdout subject MAE |
| --- | --- | --- |
| Dummy（多数类） | 0.000 | **0.2768** |
| BMI-only 逻辑回归 | 0.025 | 0.3986 |
| 全特征逻辑回归（含可用性位） | 0.241 | 0.4099 |
| **神经冠军 tcm-0005** | **0.235**（dev 0.288） | 0.4179（dev 0.3674） |

**口径说明（必读）**：subject MAE 度量概率输出到 0/1 标签的 L1 距离，常数
近零预测器天然占优（holdout 平均正类率 ≈0.28，全零预测 MAE≈0.28）——Dummy
在 MAE 口径领先是指标性质，不是模型失效。F1 口径下神经冠军（0.235）与全
特征逻辑回归（0.241）在 42 人 holdout 上不可区分（小数位差异属噪声）；
神经冠军在 dev 上优於逻辑回归（0.288 vs 0.24），但经 holdout 检验后该优势
未被证实。两个口径均如实给出。macro-F1 对全部可观测维度平均、未定义 F1
计 0（分母不随模型漂移）。

模态可用率：train/dev/holdout 的 body_features 100% 可用；
exam_features 179/191、38/41、39/42 可用（同一可用性信息同时给基线与网络）。

## 独立交付任务

- **8-score PUBLISHED 回归**（独立 manifest `ee8197ed935aa0bb…`，镜像冻结
  划分，固定配置）：逐维 MAE 均附训练均值预测器参照（summary.json
  regression_task）；该任务含 QC 异常行、未做敏感性对照（已声明限制）。
- **row-QC 敏感性**（同一冻结划分剔除 [273] 后重训练，冠军同配置；同分母
  对照）：剔除后模型 dev subject MAE 0.3781 vs 冠军 0.3674（[273] 在本划分
  属 train，缩减 dev 与完整 dev 相同）——QC 异常受试者不影响结论方向。
- **Deployable（Body/Vitals-only）消融**（冠军同超参仅换特征集）：
  dev subject MAE 0.4516、dev macro-F1 0.264——体检特征有真实贡献。

## 证据状态与限制（不可省略）

- REAL_DATA_TRAINING；agent provider 为确定性计划（AGENT_NOT_VERIFIED），
  OS/容器级隔离仍未验证（AGENT_ISOLATION_UNVERIFIED）。
- RULE 派生标签，不是作者发布分类标签，更不是临床诊断真值。
- 受试者身份 = 数据行序（subject_identity = UNVERIFIED：数据集无 ID 列）。
- 冠军由 6 候选在 dev 上选出（winner's curse）；消融单 seed；dev/holdout
  均 <50 人，未做置信区间重采样。
- 报告路径带 fail-closed 绑定校验（smoke_only / manifest / dataset 三重
  哈希核对）；派生 manifest（回归/deployable/QC）镜像冻结划分并断言成员
  一致；30/30 冻结 eval 检查 + 167 pytest 在开发机全绿。
- GO_RESEARCH_CLAIMS 仍被独立参考数据阻塞（NOT_AVAILABLE）。
- 本报告仅代表研究口径（GO_REAL_DATA_TRAINING 层级），不可临床使用，
  不构成疗效声明。
