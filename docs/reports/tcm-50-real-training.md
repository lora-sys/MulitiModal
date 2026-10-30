# TCM 真实训练报告（#50 交付，AutoResearch V1，v2）

> v2 为审查修复后的重新交付：第一轮双审查（Antigravity / Pi）的 25 项确认
> 问题冻结为 `evals/review-eval-50.md` 基准 eval 全部修复后，在开发机全量
> 重跑（新 manifest、6 轮全新 AutoResearch、报告重生成）。holdout 已如实
> 披露先后用于 v1、v2 两次最终评估。v1 数字作废，以本文为准。
> 训练与评估全部在开发机（acam-lab）完成；本仓库只保留聚合指标与哈希，
> 完整产物在 `artifacts/tcm-50-v2/`（git-ignored）。

## 冻结契约

- 数据：`mendeley-cxtph3tjsg-v1/Share data.xls`，SHA-256 `6206f432b683f8a7…`，
  274 名受试者；QC 异常行 [273] 四维计分指纹不符。
- 标准：ZYYXH/T157-2009（`tcm-label-rules-historical-verified` = VERIFIED）。
- 标签：RULE 派生类别 `derived-multilabel-v1:RULE(ZYYXH/T157-2009):YES>=40 -> 1,
  TENDENCY/NO -> 0, balanced abstain`；平和维永远 ABSTAIN。注意 CCMQ 分数
  粒度使有效切点随各维题数漂移（6 题维 ≈41.7、7 题维 ≈42.9、8 题维 ≈40.6）。
- 划分：subject-level 70/15/15 = train 191 / dev 41 / holdout 42（#27）；
  scaler 仅在 train 折内拟合（#29），退化列置零。
- 分类 manifest（v2）：`bc568ad8c83eef37c2afa7a82800461abc78c6e429465fc783f0f06e35d3e08a`。

## AutoResearch 轮次（REAL_TCM_PLAN，6 轮，primary = dev subject MAE）

| 轮 | 意图 | 决定 | dev MAE |
| --- | --- | --- | --- |
| tcm-0000 | baseline: 32-wide trunk, 40 epochs, lr 3e-3 | KEEP | 0.3108 |
| tcm-0001 | wider trunk (64/64) | REVERT | 0.3259 |
| tcm-0002 | longer schedule (80 epochs) | REVERT | 0.3140 |
| tcm-0003 | higher LR with grad clip | **KEEP（冠军）** | **0.2978** |
| tcm-0004 | smaller batches (16), 64 epochs | REVERT | 0.3064 |
| tcm-0005 | wide + high LR + long | REVERT | 0.3133 |

所有候选使用相同冻结种子与 epoch 级确定性 shuffle，仅可编辑超参不同。
冠军权重 SHA-256 `d49f4bffc012de43…`，checkpoint 携带 `smoke_only=False` +
manifest_sha256 溯源；权重文件哈希在加载前校验，state 反序列化
`weights_only=True`。

## 最终评估（holdout 在研究结束后解锁；已披露 v1/v2 两次解锁）

| 模型 | holdout macro-F1 | holdout subject MAE |
| --- | --- | --- |
| Dummy（多数类） | 0.000 | **0.2917** |
| BMI-only 逻辑回归 | 0.026 | 0.4034 |
| 全特征逻辑回归（含可用性位） | 0.275 | 0.4136 |
| **神经冠军 tcm-0003** | **0.345**（dev 0.327） | 0.3746（dev 0.2978） |

**口径说明（必读）**：subject MAE 度量概率输出到 0/1 标签的 L1 距离，常数
近零预测器天然占优（正类率 ≈0.29，全零预测 MAE≈0.29）——Dummy 在 MAE 口径
领先是指标性质，不是模型失效；神经冠军的价值在阈值化判别（F1）。两个口径
均如实给出。macro-F1 对全部可观测维度平均、未定义 F1 计 0（分母不随模型漂移）。

模态可用率（同一可用性信息同时给基线与网络）：train/dev/holdout 的
body_features 100% 可用；exam_features 178/191、38/41、40/42 可用。

## 独立交付任务

- **8-score PUBLISHED 回归**（独立 manifest `197df64dcdafec3c…`，固定配置）：
  逐维 MAE 均附训练均值预测器参照（summary.json regression_task）；
  该任务含 QC 异常行、未做敏感性对照（已声明限制）。
- **row-QC 敏感性**（同一冻结划分剔除 [273] 后重训练，冠军同配置）：
  dev subject MAE 0.3001（冠军 0.2978）——QC 异常受试者不影响结论方向。
- **Deployable（Body/Vitals-only）消融**（冠军同超参仅换特征集）：
  dev subject MAE 0.3725、dev macro-F1 0.209——体检特征有真实贡献。

## 证据状态与限制（不可省略）

- REAL_DATA_TRAINING；agent provider 为确定性计划（AGENT_NOT_VERIFIED），
  OS/容器级隔离仍未验证（AGENT_ISOLATION_UNVERIFIED）。
- RULE 派生标签，不是作者发布分类标签，更不是临床诊断真值。
- 冠军由 6 候选在 dev 上选出（winner's curse）；消融单 seed；dev/holdout
  均 <50 人，未做置信区间重采样。
- GO_RESEARCH_CLAIMS 仍被独立参考数据阻塞（NOT_AVAILABLE）。
- 本报告仅代表研究口径（GO_REAL_DATA_TRAINING 层级），不可临床使用，
  不构成疗效声明。
