# TCM 真实训练报告（#50 交付，AutoResearch V1，v4）

> v4 的 AutoResearch 由**外部 LLM agent 真实驱动**（`runtime.cli research-step`
> 单轮驱动器）：agent 逐轮读取含指标/patch 的完整研究状态，在白名单
> （model/trainer 超参）内提出下一候选，harness 负责作用域校验、冻结种子、
> 子进程隔离训练与可信评分——不是确定性剧本（v1–v3 为确定性计划，
> AGENT_NOT_VERIFIED；v4 为 AGENT_INTEGRATED，patch-only，
> AGENT_ISOLATION_UNVERIFIED：OS/container 级隔离仍未验证）。
> 划分为全新冻结 split（seed 20261103）；本次划分的 holdout 仅本报告解锁
> 一次（旧划分 v1/v2 已解锁两次，均已披露，其数字作废）。
> 训练与评估全部在开发机（acam-lab）完成；本仓库只保留聚合指标与哈希，
> 完整产物在 `artifacts/tcm-50-v4/`（git-ignored）。

## 冻结契约

- 数据：`mendeley-cxtph3tjsg-v1/Share data.xls`，SHA-256 `6206f432b683f8a7…`，
  274 名受试者；QC 异常行 [273] 四维计分指纹不符（判定与标签/模型/holdout 无关）。
- 标准：ZYYXH/T157-2009（VERIFIED）；标签：RULE 派生类别（≥40=YES），
  平和维永远 ABSTAIN；有效切点随各维题数漂移（6 题维 ≈41.7、7 题维 ≈42.9、
  8 题维 ≈40.6）。
- 划分：subject-level 70/15/15 = train 191 / dev 41 / holdout 42（#27）；
  scaler 仅 train 折内拟合（#29），退化列置零，`scaler_fit_rows=191` 入档。
- 分类 manifest（v4）：`84cd78784e210ca0db5bd15ccb704cd6f441c69d50af858e2721c29e4a662ff6`。

## AutoResearch 轮次（agent 驱动，6 轮，primary = dev subject MAE）

| 轮 | agent 意图（原文） | 决定 | dev MAE |
| --- | --- | --- | --- |
| tcm-0000 | baseline：基线参照 | KEEP | 0.3806 |
| tcm-0001 | 加宽 trunk 64/64（历次研究中最可靠单项） | KEEP | 0.3699 |
| tcm-0002 | 复测高 LR 堆叠（lr 1e-2 + clip + 80ep；两个先前划分结论相反，值得再探） | KEEP | 0.3646 |
| tcm-0003 | LR 上行探界：2e-2 + clip | KEEP | 0.3614 |
| tcm-0004 | LR 3e-2 继续上行 | **KEEP（冠军）** | **0.3613** |
| tcm-0005 | LR 平台（增益噪声级）后换容量轴：128 宽 | REVERT | 0.3939 |

冠军权重 SHA-256 `2b430e1eb9788a97…`；checkpoint 携带 `smoke_only=False` +
`manifest_sha256` + `weight_epoch="last"`。所有候选同冻结种子与 epoch 级
确定性 shuffle。该轨迹为真实自适应：agent 在 LR 连续三次 KEEP 后识别平台
（0.0001 级增益），主动换轴并在 128 宽过拟合后以 REVERT 终止——与确定性
剧本在 v3（lr 1e-2 即最优）和 v4 试运行（全部 REVERT）的产出均不同，
每轮决策可从 run manifest 的 intent/patch/metrics 复盘。

## 最终评估（holdout 本次划分仅解锁一次）

| 模型 | holdout macro-F1 | holdout subject MAE |
| --- | --- | --- |
| Dummy（多数类） | 0.000 | **0.1994** |
| BMI-only 逻辑回归 | 0.000 | 0.3670 |
| 全特征逻辑回归（含可用性位） | 0.211 | 0.3369 |
| **神经冠军 tcm-0004** | **0.195**（dev 0.366） | 0.3218（dev 0.3613） |

**口径说明（必读）**：subject MAE 度量概率输出到 0/1 标签的 L1 距离，常数
近零预测器天然占优（holdout 平均正类率 ≈0.32→全零 MAE≈0.32；本划分 Dummy
0.199 略低于正类率，属 42 人小样本波动）。MAE 口径 Dummy 领先是指标性质。
F1 口径下神经冠军 0.195 与全特征逻辑回归 0.211 在 42 人上不可区分（噪声级，
与 v3 划分结论一致：**神经模型相对逻辑回归的优势未经 holdout 证实**）。
macro-F1 对全部可观测维度平均、未定义 F1 计 0。

模态可用率：body_features 100% 可用；exam_features 176/191、39/41、41/42。

## 独立交付任务

- **8-score PUBLISHED 回归**（独立 manifest `12d6005cdb183b14…`，镜像冻结
  划分，固定配置）：逐维 MAE 附训练均值参照（summary.json regression_task）；
  含 QC 行、未做敏感性对照（已声明）。
- **row-QC 敏感性**（同分母对照）：剔除 [273] 后模型 dev subject MAE 0.3656
  vs 冠军 0.3613（[273] 属 train）——QC 异常行不影响结论方向。
- **Deployable（Body/Vitals-only）消融**：dev subject MAE 0.4259、
  dev macro-F1 0.349——体检特征有真实贡献。

## 证据状态与限制（不可省略）

- 训练模式：REAL_DATA_TRAINING（agent provider=外部 LLM agent，patch-only
  白名单提案，AGENT_ISOLATION_UNVERIFIED；OS/container 级隔离仍未验证，
  source-patch 路径仍被硬性拒绝）。
- RULE 派生标签，非作者发布分类标签，非临床诊断真值。
- 受试者身份 = 数据行序（UNVERIFIED：数据集无 ID 列）。
- 冠军由 6 候选在 dev 上选出（winner's curse）；消融单 seed；dev/holdout
  均 <50 人；神经 vs 逻辑回归优势在两个独立划分的 holdout 上均未证实。
- 报告路径 fail-closed 三重哈希绑定；派生 manifest 镜像冻结划分并断言成员；
  31/31 冻结 eval + 176 pytest 在开发机全绿。
- GO_RESEARCH_CLAIMS 仍被独立参考数据阻塞（NOT_AVAILABLE）。
- 本报告仅代表研究口径（GO_REAL_DATA_TRAINING 层级），不可临床使用。
