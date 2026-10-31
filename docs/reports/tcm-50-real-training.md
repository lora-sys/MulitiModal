# TCM 真实训练报告（#50 交付，AutoResearch V1，v5）

> v5 的 AutoResearch 由**外部 LLM agent 逐轮真实驱动**：每一轮都是
> `runtime.cli research-step` 把完整研究状态（当前配置 + 全部历史 intent/patch/指标）
> 交给出 `antigravity:gemini-3.8-flash-high`，由它提出一个白名单内的 patch，
> harness 做作用域校验、冻结种子、子进程训练与可信评分，再 KEEP/REVERT。
> 每轮在 manifest 中记录提案者、所依据状态的 SHA-256 与提案原文的 SHA-256，
> 第三方可据此复核"谁、在看到什么之后、提出了什么"。
> 划分是全新冻结 split（seed 20261104），其 holdout 仅本报告解锁一次；
> 旧划分（seed 20261030/20261101/20261102/20261103）的解锁均已在前序报告中披露，其数字作废。

## 本次重跑修复的两个实质缺陷

1. **patch 不累积**：旧实现每轮都把 patch 应用在 base_config 上，KEEP 从不成为下一轮的
   起点——候选是彼此独立的探针而非爬山式研究。v4 报告因此把 32 宽的冠军写成 64 宽。
   现在每个候选从**当前冠军的有效配置**出发，`effective_config` 逐轮入档。
2. **提案者无溯源**：旧 manifest 只记录意图字符串，无法区分"真 agent 研究"与
   "人手工写意图"。现在每轮记录 `proposer`、`context_sha256`、`proposal_sha256`，
   且 `step()` 会拒绝与当前研究状态不符的过期上下文。

两者都已固化为冻结 eval（E57、E60、E61）。

## 冻结契约

- 数据：`mendeley-cxtph3tjsg-v1/Share data.xls`，SHA-256 `6206f432b683f8a7…`，
  274 名受试者；QC 异常行 [273] 四维计分指纹不符。
- 标准：ZYYXH/T157-2009；标签：RULE 派生类别（≥40=YES），平和维永远 ABSTAIN。
- 划分：subject-level 70/15/15 = train 191 / dev 41 / holdout 42（#27）；
  scaler 仅 train 折内拟合（#29），`scaler_fit_rows=191` 入档。
- 分类 manifest：`ef5804085fd55bfac…`（seed 20261104）。

## AutoResearch 轮次（agent 驱动，6 轮，primary = dev subject MAE）

| 轮 | 意图（agent 原文摘要） | 提案者 | 决定 | dev MAE |
| --- | --- | --- | --- | --- |
| tcm-0000 | 在 lr 3e-3 下引入 grad_clip 1.0 稳定小批量优化 | antigravity:gemini-3.8-flash-high | KEEP | 0.3923 |
| tcm-0001 | lr 3e-3 → 1e-3，降低更新方差 | antigravity:gemini-3.8-flash-high | KEEP | 0.3903 |
| tcm-0002 | 优化轴收益边际，转容量轴：hidden 32 → 16 | antigravity:gemini-3.8-flash-high | REVERT | 0.3993 |
| tcm-0003 | 40 epochs 仅约 240 步，加长到 60 试更充分收敛 | antigravity:gemini-3.8-flash-high | **KEEP（冠军）** | **0.3753** |
| tcm-0004 | 沿已证实的欠训练方向继续：60 → 90 epochs | antigravity:gemini-3.8-flash-high | REVERT | 0.3833 |
| tcm-0005 | 最后一轮探未试过的 batch_size：32 → 16 | antigravity:gemini-3.8-flash-high | REVERT | 0.3812 |

冠军 **tcm-0003**：32 宽 trunk、60 epochs、lr 1e-3、grad_clip 1.0、batch 32、
`weight_epoch="last"`；权重 SHA-256 `3e2fd8e6813ed5ee…`。
轨迹是真实的假设—证据—修正：容量轴被数据否决后立刻转向训练时长，时长收益到
60 饱和（90 回退），最后转向尚未探测的批量轴同样被否决。
每轮上下文摘要不同，说明 agent 每轮看到的是最新历史而非静态提示。

## 最终评估（holdout 本次划分仅解锁一次）

| 模型 | holdout macro-F1 | holdout subject MAE |
| --- | --- | --- |
| Dummy（多数类） | 0.0000 | **0.2292** |
| BMI-only 逻辑回归 | 0.0341 | 0.3789 |
| 全特征逻辑回归（含可用性位） | 0.2342 | 0.3634 |
| **神经冠军 tcm-0003** | **0.2821**（dev 0.4587） | **0.3342**（dev 0.3753） |

**口径说明（必读）**：subject MAE 度量概率输出到 0/1 标签的 L1 距离，常数近零预测器
天然占优——本划分 holdout 平均正类率约 0.2292，全零预测的 MAE 恰为该值，故 Dummy
在 MAE 口径领先是指标性质。本划分上神经冠军在 F1 与 MAE 两个口径都优于全特征
逻辑回归，但 holdout 仅 42 人，此优势**尚未达到统计显著**，不可作为结论性证据；
此前的两个划分上该优势均未出现。跨划分结论：神经模型相对逻辑回归的优势仍未确立。

模态可用率：body_features 100% 可用；exam_features 179/191、38/41、39/42。

## 独立交付任务

- **8-score PUBLISHED 回归**（独立 manifest `d651fa5cf6bce22d…`，镜像冻结划分）：
  逐维 MAE 附训练均值参照（summary.json regression_task）；含 QC 行、未做敏感性对照。
  **逐维对比结果必须如实读出**：dev 仅 1/8 维优于训练均值预测器，holdout 8 维全部未优于。
  回归头与分类头共享训练且每维样本更少，当前证据不支持"评分预测已成立"。
- **row-QC 敏感性**（同分母对照）：剔除 [273] 后重训模型 dev subject MAE 0.3763，
  冠军在同一缩减 dev 上 0.3753——QC 异常行不改变结论方向。
- **Deployable（Body/Vitals-only）消融**：dev subject MAE 0.3911、dev macro-F1 0.1823——
  体检特征有真实贡献。

## 证据状态与限制（不可省略）

- 训练模式：REAL_DATA_TRAINING；agent provider=`antigravity:gemini-3.8-flash-high`，
  6/6 轮带 proposer + context_sha256 + proposal_sha256；agent 状态
  AGENT_ISOLATION_UNVERIFIED——OS/container 级隔离仍未验证，source-patch 路径仍被硬拒。
- RULE 派生标签，非作者发布分类标签，非临床诊断真值。
- 受试者身份 = 数据行序（UNVERIFIED：数据集无 ID 列）。
- 冠军由 6 候选在 dev 上选出（winner's curse）；消融单 seed；dev/holdout 均 <50 人。
- run manifest 哈希链仅为防意外/朴素篡改的证据层（E44），对确定写手的隔离控制仍待 #45 验证。
- GO_RESEARCH_CLAIMS 仍被独立参考数据阻塞（NOT_AVAILABLE）。
- 本报告仅代表研究口径（GO_REAL_DATA_TRAINING 层级），不可临床使用。

图表：`chart-rounds.png`、`chart-per-dim-f1.png`、`chart-support.png`、
`chart-regression-dev.png`、`chart-regression-holdout.png`（缺失柱 = 未定义，不是 0）。
