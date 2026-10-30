# TCM 真实训练报告（#50 交付，AutoResearch V1）

训练与评估全部在开发机（acam-lab）完成；本仓库只保留聚合指标与哈希。
完整产物（权重、图表、summary.json、model-card）在开发机
`~/workspace/projects/MulitiModal/artifacts/tcm-50/`（git-ignored），哈希可独立核对。

## 冻结契约

- 数据：`mendeley-cxtph3tjsg-v1/Share data.xls`，SHA-256
  `6206f432b683f8a7…`，274 名受试者；row 273 四列计分指纹异常（QC 挂起）。
- 标准：ZYYXH/T157-2009（`tcm-label-rules-historical-verified` = VERIFIED）。
- 标签：RULE 派生类别 `derived-multilabel-v1:RULE(ZYYXH/T157-2009):YES>=40 -> 1,
  TENDENCY/NO -> 0, balanced abstain`；平和维（index 0）永远 ABSTAIN。
- 划分：subject-level 70/15/15 = train 191 / dev 41 / holdout 42（#27）；
  scaler 仅在 train 折内拟合（#29）。
- 分类 manifest：`c6ee6962d77891cfc667776c5c7505d5a4af75441f593a52360d3ce4cb0cd0ca`。

## AutoResearch 轮次（REAL_TCM_PLAN，6 轮，primary = dev subject MAE）

| 轮 | 意图 | 决定 | dev MAE |
| --- | --- | --- | --- |
| tcm-0000 | baseline: 32-wide trunk, 40 epochs, lr 3e-3 | KEEP | 0.3240 |
| tcm-0001 | wider trunk (64/64) | KEEP | 0.3206 |
| tcm-0002 | longer schedule (80 epochs) | KEEP | 0.3157 |
| tcm-0003 | higher LR with grad clip | KEEP | **0.3113** |
| tcm-0004 | smaller batches (16), 64 epochs | REVERT | 0.3120 |
| tcm-0005 | wide + high LR + long | REVERT | 0.3200 |

冠军 `tcm-0003`（higher LR with grad clip），权重 SHA-256
`0752e03c3be89e8bcdf09d78be24054afc677aa69ff012f961ae58b15d127d77`，
checkpoint 携带 `smoke_only=False` + manifest_sha256 溯源。

## 最终评估（holdout 仅在研究结束后解锁一次）

| 模型 | holdout macro-F1 |
| --- | --- |
| Dummy（多数类） | 0.000 |
| BMI-only 逻辑回归 | 0.026 |
| 全特征逻辑回归 | 0.263 |
| **神经冠军（AutoResearch）** | **0.357**（dev 0.311） |

逐维 holdout F1：气虚 0.513、湿热 0.400，其余维度 0.30–0.33；
阳虚在 dev 上 0.0（正类支持仅 6/41，见 chart-support.png）。

## 独立交付任务

- **8-score PUBLISHED 回归**（独立 manifest
  `4b3e55f954320934…`，同划分种子，固定配置）：holdout 逐维 MAE 15.8–22.1
  分（0–100 制），气虚 16.5 / 湿热 15.8 最好，特禀 22.1 最差。
- **row-273 敏感性**：排除后重训练 dev subject MAE 0.3280（含 273 为 0.3113）
  ——QC 异常受试者不影响结论方向。
- **Deployable（Body/Vitals-only）消融**：dev subject MAE 0.3772、
  macro-F1 0.186（无经络/舌/脉检查特征时性能明显下降——体检特征有真实贡献）。

## 证据状态与限制（不可省略）

- 训练模式 REAL_DATA_TRAINING；agent provider 为确定性计划
  （AGENT_NOT_VERIFIED），OS/容器级隔离仍未验证（AGENT_ISOLATION_UNVERIFIED）。
- 标签是规则派生类别，不是作者发布分类标签，更不是临床诊断真值。
- 列映射状态 FINGERPRINT_CROSS_CHECKED（论文定义表在付费墙内，未逐字核对）。
- n=274、类别支持不均衡；未做置信区间重采样。
- GO_RESEARCH_CLAIMS 仍被独立参考数据阻塞（NOT_AVAILABLE）。
- 本报告与全部指标仅代表研究口径（GO_REAL_DATA_TRAINING 层级），
  不可临床使用，不构成疗效声明。
