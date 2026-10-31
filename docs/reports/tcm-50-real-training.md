# TCM 真实训练报告（#50 交付，AutoResearch V1，v7）

> v7 的 AutoResearch 由**外部 LLM agent 逐轮真实驱动**：每轮由
> `runtime.cli research-step` 把完整研究状态（当前配置 + 全部历史 intent/patch/指标）
> 交给出 `antigravity:gemini-3.8-flash-high`，由它提出白名单内的 patch，
> harness 做作用域校验、冻结种子、子进程训练与可信评分，再 KEEP/REVERT。
> 每轮记录提案者、所依据状态的 SHA-256 与提案的 SHA-256。
>
> **溯源的强度要说清楚**：这些字段由驱动方（我）写入，属于**自报归属**。
> 它证明「提交的提案与它声称依据的研究状态一致」，不证明「该 agent 真的读取了上下文」
> 或「该 agent 真的生成了这段文本」。同时 agent 的运行环境仍是
> `AGENT_ISOLATION_UNVERIFIED`（无 OS/容器级隔离），source-patch 路径继续硬拒。
>
> 划分是全新冻结 split（seed 20261106），其 holdout 仅本报告解锁一次；
> 此前所有划分的解锁逐项列在「证据状态」中。

## 选择指标的演变（三轮实测，每一轮都是 eval 固化的）

1. **v5 用 `subject_mae`**：L1 损失在二值标签上的最优常数预测是中位数，正类率 <0.5
   时就是 0——按它爬山永远赢不过全零预测器（v5 实测：Dummy 0.2292 在 holdout 上领先所有候选）。
2. **v6 改用 `subject_rmse`**（Brier 的单调根变换，proper scoring rule）：在**没有可用信号**
   的模型上，它的最优解就是预测基率常数。那一轮（v6）冠军 dev macro-F1 只有 0.021，
   而普通逻辑回归 holdout 是 0.308——指标"正确"，但对分类交付物是错的落点。
3. **v7 改用交付指标本身 `category_probs/macro_f1`（maximize）**。v6 的负结果保留在
   档案里，不删除。

## 冻结契约

- 数据：`mendeley-cxtph3tjsg-v1/Share data.xls`，SHA-256 `6206f432b683f8a7…`，
  274 名受试者；QC 异常行 [273] 四维计分指纹不符（该行在 train 折）。
- 标准：ZYYXH/T157-2009；标签：RULE 派生类别（≥40=YES），平和维永远 ABSTAIN。
- 划分：**行级互斥** 70/15/15 = train 191 / dev 41 / holdout 42。受试者级不重叠
  **未经验证**——数据集没有受试者 ID 列，只能按行序假设。
- scaler 仅 train 折内拟合（`scaler_fit_rows=191`）。
- 分类 manifest：`5339985897e02aa8…`（seed 20261106）。

## AutoResearch 轮次（agent 驱动，6 轮）

| 轮 | 意图（agent 原文摘要） | 提案者 | 决定 | dev macro-F1 |
| --- | --- | --- | --- | --- |
| tcm-0000 | lr 3e-3 → 1e-3，缓解小样本上的概率极化 | antigravity:gemini-3.8-flash-high | **KEEP（冠军）** | **0.1451** |
| tcm-0001 | batch 32 → 16，增加更新频次 | antigravity:gemini-3.8-flash-high | REVERT | 0.1019 |
| tcm-0002 | 容量 32 → 64，增强表征 | antigravity:gemini-3.8-flash-high | REVERT | 0.1401 |
| tcm-0003 | epochs 40 → 60，试探充分收敛 | antigravity:gemini-3.8-flash-high | REVERT | 0.1092 |
| tcm-0004 | epochs 40 → 30，试探更强正则 | antigravity:gemini-3.8-flash-high | REVERT | 0.1274 |
| tcm-0005 | lr 1e-3 → 5e-4，试探更保守步长 | antigravity:gemini-3.8-flash-high | REVERT | 0.1025 |

冠军 **tcm-0000**：32 宽 trunk、40 epochs、lr 1e-3、batch 32、无 grad_clip、
`weight_epoch="last"`；权重 SHA-256 `ac982febc4a66ee4…`。
**结论要说白**：6 轮里 5 个候选都输给了起点配置，自动化搜索在这一划分上
**没有找到任何改进**。这是真实结果，不是失败被藏起来——所有被否决的方向
（容量、时长、批量、学习率下探）都已记录，后续轮次可以据此跳过。

## 最终评估（holdout 本划分仅解锁一次）

| 模型 | holdout macro-F1 | holdout subject MAE |
| --- | --- | --- |
| Dummy（多数类） | 0.0000 | **0.3274** |
| BMI-only 逻辑回归 | 0.0000 | 0.4151 |
| 全特征逻辑回归（含可用性位） | 0.3397 | 0.3791 |
| **神经冠军 tcm-0000** | **0.3445**（dev 0.1451） | 0.3658（dev 0.3211） |

**必须一起读的三件事**：

1. **dev 与 holdout 的 F1 差了 0.2**（0.1451 → 0.3445）。同一个模型、同一套权重，
   两个分区给出完全不同的水平——dev 只有 41 人、每维正例 6–17 个，macro-F1 的
   随机波动远大于候选之间的差距。**这意味着本数据集上 6 轮的排序基本是噪声**，
   不能用"某轮比某轮高"讲任何故事。
2. 神经冠军与全特征逻辑回归在 holdout 上的差距不到 1.5 个百分点
   （0.3445 vs 0.3397），在 42 人、数十个正例的量级上完全不可区分。**神经模型相对线性基线的优势未被证实**，
   三个划分（v5/v6/v7）上时有时无。
3. Dummy 在 MAE 口径领先（0.3274）仍是 L1 的指标性质（holdout 正类率约 0.33，
   全零预测的 MAE 就等于它），不代表模型失效，也不代表 Dummy 可用。

## 独立交付任务

- **8-score PUBLISHED 回归**（独立 manifest `32679f1b5923c683…`，镜像冻结划分）：
  逐维 MAE 附训练均值参照（summary.json regression_task）。**逐维对比结果必须如实读出**：
  见 summary 中的逐维统计——回归头在 holdout 上没有稳定胜过训练均值预测器。
  回归头与分类头共享训练且每维样本更少，当前证据不支持"评分预测已成立"。
  该任务包含 QC 行、未做敏感性对照。
- **row-QC 敏感性**（同一冻结划分剔除 [273] 后重训练，冠军同配置）：
  dev subject MAE 0.3256（剔除后）vs 0.3211（含 QC 行）。该行位于 train 折，
  dev 分区不变，差异只落在训练侧；两个值接近，QC 行不改变任何方向性结论。
- **Deployable（Body/Vitals-only）消融**（冠军同超参，仅换特征集）：
  dev subject MAE 0.3635、dev macro-F1 0.0950。单 seed、41 人、无区间估计，
  只作方向参考——**不构成"体检特征有显著贡献"的证据**。

## 模态可用率（与 summary.availability 完全一致）

| 分区/模态 | 可用 / 行数 |
| --- | --- |
| train/body_features | 191 / 191 |
| train/exam_features | 177 / 191 |
| dev/body_features | 41 / 41 |
| dev/exam_features | 39 / 41 |
| holdout/body_features | 42 / 42 |
| holdout/exam_features | 40 / 42 |

## 证据状态与限制（每一条都会影响你能得出的结论）

- 训练模式：REAL_DATA_TRAINING；agent provider=`antigravity:gemini-3.8-flash-high`，
  6/6 轮带 proposer + context_sha256 + proposal_sha256；**自报归属**，
  agent 状态 AGENT_ISOLATION_UNVERIFIED。
- 标签为 RULE 派生类别，不是作者发布的分类标签，也不是临床诊断真值。
  有效切点随各维题数漂移（6 题维 ≈41.7、7 题维 ≈42.9、8 题维 ≈40.6）。
- 受试者身份 = 数据行序（UNVERIFIED）；划分保证行级互斥，不保证受试者级。
- 冠军在 dev 上从 6 个候选中选出（winner's curse）；消融单 seed；无置信区间重采样。
- holdout 解锁记录（逐项）：split seed 20261030 在 v1、v2 解锁；20261101 在 v3 前置报告解锁；
  20261102 在 v3 报告解锁（另有一次中止运行未产出报告）；20261103 在 v4 报告解锁；
  20261104 在 v5 报告解锁；20261105 在 v6 报告解锁（该轮指标后来被否决，但其 holdout 已消耗）。
  本报告使用的 20261106 仅本次解锁；同报告的回归、消融、QC 敏感性都在同一 holdout 上，
  属同一次解锁事件。再次重评估须换划分。
- run manifest 哈希链仅提供**防意外/朴素篡改的证据层**（无密钥、目录可写），
  真正的隔离控制仍待 #45 验证。
- `research --data-manifest` 不再允许静默跑固定 plan：无 `--allow-deterministic-stub`
  时直接拒绝，允许时每轮记为 `proposer=deterministic-stub`。
- GO_RESEARCH_CLAIMS 仍被独立参考数据阻塞（NOT_AVAILABLE）；本报告仅代表研究口径，
  不可临床使用。

**可复核的运行包**：`artifacts/tcm-50-v7/run-bundle/` 内含完整
`manifest.jsonl`（6 轮、哈希链、每轮 patch 与 effective_config）与冠军检查点
（`champion/weights.pt`，其 SHA-256 与 summary 中的冠军 artifact 哈希一致，
可用 `python -m runtime.cli checkpoint-reload --dir …/champion` 复载）。
没有这个包，本报告的"逐轮配置确实累积"无法被第三方验证（v5 就是因为缺包被审计指出）。

图表：`chart-rounds.png`、`chart-per-dim-f1.png`、`chart-support.png`、
`chart-regression-dev.png`、`chart-regression-holdout.png`（缺失柱 = 未定义，不是 0）。
