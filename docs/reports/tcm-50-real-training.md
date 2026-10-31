# TCM 真实训练报告（#50，AutoResearch V1，v8）

生成时间：2026-10-31T09:45:52.232439+00:00  
数据：`mendeley-cxtph3tjsg-v1`（SHA-256 `6206f432b683f8a7…`，274 名受试者，其中 274 名入冻结 manifest；QC 异常行 [273] 四维计分指纹不符）  
标准：ZYYXH/T157-2009；标签契约：`derived-multilabel-v1:RULE(ZYYXH/T157-2009):YES>=40 -> 1, TENDENCY/NO -> 0, balanced abstain`  
冻结 manifest：`775ba73bfe8f4da6…`；划分：train 191 / dev 41 / holdout 42（行级互斥划分；受试者级不重叠未经验证——数据集无 ID 列）

## AutoResearch 轮次

共 24 轮：KEEP ['tcm-0000']，REVERT ['tcm-0001', 'tcm-0002', 'tcm-0003', 'tcm-0004', 'tcm-0005', 'tcm-0006', 'tcm-0007', 'tcm-0008', 'tcm-0009', 'tcm-0010', 'tcm-0011', 'tcm-0012', 'tcm-0013', 'tcm-0014', 'tcm-0015', 'tcm-0016', 'tcm-0017', 'tcm-0018', 'tcm-0019', 'tcm-0020', 'tcm-0021', 'tcm-0022', 'tcm-0023']。
选择口径：subject 级 4 折交叉验证的 macro-F1 均值（池 = train+dev 共 232 人，holdout 从不进入）；KEEP 需超过冠军一个由折间 std 派生的余量，避免把噪声当改进。
冠军配置：`tcm-0000`（选择值 0.3117）。最终交付模型是按该配置在**选择池（train+dev）全量重训**的结果（232 行），权重 SHA-256 `9d642282767df088…`。
所有候选使用相同冻结种子与 epoch 级确定性 shuffle；batch_size 被改动时批次组成随之变化，其余超参独立比较。

| 轮 | 意图 | 提案者 | 决定 | 选择 macro-F1（4 折均值 ± 折间 std） |
| --- | --- | --- | --- | --- |
| tcm-0000 | baseline: 32/32 trunk, 40 epochs, lr 3e-3, batch 32 — establishes the k-fold reference | pi:deepseek-v4.1-flash | KEEP | 0.3117 ± 0.0352 |
| tcm-0001 | lr 3e-3 -> 1e-3: v7 found the lower step size best on the same data; test whether it survives k-fold averaging | pi:deepseek-v4.1-flash | REVERT | 0.2691 ± 0.0199 |
| tcm-0002 | sweep/lr-up: lr 1e-2 with grad_clip 1.0 — v7 showed the step size is the lever that moves this model most | pi:deepseek-v4.1-flash | REVERT | 0.2992 ± 0.0382 |
| tcm-0003 | sweep/epochs: 40 -> 80, test whether the baseline is under-trained | pi:deepseek-v4.1-flash | REVERT | 0.3204 ± 0.0620 |
| tcm-0004 | sweep/capacity: 32/32 -> 64/64, test whether 191 rows support a wider trunk | pi:deepseek-v4.1-flash | REVERT | 0.2938 ± 0.0434 |
| tcm-0005 | sweep/batch: 32 -> 16, more updates per epoch | pi:deepseek-v4.1-flash | REVERT | 0.3200 ± 0.0384 |
| tcm-0006 | sweep/epochs-down: 40 -> 20, test whether the baseline overfits 191 rows | pi:deepseek-v4.1-flash | REVERT | 0.2978 ± 0.0271 |
| tcm-0007 | sweep/capacity-down: 32/32 -> 16/16, stronger bottleneck for a small sample | pi:deepseek-v4.1-flash | REVERT | 0.2994 ± 0.0126 |
| tcm-0008 | sweep/batch-up: 32 -> 64, fewer but lower-variance updates | pi:deepseek-v4.1-flash | REVERT | 0.2953 ± 0.0360 |
| tcm-0009 | sweep/lr-mid: 3e-3 -> 5e-3, the untested step between the two rejected endpoints | pi:deepseek-v4.1-flash | REVERT | 0.3144 ± 0.0346 |
| tcm-0010 | combo: lr 5e-3 + epochs 80 — combine the two best-reverting single factors (0.3144 and 0.3204) | pi:deepseek-v4.1-flash | REVERT | 0.3098 ± 0.0412 |
| tcm-0011 | combo: epochs 80 + batch 16 — the two highest reverts (0.3204, 0.3200) | pi:deepseek-v4.1-flash | REVERT | 0.3270 ± 0.0439 |
| tcm-0012 | combo: lr 5e-3 + batch 16 | pi:deepseek-v4.1-flash | REVERT | 0.3045 ± 0.0141 |
| tcm-0013 | sweep/grad-clip: 1.0 at the baseline lr, isolated from the lr change | pi:deepseek-v4.1-flash | REVERT | 0.3133 ± 0.0349 |
| tcm-0014 | combo: epochs 80 + batch 16 + lr 5e-3 — add the mid lr to the current best combination (0.3270) | pi:deepseek-v4.1-flash | REVERT | 0.3225 ± 0.0278 |
| tcm-0015 | combo: epochs 80 + batch 16 + 16/16 trunk — pair the best schedule with the tightest bottleneck | pi:deepseek-v4.1-flash | REVERT | 0.3143 ± 0.0170 |
| tcm-0016 | combo: epochs 100 + batch 16 — extend the best schedule one step further | pi:deepseek-v4.1-flash | REVERT | 0.3154 ± 0.0412 |
| tcm-0017 | combo: epochs 80 + batch 8 — smallest batch at the best schedule | pi:deepseek-v4.1-flash | REVERT | 0.3148 ± 0.0241 |
| tcm-0018 | sweep/lr 2e-3 — the untested step between 1e-3 (0.2691) and 3e-3 (0.3117) | pi:deepseek-v4.1-flash | REVERT | 0.3099 ± 0.0390 |
| tcm-0019 | sweep/epochs 60 — the untested schedule between 40 (0.3117) and 80 (0.3204) | pi:deepseek-v4.1-flash | REVERT | 0.3242 ± 0.0574 |
| tcm-0020 | sweep/capacity 48/48 — between the rejected 16 and 64 trunks | pi:deepseek-v4.1-flash | REVERT | 0.2910 ± 0.0242 |
| tcm-0021 | sweep/batch 24 — between 16 (0.3200) and 32 (0.3117) | pi:deepseek-v4.1-flash | REVERT | 0.3092 ± 0.0254 |
| tcm-0022 | sweep/asymmetric 64/32 — hidden capacity without widening the fusion layer | pi:deepseek-v4.1-flash | REVERT | 0.2826 ± 0.0256 |
| tcm-0023 | sweep/grad-clip 0.5 — tighter than the 1.0 that was neutral | pi:deepseek-v4.1-flash | REVERT | 0.3093 ± 0.0327 |

## 最终评估（holdout 在研究结束后解锁；历次解锁见「证据状态与限制」，均已如实披露）

评分口径：交付模型在选择池（232 行）上重训，dev 41 人已并入该池，受信评估器禁止 train/dev 重叠，因此**只在 holdout 42 人上评分**；下方所有对照（Dummy / BMI / 全特征逻辑回归）都在同一池上拟合、同一 holdout 上评分，不靠少给基线 41 行来抬高神经模型的数字。
- 神经冠军 holdout macro-F1 0.383（macro 对全部可观测维度平均，未定义 F1 计 0，分母不随模型漂移；未定义维度 0 个）
- 神经冠军 holdout subject MAE 0.3724（诊断口径；选择指标是 macro-F1，即上方口径）
- 神经冠军 holdout subject RMSE（Brier 诊断，非选择指标）0.5447；Dummy（多数类）holdout subject RMSE 0.5373
- 逐维失败维度：holdout 上没有被完全漏判的维度
- Dummy（多数类）：holdout macro-F1 0.000，holdout subject MAE 0.2887
- BMI-only 逻辑回归：holdout macro-F1 0.014，holdout subject MAE 0.4032
- 全特征逻辑回归（含可用性指示位）：holdout macro-F1 0.215，holdout subject MAE 0.3941
- **口径说明（必读）**：AutoResearch 的选择指标就是这里报告的 macro-F1（0.5 阈值、固定观测维分母），保证爬山的落点和交付指标一致。两个概率损失代理都被实测否决：subject MAE 是 L1，二值标签下的最优常数是中位数 0（本数据正类率 ≈0.29 < 0.5），按它爬山永远赢不过全零预测器；subject RMSE（Brier）虽是 proper scoring rule，但对没有可用信号的模型，最优解是预测基率常数——实测那一轮冠军 dev macro-F1 只有 0.021，而普通逻辑回归 holdout 是 0.215。因此 MAE/RMSE 仅作诊断，选择看 F1：holdout macro-F1 0.383 对 Dummy 0.000、全特征逻辑回归 0.215。三个口径都给出，读者可自行权衡。**统计强度**：holdout 仅 42 人，macro-F1 的差异在数十个正例量级上属噪声范围，未做重采样置信区间；跨划分的重复性才是判据，单一划分上的领先不构成「该模型更好」的证据。

图表：`chart-rounds.png`、`chart-per-dim-f1.png`、`chart-support.png`、`chart-regression-dev.png`、`chart-regression-holdout.png`（图中缺失柱 = F1/MAE 未定义，不是 0）。

## 模态可用率（缺失≠被当作平均人：基线与神经模型拿到同一可用性信息）

| 分区/模态 | 可用 / 行数 |
| --- | --- |
| dev/body_features | 41 / 41 |
| dev/exam_features | 40 / 41 |
| holdout/body_features | 42 / 42 |
| holdout/exam_features | 41 / 42 |
| train/body_features | 191 / 191 |
| train/exam_features | 175 / 191 |

## 独立交付任务

本节三个任务都在 holdout 上评分，训练口径各自不同：**8-score 回归**在冻结 train 折（191 行）上训练；**row-QC 敏感性**与**Deployable 消融**在与交付模型相同的选择池（232 行）上训练。它们各自内部两臂同口径，用来回答「该组件/该特征集是否改变结果」；回归头训练的模型与交付模型不同，两个 holdout 数字不能直接相减。

- **8-score PUBLISHED 回归**（独立 manifest，固定配置）：逐维 MAE 与训练均值预测器参照见 `summary.json`（regression_task）；manifest `d112579468429c08…`。该任务包含 QC 异常行、未做敏感性对照，属已知限制。逐维对比结果：dev 在 2/8 维优于训练均值预测器；holdout 未优于训练均值预测器。回归头与分类头共享训练、每维样本量更小，当前证据不支持把它当作评分预测已成立的证据。
- **row-QC 敏感性**（选择池剔除 [273] 后重训，冠军同配置，231 行 vs 交付模型 232 行）：剔除后模型 holdout subject MAE 0.3524 vs 含 QC 行训练的冠军 0.3724。两侧都只在 holdout 上评分（同一分区、同一分母），差异只来自训练侧。
- **Deployable（Body/Vitals-only）消融**（冠军同超参、同一选择池 232 行，仅换特征集）：holdout subject MAE 0.4061，holdout macro-F1 0.200。单 seed、holdout 42 人、无区间估计——只作方向参考，不构成特征贡献的显著证据。

## 证据状态与限制

- 训练模式：REAL_DATA_TRAINING（agent provider=外部 LLM agent `pi:deepseek-v4.1-flash`；每轮记录提案者与上下文摘要（24/24 轮带完整溯源）；agent 状态 AGENT_ISOLATION_UNVERIFIED：OS/container 级隔离仍未验证）。
- 标签为 RULE 派生类别（规则阈值 ≥40=YES；TENDENCY/NO=0；平和维 ABSTAIN）——不是作者发布的分类标签，也不是临床诊断真值。注意 CCMQ 分数粒度使有效切点随各维题数漂移（6 题维 ≈41.7、7 题维 ≈42.9、8 题维 ≈40.6）。
- 列映射为 FINGERPRINT_CROSS_CHECKED（论文定义表付费墙内）；QC 异常行 [273] **包含在**主分析与全部基线中，只在 row-QC 敏感性分析里被剔除；QC 判定仅基于计分规则指纹，与标签、模型表现和 holdout 均无关，不构成 holdout 信息泄漏。
- 受试者身份 = 数据行序（row_order (UNVERIFIED: dataset has no subject ID column; identity assumed from row position)）——数据集无受试者 ID 列，无法独立核验每行确为不同受试者。
- 冠军由 24 轮候选按交叉验证选出（每轮都在上一轮冠军基础上改动，属爬山；winner's curse 仍然存在，只是从 41 人单划分换成 k 折均值加余量门槛）；消融为单 seed 固定配置；holdout 仅 42 人，小数位差异属噪声，未做置信区间重采样。
- holdout 解锁记录（逐项）：split seed 20261030 在 v1、v2 解锁；20261101 在 v3 前置报告解锁；20261102 在 v3 报告解锁（另有一次中止运行未产出报告）；20261103 在 v4 报告解锁；20261104 在 v5 报告解锁；20261105 在 v6 报告解锁（该轮的选择指标后来被否决，但其 holdout 已消耗）。本报告使用全新冻结划分（manifest `775ba73bfe8f…`，seed 20261107），其 holdout 仅本次解锁一次；同一报告的 8-score 回归、部署消融、row-QC 敏感性都在同一 holdout 上评估，属同一次解锁事件，不额外增加信息。如再次重评估须再换划分。
- 溯源强度：`proposer` / `context_sha256` / `proposal_sha256` 由驱动方（操作者）写入，属于**自报归属**——它证明「提交的提案与它声称依据的研究状态」一致」，不证明「该 agent 真的读取了上下文」或「该 agent 真的生成了它」。要升级为可证明的 agent 驱动，需要 provider 端 request/response id 或签名。
- 类别支持不均衡（见 chart-support.png）；结果仅代表研究口径，不可临床使用。
