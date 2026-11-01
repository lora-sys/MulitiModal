# 源码级 AutoResearch（v9）—— 运行记录与契约

> 这是本项目第一次跑**真正的 autoresearch**：Agent 改的是**源码**，每折有**固定墙钟预算**，KEEP/REVERT 是 **git 分支前进/回退**，并把「更简单等价也是赢」写进判据。
> 之前 v1–v8 的 harness 只允许改 6 个超参**取值**，实测 23/23 REVERT，冠军从未移动——**空搜索空间产生不了研究**。

日期：2026-10-31。运行主机：acam-lab。运行分支：`autoresearch/v9`（worktree `/home/lora/workspace/ar-v9`）。
数据：Mendeley 274 行，特征契约 **v2（63 列，含此前从未被读取的 16 列）**，冻结划分 seed `20261107`（train 191 / dev 41 / holdout 42）。

## 1. 冻结契约（Agent 不可改）

| 项 | 值 | 位置 |
| --- | --- | --- |
| 唯一可改文件 | `src/research/candidate.py` | `research/loop.py` |
| 训练墙钟预算 | **20s / 折**（常量，不是可调项） | `research/frozen.py` |
| 超预算 | 视为 **overran → REVERT**，不是"慢一点的赢家" | `runner.py` / `loop.py` |
| 选择指标 | `category_probs/macro_ap`（阈值无关的 PR 曲线面积） | `runtime/tcm_eval.py` |
| 每折 scaler | 在**该折自己的 train 行**上重拟合；折 manifest 的 holdout 为空 | `data/real_split.fold_manifest` |
| 评估器 | 在**独立的评估器子进程**中对**重建的折**评分（`research/evaluator.py`）；模型只以 **state_dict + 冻结架构白名单** 进入评估（`torch.load(weights_only=True)`，不执行任何候选代码）；候选看不到分数，也碰不到评分用的对象 | `research/runner.py` + `research/evaluator.py` + `research/model_export.py` |
| 训练子进程 | 候选代码**只在**一次性 `research/train_worker` 子进程执行：输入是裁剪后的折文件（无 dev、无 holdout、无全集路径），输出是磁盘上的权重与架构文件 | `research/train_worker.py` |
| 选择分辨率 | 4 折 × 2 重复 = **8 折**，SEM ≈ 0.016 | `loop.state()` |
| KEEP 门槛 | 增益 > **1×SEM**，且 ≥ **0.001 × 净增行数** | `loop.decide` |

`20s` 而不是上游的 300s：本模型 ~6k 参数、每折 174 行，必须保持可交互；这是**声明的**契约，改动会同时改变所有已归档轮次的含义。

**预算变更时间线（2026-11-01 审查核实，证据为 manifest 时间戳 + worktree 提交历史）**：循环首次落地时 `TIME_BUDGET_SECONDS = 60.0`（worktree 提交 `cb4c8e4`，19:32:21 +0800）；**80 秒后**、即任何轮次被打分之前，`c6ccf19`（19:33:41 +0800）就把预算改为 20s；第一个完成轮 tcm-0001 开始于 19:33:50 +0800，其 manifest `fold_seconds` 全部 ≈20.0，tcm-0003 同。主线仓库的同款修改晚至 `4d19481`（19:38:42 +0800）才合入（"judge only added lines…" 提交的一部分）。**结论：没有任何已计分轮次是在 60s 预算下测得的**，§2 表内数值同口径可比；60s 值只存在于未计分的 80 秒窗口内。

## 2. 轮次（4 轮，2 KEEP/REVERT，2 被 guard 拒收）

| 轮 | 意图 | 决策 | 选择值 macro-AP | 净增行 | git |
| --- | --- | --- | --- | --- | --- |
| tcm-0000 | （导出时工作树不干净，补丁夹带了 `frozen.py`） | **REVERT / rejected** | — | — | — |
| **tcm-0001** | 基线：32 宽 per-modality MLP + 掩码 BCE + Adam 3e-3 | **KEEP** | **0.3491** | 0 | `242ed72` |
| tcm-0002 | 3 成员集成 | **REVERT / rejected** | — | — | — |
| tcm-0003 | 3 成员集成（在预算内均分） | **REVERT / completed** | 0.3609 | +31 | 回到 `242ed72` |

**两条被 guard 拒收的记录本身就是产物**，它们暴露了两个真实缺陷（都已在同轮修掉并冻结成测试）：

1. **工作树必须干净**：脏树意味着被测文件不是 manifest 记录的父提交，而 `git diff` 生成的补丁会把别人的未提交改动夹带进本轮。现在 `assert_clean()` 直接指名违规文件。
2. **只审查新增行**：原实现扫描整个 diff，于是模块里既有的 `batch.targets["category_labels"]` 作为**上下文行**触发了"碰到冻结件"的误报。同时把 `category_labels`/`score_mask` 从禁用表移出——它们是冻结 provider **交给候选**的训练接口，不是评分侧。

**复杂度判据的第一次标定也是错的**：最初按 0.002/行**加价**，等于要求 15 行的架构改动在 SEM≈0.016 的选择上换来 +0.03——一条会否决一切、却看起来像复杂度准则的规则。现在是**增益下限**（每净增行 0.001），也就是上游那句话的字面读法：大而草率的补丁必须换来大增益，小而精准的只须过噪声门槛，而"删代码且不更差"直接 KEEP。

## 3. tcm-0003：+0.0118，未过 0.0164 门槛

`0.360878 vs best 0.349055: +0.0118 does not clear the 0.0164 noise margin`

集成**确实提升了排序**（+0.0118，8 折均值），但增益小于 1 个标准误 → REVERT。这与 v8 的"找不到东西"是**两回事**：v8 是候选之间根本没有差异，这里是**有差异但测量分辨率不够**。

按契约的正确下一步是把分辨率提上去，而不是把门槛调松：

| 动作 | SEM 预期 | 代价 |
| --- | --- | --- |
| 8 折 × 2 重复 = 16 折 | ≈ 0.0116 | 每轮 ~11 min |
| 8 折 × 3 重复 = 24 折 | ≈ 0.0095 | 每轮 ~16 min |

改分辨率必须**同口径重跑 tcm-0001 基线**，否则与已归档轮次不可比（协议变更不是免费的后台调参）。

## 4. 复现

```bash
# 开发机；运行分支是 git worktree，主检出只用来驱动
cd /home/lora/workspace/projects/MulitiModal
export PYTHONPATH=src

# 1) 打印提案者状态（无 holdout、无评估器句柄、无标签文件）
.venv/bin/python -m runtime.cli research-src-state \
  --worktree /home/lora/workspace/ar-v9 --branch autoresearch/v9 \
  --data-manifest runs/manifests/tcm-20261107-v2-features.json \
  --dataset "data/local/mendeley-cxtph3tjsg-v1/Share data.xls" \
  --out runs/research-v9

# 2) 一次一轮：给出补丁，落盘 → 训练 → 冻结评估器打分 → commit 或 reset
.venv/bin/python -m runtime.cli research-src-round \
  --worktree /home/lora/workspace/ar-v9 --branch autoresearch/v9 \
  --patch-file /tmp/pending.patch --intent "..." --proposer "<agent-id>" \
  --data-manifest runs/manifests/tcm-20261107-v2-features.json \
  --dataset "data/local/mendeley-cxtph3tjsg-v1/Share data.xls" \
  --out runs/research-v9

# 3) 证据层（哈希链 append-only）
.venv/bin/python /tmp/show_run.py     # 或直接读 runs/research-v9/manifest.jsonl
cd /home/lora/workspace/ar-v9 && git log --oneline   # 代码层：只有 KEEP 会前进
```

## 5. 证据状态与限制

- **搜索空间已经非空**：架构、损失、优化器、调度、批处理全部可改（这是 v1–v8 缺失的核心）；可改面仍被限制在**一个文件**内。
- 候选进程内可读工作树内的任何文件，包括 `data/local/` 下的原始工作簿。理论上一个候选可以把 `batch.subject_ids` 与工作簿行做关联、直接构造标签来刷分。这与上游同源的残留风险（其 `train.py` 也能读被 pin 的验证分片）**同等级**，当前防线是 2026-11-02 第二轮隔离修复（见文末）：候选代码只在一次性、输入裁剪的训练子进程执行，评估只信 state_dict + 冻结架构白名单，扣费（解锁事件）在 spawn 之前落账；OS 级隔离仍属 `AGENT_ISOLATION_UNVERIFIED`。
- `proposer` 由驱动方写入，属**自报归属**；本轮所有提案来自 `pi:deepseek-v4.1-flash`（操作者本人），不是独立 agent。轮次写入是按轮进行，未使用批量预写。
- 选择值 0.3491 是**8 折池内交叉验证**的均值，不是 holdout 成绩；v9 尚未解锁 holdout，也没有交付检查点（交付需在冠军配置上对选择池重训再用 holdout 评一次）。
- 训练标签仍是 RULE 派生（≥40=YES），受试者身份仍是行序（数据无 ID 列）；**GO_RESEARCH_CLAIMS 仍被独立参考数据阻塞**。
---

# v10：真正跑通源码级搜索、交付与 holdout 验证

日期：2026-10-31。运行分支：`autoresearch/v10`（worktree `/home/lora/workspace/ar-v10`）。
契约与 v9 相同，只有**选择分辨率**提高到 **4 折 × 3 重复 = 12 折**（SEM ≈ 0.0095–0.017）。改分辨率意味着必须同口径重跑基线，因此 v10 是独立运行，不与 v9 的数值混比。

## 1. 轮次（8 轮，2 KEEP / 6 REVERT）

| 轮 | 候选 | 决策 | macro-AP（12 折） | 净增行 | 判据 |
| --- | --- | --- | --- | --- | --- |
| tcm-0000 | v10 基线（与 v9 冠军同代码） | **KEEP** | 0.3424 | 0 | baseline |
| tcm-0001 | 集成 3 成员（27 行实现） | REVERT | 0.3605 | 27 | +0.0181 **过了噪声门槛 0.0152**，但没过 27 行的复杂度下限 0.027 |
| tcm-0002 | dropout 0.3 + weight decay 1e-2 | REVERT | 0.3492 | 3 | +0.0068，未过门槛 |
| tcm-0003 | 模型内 LayerNorm | REVERT | 0.3488 | 4 | +0.0064，未过门槛 |
| **tcm-0004** | **同样的 3 成员集成（manifest `complexity.growth` = 17）** | **KEEP** | **0.3595** | 17 | **+0.0171 > 门槛 0.0131，且 ≥ 17 行下限 0.0170** |
| tcm-0005 | 集成 5 成员 | REVERT | 0.3610 | 15 | +0.0015，未过门槛 |
| tcm-0006 | 集成 3 + weight decay | REVERT | 0.3573 | 15 | −0.0022 |
| tcm-0007 | 集成 3 + dropout + weight decay | REVERT | **0.3648** | 16 | +0.0053，未过门槛 0.0153 |

**两件必须一起读的事**：

1. **复杂度判据真的起作用了**：tcm-0001 的增益（+0.0181）已经过了噪声门槛，被拒的唯一原因是 27 行没换来 0.027。下一轮把同一机制写成 17 行（tcm-0004，manifest 记的净增口径）就通过了。这就是上游那句"a 0.001 improvement that adds 20 lines of hacky code? Probably not worth it"要的效果——**逼出更紧凑的实现，而不是否决想法**。
2. **最高分不是冠军**：tcm-0007 的绝对分数（0.3648）高于冠军（0.3595），但因未过显著性门槛被拒。这是规则的诚实代价：冠军是"**可复现地**优于前一个冠军"的配置，不是"历史上最高的单次观测"。报告必须同时给出两者，否则读者会以为冠军是 0.3648。

## 2. 交付模型与 holdout（本划分仅本次解锁）

`research-src-finalize` 把冠军配置（`git_commit 09f9421`，候选文件 SHA-256 `b78a0cd6…`）在**选择池 232 行**上重训（`refit_partition=train+dev`，holdout 42 行未参与），然后在 holdout 上评分一次。

| 模型（均在 232 行池上拟合，同一 holdout 42 人） | holdout macro-AP | holdout macro-F1@0.5 | holdout subject MAE |
| --- | --- | --- | --- |
| Dummy（多数类） | 0.2887 | 0.000 | 0.2887 |
| **BMI-only 逻辑回归（2 个特征）** | **0.3761** | 0.014 | 0.4032 |
| 全特征逻辑回归 | 0.3502 | 0.210 | 0.3981 |
| **神经交付模型（本项目冠军）** | **0.3724** | **0.362** | 0.3597 |

## 3. 泛化结论（这是要如实读的部分）

- **相对自身基线**：选择池内 12 折 CV 从 0.3424 → 0.3595（**+0.017**），这是本轮搜索唯一有统计支持的改进。
- **相对无信息/简单基线**：holdout 上清楚高于 Dummy（0.2887）与全特征逻辑回归（0.3502），但在排序指标上**没有胜过只有 2 个特征的 BMI 逻辑回归**（0.3724 vs 0.3761，差 0.004，42 人上完全不可区分）。
- **因此不得宣称**："神经模型比简单基线更好"。上一轮 v8 报告的"0.383 vs 0.215"优势被证实是**阈值产物**（同一逻辑回归按训练先验阈值可得 0.3844）；换成阈值无关的 AP 后，那个优势消失，现在这轮给出了独立复现。
- 42 人 holdout、无重采样置信区间、RULE 派生标签、受试者身份为行序（UNVERIFIED）—— **GO_RESEARCH_CLAIMS 仍未放行**。

## 4. 得到的可复现结论

1. **空搜索空间产生不了研究**：v1–v8 的 6 个超参取值，23/23 REVERT 且冠军不动；v10 在**同一数据、同一划分**上，8 轮就把选择值抬了 0.017——差别只在可改面从"取值"变成"源码"。
2. **集成是唯一被证明有效的代码级杠杆**（+0.017~0.018，两次独立复现：27 行版与 17 行版；行数一律取 manifest `complexity.growth`，不用提交信息或正文里的其它口径。）
3. **正则化类改动（dropout / weight decay / LayerNorm）在这份数据上没有可检测收益**（+0.006 量级），与"274 行、特征噪声大"一致。
4. **复杂度下限是有效的，但会与噪声门槛叠加**：两者都过的具体形态是"机制正确 + 实现紧凑"。

## 5. 复现

```bash
cd /home/lora/workspace/projects/MulitiModal && export PYTHONPATH=src
WT=/home/lora/workspace/ar-v10; B=autoresearch/v10
M=runs/manifests/tcm-20261107-v2-features.json
D="data/local/mendeley-cxtph3tjsg-v1/Share data.xls"
# 逐轮
.venv/bin/python -m runtime.cli research-src-round --worktree $WT --branch $B \
  --folds 4 --repeats 3 --data-manifest $M --dataset "$D" \
  --out runs/research-v10 --patch-file p.patch --intent "..." --proposer "<id>"
# 交付 + holdout（一次）
.venv/bin/python -m runtime.cli research-src-finalize --worktree $WT --branch $B \
  --data-manifest $M --dataset "$D" --out runs/research-v10
# 归档层
cd $WT && git log --oneline      # 只有 KEEP 前进：dc7d67b 基线 → 09f9421 冠军
```

---

# v10 phase-3：冠军之上的 5 次架构/损失攻击（全部 REVERT，按 plateau 早停）

日期：2026-11-01。同一契约（12 折、冻结划分 seed 20261107）、同一冠军（tcm-0004，0.3595）。批驱动 `/tmp/run_v10c.py` 声明了 7 个候选和 **PLATEAU=5**（连续 5 次 REVERT 即停）；实际 5 轮全 REVERT 后触发早停，最后两个组合候选（bag3_gate_perdim、bag3_lr_1e2）未运行——两个单点均无任何信号，省下约 22 分钟未损失信息。

| 轮 | 候选 | 决策 | macro-AP（12 折） | 净增行 | 判据 |
| --- | --- | --- | --- | --- | --- |
| tcm-0008 | 可学习模态门控（9 体部列 vs 54证候列不再等权） | REVERT / completed | 0.3592 | 3 | −0.0003，未过 0.0145 噪声门槛 |
| tcm-0009 | 逐维双层头（8 个维度各自 2 层） | REVERT / **crash** | — | 3 | 补丁生成 IndentationError（candidate.py line 74），runner 如实记 crash |
| tcm-0010 | 模态内 LayerNorm（折内 scaler 之外的批内标准化） | REVERT / completed | 0.3467 | 4 | −0.0128，未过 0.0124 门槛 |
| tcm-0011 | 逆流行率正样本加权（正例率 ≈0.25） | REVERT / completed | 0.3507 | 3 | −0.0088，未过 0.0144 门槛 |
| tcm-0012 | fusion 加深一层 | REVERT / completed | 0.3503 | 1 | −0.0092，未过 0.0129 门槛 |

## 学到了什么

1. **集成冠军经受住了五个方向的攻击**：模态加权（gate）、输出头结构（perdim）、批内标准化（norm）、损失加权（posweight）、深度（deep）——最好的 gate 也只是打平（−0.0003，且是加 3 行），其余全部显著更差。冠军仍是 tcm-0004（0.3595），**未产生新冠军，holdout 预算不动**（v10 已解锁一次，只剩最终交付一次）。
2. **gate 的 −0.0003 不是"gate 无效"**：−0.0003 在 12 折 SEM（≈0.0095–0.017）之内，正确读法是"无法区分"；但判据是**加代码必须证明自己**，打平的加法被拒——这与"删代码且不更差直接 KEEP"是同一条规则的两面。
3. **崩溃也是产物**：perdim 的文本变换产出了语法错误的补丁，runner 如实记为 crash REVERT，没有静默打分、也没有静默跳过；锚点断言保证了变换失败会显式失败，而不是变成 no-op。

---

# 2026-11-01：零上下文对抗审查的修正（本节为修正记录，不改写上方任何已归档数值）

一轮零上下文对抗审查判定 v10 的核心完整性声称（「候选看不到、碰不到决定其命运的分数」）在实现层面不成立。逐条修正如下；**已归档 manifest 与其数值一律不改写**，修正只作用于未来轮次与报告表述。

## 修正内容

1. **[P0] 评估器与候选同进程（已修）**。原实现在 `research/runner.py` 内 late-import `TCMTrustedEvaluator`，与 `research.candidate` 共享解释器：候选的 `train()` 可以原地替换 `runtime.tcm_eval.TCMTrustedEvaluator`，或就地污染与评估器共享的同一个 `FoldData.dev._targets`。上方 v9/v10 轮次**确实是在该路径下评分的**——第 1 节契约表的「评估器」行已按真实机制改写。修复后评分在独立的 `research/evaluator.py` 子进程完成：折从冻结 manifest 重建（确定性，含 scaler）、训练侧输入（manifest/数据集/候选文件）做训练前后完整性快照比对、评分组件在 unpickle 导入候选代码**之前**抓取、并用 audit hook 阻断候选代码对 runtime/data/core.registry 的 import。冻结层四处声称（`frozen.py` / `guard.py` / 本报告 / 设计文档）已全部对齐为实际机制。
2. **[P1] 折路径可推导 holdout 身份（已修）**。折 manifest 深拷贝了父 manifest 的全部 274 条 `subject_row_by_id`，且 provider 持有全量 records（含 holdout 行 scores）。现 `fold_manifest` / `pool_manifest` 把映射裁剪到自身 partition，`build_fold` / `cross_val_candidate` / `refit_champion` 传给 provider 的 records 同步裁剪（E72/E78 加了负例断言）。
3. **[P1] finalize 在驱动进程 import 候选代码（已修）**。`research-src-finalize` 现在只在驱动进程准备池 manifest，拟合在 `python -m research.finalize` 子进程执行（PYTHONPATH 仅工作树 src）；worker 在 import 候选模块**之前**断言候选文件 SHA-256 等于记录值（代码↔SHA 绑定）。
4. **[P2] 复杂度口径改为 AST 节点（已修，v10 冠军判定翻转）**，见下节。
5. **[P2] E71/E74 咬合力不足（已修）**；**[P3] 报告脚本指标标签、解锁台账、数据集 notes** 一并修正。

## 复杂度新口径下 tcm-0004 的重算（判定翻转，如实记录）

旧口径：非空非注释文本行。新口径（`guard.editable_complexity`）：AST 语句+表达式节点计数，防止「压缩成稠密行」与「删了再加等价代码归零」两种玩法。对 v10 冠军 tcm-0004（提交 `09f9421`）与其父提交重算：

| 口径 | 冠军文件 | 父（基线） | growth | 下限（0.001/单位） | 实测增益 |
| --- | --- | --- | --- | --- | --- |
| 旧：文本行 | 108 | 91 | 17 | 0.0170 | +0.0171 → KEEP（贴边通过） |
| 新：AST 节点 | 379 | 301 | **78** | **0.0780** | +0.0171 → **REVERT** |

**判定翻转**：在新口径下 tcm-0004 不会通过复杂度下限，v10 的有效冠军会退回 tcm-0000 基线（0.3424）。三点如实读法：(a) 已归档轮次的 KEEP 判定按当时的行口径记录，不改写；(b) 集成机制本身的效果有两次独立复现（+0.0181 / +0.0171，均过噪声门槛），**机制有效**与**17 行版是否付得起复杂度**是两个问题，新口径对后者的回答是否；(c) v10 交付模型（holdout 0.3724）与新口径下第 2 次也是最后一次解锁事件均已发生，是否需要在新口径下重选冠军属于用户的决策，不由本次修正代办。未来轮次一律按节点口径判定。

## 其它修正

- **[P3] holdout 解锁机械台账**：`research-src-finalize` 现在向 `<out>/holdout-unlock-ledger.jsonl` 追加解锁事件（时间、champion commit、候选 SHA、池/父 manifest 哈希），计数达 2 拒绝第三次（AGENTS.md 规则 11）。v10 的第一次解锁发生在台账机制存在之前，由本报告与治理文档记录，不追补台账条目。
- **[P3] 报告脚本指标标签**：`scripts/tcm_real_report.py` 原把选择指标硬编码为 macro-F1（选择指标早已改为 macro AP）；现全部从 `runtime.tcm_eval.PRIMARY_METRIC / PRIMARY_DIRECTION` 渲染（E79 防回归）。
- **[P3] 数据集 notes**：`DATASET_NCMI_PULSE` 补充「label_source 是元数据级声称，行级标签层待下载审计」，不改 eligibility。

---

# 2026-11-02：第二轮隔离修复（结构性决定：候选代码与评分彻底分离）

零上下文复审在第一轮修复之后仍找到九条可绕过路径（F-01…F-09）。本轮把一条结构性决定写死在进程拓扑里，三条核心落地位置如下：

1. **候选代码只在一次性训练子进程执行。** `research.train_worker`（新增）是唯一 import `research.candidate` 的模块：一次性、环境白名单（PATH/HOME/PYTHONPATH）、输入裁剪（fit manifest 无 dev 无 holdout、records 裁剪到训练行、config 里没有全集数据路径）。runner 从此只做编排，逐折 spawn train_worker 与 evaluator，自己不 import 候选（F-02/F-08）；finalize 的拟合同样只产出权重，holdout 评分移交给评估器子进程（F-03）；候选可见的 `FoldData.dev = None`（F-06）。
2. **评估只信 state_dict。** `research.evaluator` 不再 `torch.load` 任何可执行对象：训练子进程经 `research/model_export.py` 把模型导出为 state_dict（`weights_only=True` 可加载）+ 用 `torch.fx` 追踪、按冻结词汇表（模块类/函数/方法白名单，越界即拒绝导出）序列化的架构描述；评估器用冻结代码重建架构后 `strict=True` 载入权重（F-01）。audit hook 保留为纵深防御。影响面：已交付 v10 bundle 的 `champion/weights.pt` 只有权重、没有架构描述文件，如需用新评估路径重评分须先补一次导出——已归档数值与本报告不改写。
3. **扣费前置。** 解锁事件是「读权限被授予」，不是「读成功」：驱动在 spawn 拟合子进程**之前**向全局台账（`<仓库根>/runs/holdout-unlock-ledger.jsonl`，经 git common-dir 解析，不再随 `--out` 漂移）以 `fcntl.flock` 追加 intent 事件；worker 在 import 候选前校验预算未耗尽且本事件已在账（手跑 `python -m research.finalize` 不能绕过）；worker 崩溃照常计费（F-04）。

其余各条：`pool_manifest` 的 holdout 列表清空并连同行映射一起裁剪（需要 holdout 的评分路径只从冻结父 manifest 重建视图，`data.real_split.holdout_view`，F-05）；guard 对 `exec`/`eval`/`compile`/`__import__` 的裸名调用零容忍拒绝、复杂度对字符串常量递归 `ast.parse` 计数（精炼简化分支只在真实 AST 删除下生效，F-07）；worker 断言工作树 HEAD 等于记录 commit 且 `git status --porcelain` 除可编辑文件外为空，记录补 `worktree_commit`（F-09）。伴随契约变更：复杂度指标名改为 `ast_statement_expr_nodes+string_payloads`（字符串负载计入），与 v10 归档的纯节点口径不可直接比较。
