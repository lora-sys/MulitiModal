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
| 评估器 | 在候选进程之外的 runner 中运行，候选看不到分数 | `research/runner.py` |
| 选择分辨率 | 4 折 × 2 重复 = **8 折**，SEM ≈ 0.016 | `loop.state()` |
| KEEP 门槛 | 增益 > **1×SEM**，且 ≥ **0.001 × 净增行数** | `loop.decide` |

`20s` 而不是上游的 300s：本模型 ~6k 参数、每折 174 行，必须保持可交互；这是**声明的**契约，改动会同时改变所有已归档轮次的含义。

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
- 候选进程内可读工作树内的任何文件，包括 `data/local/` 下的原始工作簿。理论上一个候选可以把 `batch.subject_ids` 与工作簿行做关联、直接构造标签来刷分。这与上游同源的残留风险（其 `train.py` 也能读被 pin 的验证分片）**同等级**，当前防线是 guard 静态检查 + 评估器进程外运行，状态记为 `AGENT_ISOLATION_UNVERIFIED`。
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
| **tcm-0004** | **同样的 3 成员集成，13 行实现** | **KEEP** | **0.3595** | 17 | **+0.0171 > 门槛 0.0131，且 ≥ 17 行下限 0.0170** |
| tcm-0005 | 集成 5 成员 | REVERT | 0.3610 | 15 | +0.0015，未过门槛 |
| tcm-0006 | 集成 3 + weight decay | REVERT | 0.3573 | 15 | −0.0022 |
| tcm-0007 | 集成 3 + dropout + weight decay | REVERT | **0.3648** | 16 | +0.0053，未过门槛 0.0153 |

**两件必须一起读的事**：

1. **复杂度判据真的起作用了**：tcm-0001 的增益（+0.0181）已经过了噪声门槛，被拒的唯一原因是 27 行没换来 0.027。下一轮把同一机制写成 13 行（tcm-0004）就通过了。这就是上游那句"a 0.001 improvement that adds 20 lines of hacky code? Probably not worth it"要的效果——**逼出更紧凑的实现，而不是否决想法**。
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
2. **集成是唯一被证明有效的代码级杠杆**（+0.017~0.018，两次独立复现：27 行版与 13 行版）。
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
