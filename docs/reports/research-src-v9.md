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