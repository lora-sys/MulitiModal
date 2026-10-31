# AutoResearch RSI V1 设计（TCM Profile 先行）— #44 调研报告

状态：**设计冻结稿**，直接指导 #45 的实现与 #50 的真实训练验收。
日期：2026-10-30。本报告不启动任何真实长期自动训练（#45 才实现 harness，#50 才接真实数据）。

## 1. RSI V1 的定义

**有限轮次自动代码/超参迭代 + 实测得分 KEEP/REVERT**：

- Coding Agent 在白名单内修改 `train`/`model`/`config`；每轮产出受限 patch。
- 每轮固定预算训练（TCM 合成 smoke 先行，真实数据在 #50）。
- **可信 Evaluator 在 Agent 不可写的隔离环境**独立打分；Agent 自报分数一律不采信。
- 分数优于当前 best → KEEP（advance 分支）；否则 REVERT（reset）。
- 每轮实验记录持久化（RunManifest），可恢复、可审计。
- Agent 永远不能修改：标签、规则引擎、evaluator、split、holdout、标准版本、数据文件。

不是：模型自我改写标准、无界自我修改、把开发集持续优化分当作独立科研验证（#22/#47）。

## 2. 上游核查证据（2026-10-30 实查，非转抄）

| 证据项 | 结果 |
| --- | --- |
| 上游仓库 | https://github.com/karpathy/autoresearch |
| master pinned commit | `228791fb499afffb54b46200aca536f79142f117`（2026-03-26，"Merge pull request #342"；与 #44 记录的 2026-10-29 观察 SHA 一致） |
| GitHub API `license` 字段 | `null`（`gh api repos/karpathy/autoresearch` 实测） |
| 根目录 LICENSE 文件 | **不存在**（contents API 列表：.gitignore, .python-version, README.md, analysis.ipynb, prepare.py, program.md, progress.png, pyproject.toml, train.py, uv.lock） |
| README 许可声明 | README.md 第 90–92 行：`## License` / `MIT`（原文实取） |
| 维护者说明 | Issue #523（"Question: would you consider adding a LICENSE file?"，2026-04-18，closed）：维护者 svlandeg 回复 *"the readme states it's MIT, so feel free to use it as such. No need for a PR"*，并援引 #38 |

**许可结论与复用政策**：MIT 有效性由 README 声明 + 维护者 #523 评论背书，但仓库缺独立 LICENSE 文件、API 元数据为 null。因此本项目政策：

1. **优先独立实现**通用循环（本设计的 `runspec`/harness 全部原创），不从上游逐字复制文件。
2. 若确需借鉴上游具体代码片段，必须在文件头记录：上游 SHA、来源路径、MIT 声明出处（README 行号）与 #523 链接。
3. #45 代码与 RunManifest 记录 `upstream_ref_sha: 228791f…`，仅作设计出处声明，非运行依赖。

## 3. 上游 → 本项目组件映射与权限矩阵

上游三文件：`prepare.py`（只读：数据、tokenizer、dataloader、评估）、`train.py`（Agent 唯一可改）、`program.md`（人类迭代的 agent 指令）。上游指标 `val_bpb`、5 分钟 H100 预算**不适用**本项目，抽象为独立可信指标接口（#40 已提供 `TCMTrustedEvaluator`）。

| 上游组件 | 本项目对应 | Agent 可改 | 冻结依据 |
| --- | --- | --- | --- |
| `prepare.py`（数据/评估只读） | `SyntheticBatchProvider` / 真实数据契约（#37）、标签（#38）、`EvalManifest`、split | **否** | #45 硬规则 9；#27/#29；#40 冻结 manifest |
| `train.py`（唯一可改） | 模型构建 kwargs + `TrainerConfig`（epochs/lr/grad_clip/…）+ `TCMConstitutionModel` 架构超参 | **是（白名单内）** | #45；patch 静态检查拒绝越界 |
| `program.md`（人类迭代） | `docs/design/autoresearch-rsi-v1.md` + #45 的 harness 配置 | **否（本阶段）** | Agent 不改自己的指令 |
| 上游 evaluator（evaluate_bpb/val_bpb） | `TCMTrustedEvaluator`（主指标 `category_probs/subject_rmse`，proper scoring rule）+ `trusted_score()`，隔离运行 | **否** | #40；#47 |
| 上游 git 分支 advance/reset | `KEEP`→保留 artifact+config；`REVERT`→丢弃，回到 best | 由 harness 执行 | RunManifest 记录 |
| 上游 run.log / progress 表 | `RunManifest`（JSONL，见 §6） | **否**（append-only，harness 写） | 审计要求 |

**不可变集合**（Agent 不可修改、不可读取 holdout 标签）：标签/计分规则（#38）、官方标准版本、真实数据文件（#37）、分组与 holdout（#27/#40 manifest）、evaluator 代码与指标版本。

## 4. Run spec（两任务 Profile）

所有字段入 RunManifest；`*_hash` 在 harness 启动时计算并冻结。

```yaml
# runspec.tcm.synthetic.yaml —— #45 首个 smoke 用
task_id: tcm_constitution
profile: TCM_V1
frozen:
  dataset_hash: synthetic-tensor-v0        # 真实阶段由 #37 提供
  label_manifest_hash: synthetic-labels-v0 # 真实阶段由 #38 提供
  standard_version: SYNTHETIC-CONTRACT-V0
  split_hash: manifest:synthetic-manifest-v0   # EvalManifest.version + 内容 hash
  evaluator: runtime.tcm_eval.TCMTrustedEvaluator
  metric_version: tcm-eval-v0
  primary_metric: category_probs/subject_rmse
seed: 0
budget:
  max_candidates: 3            # V1 默认 3 candidates/profile
  per_run_timeout_s: 300       # CPU smoke 级别；非上游 H100 数值
  max_wall_clock_s: 1800
editable_scope:                 # patch 只允许触碰这些键
  - model.hidden_dim
  - model.fusion_dim
  - model.include_score_head
  - trainer.epochs
  - trainer.learning_rate
  - trainer.grad_clip
  - trainer.batch_size          # 经 train_provider.batch_size
model_artifact: null            # 首轮由 train_candidate 生成
```

```yaml
# runspec.physio.synthetic.yaml —— 预留，生理数据到位前不运行
task_id: physio_bpsi_efri
profile: PHYSIO_V0              # Pre-only 契约（physio-input-pre-only-v0）
frozen:
  dataset_hash: NOT_AVAILABLE   # 证据状态显式化（#22/#47）
  label_manifest_hash: NOT_AVAILABLE
  standard_version: NOT_AVAILABLE
  split_hash: NOT_AVAILABLE
  evaluator: runtime.evaluation.Evaluator
  metric_version: masked-metrics-v0
  primary_metric: efri/mse      # BPSI 仅辅助头（#41/#43）
seed: 0
budget: {max_candidates: 3, per_run_timeout_s: 300, max_wall_clock_s: 1800}
editable_scope: [trainer.epochs, trainer.learning_rate, model.hidden_dim]
model_artifact: null
```

Physio Profile 仅为占位扩展：`NOT_AVAILABLE` 是显式证据状态，生理数据到位前 #45 不得运行该 profile（与 #23–#31 全线 Blocked 一致）。

## 5. Agent 最小接口（每轮循环）

```
propose(intent)          # Agent 声明本轮假设（自然语言 + 目标键）
  → make_patch()         # 生成受限 patch：仅 editable_scope 内的 config/代码变更
  → static_checks()      # ① patch 触碰冻结路径 → 拒绝；② 禁用词扫描（label/mask/eval/
                         #    split/holdout/standard 等符号不得出现在 diff）；③ 可运行性 dry-run
  → train_candidate()    # #49 已预留的稳定入口；失败/超时 → REVERT-as-crash
  → trusted_score()      # 隔离进程/容器内对 artifact 打分；Agent 进程无写权限
  → compare(best)        # 主指标 subject_rmse 严格更小才 KEEP；平/差/崩溃 → REVERT
  → record(RunManifest)  # append-only 落盘
  → keep_or_revert()     # harness 移动 artifact 指针 / 丢弃
```

静态审查示例（#44 验收要求）：diff 中出现 `EvalManifest(`、`target_mask` 赋值、`SyntheticBatchProvider` 的 target 构造、evaluator 模块路径等任意修改 → patch 直接拒绝，不进入训练。

## 6. RunManifest（持久化与恢复）

JSONL，每轮一行，append-only，harness 拥有写权限（Agent 目录只读）：

```json
{"run_id": "tcm-0001", "parent_run": null, "decision": "KEEP",
 "intent": "wider fusion dim", "patch_summary": {"model.fusion_dim": 128},
 "artifact_dir": "runs/autoresearch/tcm-0001", "artifact_sha256": "…",
 "primary_metric": {"name": "category_probs/subject_rmse", "value": 0.4931},
 "best_so_far": 0.4931, "seed": 0, "budget_used_s": 41.2,
 "status": "completed", "smoke_only": true,
 "frozen_hashes": {"split_hash": "…", "evaluator": "runtime.tcm_eval.TCMTrustedEvaluator",
                    "metric_version": "tcm-eval-v0", "upstream_ref_sha": "228791fb…"},
 "source_commit": "<git sha>", "torch_version": "…"}
```

- **失败/超预算/异常退出**：该轮记 `status: crash|timeout`，`primary_metric.value: null`，一律 REVERT（对齐上游 "0.000000 for crashes" 惯例的更严格版本：crash 不计分）。
- **断点恢复**：harness 重启时读最后一行 RunManifest → 恢复 best 指针与剩余预算；artifact 由 `artifact_sha256` 校验（复用 #48 checksum 机制）。
- **回退**：REVERT 即回 best artifact；`decision` 全程留痕，可离线重放。
- **产物追踪**：checkpoint 目录（#49 格式）+ RunManifest 行一一对应；`runs/` 不入 Git（.gitignore 已覆盖）。

## 7. 运行环境

CPU 为最小可用环境（#49/#40 全部 CPU smoke 已验证）；MPS/CUDA 通过 `TrainerConfig.device` 可选。预算参数（轮次/超时/内存上限）全部在 runspec 中，不硬编码。**不要求 H100**；上游的 5 分钟/val_bpb 仅是设计灵感来源。

## 8. 旧 PR #20/#21 处置

两 PR（"bounded offline physiological research pipeline"、"streamline round validation and guard physiological inputs"）已关闭未合并，仅作为设计思路参考（其 round-validation 与生理输入防护思想已并入 §5 静态检查）；其历史测试**不代表当前能力**，不合并、不复用代码。

## 9. V1 最小可运行验收用例（交 #45 实现）

1. `train_candidate` 从 TCM runspec（合成）产出 artifact，`trusted_score` 返回与 manifest 一致的 versioned 指标（已具雏形：#49 protocols 测试）。
2. Harness 顺序跑 3 个 candidate：KEEP/REVERT 判定正确（构造一个必然更差、一个必然更好的 patch）。
3. 越界 patch（改 split/evaluator/标签）被静态检查拒绝且不产生训练（负例测试）。
4. Agent 自报分数与 trusted_score 不一致时以 trusted_score 为准（负例测试）。
5. harness 中途 kill 后重启：best 指针与剩余预算恢复，artifact checksum 通过。
6. 全程产物：RunManifest JSONL + 每 KEEP 轮的 #49 checkpoint；所有输出 SMOKE_ONLY 标记。
7. Physio runspec 提交但标记不可运行（NOT_AVAILABLE 校验触发即报错）。

## 10. 未决事项（不阻塞 #45 开工）

- 真实数据的 `dataset_hash`/`label_manifest_hash` 语义由 #37/#38 落地后回填 TCM Profile。
- Trusted Evaluator 的 OS 级隔离具体形态（容器 vs 独立用户进程）在 #45 实现时定，按硬规则 9 执行。
- 上游 #38（维护者引用的另一 issue）内容未核查；仅记录 #523 证据即可满足本报告许可要求。

---

## 11. 冻结修订 2026-10-31：选择指标与选择分区（v8 落地）

本节修正 §3/§5 中已被实测否决的两处设计，**不改动 §1 的权限矩阵与不可变集合**。修订依据全部来自 v5–v8 的真实运行记录（`docs/reports/tcm-50-real-training.md`、`runs/real-tcm-v5..v8`）。

### 11.1 主指标：`subject_rmse` → `category_probs/macro_f1`

| 候选指标 | 实测问题 | 结论 |
| --- | --- | --- |
| `category_probs/subject_mae` | L1 损失下二值标签的最优常数预测是中位数；本数据正类率 < 0.5 → 最优常数 = 0，按它爬山永远赢不过全零预测器（v5 实测：Dummy 0.2292 在 holdout 领先全部候选） | 否决 |
| `category_probs/subject_rmse` | Brier 的单调根变换，proper scoring rule，**数学上没错**，但对没有可用信号的模型其最优解是预测基率常数；v6 实测该轮冠军 dev macro-F1 仅 0.021，而普通逻辑回归 holdout 0.308 | 否决 |
| **`category_probs/macro_f1`** | 就是交付物的指标（0.5 阈值、固定观测维分母，未定义 F1 计 0），爬山落点与交付指标一致 | **采用（v7 起）** |

MAE/RMSE 保留为诊断口径，但不得作为 KEEP/REVERT 判据。

### 11.2 选择分区：41 人 dev 单划分 → subject 级 4 折（池 = train + dev）

v7 实测：**同一套权重** dev macro-F1 0.1451、holdout 0.3445。划分间波动远大于候选间差异，单划分无法给候选排序，v7 的 6 轮排序是噪声（v8 复验：v7 的"冠军" lr 1e-3 在 k 折下只有 0.2691，低于基线 0.3117）。

现行规则：

1. 选择分区 = `manifest["split"]["train"] + ["dev"]`（232 人），**holdout 永不进入**（`cross_val_candidate` 只读池）。
2. 折内训练不得接收 eval provider（`early_stop_patience`/`restore_best` 在真实数据路径被拒）——验证折 steering 拟合是同一种双重使用缺陷。
3. KEEP 需 `value > best + selection_margin_std × fold_std`（默认 0.5）。余量把"折间噪声"从"改进"里扣掉，避免爬山随机游走。
4. 每轮训练 k 个折模型并**丢弃**：k 折选择不产生可交付工件。

### 11.3 交付模型：池上全量重训（`research-finalize`）

`AutoResearchHarness.finalize()` 按冠军配置在池（train+dev）上重训一次，写入 `champion/`，并在检查点里记录 `selection_parent_manifest_sha256` 与 `refit_partition="train+dev"`——报告端据此验证交付模型确实源自该冻结划分，而不是相信目录名。

`RunManifestStore.champion()` 因此不再要求 `artifact_dir`：k 折轮次没有工件，若仍要求，冠军永远返回 None，爬山会退化成"永远保留第一个候选"。

### 11.4 评分口径：交付模型只在 holdout 上评分

池重训后 dev 已并入训练集，受信评估器的 train/dev 不重叠约束（`EvalManifest.__post_init__`）禁止给交付模型一个 dev 分数。因此报告里：

- 交付模型、Dummy、BMI-only、全特征逻辑回归、Deployable 消融**全部在同一池上拟合、同一 holdout 上评分**；少给基线 41 行再报差距会制造结论。
- row-QC 敏感性两侧同 holdout、同分母。
- 8-score 回归头仍在冻结 train 折上训练（它回答"回归头是否有效"，不是交付模型的一部分），报告里单独标注口径。

### 11.5 v8 实测（24 轮）

| 项 | 值 |
| --- | --- |
| 轮数 | 24（KEEP 1 = 基线，REVERT 23） |
| 覆盖 | lr 5 档 × epochs 5 档 × hidden 4 档 × fusion 4 档 × batch 5 档 × grad_clip 3 档 + 6 组组合 |
| 冠军 | `tcm-0000`（32/32、40 epochs、lr 3e-3、batch 32），选择值 0.3117 ± 0.0352 |
| 最好被否决 | `tcm-0011`（epochs 80 + batch 16）0.3270 ± 0.0439，未过 0.0220 余量 |

结论：**可编辑空间内没有配置能在超过折间噪声的程度上击败基线。** 停止判据 = 每个 scope 维度都被扫过且无 KEEP（本轮在 24 轮达到），不是轮次预算耗尽。

---

## 12. 与上游 karpathy/autoresearch 的核心核对（2026-10-31）

核对方法：`git clone https://github.com/karpathy/autoresearch` 于 `228791f`（与 §2 记录的 SHA 一致），逐文件读 `program.md`(114 行)/`prepare.py`(389 行)/`train.py`(630 行)，再逐条对照本项目实现。**结论放在最前面：上游的核心（Agent 直接改训练源码 + 分支前进）我们没有实现。**

### 12.1 上游的核心是什么

| 要素 | 上游实现 | 位置 |
| --- | --- | --- |
| Agent 改的是**源码** | "Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc." | `program.md` |
| 固定**墙钟训练预算** | `TIME_BUDGET = 300`（5 分钟），训练时长不是可调项 | `prepare.py:31` |
| keep/discard = **git 分支前进** | improved → 保留 commit（advance），equal/worse → `git reset`；产物是**代码** | `program.md` 实验循环第 3/8/9 步 |
| 冻结边界是**一个文件** | `prepare.py`（数据/分词器/dataloader/`evaluate_bpb`/预算常量）；"Modify the evaluation harness" 明令禁止 | `prepare.py:344` `evaluate_bpb` |
| **simplicity 准则是目标的一部分** | "All else being equal, simpler is better… removing something and getting equal or better results is a great outcome" | `program.md` |
| 结果表 | `results.tsv`（commit / val_bpb / memory_gb / status / description） | `program.md` |
| 运行纪律 | "NEVER STOP… do NOT pause to ask the human"；约 12 次/小时 | `program.md` |

### 12.2 我们实现的是什么（逐条对照）

| 上游要素 | 本项目 | 判定 |
| --- | --- | --- |
| Agent 改源码 | `EDITABLE_SECTIONS = ("model","trainer")`（`agent.py:29`）只接受 **6 个配置键**：`model.hidden_dim/fusion_dim`、`trainer.epochs/learning_rate/grad_clip/batch_size`（`harness.py:88-95`） | **未实现** |
| Agent 改源码（机制层） | `Proposal.source_patch` 被 `validate_source_patch` 校验后**从不落盘、从不应用**（`harness.py:341-343`，`config` 只由 `apply_config_patch` 生成）；`ExternalAgentProvider.propose` 直接 `raise NotImplementedError`（`agent.py:184-190`） | **硬禁用** |
| 固定墙钟预算 | 无训练时间预算；`per_run_timeout_s` 是超时上限，`max_wall_clock_s` 是整个循环的墙钟上限（`harness.py:86`）。训练时长由 `trainer.epochs` 决定，而它在白名单里 | **不存在（且相反）** |
| keep/discard = git 分支前进 | append-only `RunManifestStore`（JSONL + 哈希链），交付物是按冠军配置**重训的检查点**；仓库内无任何 `git commit/reset/checkout` 调用 | **不同机制，产物不同** |
| 冻结边界是一组符号 | `FORBIDDEN_SOURCE_SYMBOLS` 静态检查 + 子进程 sanitized env + 指标版本 (`tcm-eval-v0`) | **更强，但见 12.3** |
| simplicity 准则 | 全仓无对应机制（`grep -rni "simplicity\|complexity" src/ docs/` 无结果） | **未实现** |
| NEVER STOP | 有轮次预算；v8 的停止判据由驱动方事后声明（§11.5） | **弱化** |

### 12.3 我们比上游强的地方（也要说清）

上游的 Agent 直接编辑 `train.py`，而 `val_bpb` 是 `train.py` 自己打印的（`train.py:613,622`）——**Agent 可以改掉打印那一行**，上游对此的防线只有 `program.md` 的禁令。本项目把评估放在 Agent 之外的子进程、指标版本冻结、结果进哈希链 manifest，评估完整性明显更强。

所以这不是"我们做得差"，而是一次**用研究自由换评估完整性**的交易。问题是：这笔交易把 autoresearch 之所以是 autoresearch 的那一步换掉了。

### 12.4 后果：这解释了 v8 为什么"找不到任何改进"

白名单里的 6 个键是**超参数值**。在 232 行、8 个二值标签的表格数据上，这些超参数的有效效应量小于折间噪声（§11.2），因此 24 轮全部 REVERT 是必然结果，不是运气。而真正的杠杆——网络结构、损失函数、特征组选择、标签切点处理——**一个都不在可编辑面内**：

- 标签切点是 `>= 40` 的硬规则（`derived-multilabel-v1`），而 CCMQ 分数粒度使各维有效切点漂移（6 题维 ≈41.7、7 题 ≈42.9、8 题 ≈40.6），模型无法自行补偿；
- 特征分组（经络/脉搏波/舌诊/体格）的选择能力为零；
- 损失函数、池化/融合结构、类别不平衡处理全部冻结。

**因此"RSI 找不到东西"不是模型的结论，是搜索空间为空的结论。** 现有 v8 报告只应被读作：*在给定的 6 维超参空间内无可检测改进*。

### 12.5 要真正实现上游核心，需要动的边界（需项目负责人批准，Agent 不得自行改）

以下每一条都触及 AGENTS.md 硬规则 9（Agent 只能改白名单内 train/model/config）。按预期信息量排序：

| # | 动作 | 触及的硬规则 | 需要的保护 |
| --- | --- | --- | --- |
| 1 | 允许 Agent 提交**源码 patch**（先限定在一个"可改文件"内，如 `src/core/models.py` + `src/core/losses.py`），由 harness 应用到一个**隔离的工作副本**再训练 | 硬规则 9 的"白名单"从配置键扩展为文件 | 评估器/标签/split/holdout 继续在 Agent 不可写的路径；现有 `FORBIDDEN_SOURCE_SYMBOLS` + 子进程 env 白名单保留；多一层"patch 只落在声明文件"的校验 |
| 2 | 引入**固定墙钟训练预算**（如每轮 120s），把训练时长从可调项变成常量 | 无（收紧） | 同时移除 `trainer.epochs` 白名单键，否则"训练更久"仍是后门 |
| 3 | 加入 **simplicity 判据**：patch 增大代码量时，收益须过更高余量；等效且更简的改动直接 KEEP | 无（新增目标项） | 需要定义可机械计算的规模指标（行数/diff 行数），并在报告里写清权重 |
| 4 | 让**分支前进**成为记录机制（或在 manifest 之外额外记录每轮源码 diff 的哈希），使交付物包含研究产出的代码 | 无（新增记录） | 与现有 manifest 并存，不替换（哈希链的防篡改证据层要保留） |
| 5 | 逐轮**先读上下文再提案**，并把 §11.5 的时间戳自检作为硬约束（当前 22/24 轮为批量预写） | 无（收紧溯源） | 已有 `proposal_predates_context` 字段，只需把它从"记录"升级为"拒绝" |

**在这 5 条落地之前，本项目的 `AutoResearchHarness` 应被如实称为"带溯源的受限超参搜索"，而不是 RSI/autoresearch。** 文档、README 与 issue 措辞需要同步改，否则就是把一个空搜索空间的结论包装成研究结论。
