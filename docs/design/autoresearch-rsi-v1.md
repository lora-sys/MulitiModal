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
