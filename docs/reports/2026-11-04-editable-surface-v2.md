# 2026-11-04 授权：AutoResearch editable surface v2（denylist）

日期：2026-11-04。授权人：项目负责人（本会话明确指令「不要像那个一样，可以修改全部」「死代码，屎山，直接就可以删除」「自己推进一下，full-send」）。

## 授权范围

本小节按 `AGENTS.md` 第 3 节「单条 issue 与本文件冲突时，以该 issue 最新的冻结小节为准」，作为一次**限定的** editable surface 变更授权。它只放宽「可编辑文件集合」，不触碰任何科研诚信红线。

### 允许（denylist 之外）

研究 Agent 可以修改 `src/**/*.py` 中**除下方冻结清单外**的任意源码，并可以新增 `.py` 模块：
模型结构、损失函数、优化器、训练循环、特征工程、数据预处理代码、批处理、正则化、以及新的候选模块文件。
「一次只改一个文件」「只能改 `candidate*.py`」「只能改超参」这三道**实现选择**被移除。

### 冻结（denylist，任何选择/比较轮次都不可写）

| 类别 | 路径 |
| --- | --- |
| 治理与证据 | `AGENTS.md`、`RTK.md`、`README.md`、`pyproject.toml`、`docs/**`、`skills/**`、`tests/**`、`scripts/**`、`.github/**`、lockfile |
| 数据与工件 | `data/**`、`runs/**`、`artifacts/**`、`models/**`、`src/research/vendor/**` |
| 标签与标准 | `src/data/**`、`src/labels/**`、`src/evidence/**`、`src/syndrome/labels.py`、`src/syndrome/metrics.py` |
| 划分与折内拟合 | `src/research/frozen.py`、`src/research/task_sd.py`、`src/research/task_toy.py` |
| 可信评估与裁决 | `src/research/evaluator.py`、`model_export.py`、`guard.py`、`runner.py`、`spawn.py`、`train_worker.py`、`worker_supervisor.py`、`artifact_validation.py`、`doctor.py`、`finalize.py`、`loop.py`、`tasks.py`、`session.py`、`events.py`、`report.py`、`cli.py`、`agent.py`、`dashboard.py`、`source_preflight.py`、`tailnet.py` |
| 评估器运行时 | `src/runtime/tcm_eval.py`、`real_data.py`、`baselines.py`、`manifest.py` |

### 不变的红线（`AGENTS.md` 规则 1–8、10、11）

Pre-only 推理、subject-level 划分、折内拟合、数据边界、标签分层、指标语义、证据状态显式化、三层放行、Provenance、holdout 解锁预算——**全部不变**。
`Trusted Evaluator 在 Agent 不可写的隔离环境运行`（规则 9）由两件事保证：可信模块从**受信源根**加载（不再从 Agent 可写 worktree 的 `src` 加载），候选代码继续在 Docker OS 级隔离内执行。

### 明确未在本次授权内（下一阶段，需另行签发）

把候选产物契约从「`state_dict` + `torch.fx` 白名单架构」换成「框架无关的 `fit`/`predict_proba` + 纯数值预测产物」，从而支持 TF-IDF+LR / sklearn 等非 `nn.Module` 算法范式。
这需要重写可信评估链与相应回归测试，并单独重证 F-01…F-09 / H-01…H-08 隔离结论，故不并入本次「先跑通端到端」的批次。当前版本允许 Agent 改任意建模源码，但训练产物仍须能被冻结架构白名单导出。

### Holdout

Phase 1 解锁预算 2/2 + SD 1/1 已用满。本次真实多轮仅使用 toy 合成数据（显式 `SMOKE_ONLY`）的 dev/CV 选择池，不读取、不解锁任何 holdout；结果只能表述为 `GO_ENGINEERING` 工程有效，不构成科研或临床结论。
