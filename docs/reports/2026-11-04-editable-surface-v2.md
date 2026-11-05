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

---

## 落位核对（2026-11-05，只读实测 @ `43fa282`）

本节由仓库卫生审计追加，**不修改上文任何授权文字与红线声明**，只登记冻结清单在今天的实际落位，供后续清理与自动代理判定。原授权本身仍是历史事实。

| 状态 | 条目 | 实测 |
| --- | --- | --- |
| 仍冻结且存在 | `AGENTS.md`、`README.md`、`pyproject.toml`、`docs/**`、`skills/**`、`tests/**`、`scripts/**`、`src/data/**`、`src/labels/**`、`src/evidence/**`、`src/syndrome/labels.py`、`src/syndrome/metrics.py` | 12 项均有 tracked 文件且内容仍为当前实现，冻结仍然成立 |
| 仍冻结且存在，但**无活交付路径到达** | `src/runtime/tcm_eval.py`、`src/runtime/real_data.py` | 「评估器运行时」行里唯二存活项。`real_data.py` 的唯一入口是同属休眠簇的 `src/runtime/protocols.py:107 → _train_real()`（惰性 `from .real_data import`），除 `tests/runtime/test_real_data.py` 外无调用方；`tcm_eval.py` 在 `src` 内的引用面只有 `runtime/__init__.py:21` 与同簇的 `runtime/real_data.py:26`，其余均为 `tests/runtime/*`。**二者必须与休眠簇一体处置，单独删 `real_data.py` 会打断 `protocols.train_candidate`。**删除属治理裁决，不得当作卫生清理 |
| 已删除，冻结对象消失 | `src/research/` 下 21 项：`frozen.py`、`task_sd.py`、`task_toy.py`、`evaluator.py`、`model_export.py`、`guard.py`、`runner.py`、`spawn.py`、`train_worker.py`、`worker_supervisor.py`、`artifact_validation.py`、`doctor.py`、`finalize.py`、`loop.py`、`tasks.py`、`session.py`、`events.py`、`report.py`、`cli.py`、`agent.py`、`source_preflight.py` | 全部由 `5f543c4 refactor(research): replace engine with single-file experiments` 删除（单文件实验制落地）；`src/research/vendor/**` 亦在同一提交删除 |
| 已删除，冻结对象消失 | `tailnet.py`（`37f4d11`）、`src/runtime/baselines.py`、`src/runtime/manifest.py`、lockfile `uv.lock`（`3558e56`） | 同上，按类别分属两次清理 |
| 仍在库，**不得按本行判定为已删** | `src/research/dashboard.py` | 同表裸名 `dashboard.py` 指当时的旧引擎渲染入口，而该路径今天仍是活交付（`solve.sh` 与 SKILL 的只读看板）。上表 23 个裸名里 22 个已消失，仅此一个仍存在 |
| 按约定从未入库 | `RTK.md`、`.github/**`、`runs/**`、`artifacts/**`、`models/**`、`data/**` | `runs/`、`artifacts/`、`data/raw|local/` 由 `.gitignore` 排除，`git ls-files data` 为 0（本地目录存在但无 tracked 文件）；`RTK.md` 与 `.github/` 在全部历史中 0 次提交。这些是排除项，不是死代码 |

**两处会误导后续代理的歧义，必须按本节读法理解上文：**

1. 「可信评估与裁决」行使用**裸文件名**列出 `dashboard.py` 与 `cli.py`，指当时的 `src/research/dashboard.py`、`src/research/cli.py`。其中 `src/research/cli.py` 已随 `5f543c4` 删除，而 **`src/research/dashboard.py` 至今存在且是活交付**（`README.md` 与 `skills/autoresearch/` 的只读看板入口）。同名裸名条目不构成对该活文件的冻结或已删声明。
2. 同一行括注的「除 `dashboard --status` 只读渲染入口」所指旗标已不存在：现实现只接受 `--run`（必填）与 `--port`（`src/research/dashboard.py:808-809`）。

**对本授权效力的结论：** 本文件设定的冻结面主要落在 `src/research/` 的旧引擎实现上，该实现整体已由 `5f543c4` 退役；因此 F-01…F-09 / H-01…H-08 隔离结论所依赖的载体（`frozen.py`/`guard.py`/`evaluator.py`/`loop.py` 与 Docker worker 链）已不存在，「不变的红线」段落引用的 `AGENTS.md` 规则 1–8、10、11 编号在当前 `AGENTS.md`（现为 3 节，编号 1–4 / 1–3 / 1–3）中同样不再对应。红线**内容**依然全部有效，但**编号与文件级映射已失效**。后续若需重新签发 editable surface，应按当前模块重新列举，不得复用本表路径。
