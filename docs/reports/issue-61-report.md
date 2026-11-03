# #61 交付报告：独立 AutoResearch 模块、Research Doctor、Skill 与轻量 Dashboard

分支 `feat/issue-61-autoresearch-module`（PR 已开，**未合并**）。范围与验收依据：Issue #61 全文。
本报告中的每个数字都来自实际运行；未运行的项目标 `NOT_AVAILABLE`。

## 1. 修改了哪些文件，为什么

**新增（`src/research/`）**

| 文件 | 为什么 |
| --- | --- |
| `events.py` | 可信宿主**单写入者**事件流：`events.jsonl`（append-only）+ `status.json`（原子替换）。单写入者用 `fcntl.flock` 强制；撕裂尾行由读者丢弃、由下一个写者补换行；消息/指标有上限，一条失控轮次无法撑爆日志。 |
| `session.py` | 运行标识与持久化状态：`run.json`（冻结 spec，只写一次）、`STOP` 标记、活动训练子进程登记（供跨进程 `stop` 结束它）、`status/history/report` 只读视图。完成轮次的唯一权威是 `manifest.jsonl`，所以 `resume` = 从「已完成轮次数」继续，不需要第二套账本。 |
| `doctor.py` | Research Doctor：preflight（用户跨折、精确文本泄漏、标签冲突、类别映射顺序、外层 dev 是否参与选择、预算/绑定）+ postflight（失败归因、指标存在性、预算利用率、loss 趋势/NaN、测量分辨率、逐类错误、资源遥测、导出边界）。分级 `BLOCKED/WARNING/INFO/UNVERIFIED`；**不做**自动改标签/split/evaluator；不读 holdout。 |
| `agent.py` | Proposer 协议与三个实现：真实 Coding Agent（子进程、stdin 提示、JSON 校验、上下文白名单环境）、`DeterministicStubProposer`（**显式非 agent**）、`UnavailableAgent`（无 agent 时 fail closed）。 |
| `cli.py` | 唯一正式入口：`doctor/run/status/stop/resume/step/report/dashboard`。自动创建 `autoresearch/*` 专属 worktree；真实任务缺 manifest/dataset 时 fail closed。 |
| `dashboard.py` | 只读实时看板：stdlib HTTP + 单页原生 HTML/CSS/JS + 原生 SVG + 3s 轮询。路由白名单、默认 127.0.0.1、响应上限、`textContent` 渲染（无 `innerHTML`）、无文件服务。可独立进程启动（`spawn_dashboard`）。 |
| `report.py` | 确定性报告：只读持久化状态，工程拒绝与模型结果分列，缺项写 `NOT_AVAILABLE`。 |
| `task_toy.py` + `candidate_toy.py` | **与 TCM 无关**的示例任务：自生成合成数据（短文本 + 数值，3 类，用户级划分），自带记录/清单/scaler/provider/可信评估器，证明引擎的任务接缝是真的接缝。 |
| `skills/autoresearch/{SKILL.md,program.md}` | 独立 Skill：只指导 Coding Agent 调用 CLI，不含业务实现。 |

**修改（最小面）**

| 文件 | 改动 | 为什么 |
| --- | --- | --- |
| `src/research/loop.py` | 修复 `__init__` 重复参数 `editable`（**SyntaxError，基线分支上模块根本无法导入**）；新增 `research_state()`、`run_autonomous()`、`_postflight()`、`_next_action()`、`outcome` 分类、事件回调、`GitWorktree.ensure_identity()`、`fold 级进度` 流式读取 | 自主闭环、工程拒绝与模型结果分离、新 worktree 能确定性提交、看板能看到训练中进度 |
| `src/research/session.py` | `history()` 增加 `outcome` | 看板/报告读它，缺了会显示 NOT_AVAILABLE |
| `src/research/tasks.py` | 新增 Doctor hooks（可选）与 `toy` 适配器注册 | 通用核心不得硬编码任何任务规则 |
| `src/research/task_sd.py` | 实现 syndrome 的 Doctor hooks | preflight 需要真实用户键/文本/标签空间 |
| `src/research/runner.py` | 每折输出一行 progress | 折级进度必须由**协调器**转成事件，不能由候选写 |
| `pyproject.toml` | dashboard 的 E501 例外 | 页面是整段 HTML/CSS/JS 字符串，按 100 列折行不改善任何东西 |

## 2. 现在的实际架构

```
autoresearch <cmd>  (research/cli.py —— 唯一正式入口)
   │
   ├─ doctor    : research/doctor.py        （只读；BLOCKED 即拒绝训练）
   ├─ run/step  : research/loop.py ─┬─ guard.py        单文件白名单 + AST 安全 + 复杂度计价
   │                                ├─ runner.py       每折：训练子进程 → 隔离评估子进程 → 一行进度
   │                                ├─ evaluator.py    可信评分（state_dict + 冻结架构重建）
   │                                ├─ model_export.py fx 词表白名单 → UNSUPPORTED_ARCHITECTURE
   │                                └─ session.py/events.py  状态 + 事件（单写入者）
   ├─ status/stop/resume/report
   └─ dashboard : research/dashboard.py     只读、localhost、3s 轮询、原生 SVG
                    tasks.py  ── constitution / syndrome / toy（示例）
                    frozen.py ── 数据·划分·scaler·预算 冻结边界
```

复用了既有 Docker/子进程隔离、Trusted Evaluator、git worktree、Manifest、finalize/holdout 台账；**没有**第二套 Harness、数据库、前端框架或遥测平台。

## 3. 如何安装和使用 Skill

```bash
# 复制到任意支持 Skill 的 Coding Agent 的 skills 目录（本仓库不自动启用）
cp -r skills/autoresearch ~/.claude/skills/          # 或该 Agent 约定的目录
```
Skill 只教 Agent 调用 `python -m research.cli ...`；研究逻辑、训练、诊断、存储都在模块内。**未接入 lora-mode，未改任何全局配置**。

## 4. 如何启动 AutoResearch

```bash
export PYTHONPATH=src
# 1) 准备（创建 autoresearch/<name> worktree、物化示例数据、跑 preflight）
python -m research.cli run --task toy --out runs/toy-1 --repo .
# 2) 真实 Agent 驱动（有 agent CLI 时）
python -m research.cli run --task toy --out runs/toy-1 --repo . \
  --agent-command "claude -p" --max-rounds 8 --folds 4 --repeats 2 --budget 20
# 3) 无 agent CLI 时：逐轮驱动（每轮提案写在上轮结果之后）
python -m research.cli status --run runs/toy-1
python -m research.cli step  --run runs/toy-1 --proposal-file p.json --proposer "<id>"
# 4) 停止 / 恢复 / 报告
python -m research.cli stop   --run runs/toy-1
python -m research.cli resume --out runs/toy-1 --task toy
python -m research.cli report --run runs/toy-1
```
`--allow-stub` 只用于工程路径，运行与报告都会标 `autonomous_research: false`。

## 5. 如何打开实时 Dashboard

```bash
python -m research.cli dashboard --run runs/toy-1 --port 8765   # → http://127.0.0.1:8765/
# 或让 run 自己起一个独立进程：run ... --dashboard --port 8765
```
只读、仅 loopback、3s 轮询；显示当前轮/折/阶段、假设与受控变更、可信指标曲线、best-so-far、KEEP/REVERT 历史、Doctor 发现、用时与预算；无心跳超阈值显示 `STALE`；无 loss/GPU 数据时显示 `NOT_AVAILABLE`。关闭浏览器不影响训练（服务与训练是不同进程）。

## 6. 一次真实运行的完整证据

### 6.1 Agent 驱动的 4 轮（`autoresearch step`，每轮提案在**读取上一轮状态之后**写出）

Run ID `toy-20261103T051302Z-1933908` · 分支 `autoresearch/toy-agent` · task `toy` · 3 折 × 1 重复 · 每折固定 5s 预算。

| 轮 | 假设（Agent 原文摘要） | 受控变更 | outcome | 可信指标 | 门槛 | commit |
| --- | --- | --- | --- | --- | --- | --- |
| toy-0000 | the trunk is capacity-limited: width 32 → 64 raises macro-F1 | 只改 WIDTH | **KEEP** | **0.6018** | 基线 | `db1f044` |
| toy-0001 | the tail class drags macro-F1: inverse-frequency class weights… | 只改损失权重 | **INVALID_PROTOCOL** | — | — | — |
| toy-0002 | the trunk is capacity-limited: width 64 → 96 raises macro-F1 | 只改 WIDTH | REVERT | 0.6129 | +0.0112 < 0.0297 | `db1f044` |
| toy-0003 | after two capacity probes, the axis is the step size: lr 3e-3 → 1e-2 | 只改 LEARNING_RATE | REVERT | 0.6193 | +0.0175 < 0.0325 | `db1f044` |

- toy-0001 的补丁把两行替换写成含换行的单行 `+`，`git apply` 报"补丁在第 8 行损坏" → 记为 **INVALID_PROTOCOL**，不是模型劣势；这正是 issue 要求的区分。该轮不进入冠军。
- 每轮记录 `proposer="pi:claude-agent external one-round-per-call"`、`agent_kind="external-coding-agent"`、`batch_supplied=false`；`context_sha256`/`proposal_sha256` 在补齐后已入档（见 6.3）。
- 冠军 = `toy-0000`，best `0.6018`；两轮 REVERT 的 Doctor 都给出 `WARNING measurement_resolution`（差值小于一个标准误）。

### 6.2 端到端验收（真实训练 → 隔离评分 → 决策 → 事件 → 看板 → 报告 → resume）

`/tmp/i61_e2e.py` 在开发机运行，**28/28 通过**（完整输出保存在运行日志）：

```
PASS run completed wall=70.8s              PASS cli status works best=0.6018
PASS three rounds recorded 3               PASS GET / serves the page
PASS stub labelled as NOT autonomous       PASS GET /api/{status,events,history} JSON
PASS at least one KEEP and one REVERT      PASS status exposes hypothesis/progress
PASS champion commit recorded              PASS history has outcomes
PASS every round carries the research trail PASS telemetry = NOT_AVAILABLE (not invented)
PASS rounds are not batch-supplied         PASS path traversal refused (400)
PASS proposer recorded per round           PASS negative after clamped / paging safe
PASS events cover every stage              PASS POST refused (405)
PASS fold-level progress evented: 9 (round,fold) pairs
PASS event sequence monotonic              PASS no raw records in the page
PASS report written / carries the trail    PASS resume with no rounds left adds none (3 -> 3)
```

### 6.3 过程修复（每一条都由运行暴露，不是推测）

1. `loop.py` 重复参数 `editable` = SyntaxError（`ast.parse` 通过、`compile()` 拒绝）→ 基线分支上模块无法导入；修复后 110 个既有 research 测试全部通过。
2. 无尾部上下文的 hunk 隐含"文件到此结束" → 追加类补丁只生效一次；测试 fixture 与 CLI stub 都改为带双侧上下文。
3. 事件缺折级进度 → runner 输出 progress 行、协调器转事件（9 个 (round,fold)）。
4. `history()` 缺 `outcome` → 看板会显示 NOT_AVAILABLE。
5. `stop` 只在轮次边界生效 → 提案完成后、训练开始前再检查一次。
6. 新 worktree 无 git 身份 → 首次 KEEP 提交失败；`ensure_identity()` 本地设定。
7. `--block 0` 被当成"全部剩余" → 准备阶段误跑 3 轮。
8. Doctor 把"冻结 manifest 里存在 holdout"误判为 BLOCKED → 改为只有 holdout 与 train/dev **重叠**才 BLOCKED。
9. 示例数据生成器真的产生了 4 条跨折同文本 → Doctor 抓对；生成器加入唯一 token。
10. `step` 路径未记录 context/proposal 摘要 → 补齐。

## 7. 测试通过数量与未通过项

| 项 | 结果 |
| --- | --- |
| `tests/research`（含 #61 新增 4 个文件） | **128 passed** |
| 全量 `pytest tests` | **420 passed, 19 skipped**（跳过为既有原因：本地 BERT 权重/真实 TCM 数据不入 Git） |
| `ruff check .` | **All checks passed** |
| 项目自审 `scripts/review_eval.py` | **52/52**（无回归） |
| 端到端验收脚本 | **28/28** |
| Doctor 故障注入 | 19 个测试覆盖：跨折用户、同文本泄漏、标签冲突、外层 dev 参与选择、类别映射错序/越界、holdout 重叠、不可信指标、缺预算、长尾类、缺标签、无 hook（UNVERIFIED）、blocked 不进入冠军、NaN loss、失败归因、测量分辨率、逐类弱类 |
| Dashboard 安全 | 恶意文本以数据形式往返（页面无 `innerHTML`）、撕裂 JSONL、超长日志分页、路由白名单、路径遍历 400、POST/PUT/PATCH/DELETE 405、无原始记录/密钥、无事件的历史 run 不伪造实时阶段、STALE 由心跳推出 |

**未通过/未完成项：无（0 failed）**，但见下节的阻塞。

## 8. 尚未解决的风险与明确阻塞

1. **真实 Coding Agent 的 CLI 不在本机**：开发机上没有 `claude`/`codex`/`gemini` 等 CLI，也没有 LLM 客户端库，因此 `--agent-command` 路径**未在本机跑过**（代码路径存在、可配置、有超时与 JSON 校验，但 `NOT_AVAILABLE` 于真实 provider）。本轮真实验收由**操作侧 Coding Agent 逐轮调用 `autoresearch step`** 完成（6.1），这与 DoD 的"每轮根据上轮新证据提假设"一致，但不等于"外部 agent CLI 自动驱动"。若要以 CLI 形式验收，需要一个可用的 agent 命令。
2. **隔离等级未升级**：候选代码仍在子进程内，隔离状态沿用既有的 `AGENT_ISOLATION_UNVERIFIED`；本 Issue 未新增 OS/容器级验证。
3. **holdout/最终交付未动**：示例任务没有 holdout 解锁流程（`finalize` 沿用既有台账与预算），本 Issue 不消费任何真实 holdout。
4. **Dashboard 无认证**：仅 loopback；远程访问/认证/分享**明确不在本期范围**。
5. **`--dashboard` 的进程生命周期**：服务是独立进程，但未做端口冲突自动避让（固定端口失败会退出并打印原因）。
6. **复杂度政策版本化**：`selection_policy` 随运行记录（`autoresearch-policy-v2`），但**新政策尚未以 A/B campaign 验证**；历史裁决未被改写。

## 9. 结论

Issue #61 的模块目标（独立 CLI、Doctor、自主闭环、Skill、轻量只读 Dashboard、TCM-free 示例任务）已实现并通过全量回归、真实训练端到端、故障注入、看板实时/安全与恢复测试。唯一未满足的是**外部 agent CLI 自动驱动**（本机无该 CLI），已如实记为阻塞；若接受"操作侧 Coding Agent 逐轮驱动"作为真实 agent 验收，则可验收，否则保留 OPEN 并等待可用 agent 命令。