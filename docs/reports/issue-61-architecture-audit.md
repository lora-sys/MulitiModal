# #61 架构审计与最小实施方案

日期：2026-11-03。分支 `feat/issue-61-autoresearch-module`（工作树 `~/repos/MulitiModal-issue-61`）。
范围：把已有研究引擎补成**真正自主、可诊断、可观测**的独立模块，并交付独立 Skill 与轻量 Dashboard。**不重写已有系统。**

## 1. 现状审计（实测，非转述）

| 能力 | 现状 | 结论 |
| --- | --- | --- |
| 冻结边界（数据/划分/scaler/预算） | `src/research/frozen.py`：`TIME_BUDGET_SECONDS=20`、`FoldData`、`build_fold`、`selection_fold_ids`、`training_fold_data`、`FROZEN_PUBLIC_API` | **保留** |
| 补丁守卫（单文件白名单 + 禁用符号 + AST 安全 + 复杂度计价） | `src/research/guard.py`：`validate_source_patch`、`validate_frozen_imports`、`editable_complexity`（`ast_statement_expr_nodes+string_payloads`）、`validate_source_ast`、`DYNAMIC_EXECUTION_CALLS` | **保留** |
| 架构导出白名单（fx 追踪）→ `UNSUPPORTED_ARCHITECTURE` 的生产者 | `src/research/model_export.py`：`export_architecture`、`ArchitectureExportRefused`、`save_eval_model`/`load_eval_model` | **保留** |
| 可信评估（隔离子进程 + 完整性快照） | `src/research/evaluator.py`：`integrity_snapshot`、`assert_integrity`、`install_candidate_import_guard`、`score_fold`、`score_holdout` | **保留** |
| 子进程隔离与训练 worker | `src/research/spawn.py`、`train_worker.py`、`runner.py`（`_spawn_training`、`_spawn_evaluator`） | **保留** |
| 循环与裁决 | `src/research/loop.py`：`GitWorktree`（branch 必须以 `autoresearch/` 开头、`assert_clean`）、`SourceResearchLoop.state/decide/run_round/run(proposals)`，余量 `1×SEM` + 复杂度地板 `0.001/节点` | **保留，局部修改**（见 §3） |
| holdout 台账（1/2/… 次解锁、原子扣费） | `src/research/finalize.py`：`_assert_unlock_budget`、`_charge_unlock_event`、`refit_and_score`、`MAX_UNLOCKS=2`、`MAX_UNLOCKS_SD=1` | **保留** |
| task 适配层 | `src/research/tasks.py`：`TaskAdapter` + `ConstitutionAdapter` + `adapter_for`（closed pair）；`task_sd.py`（syndrome） | **保留**（#61 只**新增**一个 TCM-free 适配器） |
| CLI | `src/runtime/cli.py` 有 13 个 `research*` 子命令：`research`、`research-step`、`research-status`、`research-finalize`、`research-src-state`、`research-src-round`、`research-src-run`、`research-src-finalize`、`research-report` 等 | **保留为 deprecated wrapper**，新增唯一正式入口 |
| 自主性 | `run_round(patch, intent, ...)` 只执行外部补丁；`run(proposals)` 接受**预提交列表** | **缺失**：没有"读结果→提假设→再提案"的闭环 |
| 诊断 | 无 | **缺失**：preflight/postflight 都不存在 |
| 轮内可观测 | 只有完成的 manifest 行；无轮内事件、无 status.json | **缺失** |
| Web | 只有 sd-v1 的**静态** HTML 报告（`docs/reports/sd-v1-dashboard.html`） | **缺失**：无实时只读看板 |
| Skill | 仓库无 `skills/` | **缺失** |
| TCM-free 示例任务 | 无（两个适配器都依赖真实 TCM 数据） | **缺失** |

**运行入口现状（问题 3）**：`runtime/harness.py` 的 config-only loop、`src/research/` 的 source loop、以及 #57 的独立实验脚本三路并行。收敛规则：

- **唯一正式入口** = `python -m research.cli`（别名 `scripts/autoresearch`）：`doctor/run/status/stop/resume/report/dashboard`。
- config-only loop（`runtime/harness.py`）与全部 `runtime.cli research*` 标记 **deprecated**，保留可运行以维持回归，README/文档指向新入口，不再新增能力。
- #57 的 `scripts/issue57_*.py` 是**实验证据脚本**（已提交），不属于研究引擎入口；报告里注明其只读历史地位。

## 2. 必须保留的接口（回归面）

`tests/research/` 现有 9 个测试文件（`test_source_loop`、`test_model_export`、`test_evaluator_isolation`、`test_finalize`、`test_fold_isolation`、`test_child_spawn`、`test_train_worker`、`test_syndrome_task`）定义了不可破坏的行为：单文件守卫、AST 复杂度定价、导出词表、隔离评分、holdout 台账、fold 隔离。**#61 的改动不得让其中任何一条失效**。

## 3. 需要修改的代码（最小面）

1. `src/research/loop.py`
   - 新增**自主轮次**：`run_autonomous(session, proposer, budget)`，每轮**实时**读取 session 最新状态（含上一轮裁决、指标、Doctor 发现、剩余预算）→ 调用 proposer 产出**单一**假设 → 守卫/训练/评分 → 写证据 → 由循环自行决定下一轮。**不接受预生成候选列表**作为自主研究的实现；`run(proposals)` 保留并显式标注为**固定批跑**（非自主研究）。
   - 裁决码细化：`KEEP/REVERT/INVALID_PROTOCOL/UNSUPPORTED_ARCHITECTURE/CRASH/BUDGET_EXHAUSTED`（工程拒绝与模型劣势分开记录）。
   - 复杂度政策**版本化**（`selection_policy` 写入每轮记录），禁止同一 campaign 内事后改价。
2. `src/runtime/cli.py`：把 `research*` 子命令的帮助文本改为指向新入口并标 deprecated（行为不变）。
3. `README.md` / `AGENTS.md`：补一行"唯一研究入口"，并写明 Agent 可自主选择**假设与实验顺序**、不可改数据/划分/指标/holdout（对应 issue 问题 6）。

## 4. 新增模块（全部在 `src/research/` 内，独立于 TCM 数据）

| 文件 | 职责 | 关键约束 |
| --- | --- | --- |
| `events.py` | 可信宿主**单写入者**事件流：`events.jsonl`（append-only）+ `status.json`（原子替换）。字段 `schema_version,seq,time,run_id,round,repeat,fold,stage,status,elapsed_seconds,budget_seconds,metrics?,message?` | 只有协调器写；候选代码无写路径；损坏/半写行可容忍 |
| `session.py` | `run_id`、持久化状态、`start/stop/resume/status`、轮次与预算账本 | 重启不重跑已完成轮次；stop 用标记文件，训练子进程由其父进程结束 |
| `doctor.py` | preflight（用户跨折、精确文本重复、标签冲突与支持度、类别映射顺序、评价协议版本、外层 dev 是否参与选择、候选权限）+ postflight（loss/收敛/逐类/长尾/seed/异常/Top-k，缺项 `NOT_AVAILABLE`）→ `doctor.json` + BLOCKED/WARNING/INFO | 只用训练/开发侧数据；不读 holdout；不自动改标签/split/evaluator；无 runner 无法确认时标 `UNVERIFIED` |
| `agent.py` | proposer 协议 + **真实 Coding Agent 适配器**（子进程、stderr 抑制、超时、JSON 校验）+ **deterministic stub 明确标注非真实 Agent** | 每轮输入=最新状态；输出单个假设 + 受控变量 + 预期观测；缺真实 Agent 时自主研究 **NOT_AVAILABLE** |
| `cli.py` | 唯一入口 `doctor/run/status/stop/resume/report/dashboard` | `run` 无数据/无真实 agent/沙箱未验收 → fail closed |
| `dashboard.py` | stdlib `http.server` + 单页原生 HTML/CSS/JS + 3s 轮询 + 原生 SVG 图表 | 只读白名单路由、默认 127.0.0.1、大小上限、XSS 转义、无路径拼接、无 holdout/secret/原始数据 |
| `task_toy.py`（示例） | **与 TCM 无关**的合成任务适配器（表格二分类/多标签，合成生成，无外部数据文件） | 证明模块独立性；不引入新依赖 |
| `skills/autoresearch/SKILL.md` + `program.md` | 指导 Coding Agent 调用 CLI | 只做指导，不含业务实现；不改 lora-mode/全局配置 |

不引入：Temporal、多 Agent swarm、数据库、React/Next、Grafana/W&B/Prometheus、WebSocket、公网服务。图表用原生 SVG。

## 5. 最小实施方案（顺序，每步可验证）

1. `events.py` + `session.py`（+ 测试：seq 单调、半写行、原子 status、stop/resume 不重跑）
2. `doctor.py`（+ 故障注入回归：跨折用户、同文本泄漏、标签冲突、外层 dev 参与选择、类别顺序错误、NaN/crash）
3. `agent.py` + `loop.run_autonomous`（+ 测试：≥3 轮自适应，stub 标注非真实 Agent）
4. `task_toy.py`（TCM-free，合成数据）+ `cli.py`（doctor/run/status/stop/resume/report）
5. `dashboard.py`（+ 测试：路由白名单、XSS、路径遍历、大日志上限、部分 JSONL、STALE）
6. Skill 与 program.md
7. 真实验证：全量回归 + 示例任务真实训练/保存/加载 + Doctor 注入 + **真实 Agent ≥3 轮** + 实时 Dashboard E2E + 崩溃/刷新/无心跳/停止恢复 + 安全测试
8. PR（不合并）+ 最终报告

## 6. 已知风险与应对

- **真实 Coding Agent 不可得** → 必须如实标 `NOT_AVAILABLE`，不得用 stub 冒充；DoD 里"真实 Agent ≥3 轮"将如实记录为阻塞而不是伪造。
- **GPU/数据在 acam-lab**：示例任务是纯合成、CPU 可跑，保证模块在无 TCM 数据、无 GPU 环境下可验收。
- **复杂度政策**：不修改历史裁决；新政策以新 campaign 记录并 A/B 验证（issue 问题 4）。