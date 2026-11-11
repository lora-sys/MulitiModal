# ML 生命周期端到端验收（#112）

**结论级别**：`LOCAL_EXPERIMENTAL` / `AGENT_ISOLATION_UNVERIFIED` / `SMOKE_ONLY`
**性质**：工程验收，不是结果报告。未宣称任何预测性能、疲劳改善、因果效应或临床有效性。

## 1. 验收结论

| 项 | 结果 |
|---|---|
| 合成 CI + 本地安全负例 | **PASS** |
| `GO_ENGINEERING` | **GRANTED** |
| `GO_REAL_DATA_TRAINING` | **NOT_GRANTED** |
| `GO_RESEARCH_CLAIMS` | **NOT_GRANTED** |
| 公开合成 CI（`.github/workflows/ci.yml`） | 3 层 job：fast 契约 × Py3.11/3.12/3.13、合成 E2E × Py3.11/3.13、无 torch 诚实 skip |
| 私有数据入 Git | 无（CI 内 `git ls-files` 断言） |

三门禁由 `gate_bundle(engineering_go=True)` 强制后两者为 `NOT_GRANTED`：
**测试通过不提权真实训练或研究声明**。

## 2. 可复现命令

```bash
python -m pytest tests/lifecycle/test_contracts_fast.py -m fast -q   # 纯 stdlib 契约快测
python -m pytest tests/lifecycle -q             # 合成生命周期套件
python -m ruff check src tests scripts          # 全仓 lint
python scripts/lifecycle_verify.py              # 统一验收 + 三层检查
python scripts/lifecycle_verify.py --torch-free-smoke     # 无 torch 纯契约可导入
python scripts/lifecycle_verify.py --acceptance-only      # 仅 15 项统一检查
```

`scripts/lifecycle_verify.py` 自注入 `src/` 到 `sys.path` 和子进程 `PYTHONPATH`，
不依赖调用方环境（仓库无 `[build-system]`，实测无 PYTHONPATH 时可跑）。

## 3. 统一验收 15 项（含 10 项安全负例）

| 检查 | 阶段 | 负例 | 断言 |
|---|---|---|---|
| `contracts_immutable` | ingest | 否 | 不可变契约、假 CONNECTED 真设备拒判、trainable 硬门禁 |
| `three_gates_independent` | ready_gate | 否 | 工程 GO 不提权另两门禁 |
| `pipeline_idempotent` | snapshot | 否 | 同输入 → snapshot/QC 逐字节一致、缓存复用、零重复、不自动训练 |
| `pipeline_incremental` | reconcile | 否 | 乱序/迟到批次收敛到同一 snapshot |
| `pipeline_recovery` | normalize | **是** | 注入失败被记录；恢复后复现 snapshot |
| `pipeline_ready_gate` | ready_gate | **是** | 未批准/不可信 → ABSTAIN；approved+frozen+trusted → 准入 |
| `no_silent_skip` | validate | **是** | 被拒行必带理由，隔离区可查 |
| `training_no_auto_promotion` | train | **是** | 合成 smoke 不提升、工件 NOT_VALIDATED、三门禁独立 |
| `release_no_self_promote` | promote | **是** | registry 从不自动提升 candidate |
| `sources_capability_replay` | ingest | **是** | 假 CONNECTED 真设备拒判、BLOCKED 判定、live/replay 收敛 |
| `features_head_mismatch` | features | **是** | 负 delta 目标配 `[0,100]` 头必拒；正确头准入 |
| `features_no_fabricated_truth` | features | **是** | 无批准的疲劳真值，unavailable_reason 明确 |
| `gateway_fail_closed` | serve | **是** | 幂等、未知版本拒判、control_command 恒 null |
| `observe_no_auto_actions` | observe | **是** | 数据更新只审计，不自动训练/提升/发布 |
| `private_data_not_in_git` | ingest | 否 | 私有原始数据留在 Git 之外 |

报告落盘 `runs/lifecycle/lifecycle_acceptance.json`，含 `gates`、`checks`、
`negative_checks`、`evidence_level`、`isolation`、`limitations`。

## 4. 测试基线分层

| 层 | 标记 | 范围 | 通过数口径 |
|---|---|---|---|
| 快测 | `fast` | `tests/lifecycle/test_contracts_fast.py` | 纯 stdlib，无 I/O / 无网络 / 无 torch |
| 合成 E2E | `synthetic` | `tests/lifecycle/` | 增量幂等、恢复、drift、折内拟合、原子提升/回滚、网关负例、哈希链 |
| 安全负例 | — | `lifecycle_verify.py` | 10 项注入/拒判负例 |
| 慢测 | `slow` | 长耗时集成 / Bootstrap | CI 按需启用，**不计入常规通过数** |
| 真实数据 | `real_data` | 需 `data/local/` | 默认跳过，**不计入常规通过数** |

**不膨胀通过数**：skip 不算通过；不因测试变绿提权两个研究门禁；不把 synthetic golden
写成真实结论；不把 `GO_ENGINEERING` 写成 `DATA_MAPPING_VERIFIED`。

## 5. 逐阶段 evidence GO / NO-GO

| 阶段 | Issue | evidence | 判定 |
|---|---|---|---|
| 任务与标签协议 | #104 | `contracts.py` 14 类不可变契约 + 三门禁 | **GO_ENGINEERING** |
| 数据源与管道 | #105/#106 | `pipeline.py` 合成增量幂等 + 恢复 | **GO_ENGINEERING**（映射仍 NOT_VERIFIED） |
| 特征与标签 | #107 | 版本化注册表 + 折内拟合 + 泄漏负例 | **GO_ENGINEERING**（真实标签 BLOCKED） |
| 训练与评估 | #108 | 合成 smoke + 重复运行一致 | **GO_ENGINEERING**（真实训练 NO-GO） |
| 工件与部署 | #109 | 哈希绑定 + 原子提升/回滚 | **GO_ENGINEERING**（VALIDATED 工件 NO-GO） |
| 网关与实时服务 | #110 | 三模式隔离 + fail-closed | **GO_ENGINEERING**（真实设备 NO-GO） |
| 监控与持续迭代 | #111 | 哈希链 + 数据触发仅刷新 | **GO_ENGINEERING**（自动训练/发布 NO-GO） |

## 6. 缺失依赖与阻塞清单

| 依赖 | 状态 | 影响 |
|---|---|---|
| 厂商字段字典（94 字段语义、`is_complete`、`fatigue_level`、`脑波` 选择） | `BLOCKED_BY_DATA_OR_DEVICE` | 映射只能登记为抽取集反推假设 |
| 独立可信配对疲劳标签 | `INSUFFICIENT_EVIDENCE` | 真实训练 NO-GO |
| 生理受试者身份字典（1 member hash → 5 subject_key） | `PHYSICAL_SUBJECT_ISOLATION_UNVERIFIED` | 折是"留一 subject_key 折"，物理隔离未验证 |
| 真实设备 + 厂家控制 API | `NOT_CONNECTED` | 网关 `control_command` 恒 null |
| 独立验证 / 外部数据 | 缺失 | `GO_RESEARCH_CLAIMS` NO-GO |
| `response_hash` 构造规则 | 未声明 | 参考无法端到端独立验证 |

## 7. 遗留项（如实登记，未掩盖）

1. `requirements-dev.txt` 使用 `>=` 下界而非精确 pin，CI 会随上游小版本漂移。
   已新增**伴随式** `requirements-lock.txt`（本机 CPython 3.13.12 实测版本快照），
   但**不强制 CI 使用**：CI 矩阵跨 Py3.11/3.12/3.13，锁死到本机版本会让 3.11 job
   无法安装。CI 仍装 `requirements-dev.txt` 并打印 `pip freeze`，使漂移在 job log 可见。
   完全消除跨版本漂移需按每个 Python 版本分别生成 lock，列为后续独立任务。
2. 旧 #41–#43 的"真实 BPSI/EFRI 拟合"迁移到 #113/#114/#115 并标 BLOCKED，
   属**迁移而非实现**；本验收不声称该历史要求已完成。
3. 无 torch 的干净 CI 只跑 fast 契约与 `tests/physio`（神经用例显式 skip），
   不跑 `tests/core` 的神经栈用例；`core/__init__.py` 已改为惰性解析，
   保证纯 stdlib 契约在无 torch 环境可导入。
4. `docs/design/ml-lifecycle.md` 为单一索引；README/AGENTS.md 只加一个入口链接，
   不复制其内容，避免多处漂移。

## 8. 未解除边界（不随本次验收改变）

- 真实 94 字段完整导出与厂商字段字典：`BLOCKED_BY_DATA_OR_DEVICE`
- 独立配对疲劳标签：`INSUFFICIENT_EVIDENCE`；`GO_REAL_DATA_TRAINING(physio)` /
  `GO_RESEARCH_CLAIMS(physio)` 仍 `NOT_GRANTED`，推理固定 `NOT_AVAILABLE`
- 真实设备控制：`NOT_CONNECTED`
- 结论级别 `LOCAL_EXPERIMENTAL` / `AGENT_ISOLATION_UNVERIFIED` / `SMOKE_ONLY`
