# V0 验收记录 — 2026-10-04

本轮仅使用合成数据。真实公司数据尚未整理，所有模型标记为
`SMOKE_MODEL_NOT_VALIDATED`；下面的分数不能证明真实人体预测能力。

## 实际运行结果

| 检查 | 结果 | 原始产物 |
|---|---|---|
| Harness 故障与完整性 Benchmark | 45/45 通过 | `runs/physio-autoresearch/verification/harness-benchmark.json` 及对应 XML/log |
| 其余 Python 合约/集成测试 | 5/5 通过 | `runs/physio-autoresearch/verification/integration-final.xml` |
| 新增多目标训练与独立重载评测 | 1/1 通过 | `runs/physio-autoresearch/verification/multi-target.xml` |
| Pi/上游/真实 Python 桥接 | 4/4 通过 | `npm run test:pi` 实际输出 |
| Python 环境 | `pip check` 无损坏依赖 | 项目 `.venv` 与 `requirements-autoresearch.lock.txt` |
| Git 空白检查 | `git diff --check` 通过 | 未创建提交、推送或 PR |

Python 合计 **51 项通过**，另有 **4 项 Pi 检查通过**。历史 Demo/纸面实验矩阵、Torch 模型、预训练迁移和 Pi 真实供应商调用没有运行；另完成了下面记录的 AGY 候选实验。

## 自动模型实验

最终源码 SHA-256：`f563fe0d2140f6a3fb56fa0033672d14524a01f47d21212a09016b1dd10f8903`。
基础提交：`9fc5087dcae8b958fa7e7eee404949cd8cb3957f`。本轮代码仍是本地未提交改动。

执行 `python -m scripts.physio_experiments`，初始固定搜索快照保存在
`runs/physio-v0/experiment-report.json`，后续续跑另写 `experiment-replay-report.json`：

| 合成受试者 | 数据版本 | 完成候选 | KEEP / REVERT | Naive MAE | 当轮 Champion / MAE |
|---|---|---:|---:|---:|---|
| 5 | `c080d206bbb0caa4f3875363` | 12 | 2 / 10 | 7.2247 bpm | ElasticNet / 2.7293 bpm |
| 6 | `a7b54eef34875d3a2b8e6689` | 12 | 2 / 10 | 7.5837 bpm | ElasticNet / 2.1129 bpm |

两轮均覆盖 Naive、ElasticNet、Random Forest、XGBoost、超参数和特征组实验。
同版本再次运行增加 **0** 次实验，预算没有重置。第 6 个合成受试者触发新的版本与完整轮次。
这些数值来自人为构造的生成关系，只检验流水线；两轮分数也不是同一锁定测试集上的性能提升证明。

另验证单个受试者只产生 `FIT_ONLY_NOT_VALIDATED`，没有验证分数和 Champion。
冠军 full-fit 模型已重载，对 6 条开发记录的 pre-only 输入生成完整预测 CSV。
模型文件 2,027 bytes；当次批量 predict 耗时约 25 ms，未包含模型加载与输入 QC，
不代表部署延迟或独立泛化测试。

## AGY 提出的追加实验

通过 `agy:staffer` 的单个任务 `staffer-mutaajga-fc5861ab`，实际模型
`gemini-3.8-flash-medium` 读取演示合同与聚合实验反馈，提出一个新的 ElasticNet
候选：`alpha=0.01, l1_ratio=0.1`，仍只使用 age/pre_hr/pre_hrv。

AGY 直接调用 CLI 被 Windows ACL 拒绝，完成 **0** 次 attempt。
当前工作区执行器随后执行原候选，未改 ACL、数据、协议、评测器或预算。
同一个 6 人数据版本增加第 **13** 次实验，固定评测器判定 **KEEP**：
MAE 从 2.1129362261 降到 **1.0935733284 bpm**。这是合成开发数据搜索结果，
不能证明临床效果、在线策略优越性或独立泛化提升。

主代理重新加载 6 个折模型及 full-fit 模型，校验哈希、受试者隔离和源码 ZIP，
由保存的折外预测独立重算指标，与记录一致。完整重放保持 12/13 次尝试，
增加 0 次实验，单人无分数行为及新 Champion 的 pre-only 推理继续通过。
原始 AGY 摘要保留其失败事实；最终执行与审计另存
`runs/physio-v0/agy/host-verification.json`。AGY 外部 token/费用未测量，
不计作 Pi ledger 中的零成本调用；Pi 的真实供应商调用仍为 0。

## 独立审查和修复

本地独立审查发现并修复四个实际问题：

1. 训练时没缺失的列在验证时首次缺失，原填补器没有 indicator：改为每列固定缺失 bit。
2. 首个失败/拒绝 result 可绕过 Naive：改为成功基线闸门，并允许失败配置的有限重试。
3. 并发 Pi 会话各自计算调用预算：增加覆盖整会话的 OS 排他锁。
4. 代码升级后旧数据无法继续：增加人工显式升级、源代码 ZIP 和累计预算保留。

审查复核确认修复，限定范围内未发现新的重大阻断问题。最终测试由实际实现者运行。
最终 Benchmark 还发现 Windows 路径过长，源码快照改用 ZIP 后重新验收通过。

## 仍未验证

- Pi 的真实供应商模型响应、在线候选选择收益、实际 token/费用和在线故障恢复。
- 真实公司字段、BPSI/EFRI 定义与真实 post outcomes；独立新受试者的预测价值。
- 可执行代码自改进、Harness 候选晋升、公平的新旧 Harness 能力比较、每日自动任务。
- 任意生成代码的 OS 沙箱和真实 locked-test 权限隔离。

当前应用限制和受限 Pi 工具不能当成已经具备 OS 沙箱；Benchmark 通过不能当成 RSI 得到验证。
