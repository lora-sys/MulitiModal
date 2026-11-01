# 源码级 AutoResearch v11 报告（隔离加固后的首轮）

- 日期：2026-11-01
- 提案 agent：`claude:opus-5-5-driver`（逐轮真提案，`batch_supplied=False`）
- 引擎基线：`ce67fff`（feature/clean-slate，含第三/四轮隔离加固 H-01…H-08）
- worktree：`acam-lab:/home/lora/workspace/ar-v11`，分支 `autoresearch/v11`
- 数据：`runs/manifests/tcm-20261107-v2-features.json` + `data/local/mendeley-cxtph3tjsg-v1/Share data.xls`（与 v10 完全同一冻结划分，seed 20261107 线）

## 契约

| 项 | 值 |
| --- | --- |
| 选择协议 | folds=4 × repeats=3（12 折），margin_sem=1.0，seed=0 |
| 决策规则 | 增益 > 1×SEM **且** ≥ 0.001×净增 AST 节点；纯删代码在 SEM 带内直接 KEEP |
| 预算 | ≤14 轮 / PLATEAU=5 连续 REVERT 早停 / 4h 墙钟（实际 6 轮 + 1 冒烟，PLATEAU 触发） |
| holdout | 搜索阶段全程未触碰；交付阶段按预算解锁 #2/2（见文末「最终交付」小节） |
| 可编辑面 | `src/research/candidate.py` 单文件，guard + 冻结导入 + AST 校验全程生效 |

## 与 v10 引擎的两处硬差异（本轮的主要发现）

v10 结束后合入的隔离加固改变了搜索空间的边界，这两点是 v11 的机制级结论：

1. **复杂度度量改为 `ast_statement_expr_nodes+string_payloads`**（字符串常量按 payload 计价）。同一份 v10 冠军 Bag 补丁：v10 度量下 growth=17（最低增益 0.017），v11 度量下 **growth=86（最低增益 0.086）**。v10 实测 Bag 增益 +0.017——在新定价下即使完整复现也过不了复杂度下限。
2. **每次 fit 结束都做 `save_eval_model` 导出**（`research/model_export.py`）：`torch.fx` 追踪 + 冻结词表（模块类 / 函数 / 方法三张白名单）。自定义类作**子模块**直接拒绝（根模块豁免）。Bag 的成员是自定义 `FusionMLP` 实例 → 导出拒止，候选 crash。

结论：**v10 冠军方向（3 成员集成）在加固后引擎下被双重封死**（词表不可表达 + 复杂度不可负担）。这不是 v10 结论被推翻，而是新引擎的合法搜索空间收缩为：单模型上的常数改动（0 节点）、≤10 节点微改、以及 train() 内的自由改动（导出只追踪 forward，损失/优化器/调度不受词表限制）。

## 冒烟

- `noop`（注释级补丁，folds=2 × repeats=1，独立目录 `runs/research-v11-smoke/`）：KEEP 0.3258，全链路（含导出）验证通过。
- 同一冒烟阶段用 Bag 补丁试跑：REVERT crash，报错即上文导出拒止，growth=86 被如实记录在 smoke manifest——该方向就此关闭，未再正式提案。

## 逐轮表（runs/research-v11/manifest.jsonl）

| run_id | 提案 | 决策 | primary_value | SEM | 增益 | growth (最低增益) | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tcm-0000 | lr 3e-3→1e-2 | **KEEP** | 0.3475 | 0.0146 | — | 0 | 首轮按基线默认入账，非已证明的改进 |
| tcm-0001 | lr 1e-2→2e-2 | REVERT | 0.3417 | 0.0140 | −0.0057 | 0 (0) | 学习率向上越过 1e-2 后变差 |
| tcm-0002 | 梯度裁剪 1.0 | REVERT | 0.3553 | 0.0137 | +0.0079 | 10 (0.010) | 单模型线历史第二高，距门槛 ~0.006 |
| tcm-0003 | 权重 EMA (0.995) | REVERT | 0.3498 | 0.0146 | +0.0024 | 60 (0.060) | 新度量下定价过高，且增益平平 |
| tcm-0004 | 步级 lr 衰减 ×0.999 | REVERT | 0.3562 | 0.0145 | +0.0087 | 11 (0.011) | 单模型线历史最高值，距门槛 ~0.006 |
| tcm-0005 | 标签平滑 0.05 | REVERT | 0.3523 | 0.0127 | +0.0048 | 24 (0.024) | v9/v10 未试过的损失侧方向 |

5 轮连续 REVERT → PLATEAU=5 触发，第 6 轮后停机。

## 冠军结论

- **起点澄清**：v10 冠军 tcm-0004（Bag，0.3595）已于 2026-11-01 裁决中按严格 AST 口径翻转为 REVERT（最低增益 0.0780 > 实测增益 0.0171），v11 起点的有效冠军是 **ce67fff 基线 0.3424**（12 折，v10 tcm-0000 同协议测量）。
- **v11 冠军 = 单模型 lr 1e-2，0.3475 ± 0.0146**（tcm-0000，commit `2454fda3`，分支 `autoresearch/v11` HEAD，worktree 已停在冠军提交）。
- 如实分级：
  - 0.3475 vs 起点 0.3424：**噪声级差异**（+0.005 < 1×SEM）。且该 KEEP 是「首轮基线默认入账」，不是任何一轮里击败过当时冠军的改进——v11 未产生统计上过门槛的研究性改进，冠军线在噪声带内移动。
  - tcm-0002 / tcm-0004 的 +0.008：**未过噪声门槛**，只作为「停在门槛正下方」的方向记录，不作为结论。
  - 0.3595 仍是全项目历史最高测量值，但其 KEEP 资格按裁决不恢复；其方向在本引擎下不可表达（见硬差异）。
- 冠军文件含一行冒烟轮注释（`# v11 smoke: comment-only round...`）：这是 loop 机制如实推进的结果（冒烟 KEEP 先提交，冠军轮在其上），未改写；对语义无影响。

## 机制级结论（供后续轮次/版本使用）

1. **发现下限由 12 折 SEM (~0.013–0.015) 决定**。单模型线全部 9 个历史测量值（v10 基线 3 轮 + v11 5 轮）落在 0.3417–0.3562，没有一轮超过 +0.009。除非改变信息量（新特征/新标签/新数据），单模型微改不可能稳定过门槛——这与 #48 契约「synthetic/受限搜索不产生科学结论」一致。
2. **新复杂度定价使「结构换性能」经济整体关闭**：任何带 docstring 的类/函数新增都是 30–90 节点量级（0.03–0.09 的增益下限）。后续若想恢复集成类搜索，需要：(a) 用白名单容器（Sequential/ModuleList/ModuleDict + 白名单子模块）表达成员，(b) 无 docstring 的紧凑实现，且 (c) 证明 ≥0.04 量级增益——本数据上从未见过。
3. **v10 的 REVERT 经验继续成立**：gate / LayerNorm / posweight / deep / wider / 5 成员 / 显式 wd 均未重提；本轮新测的 4 个方向（2e-2 / clip / EMA / decay / smooth）也全部未过门槛。
4. 工程侧：`-I` 子进程启动、AST guard、candidate SHA 绑定、导出词表在 6+1 轮全部工作正常，无一次 guard 误拒；提案脚本需从当前冠军状态读取锚点（R1 KEEP 后 `LEARNING_RATE = 3e-3` 锚点失效，已改正则匹配——提案 agent 的补丁必须对准分支当前内容）。

## 合规声明

- 无临床/科研有效性声明；以上全部为工程/统计层陈述，数据为真实数据但标签仍是 RULE 派生，**GO_RESEARCH_CLAIMS 未放行**。
- 真实数据、checkpoint、runs/ 工件均未入 Git；holdout 未触碰；历史 manifest 未改写。
- 本次为真实数据上的 AutoResearch 轮次，结果表述仅限「12 折选择协议下的统计陈述」。

## 最终交付（holdout 解锁 #2/2）

日期：2026-11-01。`research-src-finalize` 对 v11 冠军（`autoresearch/v11` @ `2454fda3`，单模型 lr 1e-2）执行 refit_and_score：在选择池 232 行（train+dev）上以 seed 0、20 s 预算重训（实际拟合 20.5 s），随后 holdout 42 行在隔离 evaluator 子进程评分一次。台账事件 `c20de889347c4241aa80d2e844f64cd9`（`2026-11-01T07:12:16Z`，扣费前置）——**Phase 1 的 holdout 解锁预算 2/2 已用满**：本划分上的任何再评估需要新数据里程碑或用户明确授权（AGENTS.md 规则 11）。

### 交付工件（acam-lab `runs/research-v11/champion/`，不入 Git）

| 工件 | SHA-256（前 16 位…） |
| --- | --- |
| `weights.pt`（state_dict，12 张量，`weights_only=True` 加载已核验） | `1f4d9f37eddd132c` |
| `weights-architecture.json`（torch.fx 重放描述：graph / modules / input_spec / version；`load_eval_model` 严格重建已核验） | `e8ceaf699e811a2b` |
| `finalize.json`（candidate SHA `7fded075a91ef0b0`，pool manifest SHA `045270fa6ede9051`，父 manifest SHA `96137c20f5bcbb4a`，seed 0，refit_rows 232，holdout_rows_untouched 42） | `0db414a07bfd1a62` |

同目录 `pool-manifest.json` / `pool-records.json` 记录池契约（无 holdout 行）；交付包留在 acam-lab `runs/`，沿用 v10 惯例不复制回本机 `artifacts/`。

### holdout 逐维可观测性（42 人）

| | 维度 0 | 维度 1–8 |
| --- | --- | --- |
| 观测数 | 0（42/42 缺失，规则 ABSTAIN） | 各 42 |
| 正例率 | —（不可判） | 0.2566 |

全部汇总指标在观测维度分母（8 维）上计算；维度 0 全缺不进任何均值、不补零。

### holdout 汇总与基线对比（同一 42 人，基线同池拟合）

| 模型（均在 232 行池上拟合） | holdout macro-AP（主口径） | holdout macro-F1@0.5 | holdout subject MAE |
| --- | --- | --- | --- |
| Dummy（多数类） | 0.2887 | 0.000 | 0.2887 |
| **BMI-only 逻辑回归（2 个特征）** | **0.3761** | 0.014 | 0.4032 |
| 全特征逻辑回归 | 0.3502 | 0.210 | 0.3981 |
| 神经交付模型（v11 冠军，lr 1e-2） | 0.3702 | 0.322 | 0.3871 |

基线由隔离 evaluator 内的 `runtime.baselines` 在同一 holdout 视图上计算，三条数字与 v10 交付逐位一致（确定性路径，同池同拟合）——这同时是对口径连续性的交叉核对。

### 如实读法（v10 教训的直接延续）

- **主口径（阈值无关 macro-AP）下，神经交付模型没有胜过 BMI-only 逻辑回归**：0.3702 vs 0.3761，差 −0.006，42 人上属噪声级。这与 v10 的独立测量（0.3724 vs 0.3761）方向一致：两次独立交付都未超过 2 特征基线。
- 神经模型高于 Dummy（0.2887）与全特征逻辑回归（0.3502）；这两处差距同样只是 42 人上的单次观测，不引申。
- macro-F1（0.322）与 subject MAE（0.3871）上神经模型数字最好，但 F1 依赖 0.5 阈值（v8 的「0.383 vs 0.215」伪优势正是阈值产物），MAE 不是 EFRI 主口径——两者仅作并列诊断记录，不构成「优于基线」的依据。
- **不得宣称本表「验证」了交付模型**：v11 冠军本身对起点是噪声级 KEEP（0.3475 vs 0.3424），holdout 数字是这次噪声级冠军的一次性测量，不是任何改进的证实。
- 42 人、无置信区间、RULE 派生标签、受试者身份为行序（UNVERIFIED）——**GO_RESEARCH_CLAIMS 仍为 NOT_AVAILABLE**（缺 independent-reference-data）。以上全部为工程层陈述，无临床/科研有效性声明。

### 交付验证

acam-lab：ruff 全过，pytest 270/270，冻结 eval 52/52（E01–E79）全绿；权重 `weights_only=True` 加载与 `load_eval_model` 严格重建通过。提交：docs 一笔（报告 + README 状态句）。
