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
| holdout | 全程未触碰，未调用 finalize；解锁预算仍为「已用 1 / 上限 2」 |
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

- **冠军未更新**（相对项目记录）：v10 的 Bag 选择值 0.3595 仍是全项目最高测量值。v11 冠军 = 单模型 lr 1e-2，**0.3475 ± 0.0146**，commit `2454fda3`（分支 `autoresearch/v11` HEAD，worktree 已停在冠军提交）。
- 如实分级：
  - 0.3475 vs v10 基线 0.3424（同一数据、同分辨率）：**噪声级差异**（+0.005 < 1×SEM）。且该 KEEP 是「首轮基线默认入账」，不是任何一轮里击败过当时冠军的改进。
  - 0.3475 vs v10 冠军 0.3595：低 0.012，但两者不可直接比较——v10 冠军在新引擎下不可表达（见硬差异）。
  - tcm-0002 / tcm-0004 的 +0.008：**未过噪声门槛**，只作为「停在门槛正下方」的方向记录，不作为结论。
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
