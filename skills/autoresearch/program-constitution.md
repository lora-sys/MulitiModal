# AutoResearch 任务契约：中医体质八维回归

本程序契约定义中医体质 AutoResearch 研究线的全流程规范。本任务与证候分类线（TCM-SD）完全平行且独立，共用轻量科研基础设施（Git KEEP/REVERT/CRASH 决策闭环、`results.tsv` 契约与 `research.dashboard` 原生只读仪表盘），不引入外部 Docker、多 Agent 或复杂平台。

## 1. 研究目标与冻结条件

- **真实数据集**：Mendeley 数据源 `Share data.xls`，覆盖 274 名真实受试者，8 个 PUBLISHED 偏颇体质维度评分（依据行业标准 ZYYXH/T157-2009，分值范围 0~100）。
- **第 9 维平和质严格拒判**：平和质在源数据中完全未被观测（0/274），执行 **ABSTAIN** 策略，严禁填零、插值或伪造，任何候选模型输出形状必须严格为 `(n, 8)`。
- **主评价指标**：`constitution/masked_mae`（8 维平均绝对误差），**数值越低越好**（direction=`minimize`）。RULE 派生标签与 PUBLISHED 真实评分严格不可互换。
- **验证折划分**：仅在选择集（selection partition）内进行受试者级 4 折 × 2 次重复（共 8 折次验证），随机种子固化为 `20261105`；同一受试者严禁跨折出现；测试集 holdout 显式为空（历史 2/2 次 holdout 解锁预算已耗尽，严禁复用）。
- **完整性与防篡改哈希**：`CONSTITUTION_DATASET_SHA256` 绑定原始 Excel 文件的 sha256 校验和，选择集清单与记录哈希硬编码于 `src/research/constitution_data.py` 中。
- **执行预算**：单折时间上限 300 秒（5 分钟墙钟时间），不安装未声明的重量级依赖。
- **科学证据分级**：标记为 `LOCAL_EXPERIMENTAL`，评价隔离级别为 `AGENT_ISOLATION_UNVERIFIED`。体质线 `GO_RESEARCH_CLAIMS` 状态为 `NOT_GRANTED`，本地验证集指标严禁推导为临床有效性结论。

## 2. 严谨校验与实验前提

1. **真实数据只读校验**：
   ```bash
   python -m research.constitution_data verify
   ```
   严格校验 Excel 哈希、274 行记录、8 维有效评分、平和质未观测、特征缺失 mask、holdout 为空以及标签血统。
2. **基线对齐**：通过 `round-000.json` 沉淀基线模型（8 维训练折中位数回归），以 `record --decision KEEP` 确立初始 Champion。
3. **自主探索**：后续每轮提出单一明确假设，仅修改 `src/research/constitution_train.py`，提交 commit 并执行完整外挂打分。

## 3. 固化评价入口

```bash
export PYTHONPATH=src
python -m research.constitution \
  --manifest data/local/mendeley-cxtph3tjsg-v1/protocol-v1/selection_manifest.json \
  --dataset data/local/mendeley-cxtph3tjsg-v1/protocol-v1/selection_records.json \
  --xls "data/local/mendeley-cxtph3tjsg-v1/Share data.xls" \
  --candidate src/research/constitution_train.py \
  --out runs/constitution-v1/round-000.json \
  --folds 4 --repeats 2 --seed 20261105 --budget 300
```

> [!NOTE]
> 评分计算由 `research.constitution` 在隔离子进程外部执行，候选自报分数不予采纳。

## 4. 实验循环规则

1. **提出假设**：查阅当前 Champion、Git HEAD、`results.tsv` 及上一轮逐维误差，提出单一可检验假设。
2. **候选编辑与提交**：仅编辑 `src/research/constitution_train.py` 并提交 Git Commit。
3. **评价判定**：
   - **KEEP**：masked MAE 严格优于 Champion，且模型架构提升合理。
   - **REVERT**：指标未提升，`git reset --hard` 回退至 Champion。
   - **CRASH**：超时或异常报错，限就地排查 2 次，未恢复则果断回滚。
4. **决策落盘**：
   ```bash
   python -m research.constitution record \
     --repo . \
     --results runs/constitution-v1/results.tsv \
     --result runs/constitution-v1/round-xxx.json \
     --commit <sha> \
     --decision KEEP|REVERT|CRASH \
     --description "<假设与改动摘要>"
   ```
5. **实时看板查看**：
   ```bash
   ./solve.sh dashboard --run runs/constitution-v1
   ```

## 5. 第一轮交付成果（已冻结）

第一轮探索共计 18 轮（6 KEEP / 12 REVERT / 0 CRASH），产出冻结冠军 Commit `16c26582`：
- **模型方案**：Huber 鲁棒全局线性回归 + kNN 局部中位数回归的逐维软集成，选用前 9 维基础特征（人口学 6 维 + 人体测量 3 维）。
- **折外泛化表现**：折外 masked MAE 达到 **14.1437**（显著优于中位数基线的 15.65 与单一模型的表现）。
- **交付与验收入口**（与证候线标准交付同构，工件与协议物理隔离）：
  ```bash
  export PYTHONPATH=src
  python -m research.constitution_delivery deliver   # 最终全量拟合 + 工件导出 + 自动化验收
  python -m research.constitution_delivery verify    # 纯只读核对（哈希与鲁棒性校验，不重新训练）
  ```
- **核心工件与输出清单**：
  - 持久化模型工件：`artifacts/constitution-v1-champion/`（包含 `model.joblib`、`labels.json`、`registry_entry.json`，内嵌全量 sha256 清单与完整 provenance）。
  - 交付审计报告：`runs/constitution-delivery/v1/`（包含 `delivery-report.json`、`per-dimension.json`、`verification.json`）。
  - 8 维安全推理接口：`ConstitutionInference`，严格输出 8 个 PUBLISHED 维度；**第 9 维平和质恒定返回 ABSTAIN**；对缺失或超界输入严格防御（返回 `ABSTAIN` 或 `OUT_OF_DOMAIN`）。
  - 所有逐维指标均严格源于 4 折 × 2 次重复的折外预测，非训练集内虚高自评。

## 6. 严令禁止事项

- 严禁篡改评价器逻辑、真实数据集文件、标签空间、验证折划分、随机种子或时间预算。
- 严禁修改证候任务线及其候选文件 `src/research/train.py`。
- 严禁对缺失特征盲目补零，严禁对未观测的平和质伪造预测值。
- 严禁访问或重开已耗尽预算的历史 holdout 测试集。
- 严禁引入 Docker、分布式 Agent 集群或破坏本地轻量闭环。
