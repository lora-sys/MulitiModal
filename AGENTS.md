# MulitiModal：工程规范与项目事实手册

本文档供所有参与本仓库研发与实验的 Agent 查阅与执行。改动前须通读相关任务文件；若具体任务最新冻结契约与本文档产生冲突，以任务最新冻结契约或交付模块为准，严禁私自改写研究目标与评估口径。

---

## 1. 核心研发与工程规范

1. **奥卡姆剃刀与极简原则**：
   - 实现保持简单、职责清晰；优先复用现有依赖与已验证的轻量级方案。
   - 已废弃的过时路径应彻底清理，不保留无意义的过渡兼容层与废弃注释。
2. **AutoResearch 双轨单文件约束**：
   - 证候任务线（TCM-SD）：唯一可修改文件为 `src/research/train.py`。
   - 体质任务线（Constitution）：唯一可修改文件为 `src/research/constitution_train.py`。
   - 外部冻结评价器（`prepare.py` / `constitution.py`）、输入数据、标签字典、交叉验证划分、时间预算与随机种子一律冻结，严禁在实验轮次中改动。
3. **闭环决策与记录契约**：
   - 每一轮必须基于上一轮证据提出单一可验证假设；
   - 严格依据外部评价器输出裁决：**KEEP**（指标严格提升且实现紧凑）、**REVERT**（未提升则强制回滚至 Champion）、**CRASH**（异常报错或超时，限修 2 次）；
   - 所有决策原子化追加写入各任务的 `runs/<tag>/results.tsv`。
4. **代码提交规范**：
   - 遵循 Conventional Commits 规范，一个 commit 对应一个独立可验证的目标；
   - Schema 改动必带契约测试，Adapter 改动必带 shape/mask/负例测试，Bug 修复必带回归测试。
   - 运行环境使用本机 Python 与可用 CPU/GPU，不强求引入 Docker、多 Agent 集群或分布式编排。

---

## 2. 数据与科学证据边界红线

1. **受试者级物理隔离与防泄露**：
   - 同一受试者的所有时序或多模态记录必须严格归属于同一折（Partition），严禁跨折拆分；
   - 特征标准化（均值、标准差、中位数）、缺失值插补、特征选择与权重拟合仅限在当前训练折内部拟合，严禁接触测试或验证样本；
   - 历史 holdout 测试集解锁预算（体质线 2/2 次、证候线 1/1 次）已彻底耗尽，模型探索与候选筛选绝对禁止触碰历史 holdout。
2. **真实性与不猜测原则**：
   - 体质线第 9 维平和质在源数据中完全未观测（0/274），模型输出必须为 `(n, 8)`，推理端统一返回 `ABSTAIN`，严禁补零、插值或伪造；
   - 标签体系（RULE、PUBLISHED、LLM_SILVER）必须保留原始血统来源且不可互换；LLM_SILVER 不作为参考真值；规则缺项一律 `ABSTAIN`。
3. **证据分级声明**：
   - 开发机本地实验结果一律标识为 `LOCAL_EXPERIMENTAL`，隔离级别记为 `AGENT_ISOLATION_UNVERIFIED`；
   - `GO_ENGINEERING`、`GO_REAL_DATA_TRAINING`、`GO_RESEARCH_CLAIMS` 三级门禁独立判定，任何本地 dev/CV 指标均不得外推为临床有效性定论；
   - 原始数据集只读，导入前后核对 sha256；真实数据、二进制模型工件、日志、`runs/` 及 `artifacts/` 严禁提交至 Git 仓库。
   - 合成数据必须显式标注 `SMOKE_ONLY`。

---

## 3. 三大研究主线事实与交付状态

### 1. 证候分类研究线 (Syndrome Track)
- **数据规格**：TCM-SD 数据集，54,152 例中文症状主诉，148 类证候分类，数据许可为 CC BY-NC-SA 4.0（非商用）。
- **基线与交付**：protocol-v2 下字符 1~4 gram TF-IDF + LogisticRegression 的 148 类 `macro-F1` 约为 0.3199，优于同等条件下的复杂神经方案，作为交付候选。
- **服务接口**：[src/syndrome/service.py](src/syndrome/service.py) 提供双模式（症状分析与实时非诊断知识）安全产品契约，支持 fail-closed 拒判保护与工件哈希校验。

### 2. 体质多维回归研究线 (Constitution Track)
- **数据规格**：Mendeley `Share data.xls`，274 名真实受试者，8 个已观测维度连续评分（依据 ZYYXH/T157-2009 标准，0~100 分）。
- **已交付 Champion**：第一轮 18 轮探索已顺利冻结并交付（Commit `16c26582`，Huber 线性与 kNN 局部中位数的逐维自适应软集成，选用 9 维基础特征），折外 `masked MAE` 达到 **14.1437**。
- **交付工件与接口**：
  - 持久化工件：`artifacts/constitution-v1-champion/`（模型权重、标签定义、带 sha256 清单的 `registry_entry.json`）；
  - 交付验收入口：[src/research/constitution_delivery.py](src/research/constitution_delivery.py)（`deliver` 训练并固化工件，`verify` 快速只读审计）。

### 3. 生理状态与实时知识研究线 (Physiology Track)
- **业务定位**：面向按摩椅前置生理指标（心率、血氧、皮温、HRV 等）的健康主题知识检索；
- **防越界护栏**：实时知识模式绝不从生理数值推断中医证候或体质，严格切断生理数值与疾病/证候的伪造关联；当前未配对真值数据状态为 `BLOCKED`。
