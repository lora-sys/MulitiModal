# 按摩前生理预测模型：下一阶段最小实施方案（草案，未验证）

**状态**：`NOT_VALIDATED` / `NOT_AVAILABLE`。本文是实施设计，不是结果报告。
未采集任何真实配对数据，未训练任何生理模型，未验证任何预测效果。

## 1. 依据与现状

| 项 | 值 |
|---|---|
| 原始 PDF | `docs/source/pre-massage-fatigue-prediction-framework.pdf` |
| PDF sha256 | `29727daced33cc95506f580ea542f4bd1386c9a44b29cf1e2836627439eed781` |
| 文档版本 | 用户草案，正文含 `[N]`、`[value]`、`[mean]`、`[range]` 等占位符（20 行、约 65 处） |
| 已实现代码 | `src/core/specs.py`（Pre-only 输入契约、EFRI/BPSI 目标）、`src/core/models.py` 的 `PhysioBPSIEFRIModel`、`tests/core/test_specs.py` / `test_models.py` 的 During/Post 泄漏负例 |
| 证据登记 | `physio-real-data` claim = `NOT_AVAILABLE`；`physio-engineering-smoke` = 合成张量上的 forward/loss/checkpoint 路径可执行 |
| GO 门 | `GO_ENGINEERING(physio)` GRANTED；`GO_REAL_DATA_TRAINING(physio)` **NOT_GRANTED**（缺 `physio-real-data`、`physio-label-rules-verified`）；`GO_RESEARCH_CLAIMS(physio)` **NOT_GRANTED** |

结论：**工程骨架已具备，真实数据、标签规则、评估与任何效果声明都缺失**。
下一阶段只能做到"端到端可运行但仅合成 smoke"，不能声称真实训练或泛化。

## 2. 输入（Pre-only，PDF §1.3 / Eq.1）

推理输入严格限定 `x = [x_body, x_phys]`，共 12 + 7 = 19 维，全部为按摩前可得：

- `x_body`（12）：age, sex, height, weight, body fat percentage, BMI,
  basal metabolic rate, visceral fat, muscle mass, body water percentage,
  body score, body age。
- `x_phys`（7）：heart rate, microcirculation, ECG-derived fatigue index,
  blood oxygen saturation, health score, systolic BP, diastolic BP。

**禁止进入推理**（PDF §1.3、AGENTS.md 生理线硬约束）：
- 按摩中 EEG 9 项（EEG fatigue value、mean/max attention、meditation
  duration/proportion、mind-wandering frequency、stress-relief index、
  comprehensive score、mean mindfulness index）；
- 按摩后 7 项生理测量及任何 post−pre 变化量；
- 以上数据的别名、差分、比值、滞后特征或衍生统计量。

这些字段**只允许在训练期用于构造标签**（见 §3），不得通过别名或衍生
特征进入推理。现有 `make_physio_input_spec()` 与
`ModelBatch.validate()` 已实现该边界，并有 `extra_during_post=True` 的
拒判负例测试；实施时复用，不重写。

预处理：数值特征用训练折均值/标准差标准化，类别编码与缺失/异常规则
按 PDF 占位符待定（`[binary or one-hot encoding]`、`[the selected method]`、
`[the selected rule]`）；**所有参数只在训练折内拟合**，再应用到验证/测试折
（PDF §1.3 / Eq. 后段落，AGENTS.md 同款约束）。缺项 ABSTAIN，不补零或猜测。

## 3. 标签（训练期构造，非真值）

| 标签 | 定义 | 来源层级 | 状态 |
|---|---|---|---|
| BPSI | Eq.(2)：标准化并**方向对齐**的 Pre 特征加权和；主方案为 PCA/FA 状态导向权重，线性缩放到 0–100；另有 equal / PLS / ElasticNet / 神经权重作为对照 | `RULE`（由输入派生，非独立金标准） | 权重策略未定 |
| EFRI | Eq.(3)：对齐的 EEG 响应 `e` 与 post−pre 生理响应 `r` 的加权和；含 `EFRI_EEG_ONLY` / `EFRI_POST_ONLY` / `EFRI_JOINT` 三种定义，各有权重块 | `RULE`（复合指标，非绝对疲劳真值） | `0.5/0.5 EFRI_JOINT_EQUAL_BLOCK_V0` 是实验设计选择，**不是验证系数** |

PDF 明确警示：
- BPSI 不是独立金标准疲劳度量；PLS/ElasticNet 权重用到 EFRI 信息，解释须与状态导向主方案区分。
- 变量方向必须先对齐（疲劳指数下降=改善，心率/血压升高含义依基线而定）。
- meditation 时长/比例、mindfulness、stress-relief、comprehensive score **不得自动当作客观疲劳标签**，纳入需统计或敏感性分析支持。
- 无对照/假按摩组时，EFRI **不得解释为严格因果疗效**；不得宣称临床疗效。
- PDF 统计检验（§2.8）、样本量 `[N]`、指标表 `[value]` 全部为占位符。

标签版本须随 provenance 记录 `standard_version` / `label_version` / 权重块 / 方向表 / 缺失掩码；`RULE`、`PUBLISHED`、`LLM_SILVER` 不互换，`LLM_SILVER` 不作参考真值。

## 4. 神经网络架构（PDF §1.4，已有代码）

`PhysioBPSIEFRIModel`（`src/core/models.py`）已实现，不新建训练平台：

```
x_body(12) -> f_body -> h_body ┐
                                ├-> concat -> f_fusion -> h ┬-> g_EFRI  -> EFRI (主输出)
x_phys(7) -> f_phys -> h_phys ┘                            └-> g_BPSI -> BPSI (辅助头)
L = λB·L_BPSI + λE·L_EFRI + λR·L_reg                       (Eq.9)
```

- 单主输出 EFRI + 可选 BPSI 辅助头（`#41` 单主输出规则；BPSI 是基线/辅助，
  不是替代性能声明）；`bpsi_aux_weight` 默认 0.3。
- 轻量 MLP 分支（hidden=64, fusion=64），small-N 场景配合 dropout / weight
  decay / early stopping；`ModelBatch.validate` 强制 Pre-only 输入契约。
- 掩码损失：`masked_mse_loss` / `masked_mae_loss`，目标缺项不参与、不补零。
- 评估指标（PDF Eq.10-12）：MAE、RMSE、R²；折级误差配对检验；权重稳定性 / 特征选择频率 / 方向一致性（BPSI）。
- 划分：同一受试者全部记录同属一个 partition；分组 CV；独立外部测试需新数据里程碑。
- 明确不引入：Docker、多 Agent、新训练平台、新依赖。

## 5. 数据缺口（实施前必须显式登记）

1. **没有配对真实数据**：`physio-real-data` = NOT_AVAILABLE；无 pre/during/post
   同人配对记录。`#23–#31`、`#41–#43` 真实数据阶段保持 BLOCKED。
2. **样本量与统计参数未定**：PDF 中 `[N]`、`[mean]`、`[range]`、`[SD]` 为空，
   无法做功效分析或确定折数；表格、图、显著性检验均为 `[value]` 占位符。
3. **标签规则未验证**：方向对齐表、权重块、缺失/异常规则、EEG 分量纳入依据
   都缺实证；`physio-label-rules-verified` claim 不存在。
4. **无独立外部数据**：`independent-reference-data` 缺失，
   `GO_RESEARCH_CLAIMS` 未放行；不得宣称跨来源泛化。
5. **无对照/假按摩组**：EFRI 不可作因果结论。
6. **历史 holdout**：生理线未解锁；本阶段任何评估都不得使用历史 holdout。

## 6. 最小可运行的下一阶段实施方案（分阶段门控）

复用现有 `skills/autoresearch/`（Skill、Git KEEP/REVERT 循环、
`research.dashboard`），但**先不启动 AutoResearch 实验轮次**：
没有真实数据与冻结标签规则，任何 KEEP/REVERT 都只会是合成 smoke，
不得记作真实训练成果（AGENTS.md：未经独立认证的本地记录只可用于合成 smoke）。

### 阶段 A：契约固化（可立即做，纯工程）

1. 新建 `autoresearch/physio-v0` 分支；在 `skills/autoresearch/` 增加
   `program-physio.md` 或等价研究简报，写明本文 §2–§5 的冻结条件。
2. 补齐 `src/data/` 的**生理数据适配器**（含形状、mask、负例测试）：
   - 输入契约 = `make_physio_input_spec()` 的 12+7 维，显式拒绝 During/Post 字段；
   - 原始数据只读、导入前后哈希核对；
   - 每行携带 provenance：`source_file`、hash、`label_tier=RULE`、
     `label_version`、`standard_version`、缺失掩码；
   - 缺项 ABSTAIN，不补零、不猜测。
3. 建立**合成 smoke 协议**（显式标记 `SMOKE_ONLY`）：仅验证
   forward / compute_loss / checkpoint / Pre-only 拒判 / 8 维之外的
   EFRI·BPSI 掩码形状，复用 `tests/core/` 现有负例，不产生任何可被
   当作科研分数的指标。**不伪造"真实"数据文件**，合成样本显式命名
   `synthetic`。
4. 验收：`pytest -q` 全套 + ruff；`program-physio.md` 中登记
   `AGENT_ISOLATION_UNVERIFIED`、`LOCAL_EXPERIMENTAL`、`SMOKE_ONLY`。

### 阶段 B：真实数据里程碑（前置条件，不提前执行）

必须同时满足才解锁：
- 获得 pre/during/post 同人配对数据与许可；导入前后哈希核对；
- 完成标签规则审计并登记 `physio-label-rules-verified`（方向表、权重块、
  缺失/异常规则、EEG 分量依据）；
- 冻结评价协议（划分、seed、预算、主指标、mask、代码哈希）与
  `GO_REAL_DATA_TRAINING(physio)` 重新判定。

在此之前 `GO_REAL_DATA_TRAINING(physio)` 保持 `NOT_GRANTED`，
不训练、不评估、不产出分数、不更新 `results.tsv`。

### 阶段 C：解锁后（本方案不包含，仅列出顺序）

真实数据导入/QC → 标签引擎（RULE，保留来源）→ 冻结 selection manifest →
合成 smoke 通过后跑真实固定协议 → 首次 KEEP/REVERT → 独立外部验证 →
再谈交付与 `GO_RESEARCH_CLAIMS`。三条 GO 独立判定，不得由 dev/CV 分数推导。

## 7. 明确不做（本阶段）

- 不伪造或"占位"任何真实生理数据、标签、权重系数、样本量或指标数值；
- 不把 PDF 占位符 `[value]` 当作结果，不宣称已验证生理预测效果、
  临床疗效、疲劳改善或因果效应；
- 不把合成 smoke 分数写成真实训练成果或科研结论；
- 不修改证候冠军 `src/research/train.py`、体质冠军
  `src/research/constitution_train.py` 或两条线已冻结的评价协议；
- 不使用历史 holdout、不新增 Docker / 多 Agent / 新训练平台 / 新依赖；
- 不在 `GO_REAL_DATA_TRAINING(physio)` 未放行前启动真实训练。
