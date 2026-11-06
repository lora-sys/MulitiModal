# 推理正确性、预测价值与实际输入场景复核（2026-11-06）

证据等级：`LOCAL_EXPERIMENTAL`；评估器隔离 `AGENT_ISOLATION_UNVERIFIED`。
在 `acam-lab`（lora 用户）执行；结论不构成诊断、临床疗效或外部泛化有效性声明。
历史 holdout 解锁预算已耗尽，本阶段未使用任何 holdout；未重启 AutoResearch，
未改冠军源码/配置/seed，未写 `results.tsv`。

## 0. 执行口径与交付物

| 项 | 内容 |
| --- | --- |
| 代码 | `9290e58` / `9df1337` / `1f574b0`（本地 + origin + 生产机三方一致） |
| 生产机全套测试 | `454 passed, 2 skipped`（2 个 skip 是 Zenodo 4431679 引用文件未缓存，与本阶段无关） |
| 工件 | `artifacts/constitution-v1-champion` / `artifacts/sd-v2-ar1-champion`（gitignored，随磁盘保留） |
| 复核报告 | `runs/value-review/{constitution,syndrome,input-scenarios}.json`（gitignored，按规范不入库） |
| 冻结源码 sha256 | `bd1bed98…`（`train.py`）、`53240a28…`（`constitution_train.py`），交付前后未变 |

命令（生产机，`PYTHONPATH=src`）：

```bash
python -m research.constitution_delivery deliver   # 确定性重存工件；预测逐位一致
python -m research.constitution_delivery verify    # registry_hashes_verified=true, retraining_performed=false
python -m research.constitution_value_review --out runs/value-review/constitution.json
python -m syndrome.value_review --out runs/value-review/syndrome.json
python -m scripts.tcm_input_scenarios --out runs/value-review/input-scenarios.json
```

### 任务 A：推理正确性（根因与修复）

1. **体质工件加载**：`python -m ... deliver` 曾把 `ConstitutionPredictor` 定义在
   `__main__`，joblib 写入不可解析的 `__main__.ConstitutionPredictor`，独立进程加载失败。
   根因修复：CLI 入口改为委托给 importable 模块；`save_artifact` 拒绝 `__main__` 绑定的类；
   删除 demo 里的 `sys.modules["__main__"]` 别名 shim（不保留兼容层/回退/迁移）。
   用冻结 seed/数据/配置确定性重存工件；重存前后 5 个真实样本预测 `max abs diff = 0.0`。
   新进程无任何别名即可加载两个真实工件；`verify` 通过（`registry_hashes_verified=true`）。
2. **异常输入**：非映射行、`int/float/bool/str/bytes/list` 症状文本、非 mapping 的 fields、
   `None` 后端 spec、直接构造的坏请求、缺失/损坏工件目录 → 明确 `OUT_OF_DOMAIN` 或
   `ABSTAIN` + 原因字段，不再抛异常，也不产出候选/分数。
3. **缺失值**：冠军只用前 9 项特征；这 9 项全缺时旧实现仍返回 `OK` 并给出分数（误导）。
   现为 `ABSTAIN`；全 63 项缺失亦 `ABSTAIN`；部分缺失仍 `OK` 并列出 `missing_information`。
4. **极端值**：`1e9` 曾给出 3.3e9 的伪造分数，`1e308` 曾崩溃。现在捕获非有限算术、
   校验输出落在已发布的 0–100 量表内，越界即 `OUT_OF_DOMAIN`。

### 任务 B：预测价值复核（只读）

**体质**（冻结 4 折 × 2 重复折外，`holdout_subjects=0`，`n_subjects=274`）：

| 指标 | 值 |
| --- | --- |
| 冠军折外 masked MAE | 14.1444 |
| 训练折均值基线 MAE | 14.4974 |
| 配对差值（冠军 − 基线） | −0.3530，bootstrap 95% CI `[−0.6087, −0.0870]`（不含 0） |
| 改善受试者 / 折 | 152 / 274；8 / 8 折 |
| 逐维 CI 不含 0 的维度 | 4 / 8（阴虚、痰湿、气郁、特禀） |

**证候**（冻结 protocol-v2 dev 分区，n=2844，`holdout_subjects=0`）：

| 指标 | 值 |
| --- | --- |
| Top-1 / Top-3 / Top-5 | 0.612869 / 0.843179 / 0.902250 |
| macro-F1（95% bootstrap CI） | 0.320213（`[0.270196, 0.325112]`） |
| 逐类 | 148 类，144 类有 support，**55 类 F1=0**，min/max support 0 / 543 |
| 拒判 | dev 2562 `OK` / 282 `UNCERTAIN`；决策覆盖 0.900844，拒判覆盖 0.099156 |
| 对抗探针 | 7 / 7 全拒（2 `ABSTAIN` + 5 `OUT_OF_DOMAIN`） |
| 阈值 | 0.2780843（工件自带，dev 上重算一致）；阈值上/下准确率 0.649883 / 0.276596 |

### 任务 C：实际输入场景（`SMOKE_ONLY` 合成输入，`ROBUSTNESS_ONLY`）

17 个场景，覆盖 4 类；`accuracy_reported=false`、`expert_labels=false`。

| 类别 | 场景数 | 结果 |
| --- | --- | --- |
| 普通人日常表达 | 5 | 3 `OK`、2 `UNCERTAIN`（低置信），0 崩溃、0 错误类型 |
| 重复/矛盾症状 | 5 | 4 `OK`、1 `UNCERTAIN`，0 崩溃 |
| 缺失身体信息 | 5 | 全缺 / 前 9 项全缺 → `ABSTAIN`；部分缺失 → `OK`；维度错 → `OUT_OF_DOMAIN`；未提供 → 未评估 |
| 极端合理输入（观测 min/max，非外推） | 2 | 2 `OK`（分数仍在 0–100 量表内） |

## 1. 八维体质评分相对于简单基线，究竟有没有稳定预测价值？

**有，但只在「内部选择集折外 CV、幅度很小」这个意义上成立，且不是逐维都稳定。**

- 冠军 14.1444 vs 训练折均值基线 14.4974；配对差值 −0.3530，95% CI
  `[−0.6087, −0.0870]` 不含 0，8/8 折全部改善、152/274 受试者改善。方向上稳定，
  但绝对幅度只有 0.35 个 MAE 点（约 2.4% 相对改善）。
- 逐维看只有 4/8 维 CI 不含 0（阴虚 −0.6331、痰湿 −0.4520、气郁 −0.5746、
  特禀 −0.7013）；气虚 −0.0120、血瘀 −0.0108 基本与基线无差别，阳虚/湿热的 CI 跨 0。
  因此不能说「八个维度都稳定优于基线」。
- 这是同一批 274 名受试者的内部折外比较：评估器隔离未认证
  （`AGENT_ISOLATION_UNVERIFIED`），无独立外部数据源，历史 holdout 预算已耗尽。
  不能外推为跨机构/跨设备/跨时间的稳定价值，更不是临床有效性。

## 2. 体质模型需要的前 9 项特征，我们实际能不能获得？

**能获得，且不需要专用仪器；但「能拿到数值」不等于「已验证的采集链路」。**

冻结选择集表头实测：

```
0 gender, 1 age, 2 education, 3 occupation, 4 chronic dz, 5 surgery hx,
6 BH, 7 BW, 8 BMI
```

- 前 6 项是问卷/自报人口学与病史；后 3 项是身高、体重、BMI（普通体重秤/身高尺即可）。
  这与其余 54 项需要经络、脉搏波、舌诊设备的特征不同——冠军恰好只用这 9 项，
  所以从「可获得性」看是低成本的。
- 但当前没有验证过的**实时采集到模型编码**的链路：源数据来自 Mendeley
  `cxtph3tjsg/1`（`Share data.xls`，274 人研究协议），缺失语义是 `null + feature_mask=False`，
  禁止补零。真实产品里把用户自报值映射到这套编码、单位、缺失约定尚未验证。
- 实时生理数值（心率/血氧/皮温/HRV）**不是**这 9 项；`PHYSIOLOGY_ONLY` 演示明确
  `ABSTAIN`（`constitution=UNAVAILABLE`、`syndrome=UNAVAILABLE`），
  `#54 生理→证候` 仍为 `BLOCKED`。第 9 维平和质在源数据中完全未观测（0/274），
  永远 `ABSTAIN`，不补零、不伪造。

## 3. 证候模型能不能处理用户自然说出的症状？

**能接住并给出结构化响应，但「能输出」不等于「已验证正确」，自然口语没有专家标签。**

- 5 句日常表达：3 句 `OK`（Top-3 候选，全部 `calibrated=false`）、2 句 `UNCERTAIN`
  （最高分低于 dev 阈值而拒判）；0 崩溃、0 异常错误类型。重复/矛盾症状 4 `OK`、1 `UNCERTAIN`。
  英文、乱码、把证候名当症状、空输入 → `OUT_OF_DOMAIN` / `ABSTAIN`，不产出证候标签。
- 在冻结 protocol-v2 dev（TCM-SD 书面主诉，有标签）上 Top-3 0.843179、Top-5 0.902250，
  但 macro-F1 仅 0.320213，且 55/148 类 F1=0、最小 support 为 0，长尾类别不可用。
  拒判机制在工作：dev 拒判覆盖 0.099156，阈值上准确率 0.649883 vs 阈值下 0.276596。
- 场景文本是 `SMOKE_ONLY` 合成输入，无专家标签，按任务书**不报告预测准确率**；
  这里的 `OK`/`UNCERTAIN` 只是鲁棒性与拒判行为记录，不能当作口语场景的正确率。
  分数未校准，不是概率、置信度或诊断。

## 4. 两个模型遇到不合理输入时，会不会给出误导性结果？

**修复后会明确拒判；修复前确实存在误导性结果，这正是本次修掉的问题。**

修复前（实测）：`1e9` 输入给出 3.3e9 的伪分数（`status=OK`）、前 9 项必需特征全缺仍
`OK` 并输出 8 个分数、`1e308` 直接崩溃、`int/float/bytes` 症状文本抛
`AttributeError/TypeError`、工件目录缺失/损坏时 demo 抛 `AttributeError`（`spec=None`）。
这些都是「给了看起来可信的结果」或「崩溃」。

修复后（生产机实测）：

- 非映射、非字符串、非有限/溢出、输出越出 0–100 量表 → `OUT_OF_DOMAIN` + 原因；
  全缺/必需 9 项全缺 → `ABSTAIN` + 原因；两者都不返回候选或分数。
- 工件缺失、目录不存在、registry 损坏 → `ABSTAIN` + `backend_availability=NOT_AVAILABLE`，
  不抛异常、不静默 fallback。C 报告 17 个场景 `error_types` 全为空，0 崩溃。
- 极端**但合理**（真实观测范围内的 min/max）输入仍返回 `OK` 且分数在 0–100 内——
  这是有意保留：不把合理边界值误判为越域。第 9 维平和质仍永远 `ABSTAIN`。
- 无配对训练证据时（仅实时生理数值）明确 `ABSTAIN`，不从生理数值推断体质/证候。

## 5. 哪些输出可以作为内部研究结果保留，哪些还缺乏验证？

**可保留为内部研究结果（`LOCAL_EXPERIMENTAL`）**：

- 折外 CV 数字：冠军 14.1444、均值基线 14.4974、配对差值 −0.3530 与 CI，逐维 MAE/CI。
- protocol-v2 dev 上的 Top-1/3/5、macro-F1 与 CI、逐类 support/precision/recall/f1、
  阈值与拒判覆盖率（决策覆盖 0.900844 / 拒判覆盖 0.099156）。
- 工件 provenance、sha256 清单与 `verify` 结果（`registry_hashes_verified=true`，
  `retraining_performed=false`）；推理示例与鲁棒性/拒判场景记录（`SMOKE_ONLY`）。
- 八维体质研究性估计接口本身：8 个已观测维度、未校准、第 9 维 `ABSTAIN`。
- GO 门禁现状：`tcm_constitution` 的 `GO_ENGINEERING=GRANTED`、`GO_REAL_DATA_TRAINING=GRANTED`。

**仍缺乏验证、不得作为结论外推**：

- 外部/异源泛化：无独立外部数据源，跨机构/设备/时间泛化未验证。
- 评估器隔离 `AGENT_ISOLATION_UNVERIFIED`，无独立防篡改认证；历史 holdout 预算已耗尽，
  无留出测试集复核。
- 校准与临床语义：所有分数未校准，不是概率/置信度/诊断；`GO_RESEARCH_CLAIMS=NOT_GRANTED`，
  不构成临床疗效或有效性结论。
- 自然口语症状无专家标签：只能做鲁棒性检查，口语场景准确率未验证。
- 55/148 类 F1=0、min support 0，长尾类别不可用；平和质（第 9 维）完全未观测。
- 实时生理→体质/证候链路：`physio_bpsi_efri` 的 `GO_REAL_DATA_TRAINING=NOT_GRANTED`，
  `tcm_sd_syndrome` 门禁 `NOT_ASSESSED`；`#54` 仍 `BLOCKED`。
- 许可：TCM-SD `CC BY-NC-SA 4.0`（仅非商业科研）；Mendeley `CC BY 4.0`。
  真实数据、二进制工件、日志、`runs/` 与 `artifacts/` 均不入库。
