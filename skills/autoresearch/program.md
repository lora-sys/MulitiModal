# AutoResearch program: TCM-SD 证候分类

本文件指导 Coding Agent 参考 Karpathy AutoResearch 进行连续、真实、可复查的模型研究。本轮只完成中医证候模型的优化、交付与泛化验收，不开发新的研究基础设施，也不启动体质或按摩前生理预测训练。

## Goal

当前任务：根据中文短症状主诉预测 TCM-SD 的 148 类中医证候。主指标为 148 类 macro-F1，越高越好。使用冻结的 protocol v2 开发选择集，4 folds × 2 repeats，并始终按受试者分组。历史 TF-IDF 字符 1–3 gram + LogisticRegression 基线约 0.319921，新版本机复现实测约 0.319900。它们是参考值，不是预先假定的新实验结果。

目标是选出在固定协议下表现最好、复杂度合理、可保存和重载的模型，并给出逐类表现、推理能力和泛化证据。开发集表现提高不能直接证明临床有效性或跨医院泛化。

项目的长期目标仍是按摩椅生理健康预测，包括独立的中医体质研究和 PDF 中的 BPSI、EFRI 预测。当前不使用这两类任务的标签、输入或评价指标。

## Setup

1. 阅读 `AGENTS.md`、`skills/autoresearch/SKILL.md`、本文件，以及 `src/research/prepare.py`、`src/research/train.py`。建立独立的 `autoresearch/<tag>` 分支或 worktree，核对 Git 状态。
2. 使用开发机 Python 环境以及已有 CPU/GPU，不启用 Docker、sandbox、额外 Agent 服务或训练调度平台。检查实际依赖和数据访问情况，不因设备上有 CUDA 就假定 sklearn 使用了 GPU。
3. 检查 `runs/sd-v2/short/selection_manifest.json` 和 `selection_records.json` 是否存在，核对 `prepare.py` 内固定的 manifest 和 dataset SHA-256、148 类标签、受试者分组、折内单向精确文本去重。真实数据和模型工件不提交 Git。
4. 冻结本次比较的 seed、folds、repeats、每折训练预算、依赖、硬件和评价代码版本。允许为不同算力预算开新研究组，但新组必须重跑基线；不同协议或预算的分数不得直接排序。
5. 为本次研究建立忽略 Git 的 `runs/<tag>/` 目录，存放日志、`round-*.json`、`results.tsv` 和 `status.json`。先检查既有结果是否完整；不能覆盖已有轮次。
6. 第一轮运行未经修改的 `train.py` 基线。若本研究组已有同一协议、同一源码的完整基线记录，可以校验后复用；不得凭历史报告填入新分数。

## Experiment rules

- **唯一可编辑候选**：`src/research/train.py`。可研究特征、模型结构、损失、优化器、正则化、训练和预测策略。不要预设 BERT 或其他神经模型必然优于 TF-IDF + LR。
- **冻结内容**：`prepare.py`、评价器、数据、划分、标签顺序、预处理拟合边界、metric、seed 方案和本研究组的预算。不能通过修改评价规则提高分数。
- 词表、TF-IDF、标准化器、特征筛选、阈值和模型必须在对应训练折内拟合；不能使用验证折标签拟合。保留训练集真实重复记录，并按现有协议对验证侧执行去重。
- 历史证候 holdout 的 1/1 解锁预算已经用尽，体质 holdout 的 2/2 也已用尽。禁止再次读取、重新划分或用于任何模型选择。
- 不私自增加或升级依赖。若新研究路线必须引入不同权重、依赖或预算，先明确版本、许可证和新的比较口径；不修改既有候选比较历史。
- 每轮固定一个可检验假设，不预先排好一长串实验假装根据结果自主研究。优先分析长尾类别、同文本不同标签、错误预测、过拟合、特征覆盖和训练成本。允许尝试神经网络，也允许证明简化方案更好。
- **没有默认总时长或固定总轮数。** 启动后逐轮继续，不逐轮请示；用户明确停止、要求进入交付验收、实验环境/协议失效或安全边界受阻时停止。Coding Agent 会话结束不会在后台继续运行。

## Fixed evaluation

每轮用相同的 protocol v2 正式评价；合成数据和减少 folds 的运行仅供 smoke 或调试，不进入正式排行榜。

```bash
export PYTHONPATH=src
python -m research.prepare \
  --manifest runs/sd-v2/short/selection_manifest.json \
  --dataset runs/sd-v2/short/selection_records.json \
  --candidate src/research/train.py \
  --out runs/<tag>/round-000.json \
  --folds 4 --repeats 2 \
  --seed 20261104 --budget <per_fold_seconds>
```

将 `<tag>` 换成当前研究目录，并为后续轮次顺序使用 `round-001.json` 等新路径。`<per_fold_seconds>` 是在本研究组首次基线前确定的每折固定秒数，不代表总研究时长。现有程序约束单折最大 3600 秒，不要临时修改这个限制让候选占优。

评估器计算 148 类 macro-F1，不能使用 `train.py` 自报的指标。读取每折结果、均值、标准差、训练耗时、实际使用的计算设备、候选 commit 和输入哈希。所有真实本机结果标记 `LOCAL_EXPERIMENTAL`，不冒充独立防篡改认证结果。

## Autonomous loop

1. 读取最新 champion、Git HEAD、`results.tsv`、逐折 JSON、上一轮错误和失败假设。
2. 提出一个可证伪的改进假设。说明它针对什么具体问题以及预期改变哪些错误。
3. 保存当前冠军的完整 commit，只改 `train.py`，检查 diff，提交候选 commit。
4. 用相同命令、同一固定协议执行正式评价。将输出保存到当前轮日志，`status.json` 由评估入口更新，供现有 Dashboard 读取。
5. 训练报错、超时、OOM、输出不完整或分数无效时记录 `CRASH`，检查末尾错误。明显实现错误最多修复两次；失败不视为 0 分、不晋级。
6. 只有完整合法的 macro-F1 严格高于已有冠军时，才能按现有 `prepare.py` 的严格规则记录 `KEEP`；否则记录 `REVERT` 并回退。分数很接近时审查同折配对差异和重复结果，不能将随机微小提升宣传成可靠进步。复杂度和推理成本用于解释与最终候选权衡，不擅自改变晋级规则。
7. 使用 `python -m research.prepare record` 写入 `results.tsv`，保留完整 commit、评价协议、决策和简短假设。每折分数和设备等详细数据保存在对应 `round-*.json`，不要求把这些字段伪装成 TSV 现有列。记录完成后，将工作树保持在 champion。
8. 用最新证据选择下一轮实验。持续执行直到停止条件出现。

运行结果必须可供现有只读 Dashboard 查看：

```bash
./solve.sh dashboard --run runs/<tag>
```

Dashboard 负责显示当前状态、轮次指标、逐折表现和 KEEP/REVERT 历史，不负责控制训练。

## Model finalization

研究者要求进入交付验收时，冻结当前 champion commit 和所有研究结果，不再根据交付验收结果修改研究冠军。

1. 在冻结的开发选择集上用冠军配置完成最终拟合，保存可重新加载的模型工件及标签顺序、预处理器、依赖版本、种子、数据哈希与源码 commit。优先复用现有 `src/syndrome/` 的训练、推理和服务代码；不要为了导出模型另起一套 AutoResearch 基础设施。
2. 测试模型重载前后对相同样例的预测一致性，并检查短主诉、空输入、非法文本、未覆盖证候和推理延迟。
3. 提供 Top-1 与可用的 Top-k 候选、逐类 Precision/Recall/F1、头尾类别分层、混淆情况、拒判和不确定性行为。未校准分数不得标为真实概率。
4. 交付模型文件路径、Git commit、环境、模型卡、`results.tsv`、逐轮 JSON、错误分析和 Dashboard 截图。没有实际生成或验收的工件必须标记 `NOT_AVAILABLE`，不能宣布模块已经完成。

## Generalization acceptance

泛化验收在冠军冻结之后进行，不能把验收集变成下一轮调参数据。

1. **同来源、不同受试者**：汇报冻结的 protocol v2 分组 CV、每折波动、逐类支持度和头尾表现。多轮开发集选择后的 CV 仍是内部开发证据，不是新的独立测试集。
2. **输入鲁棒性**：测试口语化主诉、错别字、否定症状、缩写、缺失信息、重复文本、长输入、无关文本和明显分布外输入。只有配有经核实参考标签的样本才可报告准确率或 F1；无标签测试只能报告行为一致性、拒判与错误样例。
3. **跨来源验证**：只有取得新受试者来源、独立标签且能审计映射到既定证候体系的数据时，才按事前冻结的协议执行一次外部评估；严禁用其分数反复选候选或调阈值。若无法取得，明确写 `NOT_VALIDATED`，不得据此声称外部泛化、临床准确性或商业适用性。
4. 保存通过项目检查的内部交付结论，以及尚未通过的外部验证项目。TCM-SD 数据许可为 CC BY-NC-SA 4.0，现有研究成绩不授予商业按摩椅部署许可。

## Handoff

当前证候模型达到内部工程交付标准之后，才将九体质作为另一份独立研究任务启动。现有 Mendeley 274 名受试者仅有八种偏颇体质评分，平和质缺失，不能伪造完整九体质标签。

按摩前生理预测属于再下一阶段。按 PDF 单独建立 Body 和 Physio 输入、BPSI 与 EFRI 目标及评价规则。训练期间可以使用按摩中 EEG 与按摩后变化构造目标，但推理只能使用按摩前信息，不得与 TCM-SD 证候任务混用。
