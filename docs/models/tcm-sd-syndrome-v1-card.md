# TCM-SD 证候倾向模型 v1（模型卡草稿）

状态：**草稿（#53 阶段 1）** · 最后更新：2026-11-02 · 分支 `feature/tcm-sd-syndrome-v1`

## 一句话

在 TCM-SD 真实公开临床病例上训练的中文症状文本 → 148 证候单标签倾向模型
（多数类 Dummy、字符 n-gram TF-IDF + LR 两个简单基线 + bert-base-chinese 微调），
配套标准化推理接口（ABSTAIN/OUT_OF_DOMAIN 拒判）与 realtime_feedback 知识参考接口。

**本模型不构成诊断准确率或临床有效性声明。所有真实数据上的结果均为内部
训练/验证口径，未经外部验证（独立用户/专家验证：NOT_AVAILABLE）。**

## 用途 / 非用途

- **用途**：用户主动提供中文症状描述时，给出候选证候倾向（Top-k + 分数）与
  有来源的证候知识解释；无症状时仅返回非个体化的知识参考。
- **非用途**：
  - 不是医疗诊断设备，不支持任何疗效声明（AGENTS.md：不产出临床疗效声明）；
  - 不从生理数值推断证候（realtime_feedback 结构性禁止，见 `src/syndrome/feedback.py`）；
  - 不用于商业部署（数据许可 CC BY-NC-SA 4.0 仅限非商业科研用途）。

## 数据

| 项 | 值 |
| --- | --- |
| 来源 | TCM-SD（https://github.com/Borororo/ZY-BERT ） |
| 原始规模 | 54,152 例（官方文件 43,180/5,486/5,486），52,295 用户 |
| 加载后规模 | 54,151 例 / 52,294 用户：泄漏记录剔除（见下）后经 subject-level 重划分的实际输入 |
| 标签 | 单标签 `norm_syndrome`，148 类（taxonomy `tcm-sd-syndrome-vocab-v1`） |
| 许可 | CC BY-NC-SA 4.0（非商业） |
| 划分 | subject-level：同一 `user_id` 全部病例只属一个 partition（seed 20261101）|
| 佐证知识 | syndrome_knowledge.json 1,027 条（148 条与本任务词表同名） |

类别高度不平衡（去重后训练集最大类 7,566 / 最小 9），因此**macro-F1 为主口径**。
官方 train/dev/test 文件边界仅作对照记录；实际划分按 `subject_partitions`
（user_id 跨官方文件重叠）。

**泄漏记录剔除（R1 F2）**：审计发现 1 例 train 记录的 norm_syndrome 字符串
逐字出现在自身 detection 文本（user 484293）。加载层
（`src/data/tcm_sd.py::EXCLUDED_LEAK_RECORDS`）在数据边界跳过该例，原始文件
保持只读；全部下游（训练/评估/清单）一律看到剔除后的 54,151 例流。

**评估集单向去重（R1 F1 + 主线仲裁 2026-11-02）**：评估侧病例若其**本场景
输入文本**与先前 partition 任一病例逐字节相同即被剔除（clinical 基准=
`build_clinical_text`，short 基准=`build_short_text`；基准必须随场景切换，
否则「主诉相同、现病史不同」的可记忆副本会漏进 short 场景）；test 对
train∪dev 同理；**train 侧全量保留**（train 侧重复是真实再诊数据）。
subject-level 重划分后实际病例数：train 44,407 / dev 5,412（泄漏记录剔除后
train 44,406）；clinical 去重后 dev **4,817**（剔 595）/ test（封存）3,838
（剔 495）；short 去重后 dev **3,347**（剔 2,065）/ test（封存，解锁时按同
基准计算）。去重策略与理由见
`data.tcm_sd.deduplicate_evaluation_records`。

## 模型与训练

| 模型 | 说明 |
| --- | --- |
| `dummy-most-frequent` | 全体预测训练集最频类（朴素下界） |
| `tfidf-char-1-3-logreg` | 字符 1–3 gram TF-IDF + LogisticRegression（OvR-liblinear，`class_weight=balanced`），seed 20261101。选 liblinear 原因：本环境 sklearn 1.9/scipy 1.18 多项式 lbfgs 梯度路径有严重性能缺陷（2000×500/20 类 ~51s）；OvR-liblinear 8k 文档×148 类 ~15s（2026-11-02 实测） |
| `tcm-sd-syndrome-bert-v1` | bert-base-chinese 微调，max_len 256，lr 2e-5，batch 32，逐 epoch 按 dev macro-F1 选最佳权重；6 epoch 优于 3 epoch（clinical 0.3262 vs 0.2283，第 6 epoch 已趋平 0.3197→0.3262），以 6 epoch 为准；权重本地分发（不入 Git） |

两种文本场景分别评估：`clinical`（主诉+现病史+四诊摘要完整文本）与
`short`（仅主诉）。训练/模型选择/阈值选择只使用 train/dev partition；
test partition 封存，最终解锁测试由主线按 AGENTS.md 硬规则 11 执行。

## 指标（dev partition，真实数据，按场景基准去重后）

主表为**按场景输入文本基准去重后**的数字（主线仲裁 2026-11-02：去重基准=
模型实际看到的文本——clinical 场景剔 595 例（n=4,817），short 场景剔 2,065 例
（n=3,347）；与多数类 Dummy 的配对差值 bootstrap 同口径重跑，n_boot=1000,
seed 20261101）：

| 场景 | 模型 | macro-F1（95% CI） | accuracy | Top-3 | Top-5 |
| --- | --- | --- | --- | --- | --- |
| clinical | dummy-most-frequent | 0.00203 | — | — | — |
| clinical | tfidf-char-1-3-logreg | **0.5754**（0.5007–0.5631） | 0.8360 | 0.9520 | 0.9695 |
| clinical | bert-base-chinese（6 ep，重评分） | 0.2249（—） | — | — | — |
| short | tfidf-char-1-3-logreg | **0.3449**（0.2845–0.3421） | 0.5907 | 0.8500 | 0.9032 |
| short | bert-base-chinese（6 ep，重评分） | 0.1658（—） | — | — | — |

（BERT 重评分只出了 macro-F1 与 n；CI/Top-k 未重算，如实留空。）

与多数类 Dummy 的配对差值：clinical +0.5734（CI [0.4988, 0.5611]，p≈0）；
short +0.3427（CI [0.2823, 0.3399]，p≈0）。

**去重前（含记忆效应）**：clinical TF-IDF 0.5895 / BERT 0.3262（n=5,412）；
short TF-IDF 0.3971 / BERT 0.2454。实测按场景基准去重降幅：clinical −0.0141、
short −0.0386（short 38.1% 的 dev 病例主诉与 train 逐字节相同——临床全文基准
抓不到「主诉相同、现病史不同」的可记忆副本，这是 first 修复轮用临床全文基准
只收到 −0.0136 的原因；按场景基准仲裁后修正）。BERT 在 short 下降 −0.0801，
幅度大于 TF-IDF——被剔除的模板行正是 BERT 得分高的行。
**通道披露**：BERT 的最佳 epoch 选择与 UNCERTAIN 阈值选择都发生在去重前的
dev 上，重评分原样继承（选择用 dev ≠ 评分用 dev，未重选）；TF-IDF 无选择
通道（配置冻结，seed 相同重跑），其前后差是纯去重效应。

**字符 n-gram TF-IDF 基线在两个场景都大幅优于 BERT 微调**——主因是训练文本
高度模板化（病历套话含强判别性字符 n-gram），以及尾部类别支持度过低限制
BERT 逐类表现。工程结论：以 TF-IDF 为冻结基线模型，BERT 结果如实记录为
「未胜过简单基线」。

Bootstrap 区间上界低于点估计（如 clinical 0.5754 vs CI high 0.5631）：重采样
会抽空部分支持度=1 的尾部类，重采样 macro-F1 系统性偏低，属于极小类
bootstrap 已知偏差，如实记录，不视为 bug。

详细数字（逐类 P/R/F1、环境 provenance、去重统计）见远程
`runs/tcm-sd-syndrome-v1/baselines-report.json`（按场景基准去重，权威版本；
历史中间版本在 baselines-report-deduped.json 与
`runs/tcm-sd-syndrome-v1-prededup-backup/`）、
`runs/tcm-sd-syndrome-v1/bert-*/dev-evaluation-deduped.json`
（git-ignored，acam-lab 留存）。

推理分数语义：模型原始 softmax 概率，**未经校准**（结果内 `calibrated: false`
显式声明）。UNCERTAIN 阈值在**去重前** dev 上以 ≥90% 覆盖率约束下选最大过滤后
准确率（选择通道见上）：BERT clinical 阈值 0.4392（覆盖率 0.900，过滤后
accuracy 0.8487）、BERT short 阈值 0.4036（覆盖率 0.900，过滤后 accuracy
0.7768）。

## 鲁棒性

- 合成套件（`tests/syndrome/test_robustness.py`）：短输入、否定症状（「无发热」）、
  口语化、越界（英文/乱码/证候名当输入）、缺失字段——接口不崩、状态合法、
  拒判路径不给候选。
- 真实数据数字（按场景基准去重后 dev partition，TF-IDF 短文本场景，阈值
  0.2061）：文本截断到 20 字符后 macro-F1 0.3449 → 0.3434（下降 0.0015，
  短文本场景天然对截断不敏感）；7 个对抗探针拒判 6 个（85.7%）：空输入
  ABSTAIN ×2、英文/乱码/证候名（「气虚血瘀证」）OUT_OF_DOMAIN ×4。
- 已知盲点：「脾肾阳虚」（词表内名称为「脾肾阳虚证」，无「证」字）未被
  证候名拒判逻辑捕获，按常规输入返回 OK——拒判仅匹配词表精确同名，
  前缀/别名形式的证候名输入不可拒，如实记录。
- 真实否定症状改写无真值：**NOT_AVAILABLE**（仅有合成接口行为测试）。

## 局限（重要）

1. 类别极不平衡，尾部 100+ 类支持度低，逐类指标不稳定（bootstrap 区间如实变宽）；
2. 训练分布来自住院病历书写风格，对口语化用户输入的泛化未经验证；
3. subject-level 划分消除了同人泄漏，但语料来自同一来源医院体系，跨来源
   泛化（TCM-Lung / TCMEval-SDT）在本阶段未评估（NOT_COMPUTABLE，待映射审计）；
4. 分数未校准，不能作概率解释；
5. 「证候倾向」不是诊断；无设备体征/舌脉输入时信息先天不足，接口会显式列出
   缺失字段并以 UNCERTAIN/ABSTAIN 拒判；
6. 非商业数据许可不授予商业部署权。

## 复现

```bash
# 远程开发机（acam-lab，权重与数据经 rsync 放 data/local/，均不入 Git）
.venv/bin/python -m pip install -r requirements-dev.txt   # 含 transformers>=4.40、accelerate>=0.30
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   # 见下「环境缺陷」
python -m syndrome.experiment baselines --raw-dir data/local/tcm-sd --out runs/tcm-sd-syndrome-v1 --solver ovr-liblinear
python -m syndrome.experiment robustness --raw-dir data/local/tcm-sd --out runs/tcm-sd-syndrome-v1
python -m syndrome.experiment bert --raw-dir data/local/tcm-sd \
    --weights data/local/models/bert-base-chinese --out runs/tcm-sd-syndrome-v1 --scenario short
# 已保存 BERT 权重在去重后 dev 上的重评分（不重训、不重选阈值，R1 F1）
python -m syndrome.experiment bert-rescore --raw-dir data/local/tcm-sd \
    --out runs/tcm-sd-syndrome-v1-e6 --scenario short
pytest tests/syndrome
```

推理：

```python
from syndrome.experiment import build_bert_inference

inf = build_bert_inference("runs/tcm-sd-syndrome-v1/bert-short")
result = inf.predict_one("最近总是失眠心悸，食欲差", fields={"age": 45, "gender": "女"})
print(result.to_dict())
```

证据状态词汇：`NOT_AVAILABLE` / `UNVERIFIED` / `SMOKE_ONLY` / `NOT_VALIDATED`，
按 AGENTS.md 硬规则 7 显式使用，不做隐式补全。

环境缺陷记录（acam-lab，sklearn 1.9.1 / scipy 1.18.1 / numpy 2.5.4）：多项式
lbfgs 走有限差分梯度（真实规模不可用）、OvR n_jobs>1 触发 joblib
`WRITEBACKIFCOPY` 崩溃、OpenBLAS 多线程空转——因此冻结 `ovr-liblinear`
（n_jobs=1，单线程 BLAS）。`--solver multinomial-saga` 备选口径已实现：
真实规模训练（43k 文档 × 148 类，max_iter=2000）运行 ~96 CPU 分钟仍未收敛、
无任何产出，已终止；**saga vs liblinear 的 dev 对比为 NOT_AVAILABLE**（原因：
saga 在该规模收敛预算不可行），基线求解器以 ovr-liblinear 冻结。
