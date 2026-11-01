# TCM-SD 证候倾向模型 v1（模型卡草稿）

状态：**草稿（#53 阶段 1）** · 最后更新：2026-11-01 · 分支 `feature/tcm-sd-syndrome-v1`

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
| 规模 | 43,180 训练 / 5,486 dev / 5,486 test 临床病例 |
| 标签 | 单标签 `norm_syndrome`，148 类（taxonomy `tcm-sd-syndrome-vocab-v1`） |
| 许可 | CC BY-NC-SA 4.0（非商业） |
| 划分 | subject-level：同一 `user_id` 全部病例只属一个 partition（seed 20261101）|
| 佐证知识 | syndrome_knowledge.json 1,027 条（148 条与本任务词表同名） |

类别高度不平衡（训练集最大类 7,380 / 最小 7），因此**macro-F1 为主口径**。
官方 train/dev/test 文件边界仅作对照记录；实际划分按 `subject_partitions`
（user_id 跨官方文件重叠）。

## 模型与训练

| 模型 | 说明 |
| --- | --- |
| `dummy-most-frequent` | 全体预测训练集最频类（朴素下界） |
| `tfidf-char-1-3-logreg` | 字符 1–3 gram TF-IDF + LogisticRegression（OvR-liblinear，`class_weight=balanced`），seed 20261101。选 liblinear 原因：本环境 sklearn 1.9/scipy 1.18 多项式 lbfgs 梯度路径有严重性能缺陷（2000×500/20 类 ~51s）；OvR-liblinear 8k 文档×148 类 ~15s（2026-11-02 实测） |
| `tcm-sd-syndrome-bert-v1` | bert-base-chinese 微调，max_len 256，lr 2e-5，batch 32，逐 epoch 按 dev macro-F1 选最佳权重；6 epoch 优于 3 epoch（clinical 0.3262 vs 0.2283，第 6 epoch 已趋平 0.3197→0.3262），以 6 epoch 为准；权重本地分发（不入 Git） |

两种文本场景分别评估：`clinical`（主诉+现病史+四诊摘要完整文本）与
`short`（仅主诉）。训练/模型选择/阈值选择只使用 train/dev partition；
test partition 封存，最终解锁测试由主线按 AGENTS.md 硬规则 11 执行。

## 指标（dev partition，真实数据，n=5,412 病例 / 148 类全部出现）

| 场景 | 模型 | macro-F1（95% CI） | accuracy | Top-3 | Top-5 |
| --- | --- | --- | --- | --- | --- |
| clinical | dummy-most-frequent | 0.00197 | — | — | — |
| clinical | tfidf-char-1-3-logreg | **0.5895**（0.5143–0.5811） | 0.8503 | 0.9573 | 0.9728 |
| clinical | bert-base-chinese（6 ep） | 0.3262（0.2921–0.3318） | 0.7984 | 0.9172 | 0.9462 |
| short | tfidf-char-1-3-logreg | **0.3971**（0.3403–0.3957） | 0.6742 | 0.8830 | 0.9278 |
| short | bert-base-chinese（6 ep） | 0.2454（0.2234–0.2537） | 0.7284 | 0.8934 | 0.9274 |

与多数类 Dummy 的配对差值（n_boot=1000, seed 20261101）：clinical
+0.5876（CI [0.5123, 0.5791]，p≈0）；short +0.3952（CI [0.3383, 0.3937]，p≈0）。
**字符 n-gram TF-IDF 基线在两个场景都大幅优于 BERT 微调**——主因是训练文本
高度模板化（病历套话含强判别性字符 n-gram），以及尾部类别支持度过低限制
BERT 逐类表现。工程结论：以 TF-IDF 为冻结基线模型，BERT 结果如实记录为
「未胜过简单基线」。

Bootstrap 区间上界低于点估计（如 clinical 0.5895 vs CI high 0.5811）：重采样
会抽空部分支持度=1 的尾部类，重采样 macro-F1 系统性偏低，属于极小类
bootstrap 已知偏差，如实记录，不视为 bug。

详细数字（逐类 P/R/F1、环境 provenance）见远程
`runs/tcm-sd-syndrome-v1/baselines-report.json`、
`runs/tcm-sd-syndrome-v1-e6/bert-*/dev-evaluation.json`（git-ignored，acam-lab 留存）。

推理分数语义：模型原始 softmax 概率，**未经校准**（结果内 `calibrated: false`
显式声明）。UNCERTAIN 阈值在 dev 上以 ≥90% 覆盖率约束下选最大过滤后准确率：
BERT clinical 阈值 0.4392（覆盖率 0.900，过滤后 accuracy 0.8487）、BERT short
阈值 0.4036（覆盖率 0.900，过滤后 accuracy 0.7768）。

## 鲁棒性

- 合成套件（`tests/syndrome/test_robustness.py`）：短输入、否定症状（「无发热」）、
  口语化、越界（英文/乱码/证候名当输入）、缺失字段——接口不崩、状态合法、
  拒判路径不给候选。
- 真实数据数字（dev partition，TF-IDF 短文本场景，阈值 0.2068）：文本截断到
  20 字符后 macro-F1 0.3971 → 0.3953（下降 0.0018，短文本场景天然对截断不
  敏感）；7 个对抗探针拒判 6 个（85.7%）：空输入 ABSTAIN ×2、英文/乱码/
  证候名（「气虚血瘀证」）OUT_OF_DOMAIN ×4。
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
