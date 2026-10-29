# Dataset Audit Report — Mendeley cxtph3tjsg/1 与 Zenodo 4431679（#37）

审计日期：2026-10-30；执行环境：开发机 acam-lab（文件只读，原始文件不入 Git）。
机器可读注册表：`src/data/datasets.py`（DATASET_REGISTRY）；证据状态：`docs/evidence/registry.json`。

## 1. Mendeley cxtph3tjsg/1（Su et al. 2019, DOI 10.17632/cxtph3tjsg.1）

| 项 | 结果 |
| --- | --- |
| 获取方式 | 官方文件列表 API `https://data.mendeley.com/api/datasets/cxtph3tjsg/files?version=1`（落地页 SPA 不含文件数据；`api.data.mendeley.com/datasets/publics/...` 需鉴权返回 Unauthorized） |
| 文件 | `Share data.xls`，289,792 字节 |
| SHA-256 | `6206f432b683f8a7b9a75dae23e598a4f01a41a4f8af84b099924b9d85d08a96`（磁盘实测 == API 元数据 ✓） |
| 许可 | CC BY 4.0（落地页声明；署名使用，无再识别风险评估的例外） |
| 结构 | 1 个 sheet `data`，275 行 × 71 列 = **274 名受试者 × 71 特征/标签列** |
| 受试者 | **无显式 ID 列**；行序即匿名身份；整行重复 = 0 |
| 标签 | 列 63–70：`YaD, YiD, QiDf, PW, WH, BS, SDa, QDp` —— **8 个体质评分，0–100，274/274 全部观测、零缺失**；范围 [0, 100.0]；与论文"八种偏颇体质"一致 |
| 第 9 维 | **平和质列不存在** → 按规则记为 unobserved（score_mask=false），永不伪造（#37 冻结小节 3） |
| 特征 | 人口学 0–5、体格 6–8（BH/BW/BMI）、经络 9–20（1 列缺失）、脉搏波 36–50（SYS…MWTCI）、舌诊 51–61；列 62（`b`）未定义、跳过 |
| 缺失 | 仅 h2/h3 经络列各缺 1 行；标签列零缺失 |
| 标准版本 | 分数遵循 **CCMQ-2009**（论文 2019）；禁止静默换算为 GB/T 46939—2025（#38 认证映射前） |
| 列缩写语义 | YaD/YiD/QiDf/PW/WH/BS/SDa/QDp → 体质维度的权威映射待 #38 对照官方 CCMQ 计分认证；当前不作语义断言 |

**结论：`train_candidate`（8/9 维 masked 评分回归，PUBLISHED 标签，CCMQ-2009）**；不宣称九维完整、不宣称可做九类互斥分类训练（本文件无分类标签）。

## 2. Zenodo 4431679（CCMQ 研究者计分代码）

| 项 | 结果 |
| --- | --- |
| 文件 | `CCMQ SAS code.sas`（md5 3b622e25…）、`CCMQ SPSS syntax.sps`（d843895a…）、`TEST_DATA.xlsx`（363d3e38…，14 人 × 63 题作答） |
| 许可 | CC BY 4.0 |
| 定位 | **`rules_test_only`**：仅用于 #38 规则引擎实现与研究计分的一致性校验；**不是监督训练集**（14 人）；研究者代码 ≠ 官方计分器（#47） |

## 3. 准入判定汇总

| 数据集 | eligibility | 理由 |
| --- | --- | --- |
| mendeley-cxtph3tjsg-v1 | **train_candidate** | 真实逐受试者 8 维 PUBLISHED 标签 + 独立匹配特征 + CC BY 4.0 + 逐行 subject 划分可验证（冻结小节 3 的准入条件按 8 维满足；第 9 维 mask） |
| zenodo-4431679 | rules_test_only | 计分参考代码 + 微型测试数据 |
| Harvard DVN/COJZMQ、Dryad TMC-Tongue、NCMI 9703 人库 | pending_verification | 未下载/未审计（#37 冻结小节"Additional data discovery"）；Dryad 仅 20 个舌象标签，非九体质 |
| Figshare 10.6084/m9.figshare.15087891.v1 | **rejected** | COVID 医护数据，与中医体质无关（已证伪，#47 同步记录） |

## 4. 对 GO 门的影响

`mendeley-dataset-files` 证据升为 VERIFIED（见 registry.json 更新）后，TCM 的
GO_REAL_DATA_TRAINING 剩余阻塞项仅 **tcm-label-rules-verified**（#38：官方/研究者
计分一致性 + 标准版本认证）。GO_RESEARCH_CLAIMS 仍需 independent-reference-data。
本报告不改变任何 GO 判定，判定仍由 `src/evidence.evaluate_gates` 机械推导。

## 5. 可复现命令

```bash
# 开发机 acam-lab 上（文件已在 data/local/，哈希见上表）
cd /home/lora/workspace/projects/MulitiModal
.venv/bin/python -c "
from src.data import DATASET_REGISTRY, verify_files
for rid, rec in DATASET_REGISTRY.items():
    print(rid, verify_files(rec))
"
PYTHONPATH=src .venv/bin/python -m pytest tests/data -q   # 含真实文件审计测试
```
