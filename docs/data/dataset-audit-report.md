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
| 标准版本 | **ZYYXH/T157-2009（CCMQ-60）**，条目数指纹验证（见下节）；禁止静默换算为 GB/T 46939—2025（#38 认证映射前） |
| 列缩写语义 | 单一审计映射 `SOURCE_COLUMN_MAPPING`（`src/data/mendeley.py`，状态 `FINGERPRINT_CROSS_CHECKED`，2026-10-30）：YaD→阳虚、YiD→阴虚、QiDf→气虚、PW→痰湿、WH→湿热、BS→血瘀、SDa→特禀、QDp→气郁；论文原文直接确认 `QiDf=Qi-deficiency`，其余缩写为标准中医英文对应并与条目数指纹一致；论文缩写定义表在付费墙内，逐条对照原文仍开放 |

### 1.1 列→九维规范顺序映射与条目数指纹（2026-10-30，#38 冻结修正）

CCMQ 计分公式 `(sum−n)/(4n)×100` 决定 n 项子量表只能产生 `100/(4n)` 的整数倍，
因此每列分数的可行条目数（fingerprint）可与 ZYYXH/T157-2009 子量表条目数
（PMC10164641 Table 1：气虚 8、阳虚 7、阴虚 8、痰湿 8、**湿热 6**、血瘀 7、气郁 7、
特禀 7；判定阈值偏颇质 30/40、平和质 60+30/40）逐列比对：

| 列 | 实测指纹（唯一可行 n） | 标准条目数 | 映射 |
| --- | --- | --- | --- |
| YaD | 7（220 行唯一确定） | 阳虚 7 | yang_deficiency |
| YiD | 8（136 行唯一确定） | 阴虚 8 | yin_deficiency |
| QiDf | 8（128 行唯一确定） | 气虚 8 | qi_deficiency |
| PW | 8（134 行唯一确定） | 痰湿 8 | phlegm_damp |
| WH | 6（168 行唯一确定，最具区分度） | 湿热 6 | damp_heat |
| BS | 7（245 行唯一确定） | 血瘀 7 | blood_stasis |
| SDa | 7（240 行唯一确定） | 特禀 7 | special_diathesis |
| QDp | 7（217 行唯一确定） | 气郁 7 | qi_depression |

8 列指纹在上述映射下与标准条目数**全部一致**；缩写语义与指纹相互独立、结论相同。
适配器按规范顺序 `[平和(缺失), 气虚, 阳虚, 阴虚, 痰湿, 湿热, 血瘀, 气郁, 特禀]`
输出并以 `score_dims` 携带名称，测试逐维按名称断言（#38 修正前的
`sorted(score_idx)` 源列序输出即静默维度错位 bug，已移除）。

**QC 异常：subject row 273（0 基，最后一名受试者）** 的 YaD/PW/WH/QDp 四列值
违反本列指纹（例如 YaD=46.875 仅符合 8 项粒度），疑为该名受试者计分口径不同或
录入异常；其余 273 行全部一致。该行禁止混入 #38 规则交叉校验；训练期使用时须
在 QC 报告中单列。

**结论：`train_candidate`（8/9 维 masked 评分回归，PUBLISHED 标签，ZYYXH/T157-2009 条目数指纹验证）**；不宣称九维完整、不宣称可做九类互斥分类训练（本文件无分类标签；评分→分类转换须经 #38 规则引擎的已验证阈值 30/40/60，产物为 RULE 施于 PUBLISHED 评分的派生标签，不是本文件的原始标签）。

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

`mendeley-dataset-files` 与 `tcm-label-rules-historical-verified`（#38：SAS 字节
交叉校验 + ZYYXH/T157-2009 阈值 + 版本指纹，见 §1.1）均已 VERIFIED（2026-10-30），
TCM 的 GO_REAL_DATA_TRAINING 由 `src/evidence.evaluate_gates` 机械推导为 **GRANTED**；
GO_RESEARCH_CLAIMS 仍缺 independent-reference-data。GB/T 46939—2025 主张保持
NOT_VERIFIED，仅约束未来换版工作，不阻塞本数据路径。

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
