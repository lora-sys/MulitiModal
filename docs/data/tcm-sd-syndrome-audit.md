# TCM-SD 证候数据审计报告（#53 阶段 1）

审计日期：2026-11-01。审计人：实现子 Agent（#53 分支 `feature/tcm-sd-syndrome-v1`）。
数据位置：`data/local/tcm-sd/`（git-ignored，原始文件只读）。清单可重放：
`build_tcm_sd_manifest()`（src/data/tcm_sd.py），本次运行输出
`runs/data-manifests/tcm-sd-manifest.json`。

## 0. 结论速览

| 项 | 结论 |
| --- | --- |
| 主数据可用性 | **可用作 train_candidate**：54,152 例全部有 `norm_syndrome`，148 类封闭集，0 个 vocab 外标签 |
| 标签泄漏（证候名出现在输入文本） | 1 / 54,152（train，user 484293）——可在预处理阶段剔除，不构成阻塞 |
| 医生结论性/处方性文本 | 结论性辨证措辞（证属/辨证为/诊为/诊断为）出现在 9,522 例（17.58%），处方类措辞 305 例（0.56%）——见 §5 的两个评估口径 |
| Subject-level 划分 | **必须重划分**：官方 split 的 user_id 跨文件重叠（19/18/13）；真实独立用户数 52,295，不是早期探针口径的 52,345 |
| 精确文本重复 | 2,635 组重复（涉及 8,612 例，占 15.9%）；334 组组内 norm 标签互相冲突；1,382 组跨不同用户 |
| 许可证 | CC BY-NC-SA 4.0：仅科研非商业；**不授予商业部署权**，模型作为数据衍生品对外部署前需单独许可确认 |
| 标签性质 | 单标签（每例一个 norm_syndrome）；148 类封闭集 → 单头 softmax 输出，macro-F1 为主指标 |
| 外部候选 TCM-Lung | **NOT_COMPUTABLE**：标签空间是匿名整数 ID（50 类），无任何名称解码，映射覆盖率 0/148；且无 LICENSE |
| 外部候选 TCMEval-SDT | 部分可对齐：精确匹配 0/407 个不同标签；去"证"尾模糊匹配 42/407（实例级 93/493 = 18.9%）；只能在该对齐子集上算跨库指标，其余 NOT_COMPUTABLE |

证据状态说明：本报告所有数字均来自 2026-11-01 在本机对实际文件的全量扫描
（脚本对 54,152 行逐一解析），非抽样推断；§5 的措辞统计是**标记词级**审计，
未经人工逐例复核，结论强度以"出现率下限"理解。

## 1. 来源、哈希与完整性

来源：https://github.com/Borororo/ZY-BERT （train/dev 来自 `TCM_SD_train_dev.zip`，test 为带答案的 `test.json`）。许可：CC BY-NC-SA 4.0。

| 文件 | 行数 | 字节 | SHA-256（前 16 位） |
| --- | --- | --- | --- |
| train.json | 43,180 | 57,187,960 | `d29d3be5eccc82a1` |
| dev.json | 5,486 | 7,577,225 | `34e0e85f03d2d3e4` |
| test.json | 5,486 | 7,398,155 | `cfcd6f20e74c6658` |
| syndrome_vocab.txt | 148 | 2,462 | `63ee108754d90914` |
| syndrome_knowledge.json | 1,027（JSONL） | 1,230,148 | `c98b830f4d13c44b` |

完整哈希固化在 `src/data/tcm_sd.py` 的 `EXPECTED_SHA256`，加载前强制校验，
不一致即拒绝载入（哈希不匹配 → `ValueError`）。字段完整性：8 个字段
（user_id / lcd_id / lcd_name / syndrome / norm_syndrome / chief_complaint /
description / detection）在全部 54,152 例中**无缺失键**；空值仅有
chief_complaint 19 例（train 14 / dev 3 / test 2）、detection 4 例
（train 3 / dev 1）；**没有全文本为空的病例**。

## 2. 标签空间与 vocab 一致性

- `syndrome_vocab.txt`：148 行、148 个唯一证型名。全部 148 个 vocab 术语在
  `syndrome_knowledge.json`（1,027 条，Name 唯一）中都有知识条目；反向多余 879 条
  （知识库是超集），47 条知识条目存在空字段（加载后原样保留，不补零）。
- **监督目标是 `norm_syndrome`**：三个文件中 norm_syndrome 与 vocab 的差集为空
  （0/54,152 出界）。原始 `syndrome` 字段是未归一化的医生原始写法（train 278 种
  写法、136 种在 vocab 外），与 norm 不一致的有 6,725 例（12.42%）——原始字段仅作
  溯源保留，禁止作为训练标签。
- 类别分布（norm_syndrome，全量）：148 类，最少 11 例（瘀血内阻证、气血失调证、
  肾阳虚证、阳虚痰凝证），最多 9,226 例。长尾极重：约 20 个类别不足 15 例，
  逐类 P/R/F1 对这些类必须连带"支持样本数不足"声明，不得单独解读。
- 标签性质：**单标签**。256 个用户跨多次就诊出现 >1 种 norm_syndrome（按用户聚合
  才有多标签语义；按病例永远是单标签）。按 #53，这决定输出头为 148 维单头分类。

## 3. 用户与 subject-level 划分

- 独立用户：train 41,724 / dev 5,305 / test 5,316，**全量并集 52,295**。
- 官方 split 的 user_id 重叠：train∩dev = 19，train∩test = 18，dev∩test = 13
  （共 50 人跨文件）。早期探针报告的"52,345 用户"是三个文件各自去重后的**直接
  相加**，把 50 个跨文件用户重复计了一次；本报告与
  `DatasetRecord(n_subjects=52295)` 以并集 52,295 为准。
- 单用户最多 8 条记录（train，1 人）；绝大多数用户 1 条（40,296 人）。
- **结论**：官方 split 违反 subject-level 不变量（AGENTS.md 硬规则 2），一律弃用。
  训练/评估使用 `src/data/tcm_sd.py::subject_partitions(records, seed=20261101)`：
  按每用户多数 norm_syndrome 分层、层内固定种子洗牌、目标比例 0.82/0.10/0.08。
  实测落点 train/dev/test = 42,889 / 5,223 / 4,183 用户（82.01% / 9.99% / 8.00%），
  三个分区两两不交、并集恰为全体用户，同输入同 seed 结果逐位一致（有测试）。
  该划分是确定性的，可独立重放。

## 4. 精确文本重复

以"chief_complaint + description + detection 拼接全文"逐字节比较：

- 重复组：**2,635 组**，涉及 **8,612 例**（15.90% 的病例）。
- 组内 norm_syndrome 互相冲突的：**334 组**（同一文本被贴不同证型标签——直接
  训练会互为噪声）。
- 跨不同用户的重复组：**1,382 组**（其余为同人多次就诊的重复录入）。

处置建议（供 #53 后续阶段采用）：重复组不跨分区移除（会引入分区依赖的偏差），
而是在训练折内按组去重或对组内样本加权；跨分区"同文本不同标签"的 334 组在
逐类指标解读时单独声明。冻结划分本身不含去重逻辑，保持可重放。

## 5. 泄漏扫描与两个评估文本口径

对全量 54,152 例扫描（输入文本 = chief_complaint + description + detection）：

- **证候名直接泄漏**：norm_syndrome 字符串出现在文本中的仅 **1 例**
  （train，user 484293，`detection` 内含"辩证当属湿热瘀阻证"）；dev/test 为 0。
  **已实现剔除**（R1 F2）：加载层 `src/data/tcm_sd.py::EXCLUDED_LEAK_RECORDS`
  在 `load_tcm_sd_records` 边界跳过该例（原始文件保持只读不动），全仓下游
  一律看到剔除后的 54,151 例流。其余 54,151 例可认为无标签词泄漏
  （词面级，未做同义改写检测——同义改写泄漏状态记为 `UNVERIFIED`）。
- **医生结论性辨证措辞**（证属 / 辨证为 / 诊为 / 诊断为）：**9,522 例（17.58%）**，
  几乎全部位于 description（现病史叙述中的辨证结论），chief/detection 各 1 例。
  这些是医生结论性文本混入输入，属于任务设计层面（非字符串级）的软泄漏。
- **处方/治疗类措辞**（处方：/方药/方用/治以/治法）：305 例（0.56%）；
  西医治疗类措辞（予以/静滴/口服/静点）12,912 例（23.84%），均为住院经过叙述。

两个冻结的评估口径（#53 要求，训练与部署输入特征口径一致）：

1. **临床完整口径**：`build_clinical_text`（主诉+病史+四诊/检查拼接）——受上述
   17.6% 结论性措辞影响，指标偏乐观，只作内部上限参考。
2. **用户简短口径**：`build_short_text`（仅主诉；19 例为空串）——贴近真实用户
   输入，是主报告口径。两口径的性能差必须在 #53 交付时一并报告。

## 6. 许可证结论

- TCM-SD 数据：**CC BY-NC-SA 4.0**（作者 Borororo/ZY-BERT）。
- 含义：可用于科研与非商业用途；**禁止商业部署**。按摩椅产品若商用该证候模型，
  该模型是 NC-SA 数据的衍生品，ShareAlike 与 NC 双重约束，需单独取得作者许可
  或更换训练数据——部署前这项是硬门槛，不是免责声明能绕开的。
- 病例含真实患者 ID（user_id 为真实编号字符串）与临床叙述：数据本身不入 Git
  （已由 .gitignore `data/local/` 覆盖并验证），任何对外产物只允许哈希与统计。

## 7. 外部候选审计

### 7.1 TCM-Lung（PresRecST，https://github.com/2020MEAI/PresRecST）

- 已下载本机审计：`data/local/tcm-lung-presrecst/TCM_Lung.xlsx`
  （SHA-256 `68e911dd0fe0c35a…`，274,951 字节）。
- 结构：Data 表 4,484 行 × 4 列（SymptomID / SyndromeID / TreatmentID / HerbID），
  全部是**匿名整数 ID**；Syndrome Dictionary 仅 50 个裸整数（1–50），Symptom
  Dictionary 974 个裸整数——**没有任何证候或症状的名称解码**。SyndromeID 列
  114 种取值（含 "8,9" 这类多证组合，多标签）。
- **结论：NOT_COMPUTABLE**。与 `tcm-sd-syndrome-vocab-v1` 的一对一映射验证在
  根本上不可进行：映射覆盖率 **0/148**，也没有任何文本可做跨库去重。跨库 F1
  声明被此阻断。
- 附带：仓库无 LICENSE 文件、README 无许可声明 → 复用许可 `NOT_AVAILABLE`；
  无受试者 ID（行是症状集→证 ID 的映射，不是患者记录）。这两条独立成立，
  任一条都足以否决其作为 #53 评估集。

### 7.2 TCMEval-SDT（https://doi.org/10.6084/m9.figshare.27184596 ，figshare 现为 v5）

- 已下载本机审计：3 个 JSON（Train/Validation/Test，v1）+ 2 个 result 文件 +
  README，存放于 `data/local/tcmeval-sdt/`（哈希见 `DATASET_TCMEVAL_SDT`）。
  许可 **CC BY 4.0**（figshare 元数据），300 例古典医籍病案（200/50/50），
  病例级 ID（病例N），无受试者身份。
- 标签可得性：Train 200/200 带金标 `TCM Syndrome` 文本；**Validation/Test 的
  金标不在 JSON 内**（答案字段为空串），但 figshare 后上传的
  `Validation_data_result.txt` / `Test_data_result.txt` 以
  "病例ID@症状@病机答案字母@证候答案字母@解析" 格式给出答案字母，与各记录
  自带的 `Options of TCM Syndrome`（逐例不同选项）联查可 100% 解码
  （50/50 + 50/50，无未解析字母）。
- **标签对齐结论**（对 148 vocab）：
  - 精确字符串匹配：**0 / 407** 个不同标签。源标签是自由发挥的古典短语
    （"上热下寒""伤寒夹食之证"），不是标准证型命名。
  - 模糊口径（去掉尾字"证"再比对）：**42 / 407** 个不同标签、实例级
    **93 / 493（18.9%）** 可对齐（如 气滞血瘀、阴虚火旺、肝阳上亢 等）。
  - 源标签为**多标签**（train 117/200 例 ≥2 个证）。
- **结论**：跨库评分只能在"去证尾对齐子集 + 单标签记录"上做，其余标记
  `NOT_COMPUTABLE`；映射表按模糊口径 42 项冻结前需人工复核同义性
  （如"血热"→"血热证"是否同一概念），当前状态 `UNVERIFIED`。
- 跨库重复：TCMEval 全部 Clinical Data / Clinical Information 的 ≥12 字片段与
  TCM-SD 95,009 个非空文本字段精确匹配 **0 例**——未发现跨库重复病案。

### 7.3 汇总

| 数据集 | 规模 | 许可 | 对齐覆盖率 | 可用性 |
| --- | --- | --- | --- | --- |
| TCM-SD | 54,152 例 / 52,295 人 | CC BY-NC-SA 4.0 | 本体（148/148） | train_candidate |
| TCM-Lung (PresRecST) | 4,484 行（匿名 ID） | not stated | NOT_COMPUTABLE（0/148） | rejected |
| TCMEval-SDT | 300 例 | CC BY 4.0 | 模糊 42/407（实例 18.9%） | pending_verification（仅对齐子集可评分） |

## 8. 验证方式（可独立重放）

- 全量扫描脚本：对三个 JSONL 逐行 `json.loads`，无抽样；哈希用 `shasum -a 256`。
- 代码层验证：`tests/data/test_tcm_sd.py`（合成 fixture：哈希拒绝、分区不变量、
  文本构造、vocab/knowledge 加载、清单可重放；真实数据测试在 `data/local/tcm-sd`
  缺失时 skip）。
- 清单重放：`PYTHONPATH=src python -c "from data.tcm_sd import build_tcm_sd_manifest,
  RAW_DIR_NAME; build_tcm_sd_manifest(RAW_DIR_NAME)"`，输出与
  `runs/data-manifests/tcm-sd-manifest.json` 一致（runs/ 不入 Git）。
