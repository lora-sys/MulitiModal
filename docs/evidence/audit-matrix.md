# #47 证据审计矩阵（2026-10-30）

机器可读源：[registry.json](registry.json)（`src/evidence` 加载并据此机械推导 GO 门）。
状态词汇：VERIFIED / UNVERIFIED / NOT_VERIFIED / NOT_AVAILABLE / NOT_APPLICABLE（#22/#47）。

| claim_id | 来源 | 支持的事实 | 状态 | 缺口 / 阻塞 |
| --- | --- | --- | --- | --- |
| pdf-spec | 仓库内 PDF（SHA-256 29727dace…d781） | BPSI Eq.2 / EFRI Eq.3 / 双分支网络 Eq.4–9 的**设计意图** | VERIFIED | 结果为占位符；非实验证据 |
| tcm-engineering-smoke | tests/（commits 65df01b…ca22434） | TCM 工程闭环（模型/训练/评估/推理/RSI harness）合成数据端到端可用 | VERIFIED | SMOKE_ONLY；不支撑任何科研/临床结论 |
| physio-engineering-smoke | tests/runtime | 生理 V0 前向/损失/checkpoint 合成可用，During/Post 拒绝生效 | VERIFIED | SMOKE_ONLY；真实数据 NOT_AVAILABLE |
| mendeley-dataset-files | data.mendeley.com/cxtph3tjsg/1（SHA 6206f432…，274 人 × 71 列，8 维评分零缺失） | 文件已下载并逐列审计；8/9 维 PUBLISHED 评分 + 独立特征 + CC BY 4.0；train_candidate（CCMQ-2009） | VERIFIED | 剩余：#38 计分语义映射与标准版本认证；第 9 维按规则 mask，永不伪造 |
| tcm-label-rules-verified | openstd GB/T 46939—2025 | —（官方全文未取得，逐题比对未做） | **NOT_VERIFIED** | GO_REAL_DATA_TRAINING(tcm) 的唯一剩余阻塞项；第三方转录不作权威版本 |
| independent-reference-data | — | —（不存在） | **NOT_AVAILABLE** | 阻塞双模型 GO_RESEARCH_CLAIMS |
| physio-real-data | — | —（未采集） | **NOT_AVAILABLE** | 阻塞 #26/#42 与 GO_REAL_DATA_TRAINING(physio) |
| autoresearch-upstream-license | karpathy/autoresearch @228791fb（README SHA 3958fd41…） | MIT 可用（README 声明 + 维护者 issue #523 确认；API license=null、无 LICENSE 文件已记录） | VERIFIED | 复制片段须记录出处；本项目独立实现循环 |
| autoresearch-agent-isolation | #45 里程碑 1 | 仅静态检查 + 子进程分离已验证 | **NOT_VERIFIED** | 阻塞真实 Agent E2E；当前状态 AGENT_ISOLATION_UNVERIFIED |
| zenodo-ccmq-scripts | zenodo.org/records/4431679（md5 已核） | 研究者计分代码 + 14 人 TEST_DATA 取回；rules_test_only | VERIFIED | 研究者代码≠官方计分器；不作监督训练集 |
| su-2019-paper | doi:10.1016/j.eujim.2019.04.001 | —（未取回） | UNVERIFIED | 与 #37 一并取回 |

已证伪候选（不注册）：Figshare 10.6084/m9.figshare.15087891.v1 为 COVID 医护人员数据，与中医体质无关（#47 明确排除）。

## GO 门（由 `src/evidence.evaluate_gates` 机械推导，禁止由 smoke 分数升级）

| 模型 | GO_ENGINEERING | GO_REAL_DATA_TRAINING | GO_RESEARCH_CLAIMS |
| --- | --- | --- | --- |
| tcm_constitution | **GRANTED** | NOT_GRANTED（仅缺 tcm-label-rules-verified；mendeley 文件已 VERIFIED） | NOT_GRANTED（另缺 independent-reference-data） |
| physio_bpsi_efri | **GRANTED** | NOT_GRANTED（缺 physio-real-data、physio-label-rules-verified） | NOT_GRANTED（另缺 independent-reference-data） |

负例保障（tests/evidence）：无哈希/访问日期的 claim 不能标 VERIFIED；NOT_VERIFIED/NOT_AVAILABLE 必须写明缺口或阻塞；把数据 claim 伪标 VERIFIED 之前会被构造器拒绝——任何"来源错误/缺标准原文/标签不匹配"情况都只能停在 NOT_GRANTED。
