# Issue #58 — 双模式产品接口（dual-mode service contract）

Status: IMPLEMENTED; SYNTHETIC_TESTS_PASS_ON_ACAM_LAB; TWO_REVIEW_ROUNDS_GO_WITH_FINDINGS (fixed in-tree)

## 交付内容

- `src/syndrome/service.py`：统一双模式契约 `tcm-service-schema-v1`。
  - `SYMPTOM_ANALYSIS`：症状文本 → 版本化 `ModelRegistry` 后端 → Top-k 候选（全部 `PREDICTED` + `calibrated:false`）。后端可替换（线性 ↔ 神经 = 换 registry 条目），默认后端为 TF-IDF+LR——接口不预设神经模型获胜。
  - `REALTIME_KNOWLEDGE`：设备测量 → `MEASURED` 实测回执 + `KNOWLEDGE` 非诊断性主题知识；结构性不加载、不调用证候分类器（AST 断言 + 子进程 import 证明 + `service._inferences == {}` 行为断言三重保证）。
- `src/syndrome/feedback.py`：移除 `METRIC_KEYWORDS`（指标关键词 → 证候条目检索会被误读为个人证候关联），替换为指标名 → 一般健康主题（`METRIC_TOPICS`，与数值和个人无关）；知识条目 `source/license/version` 缺一即 fail-closed 排除；每条测量带 `disposition` 回执（accepted / low_quality / unknown_metric / missing_value / invalid_value / out_of_range / invalid_timestamp / timestamp_regression）与重连标记。
- `scripts/syndrome_service_demo.py`：CLI 演示；有真实工件时注册后端并输出真实候选，无工件时 `NOT_AVAILABLE` + ABSTAIN（无合成模型回退）；实时模式六条 SYNTHETIC_MOCK 负例场景。
- 阈值只透传（`threshold` / `threshold_scope` / selection 绑定进响应 `provenance`），服务层无重选逻辑。

## 评审与修复记录

两路独立评审（antigravity gemini-3.8-flash-high、pi）结论均为 GO-WITH-FINDINGS，发现高度收敛。已修复：

- P1：demo 加载工件后未 `register` 进 registry（症状模式永不可用）；
- P1：manifest 未强制绑定 `model.joblib` / `labels.json`（漏条目的文件被未校验加载）——现在 spec 构建与加载双层强制；
- P1：测量 `value` 非数值类型（字符串/布尔）在 schema 层拒收、绕过 schema 时按 `invalid_value` 拒收留回执，NaN 落 `out_of_range`；
- P1：损坏工件使 `analyze_symptoms` 抛异常 → 改为结构化 `NOT_AVAILABLE` + ABSTAIN；
- P1/P2：`measured_summary` 是唯一未打来源标签的载荷通道 → 包裹为 `{source_tag, asserted_by:"caller", summary}`；
- P2：AST 结构断言可被 `from syndrome import inference` / 相对导入 / `importlib` 绕过 → 三种写法全部封堵 + 新增子进程运行时证明；
- P2：全局 `last_seen_ts` 误标多设备交错流 → 时间戳回归改按设备来源判定；
- P2：响应 `from_dict` 未校验 mode、`KeyError` 不一致 → `_require_mode` + 统一 `ValueError`；
- P2：`METRIC_TOPICS` / `PLAUSIBLE_RANGES` 两表无对齐护栏 → 缺表指标退化为 `unknown_metric` + 集合相等测试；
- P2：`knowledge_field`/`_SNAKE` 死代码与反向依赖 → 移入 `inference.py`；
- P2：请求级 `timestamp` 必填但被忽略 → 用与测量相同的 fail-closed 规则解析，状态入 `provenance.request_timestamp_status`；
- P3（低成本的已随手修）：`registry or ModelRegistry()` 的真值替换、第二次 `default=True` 显式拒绝、空 `session_id` 保留回执、重复主题 note 去重、非 Mapping 知识库按 NOT_AVAILABLE、`PLAUSIBLE_RANGES` 单位约定入注释、`knowledge_availability` 对称字段、registry_entry 未签名信任根边界与推理缓存语义写入模块文档。

## 明确未做（留给后续 issue）

- `save_tfidf_logreg_artifact` 尚无 shipped CLI 入口（`syndrome.experiment` 增加 `save-backend` 子命令）——待 #50 真实训练产物落地时接线，届时以 dev 选定阈值入档。
- registry_entry.json 签名 / 外部 manifest 绑定（当前防意外损坏，不防恶意替换；边界已文档化）。
- 推理缓存失效策略（科研一次性会话场景不适用长驻服务）。

## 边界与许可

- 证候候选是**倾向**输出，不是诊断；分数未校准。
- 默认后端训练数据 TCM-SD 为 CC BY-NC-SA 4.0（非商业）；商业使用需独立许可与医疗内容审核门槛。
- 实时知识模式不输出个人证候标签（#54 未来阶段，当前不解锁）。

## 证据

- acam-lab 全量：`PYTHONPATH=src pytest tests/ -q` → 550 passed / 26 skipped / 0 failed；`ruff check .` 全净（评审修复前基线，head `542f4c2` 同步副本）。评审修复后的复跑结果见 PR #60 最新 Evidence。
- demo（acam-lab）：无工件时症状模式 `backend_availability=NOT_AVAILABLE` + 三条结构化 ABSTAIN；实时模式六场景全部 fail-closed 行为符合预期。
