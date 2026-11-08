# 实时生理疲劳与安全推荐：工程契约 V1（#95–#100）

**状态**：`LOCAL_EXPERIMENTAL` / `SMOKE_ONLY` 工程实现，**未对真实设备或真实受试者验证**。
本文件冻结模块边界与公开 API，供 #96/#97/#98/#99/#100 并行开发；不声明任何疲劳模型效果。

## 1. 共用输入（#96，已冻结）

- `src/physio/schema.py`：`PhysioEventV1`、`PhysioSessionV1`、`MeasurementV1`、
  `ProtocolContextV1`、`ProvenanceV1`、`assemble_session`、`mark_stale`。
  - `metric` / `value` 只在 `valid=True` 时可被下游使用；
  - `quality ∈ {ok, degraded, rejected, stale}`，`rejected/stale` 不得进入推理；
  - 时间戳必须带时区，`received_at < measured_at` 直接拒绝。
- `src/physio/metrics.py`：`FieldRouter`、`route_measurement`、`METRIC_SPECS`。
  - 未知字段/单位保留原值但 `valid=False`，不猜正常值；
  - SpO2 fraction→%，温度 °F→°C，血压 kPa→mmHg；
  - 零值语义按字段固定：`rr/bp/bmr` 的 0 表示缺失；`spo2` 的 0 非法；
    `fatigue_index` 的 0 是厂商代理分数，不能当独立标签。
- `src/physio/adapters.py`：设备、厂商服务、离线 CSV/JSON 三类适配器。
  - 设备/服务 `capabilities().status == NOT_CONNECTED`，未提供协议前不声称接通；
  - 离线适配器 `realtime=False`，不得被实时推理服务 import。
- `src/physio/stream.py`：`OnlineIngestor` 负责断线、重连、乱序、重复与 gap。

## 2. 模块所有权（并行开发互不修改同一文件）

| Issue | 文件 | 说明 |
|---|---|---|
| #97 | `src/physio/windows.py`, `preprocessing.py`, `model.py`, `artifacts.py`, `inference.py`, `training.py` | 滑窗特征、折内 scaler、模型、工件绑定、推理会话与 torch-optional smoke 训练 |
| #98 | `src/physio/programs.py`, `response.py`, `dataset.py` | 程序能力表/上下文解析、Post−Pre 标签、mask/折内插补/受试者划分与标签准入 |
| #100 | `src/physio/tcm_exam.py` | TCMExamV1、同义词 parser、审核知识侧车 |
| #99 | `src/physio/recommendation.py` | 推荐服务与安全状态机（依赖 #97/#98 的公开类型） |
| #96 | `src/physio/legacy_export.py` | 94 字段离线导出注册表、逐行 QC 与确定性导入 |
| #95 | `src/physio/acceptance.py`, `evaluation.py`, `benchmark.py`, `scripts/physio_realtime_demo.py`, `scripts/physio_benchmark.py`, `scripts/physio_verify.py` | 统一验收、证据门禁、benchmark 与模拟 E2E |

不得修改 `src/core/{specs,batch,models}.py` 的已冻结旧语义，也不得触碰两条 AutoResearch 候选文件。

## 3. 统一门禁

- `REALTIME_TARGET_VERSION = "fatigue_state_v1"`；没有真实标签/独立验证时
  `validation_status="NOT_VALIDATED"`，推理必须输出 `NOT_AVAILABLE`。
- `vendor_fatigue`/`fatigue_index` 只是厂商代理分数，不是临床或独立疲劳真值。
- 推理输出必须带：`target_version`、`feature_version`、`model_version`、
  `input_cutoff`、`quality`、`coverage`、`reason`。
- 空窗、全部字段缺失、窗口过期、模型/工件不可用 → `ABSTAIN` 或 `NOT_AVAILABLE`，
  禁止用 0 或均值伪造。
- 中医知识必须带 `source/license/version/review_status`；缺一项即 fail-closed，
  个人体质结论不得由几项舌象单独推导。
- 离线 CSV 不是实时入口；设备控制接口保持 `NOT_CONNECTED`。

## 4. 统一输出字段（供 #99/#95 消费）

```json
{
  "status": "OK | ABSTAIN | NOT_AVAILABLE",
  "reason": "human-readable, no private values",
  "target_version": "fatigue_state_v1",
  "feature_version": "physio-window-v1",
  "model_version": "NOT_AVAILABLE | <artifact version>",
  "input_cutoff": "2026-11-07T10:01:00+00:00",
  "quality": "ok | degraded | rejected | stale",
  "coverage": 0.0,
  "measured": [{"metric": "heart_rate", "value": 72.0, "unit": "bpm", "valid": true}],
  "prediction": null,
  "candidates": []
}
```

预测值只有在 `validation_status="VALIDATED"` 且工件匹配时才可出现。

## 5. 离线导入与数据纪律（#95/#96/#98）

- **94 字段导出是研究输入，不是在线输入维度**。`legacy_export.ExportFieldRegistry`
  冻结 `raw_name -> canonical_metric/phase/unit/origin/label_role` 映射并带 sha256；
  字段语义未确认时显式标 `ambiguous_semantics` 并拒绝导入，不按名称猜测。
- **逐行 QC 不静默修复**：未知字段、快照/完成态指标冲突、同一 `record_id` 跨阶段
  （Pre/Post 冒充）、重复记录、sequence 碰撞、session 身份碰撞、naive/倒序时间戳、
  跨日跨度、EEG 起止时间与 `duration` 冲突、imputed/derived 值都会写入
  `ImportIssue`。补值只保留审计标记，不参与标签。
- **数据纪律**：`FeatureRow` 分离 `observed_mask` / `valid_mask` / `imputed_mask`；
  `FoldImputer` 只在训练折有效观测上拟合，QC 无效值不插补；`SubjectFold` 保证
  同一受试者只落一折，共用/未知账号默认拒绝泛化；`TargetAdmission` 拒绝同记录、
  超时和补值端点，`y_pri` 恒为 `None/False`。
- **训练与证据**：`training` 在无有效标签或无可信身份时抛 `TrainingRefused`；
  `evaluation` 在样本或受试者不足时返回 `INSUFFICIENT_EVIDENCE`；`benchmark`
  只报告实测数值，不设医学或产品门限。
- **真实数据状态**：两台机器均无 94 字段导出、5 场共用账号会话和独立疲劳标签，
  真实字段名映射与训练有效性标 `BLOCKED_BY_DATA_OR_DEVICE`。
