"""#58 双模式统一产品服务契约（症状分析 / 实时中医知识）。

在 syndrome.inference（症状 → 证候倾向）与 syndrome.feedback（测量 → 主题知识）
之上扩展统一请求/响应 schema；不重写训练循环，不另建并行推理基础设施。

两种模式（一个服务入口，schema 各自独立）：
- SYMPTOM_ANALYSIS：用户主动提供的中文症状文本 → 可插拔推理后端（版本化
  registry 条目）→ Top-k 候选（calibrated=false）+ 依据 + 缺失项 + status；
- REALTIME_KNOWLEDGE：timestamp / session_id / 设备测量摘要 → 只返回实测
  指标回执（MEASURED）与非诊断性主题知识（KNOWLEDGE）；结构性不调用症状
  分类器（见 feedback.py 模块注释与 tests 的断言），不输出个人证候标签。

许可与边界（AGENTS.md / 模型卡）：
- 本服务是**科研原型**：症状模式默认后端在 TCM-SD（CC BY-NC-SA 4.0，非商业）
  上训练，派生知识同许可；**商业使用需要独立许可与医疗内容审核门槛**，本接口
  不授予任何商业部署权；
- 输出为证候**倾向**候选，不是诊断；分数未经校准，不得当作可信度解释；
- 实时知识模式不从生理数值推断证候（#54 未来阶段，当前不解锁）。

阈值出处：推理工件与 dev 选择清单严格绑定（#56 协议）。本服务只**透传**
threshold / threshold_scope / selection 绑定字段（见响应 provenance），绝不
重选或绕过阈值；scope 缺失时显式标记 UNVERIFIED。

工件信任边界：registry_entry.json + 工件哈希可防**意外损坏**（磁盘位翻转、
截断、部分写入），但不能防**恶意替换**——registry_entry.json 本身未签名，
重写条目与工件可一致地绕过校验。信任根是保存 registry_entry.json 的运行目录；
需要对抗恶意篡改时须把条目哈希绑定到外部运行 manifest。

Fail-closed 路径：空症状（ABSTAIN）、后端不可用（NOT_AVAILABLE + ABSTAIN）、
低质量/越界/未知指标/缺值/时间戳回退测量（拒收，仅留实测回执）、知识条目出处
缺失（排除该条目）、schema 版本不匹配（拒绝请求）。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from syndrome.feedback import (
    SOURCE_KNOWLEDGE,
    SOURCE_MEASURED,
    MeasurementSample,
    TopicKnowledge,
    realtime_feedback,
)
from syndrome.feedback import (
    _parse_timestamp as _parse_request_timestamp,
)
from syndrome.inference import ProbabilisticPredictor, SyndromeInference
from syndrome.labels import LABEL_TAXONOMY_ID, STATUS_ABSTAIN, LabelSpace

SERVICE_SCHEMA_VERSION = "tcm-service-schema-v1"
BACKEND_ARTIFACT_SCHEMA = "tcm-service-backend-v1"

MODE_SYMPTOM = "SYMPTOM_ANALYSIS"
MODE_REALTIME = "REALTIME_KNOWLEDGE"

SOURCE_PREDICTED = "PREDICTED"
BACKEND_AVAILABLE = "AVAILABLE"
BACKEND_NOT_AVAILABLE = "NOT_AVAILABLE"

PREDICTOR_ARTIFACT_NAME = "model.joblib"
LABELS_ARTIFACT_NAME = "labels.json"
REGISTRY_ENTRY_NAME = "registry_entry.json"

# 症状模式默认后端的数据许可说明（模型卡：TCM-SD 非商业）。
DEFAULT_DATA_LICENSE = "TCM-SD (CC BY-NC-SA 4.0, non-commercial research only)"


def _artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- 请求


@dataclass
class SymptomAnalysisRequest:
    """症状分析请求：用户主动提供的中文症状文本 + 可选基本字段。"""

    symptom_text: str
    fields: dict | None = None
    model_id: str | None = None
    schema_version: str = SERVICE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "symptom_text": self.symptom_text,
            "fields": dict(self.fields) if self.fields else None,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> SymptomAnalysisRequest:
        _require_schema(data)
        symptom_text = data.get("symptom_text")
        if not isinstance(symptom_text, str):
            raise ValueError("symptom_text must be a string")
        fields = data.get("fields")
        if fields is not None and not isinstance(fields, Mapping):
            raise ValueError("fields must be an object")
        model_id = data.get("model_id")
        if model_id is not None and not isinstance(model_id, str):
            raise ValueError("model_id must be a string")
        return cls(
            symptom_text=symptom_text,
            fields=dict(fields) if fields else None,
            model_id=model_id,
        )


@dataclass
class RealtimeKnowledgeRequest:
    """实时知识请求：测量摘要 + 可选结构化生理摘要；不含任何症状文本。"""

    session_id: str
    timestamp: str
    measurements: list[MeasurementSample] = field(default_factory=list)
    physio_summary: dict | None = None
    schema_version: str = SERVICE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "measurements": [
                {
                    "source": m.source,
                    "metric": m.metric,
                    "value": m.value,
                    "unit": m.unit,
                    "timestamp": m.timestamp,
                    "quality": m.quality,
                }
                for m in self.measurements
            ],
            "physio_summary": dict(self.physio_summary) if self.physio_summary else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> RealtimeKnowledgeRequest:
        _require_schema(data)
        session_id = data.get("session_id")
        timestamp = data.get("timestamp")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(timestamp, str) or not timestamp:
            raise ValueError("timestamp must be a non-empty string")
        raw_measurements = data.get("measurements", [])
        if not isinstance(raw_measurements, list):
            raise ValueError("measurements must be a list")
        measurements = [_measurement_from_dict(m) for m in raw_measurements]
        physio_summary = data.get("physio_summary")
        if physio_summary is not None and not isinstance(physio_summary, Mapping):
            raise ValueError("physio_summary must be an object")
        return cls(
            session_id=session_id,
            timestamp=timestamp,
            measurements=measurements,
            physio_summary=dict(physio_summary) if physio_summary else None,
        )


def _measurement_from_dict(data: Mapping) -> MeasurementSample:
    try:
        source, metric = data["source"], data["metric"]
        value, unit = data["value"], data["unit"]
        timestamp, quality = data["timestamp"], data["quality"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed measurement record: missing {exc}") from exc
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float))
    ):
        raise ValueError(
            f"measurement {source}/{metric}: value must be a number or null, "
            f"got {type(value).__name__}"
        )
    return MeasurementSample(
        source=source,
        metric=metric,
        value=value,
        unit=unit,
        timestamp=timestamp,
        quality=quality,
    )


def _require_schema(data: object) -> None:
    if not isinstance(data, Mapping):
        raise ValueError("request payload must be an object")
    version = data.get("schema_version", SERVICE_SCHEMA_VERSION)
    if version != SERVICE_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {version!r}")


def _require_mode(data: Mapping, expected: str) -> None:
    mode = data.get("mode")
    if mode is not None and mode != expected:
        raise ValueError(f"mode mismatch: expected {expected}, got {mode!r}")


# --------------------------------------------------------------------------- 响应


@dataclass
class SymptomAnalysisResponse:
    """症状分析响应：候选全部 PREDICTED + calibrated=false；知识引用 KNOWLEDGE。"""

    status: str
    candidates: list[dict] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    knowledge_reference: list[dict] = field(default_factory=list)
    message: str = ""
    model_id: str = ""
    model_version: str = ""
    backend_availability: str = BACKEND_AVAILABLE
    label_taxonomy_version: str = LABEL_TAXONOMY_ID
    data_license: str = "NOT_AVAILABLE"
    provenance: dict = field(default_factory=dict)
    schema_version: str = SERVICE_SCHEMA_VERSION
    mode: str = MODE_SYMPTOM

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "status": self.status,
            "candidates": [dict(c) for c in self.candidates],
            "evidence": list(self.evidence),
            "missing_information": list(self.missing_information),
            "knowledge_reference": [dict(r) for r in self.knowledge_reference],
            "message": self.message,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "backend_availability": self.backend_availability,
            "label_taxonomy_version": self.label_taxonomy_version,
            "data_license": self.data_license,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> SymptomAnalysisResponse:
        _require_schema(data)
        _require_mode(data, MODE_SYMPTOM)
        try:
            status = data["status"]
        except KeyError as exc:
            raise ValueError(f"symptom response missing field: {exc}") from exc
        return cls(
            status=status,
            candidates=list(data.get("candidates", [])),
            evidence=list(data.get("evidence", [])),
            missing_information=list(data.get("missing_information", [])),
            knowledge_reference=list(data.get("knowledge_reference", [])),
            message=data.get("message", ""),
            model_id=data.get("model_id", ""),
            model_version=data.get("model_version", ""),
            backend_availability=data.get("backend_availability", BACKEND_AVAILABLE),
            label_taxonomy_version=data.get("label_taxonomy_version", LABEL_TAXONOMY_ID),
            data_license=data.get("data_license", "NOT_AVAILABLE"),
            provenance=dict(data.get("provenance", {})),
        )


@dataclass
class RealtimeKnowledgeResponse:
    """实时知识响应：measured 全部 MEASURED，knowledge 全部 KNOWLEDGE，
    syndrome_judgment 恒为 None，classifier_invoked 恒为 False。"""

    status: str
    session_id: str
    measured: list[dict] = field(default_factory=list)
    measured_summary: dict | None = None
    knowledge: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    message: str = ""
    syndrome_judgment: None = None
    classifier_invoked: bool = False
    knowledge_availability: str = BACKEND_AVAILABLE
    provenance: dict = field(default_factory=dict)
    schema_version: str = SERVICE_SCHEMA_VERSION
    mode: str = MODE_REALTIME

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "status": self.status,
            "session_id": self.session_id,
            "measured": [dict(m) for m in self.measured],
            "measured_summary": dict(self.measured_summary) if self.measured_summary else None,
            "knowledge": [dict(k) for k in self.knowledge],
            "notes": list(self.notes),
            "message": self.message,
            "syndrome_judgment": self.syndrome_judgment,
            "classifier_invoked": self.classifier_invoked,
            "knowledge_availability": self.knowledge_availability,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> RealtimeKnowledgeResponse:
        _require_schema(data)
        _require_mode(data, MODE_REALTIME)
        try:
            status, session_id = data["status"], data["session_id"]
        except KeyError as exc:
            raise ValueError(f"realtime response missing field: {exc}") from exc
        return cls(
            status=status,
            session_id=session_id,
            measured=list(data.get("measured", [])),
            measured_summary=data.get("measured_summary"),
            knowledge=list(data.get("knowledge", [])),
            notes=list(data.get("notes", [])),
            message=data.get("message", ""),
            knowledge_availability=data.get("knowledge_availability", BACKEND_AVAILABLE),
            provenance=dict(data.get("provenance", {})),
        )


# --------------------------------------------------------------------------- 后端 registry


@dataclass(frozen=True)
class LoadedBackend:
    """一次后端加载的结果：predictor + 冻结词表 + dev 选定阈值 + 出处。"""

    predictor: ProbabilisticPredictor
    label_space: LabelSpace
    threshold: float
    threshold_scope: str = "UNVERIFIED"
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BackendSpec:
    """版本化模型 registry 条目。换后端（线性 ↔ 神经）= 换条目，不改 API。

    loader 惰性调用（工件加载/反序列化只在首次用到该后端时发生）。"""

    model_id: str
    model_version: str
    taxonomy_version: str
    loader: Callable[[], LoadedBackend]
    artifact_dir: str | None = None
    artifacts: Mapping[str, str] = field(default_factory=dict)  # name -> sha256
    data_license: str = "NOT_AVAILABLE"
    provenance: Mapping[str, object] = field(default_factory=dict)


class ModelRegistry:
    """model_id → BackendSpec。taxonomy 版本不符的条目直接拒收（fail-closed）。"""

    def __init__(self) -> None:
        self._specs: dict[str, BackendSpec] = {}
        self.default_model_id: str | None = None

    def register(self, spec: BackendSpec, default: bool = False) -> None:
        if not spec.model_id:
            raise ValueError("backend model_id must not be empty")
        if spec.taxonomy_version != LABEL_TAXONOMY_ID:
            raise ValueError(
                f"backend taxonomy {spec.taxonomy_version!r} does not match "
                f"the frozen {LABEL_TAXONOMY_ID!r}"
            )
        if spec.model_id in self._specs:
            raise ValueError(f"backend model_id already registered: {spec.model_id!r}")
        if default and self.default_model_id is not None:
            raise ValueError(
                f"default backend already registered: {self.default_model_id!r}; "
                "reassigning the default must be explicit (new registry)"
            )
        self._specs[spec.model_id] = spec
        if default or self.default_model_id is None:
            self.default_model_id = spec.model_id

    def get(self, model_id: str | None) -> BackendSpec | None:
        if model_id is None:
            model_id = self.default_model_id
        if model_id is None:
            return None
        return self._specs.get(model_id)

    def model_ids(self) -> tuple[str, ...]:
        return tuple(self._specs)


# --------------------------------------------------------------------------- TF-IDF+LR 工件


def save_tfidf_logreg_artifact(
    predictor: ProbabilisticPredictor,
    labels: Sequence[str],
    artifact_dir: str,
    *,
    model_version: str,
    threshold: float,
    threshold_scope: str = "UNVERIFIED",
    data_license: str = DEFAULT_DATA_LICENSE,
    provenance: dict | None = None,
    model_id: str = "tfidf-char-1-3-logreg",
) -> BackendSpec:
    """保存已拟合的 TF-IDF+LR 预测器为服务端默认后端工件（真实训练产物才可入档；
    调用方负责 provenance 的真实性）。返回可直接 register 的 BackendSpec。"""
    import joblib

    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    predictor_path = directory / PREDICTOR_ARTIFACT_NAME
    labels_path = directory / LABELS_ARTIFACT_NAME
    joblib.dump(predictor, predictor_path)
    with open(labels_path, "w", encoding="utf-8") as fh:
        json.dump(list(labels), fh, ensure_ascii=False)
    artifacts = {
        PREDICTOR_ARTIFACT_NAME: _artifact_sha256(predictor_path),
        LABELS_ARTIFACT_NAME: _artifact_sha256(labels_path),
    }
    entry = {
        "schema_version": BACKEND_ARTIFACT_SCHEMA,
        "model_id": model_id,
        "model_version": model_version,
        "kind": "tfidf-logreg",
        "taxonomy_version": LABEL_TAXONOMY_ID,
        "artifacts": artifacts,
        "threshold": float(threshold),
        "threshold_scope": threshold_scope,
        "data_license": data_license,
        "provenance": dict(provenance or {}),
    }
    with open(directory / REGISTRY_ENTRY_NAME, "w", encoding="utf-8") as fh:
        json.dump(entry, fh, ensure_ascii=False, indent=2)
    return _spec_from_entry(directory, entry)


def tfidf_logreg_spec(artifact_dir: str) -> BackendSpec | None:
    """工件目录不存在时返回 None（调用方按 NOT_AVAILABLE 处理）；存在但损坏则抛错。"""
    entry_path = Path(artifact_dir) / REGISTRY_ENTRY_NAME
    if not entry_path.is_file():
        return None
    try:
        with open(entry_path, encoding="utf-8") as fh:
            entry = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed backend registry entry: {entry_path}") from exc
    if not isinstance(entry, dict) or entry.get("schema_version") != BACKEND_ARTIFACT_SCHEMA:
        raise ValueError(f"unsupported backend artifact schema at {entry_path}")
    return _spec_from_entry(Path(artifact_dir), entry)


def _spec_from_entry(directory: Path, entry: dict) -> BackendSpec:
    threshold = entry.get("threshold")
    if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("backend artifact threshold must be a probability in [0, 1]")
    artifacts = entry.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("backend artifact manifest is missing artifact hashes")
    for required in (PREDICTOR_ARTIFACT_NAME, LABELS_ARTIFACT_NAME):
        if required not in artifacts:
            raise ValueError(f"backend artifact manifest missing required entry: {required}")
    model_id = entry.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("backend artifact model_id is missing")
    return BackendSpec(
        model_id=model_id,
        model_version=str(entry.get("model_version", "")),
        taxonomy_version=str(entry.get("taxonomy_version", "")),
        loader=lambda: _load_tfidf_logreg(directory, entry),
        artifact_dir=str(directory),
        artifacts={str(k): str(v) for k, v in artifacts.items()},
        data_license=str(entry.get("data_license", "NOT_AVAILABLE")),
        provenance={
            "kind": entry.get("kind", "tfidf-logreg"),
            "threshold_scope": entry.get("threshold_scope", "UNVERIFIED"),
            **(entry.get("provenance", {}) if isinstance(entry.get("provenance"), dict) else {}),
        },
    )


def _load_tfidf_logreg(directory: Path, entry: dict) -> LoadedBackend:
    import joblib

    # manifest 必须绑定实际加载的全部文件：缺条目的文件即使存在也拒载
    # （否则 labels.json 可被换名/重排而哈希校验仍"通过"）。
    for required in (PREDICTOR_ARTIFACT_NAME, LABELS_ARTIFACT_NAME):
        if required not in entry["artifacts"]:
            raise ValueError(
                f"backend artifact manifest missing required entry: {required}"
            )
    for name, sha256 in entry["artifacts"].items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"backend artifact missing: {path}")
        actual = _artifact_sha256(path)
        if actual != sha256:
            raise ValueError(
                f"backend artifact hash mismatch for {name}: expected {sha256}, got {actual}"
            )
    with open(directory / LABELS_ARTIFACT_NAME, encoding="utf-8") as fh:
        labels = json.load(fh)
    predictor = joblib.load(directory / PREDICTOR_ARTIFACT_NAME)
    if not hasattr(predictor, "predict_proba"):
        raise ValueError("backend artifact does not expose predict_proba")
    return LoadedBackend(
        predictor=predictor,
        label_space=LabelSpace(tuple(labels)),
        threshold=float(entry["threshold"]),
        threshold_scope=str(entry.get("threshold_scope", "UNVERIFIED")),
        provenance={
            "artifact_dir": str(directory),
            "artifacts_sha256": dict(entry["artifacts"]),
            "kind": entry.get("kind", "tfidf-logreg"),
        },
    )


# --------------------------------------------------------------------------- 服务


class SyndromeService:
    """#58 统一双模式服务。症状模式走 registry 后端；实时知识模式结构性不接触
    任何后端（不加载、不调用），生理数据在类型层面无法进入证候分类器。"""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        knowledge: Mapping[str, object] | None = None,
        topic_knowledge: Mapping[str, TopicKnowledge] | None = None,
        reconnect_gap_seconds: float = 300.0,
        provenance: dict | None = None,
    ) -> None:
        self.registry = registry if registry is not None else ModelRegistry()
        self.knowledge = knowledge
        self.topic_knowledge = topic_knowledge
        self.reconnect_gap_seconds = reconnect_gap_seconds
        self._base_provenance = dict(provenance or {})
        # model_id → 已加载推理实例。缓存按进程生命周期存活、不自动失效：
        # 更换工件目录须重建 service（科研场景一次性会话，非长驻服务）。
        self._inferences: dict[str, SyndromeInference] = {}

    # ------------------------------------------------------------------ 症状分析

    def analyze_symptoms(self, request: SymptomAnalysisRequest) -> SymptomAnalysisResponse:
        if not isinstance(request, SymptomAnalysisRequest):
            raise TypeError("analyze_symptoms expects a SymptomAnalysisRequest")
        if request.schema_version != SERVICE_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {request.schema_version!r}")
        spec = self.registry.get(request.model_id)
        if spec is None:
            return SymptomAnalysisResponse(
                status=STATUS_ABSTAIN,
                message=(
                    "推理后端不可用（NOT_AVAILABLE），不作证候判断；"
                    "后端=版本化 registry 条目，需先注册有效模型工件"
                ),
                model_id=request.model_id or "",
                backend_availability=BACKEND_NOT_AVAILABLE,
                provenance=dict(self._base_provenance),
            )
        try:
            inference = self._inference(spec)
        except Exception as exc:  # noqa: BLE001 - 任何工件加载失败都必须 fail-closed
            return SymptomAnalysisResponse(
                status=STATUS_ABSTAIN,
                message=(
                    f"后端工件加载失败（{BACKEND_NOT_AVAILABLE}），不作证候判断：{exc}"
                ),
                model_id=spec.model_id,
                model_version=spec.model_version,
                backend_availability=BACKEND_NOT_AVAILABLE,
                provenance=dict(self._base_provenance),
            )
        result = inference.predict_one(request.symptom_text, request.fields)
        return SymptomAnalysisResponse(
            status=result.status,
            candidates=[
                {
                    "label": c.label,
                    "score": round(c.score, 6),
                    "source_tag": SOURCE_PREDICTED,
                    "calibrated": False,
                }
                for c in result.candidates
            ],
            evidence=list(result.evidence),
            missing_information=list(result.missing_information),
            knowledge_reference=[
                {**r, "source_tag": SOURCE_KNOWLEDGE} for r in result.knowledge_reference
            ],
            message=result.message,
            model_id=spec.model_id,
            model_version=spec.model_version,
            backend_availability=BACKEND_AVAILABLE,
            label_taxonomy_version=result.label_taxonomy_version,
            data_license=spec.data_license,
            provenance={**self._base_provenance, **result.provenance},
        )

    def _inference(self, spec: BackendSpec) -> SyndromeInference:
        cached = self._inferences.get(spec.model_id)
        if cached is not None:
            return cached
        loaded = spec.loader()
        if loaded.label_space.taxonomy_id != spec.taxonomy_version:
            raise ValueError(
                f"backend {spec.model_id!r} label space taxonomy mismatch: "
                f"{loaded.label_space.taxonomy_id!r} != {spec.taxonomy_version!r}"
            )
        inference = SyndromeInference(
            loaded.predictor,
            loaded.label_space,
            model_version=spec.model_version,
            provenance={
                "model_id": spec.model_id,
                "artifacts_sha256": dict(spec.artifacts),
                "artifact_dir": spec.artifact_dir,
                **dict(spec.provenance),
                **loaded.provenance,
            },
            threshold=loaded.threshold,
            knowledge=self.knowledge,
            threshold_scope=loaded.threshold_scope,
        )
        self._inferences[spec.model_id] = inference
        return inference

    # ------------------------------------------------------------------ 实时知识

    def realtime_knowledge(self, request: RealtimeKnowledgeRequest) -> RealtimeKnowledgeResponse:
        if not isinstance(request, RealtimeKnowledgeRequest):
            raise TypeError("realtime_knowledge expects a RealtimeKnowledgeRequest")
        if request.schema_version != SERVICE_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {request.schema_version!r}")
        feedback = realtime_feedback(
            request.measurements,
            self.topic_knowledge,
            reconnect_gap_seconds=self.reconnect_gap_seconds,
        )
        # 请求级 timestamp 用与测量相同的 fail-closed 规则解析：无效不崩溃、不臆断时区。
        request_ts = _parse_request_timestamp(request.timestamp)
        if request_ts is None:
            feedback.notes.append("请求 timestamp 无法解析为带时区的 ISO 8601：按 UNVERIFIED 处理")
        # session_id 缺失同样只降级为 ABSTAIN，实测回执照常保留。
        status = feedback.status
        if not request.session_id.strip():
            status = STATUS_ABSTAIN
            feedback.notes.append("缺少有效 session_id：状态降级为 ABSTAIN，回执保留")
        knowledge_available = (
            BACKEND_AVAILABLE if self.topic_knowledge else BACKEND_NOT_AVAILABLE
        )
        return RealtimeKnowledgeResponse(
            status=status,
            session_id=request.session_id,
            measured=list(feedback.measured),
            measured_summary=(
                {
                    "source_tag": SOURCE_MEASURED,
                    "asserted_by": "caller",
                    "summary": dict(request.physio_summary),
                }
                if request.physio_summary
                else None
            ),
            knowledge=list(feedback.references),
            notes=list(feedback.notes),
            knowledge_availability=knowledge_available,
            message=(
                "实测指标摘要与非诊断性主题知识；无个人证候判断（#53/#58 契约）"
            ),
            provenance={
                **self._base_provenance,
                "topic_mapping": "METRIC_TOPICS (metric-name-level, value-independent)",
                "n_topic_knowledge_entries": (
                    len(self.topic_knowledge) if self.topic_knowledge else 0
                ),
                "request_timestamp_status": (
                    "PARSED" if request_ts is not None else "UNVERIFIED"
                ),
            },
        )

