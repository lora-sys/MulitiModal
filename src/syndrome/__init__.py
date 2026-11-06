"""TCM-SD 中文证候倾向模型与统一双模式产品接口。

标签：单标签 148 证型（norm_syndrome，taxonomy tcm-sd-syndrome-vocab-v1）。
接口：单条/批量推理（ABSTAIN/OUT_OF_DOMAIN 拒判）+ realtime_feedback 主题知识
+ 统一双模式服务契约（service.py：症状分析 / 实时知识）。
边界：生理数值不进入证候分类；真实数据与工件不入 Git；synthetic 结果不表述
科研/临床有效性；商业使用需独立许可（见 service.py 模块注释）。
"""

from syndrome.feedback import (
    ABSTAIN_REASON,
    SOURCE_KNOWLEDGE,
    SOURCE_MEASURED,
    FeedbackResult,
    MeasurementSample,
    TopicKnowledge,
    realtime_feedback,
)
from syndrome.inference import (
    Candidate,
    InferenceResult,
    SyndromeInference,
    is_out_of_domain,
    select_uncertainty_threshold,
)
from syndrome.labels import (
    LABEL_TAXONOMY_ID,
    STATUS_ABSTAIN,
    STATUS_OK,
    STATUS_OUT_OF_DOMAIN,
    STATUS_UNCERTAIN,
    UNKNOWN,
    LabelSpace,
)
from syndrome.service import (
    BACKEND_AVAILABLE,
    BACKEND_NOT_AVAILABLE,
    MODE_REALTIME,
    MODE_SYMPTOM,
    SERVICE_SCHEMA_VERSION,
    SOURCE_PREDICTED,
    BackendSpec,
    LoadedBackend,
    ModelRegistry,
    RealtimeKnowledgeRequest,
    RealtimeKnowledgeResponse,
    SymptomAnalysisRequest,
    SymptomAnalysisResponse,
    SyndromeService,
    tfidf_logreg_spec,
)

__all__ = [
    "ABSTAIN_REASON",
    "BACKEND_AVAILABLE",
    "BACKEND_NOT_AVAILABLE",
    "BackendSpec",
    "Candidate",
    "FeedbackResult",
    "InferenceResult",
    "LABEL_TAXONOMY_ID",
    "LabelSpace",
    "LoadedBackend",
    "MODE_REALTIME",
    "MODE_SYMPTOM",
    "MeasurementSample",
    "ModelRegistry",
    "RealtimeKnowledgeRequest",
    "RealtimeKnowledgeResponse",
    "SERVICE_SCHEMA_VERSION",
    "SOURCE_KNOWLEDGE",
    "SOURCE_MEASURED",
    "SOURCE_PREDICTED",
    "STATUS_ABSTAIN",
    "STATUS_OK",
    "STATUS_OUT_OF_DOMAIN",
    "STATUS_UNCERTAIN",
    "SyndromeInference",
    "SyndromeService",
    "SymptomAnalysisRequest",
    "SymptomAnalysisResponse",
    "TopicKnowledge",
    "UNKNOWN",
    "is_out_of_domain",
    "realtime_feedback",
    "select_uncertainty_threshold",
    "tfidf_logreg_spec",
]
