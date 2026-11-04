"""TCM-SD 中文证候倾向模型（#53 阶段 1）。

标签：单标签 148 证型（norm_syndrome，taxonomy tcm-sd-syndrome-vocab-v1）。
接口：单条/批量推理（ABSTAIN/OUT_OF_DOMAIN 拒判）+ realtime_feedback 知识参考。
边界：生理数值不进入证候分类；真实数据与工件不入 Git；synthetic 结果不表述
科研/临床有效性。
"""

from syndrome.feedback import (
    ABSTAIN_REASON,
    FeedbackResult,
    MeasurementSample,
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

__all__ = [
    "ABSTAIN_REASON",
    "Candidate",
    "FeedbackResult",
    "InferenceResult",
    "LABEL_TAXONOMY_ID",
    "LabelSpace",
    "STATUS_ABSTAIN",
    "STATUS_OK",
    "STATUS_OUT_OF_DOMAIN",
    "STATUS_UNCERTAIN",
    "SyndromeInference",
    "UNKNOWN",
    "MeasurementSample",
    "is_out_of_domain",
    "realtime_feedback",
    "select_uncertainty_threshold",
]
