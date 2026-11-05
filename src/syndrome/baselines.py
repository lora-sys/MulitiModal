"""可复现的简单文本基线（#53 Scope 2）：

1. 多数类 Dummy（most_frequent）——朴素下界；
2. 字符 n-gram（1–3）TF-IDF + LogisticRegression——简单强基线。

两者只在训练分区内 fit（折内拟合原则的整分区间版本），并记录 sklearn/Python
版本与 seed 供 provenance 入档。真实数据上的复现命令见模型卡。
"""

from __future__ import annotations

import platform
import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

#: 冻结 TCM-SD 词表大小（data.tcm_sd.AUDIT_CONCLUSIONS["vocab_size"]）。
#: 这里用模块级常量而非硬导入：评估胶水测试会向 sys.modules 注入 fake
#: data.tcm_sd，模块级导入会在 fake 生效时失败；子词表合成夹具通过
#: TfidfNgramBaseline(n_classes=...) 覆盖。
TCM_SD_CLASS_COUNT = 148

MAJORITY_BASELINE_NAME = "dummy-most-frequent"
NGRAM_BASELINE_NAME = "tfidf-char-1-3-logreg"


def environment_record(seed: int, extra: dict | None = None) -> dict:
    """基线/BERT 共用的环境记录（#53: 记录 sklearn 版本、seed 等供复现）。"""
    record = {
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": seed,
    }
    if extra:
        record.update(extra)
    return record


class MostFrequentBaseline:
    """多数类 Dummy：对任何输入预测训练集最频类。无随机性。

    seed 仅入档（provenance 复现记录），不参与任何计算（R1 F4：不得
    硬编码 0 冒充真实 seed）。"""

    name = MAJORITY_BASELINE_NAME
    score_semantics = "one-hot majority score; not a calibrated probability"

    def __init__(self, seed: int = 20261101) -> None:
        self.seed = seed
        self.majority_: int | None = None
        self.n_classes_: int | None = None

    def fit(self, texts: list[str], labels: list[int]) -> MostFrequentBaseline:
        if not labels:
            raise ValueError("training labels must not be empty")
        self.majority_ = Counter(labels).most_common(1)[0][0]
        self.n_classes_ = max(labels) + 1
        return self

    def predict(self, texts: list[str]) -> list[int]:
        if self.majority_ is None:
            raise RuntimeError("baseline is not fitted")
        return [self.majority_] * len(texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """one-hot 概率（未校准，仅为统一接口；dummy 概率无置信意义）。"""
        if self.majority_ is None or self.n_classes_ is None:
            raise RuntimeError("baseline is not fitted")
        probs = np.zeros((len(texts), self.n_classes_), dtype=np.float64)
        probs[:, self.majority_] = 1.0
        return probs

    def provenance(self) -> dict:
        return {
            "name": self.name,
            "score_semantics": self.score_semantics,
            "params": {},
            "environment": environment_record(seed=self.seed),
        }


@dataclass
class TfidfNgramConfig:
    """冻结的 TF-IDF 基线配置；改动须经 dev macro-F1 选择并记录。"""

    ngram_low: int = 1
    ngram_high: int = 3
    min_df: int = 2
    sublinear_tf: bool = True
    C: float = 4.0
    class_weight: str | None = "balanced"  # macro-F1 主口径下补偿类别不平衡
    max_iter: int = 2000
    seed: int = 20261101
    # "ovr-liblinear"（默认）：148 次二分类 liblinear，小数据快、真实 44k 慢；
    # "multinomial-saga"：一次多项式拟合，真实规模可用（8k 文档×148 类 64s/30 iters，
    # 2026-11-02 实测）。两者以 dev macro-F1 择优冻结，选择随 provenance 入档。
    solver: str = "ovr-liblinear"
    # 交付冠军（commit d8371c7b）需要 char_wb 边界特征与二值化计数。默认值维持
    # #53 基线行为，既有调用与测试不受影响。
    analyzer: str = "char"
    binary: bool = False


class TfidfNgramBaseline:
    """字符 1–3 gram TF-IDF + LogisticRegression（OvR-liblinear）。

    选 liblinear 的原因：本环境 sklearn 1.9 + scipy 1.18 的多项式 lbfgs
    未走解析梯度（有限差分），2000×500/20 类即需 ~51s，真实规模不可用；
    OvR-liblinear 单线程 8k 文档 ×148 类约 15s（2026-11-02 实测，acam-lab）。
    predict_proba 为 OvR sigmoid 归一化，非多项式 softmax、未校准。
    """

    name = NGRAM_BASELINE_NAME
    score_semantics = "OvR sigmoid-normalized scores; not softmax; uncalibrated"

    def __init__(
        self, config: TfidfNgramConfig | None = None,
        n_classes: int | None = None,
    ) -> None:
        self.config = config or TfidfNgramConfig()
        # 默认对齐冻结的 148 类词表；子词表合成夹具显式传入更小的类数
        self._n_classes = TCM_SD_CLASS_COUNT if n_classes is None else n_classes
        self._pipe: Pipeline | None = None

    def _build(self) -> Pipeline:
        cfg = self.config
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer=cfg.analyzer,
                        ngram_range=(cfg.ngram_low, cfg.ngram_high),
                        min_df=cfg.min_df,
                        binary=cfg.binary,
                        sublinear_tf=cfg.sublinear_tf,
                    ),
                ),
                (
                    "clf",
                    # n_jobs=1：多进程 worker 序列化稀疏矩阵在本环境触发
                    # "WRITEBACKIFCOPY base is read-only"（numpy 2.5 + joblib ABI 问题）
                    OneVsRestClassifier(
                        LogisticRegression(
                            C=cfg.C,
                            class_weight=cfg.class_weight,
                            max_iter=cfg.max_iter,
                            solver="liblinear",
                            random_state=cfg.seed,
                        ),
                        n_jobs=1,
                    )
                    if cfg.solver == "ovr-liblinear"
                    else LogisticRegression(
                        C=cfg.C,
                        class_weight=cfg.class_weight,
                        max_iter=cfg.max_iter,
                        solver="saga",
                        random_state=cfg.seed,
                    ),
                ),
            ]
        )

    def fit(self, texts: list[str], labels: list[int]) -> TfidfNgramBaseline:
        if len(texts) != len(labels):
            raise ValueError(f"length mismatch: {len(texts)} vs {len(labels)}")
        if not labels:
            raise ValueError("training labels must not be empty")
        if any(
            isinstance(label, bool) or not isinstance(label, int) or label < 0
            for label in labels
        ):
            raise ValueError("training labels must be non-negative ints")
        if any(label >= self._n_classes for label in labels):
            raise ValueError(
                f"training label outside the {self._n_classes}-class taxonomy"
            )
        self._pipe = self._build()
        self._pipe.fit(texts, labels)
        return self

    def _require(self) -> Pipeline:
        if self._pipe is None:
            raise RuntimeError("baseline is not fitted")
        return self._pipe

    def predict(self, texts: list[str]) -> list[int]:
        return [int(x) for x in self._require().predict(texts)]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """按冻结的 TCM-SD label id 返回概率，训练 fold 未见类补 0。"""
        pipe = self._require()
        classifier = pipe.named_steps["clf"]
        observed_classes = np.asarray(classifier.classes_, dtype=np.int64)
        observed_probs = pipe.predict_proba(texts)
        probs = np.zeros((len(texts), self._n_classes), dtype=np.float64)
        probs[:, observed_classes] = observed_probs
        return probs

    def provenance(self) -> dict:
        cfg = self.config
        return {
            "name": self.name,
            "score_semantics": self.score_semantics,
            "params": {
                "ngram_range": [cfg.ngram_low, cfg.ngram_high],
                "min_df": cfg.min_df,
                "sublinear_tf": cfg.sublinear_tf,
                "C": cfg.C,
                "class_weight": cfg.class_weight,
                "max_iter": cfg.max_iter,
                "solver": cfg.solver,
                "analyzer": cfg.analyzer,
                "binary": cfg.binary,
                "solver_rationale": (
                    "multinomial lbfgs FD-gradient perf defect in sklearn1.9/scipy1.18"
                ),
            },
            "environment": environment_record(seed=cfg.seed),
        }
