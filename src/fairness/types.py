"""Constants and datatypes for the fairness audit package."""

from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable, Dict, List

import numpy as np

AUDIT_COLUMNS: List[str] = [
    "SK_ID_CURR",
    "y",
    "score",
    "fold",
    "CODE_GENDER",
    "AGE_BAND",
    "GENDER_AGE",
    "PTI",
    "PTI_Q",
]
GROUPING_COLUMNS: List[str] = ["CODE_GENDER", "AGE_BAND", "GENDER_AGE"]
Y_COL = "y"
SCORE_COL = "score"
FOLD_COL = "fold"
GENDER_COL = "CODE_GENDER"
GENDER_AGE_COL = "GENDER_AGE"
OOF_SCORE_COL = "y_pred_proba"
ANNUITY_COL = "AMT_ANNUITY"
INCOME_COL = "AMT_INCOME_TOTAL"
PTI_COL = "PTI"
PTI_Q_COL = "PTI_Q"
YHAT_COL = "yhat"

# Columns of ``gaps.csv`` and metric names read by the comparison modules.
PAIR_ID_COL = "pair_id"
METRIC_COL = "metric"
DIFF_COL = "diff"
DIFF_CI_LOW_COL = "diff_ci_low"
DIFF_CI_HIGH_COL = "diff_ci_high"
RATIO_COL = "ratio"
RATIO_CI_LOW_COL = "ratio_ci_low"
RATIO_CI_HIGH_COL = "ratio_ci_high"
APPROVAL_RATE_METRIC = "approval_rate"
FPR_METRIC = "fpr"
FNR_METRIC = "fnr"
PPV_METRIC = "ppv"
DEFAULT_AMONG_APPROVED_METRIC = "default_among_approved"
AUC_METRIC = "auc"
BRIER_METRIC = "brier"
LOG_LOSS_METRIC = "log_loss"

# Pooled row of ``group_counts.csv`` / ``group_metrics.csv``.
ALL_GROUPING = "ALL"
ALL_GROUP = "all"

# Display labels of the gap metrics in reports and figures.
METRIC_LABELS: Dict[str, str] = {
    APPROVAL_RATE_METRIC: "Δ approval rate",
    FPR_METRIC: "EOD (FPR gap)",
    FNR_METRIC: "ΔFNR",
    DEFAULT_AMONG_APPROVED_METRIC: "Δ default among approved",
    PPV_METRIC: "ΔPPV",
}

OOF_COLUMNS: List[str] = ["SK_ID_CURR", "TARGET", "y_pred_proba", "fold"]


def pti_quintile_labels(n_quantiles: int) -> List[str]:
    """Return the ``PTI_Q`` labels ``Q1 .. Q<n_quantiles>`` in order.

    Args:
        n_quantiles: Number of PTI quantile bins.

    Returns:
        Ordered list of labels.
    """
    return [f"Q{i}" for i in range(1, n_quantiles + 1)]


@dataclass(frozen=True)
class MetricSpec:
    """Subpopulation, event, and denominator name for one rate metric.

    Attributes:
        name: Metric name (key of ``group_metrics.METRIC_SPECS``).
        subpopulation: ``(y, yhat) -> bool mask`` of the rows the rate is over.
        event: ``(y, yhat) -> bool mask`` of the counted event.
        denominator: Name of the denominator (``n``, ``negatives``, ...).
    """

    name: str
    subpopulation: Callable[[np.ndarray, np.ndarray], np.ndarray]
    event: Callable[[np.ndarray, np.ndarray], np.ndarray]
    denominator: str


@dataclass(frozen=True)
class PairSpec:
    """Two groups defined by column filters; gaps are a − b."""

    pair_id: str
    a: Dict[str, str]
    b: Dict[str, str]


@dataclass(frozen=True)
class AuditConfig:
    """Resolved fairness audit settings from ``configs/fairness.yaml``."""

    seed: int
    n_boot: int
    confidence: float
    variant: str
    models: List[str]
    robustness_ranks: List[int]
    calibration_bins: int
    dir_reference_line: float
    groupings: Dict[str, List[str]]
    pairs: List[PairSpec]
    primary_pair: str
    primary_metrics: List[str]
    gap_metrics: List[str]
    cell_summary_metrics: List[str]
    robustness_metrics: List[str]
    cdp_pairs: List[str]
    cdp_n_quantiles: int
    cdp_expected_cut_points: List[float]
    cdp_expected_missing: int
    proxy_watch_list: Dict[str, List[str]]
    importance_type: Dict[str, str]

    @property
    def z(self) -> float:
        """Normal quantile for two-sided ``confidence`` (Wald / plug-in intervals)."""
        tail = (1.0 - self.confidence) / 2.0
        return float(NormalDist().inv_cdf(1.0 - tail))


@dataclass
class ThresholdResult:
    """Group-blind base-rate threshold summary."""

    threshold: float
    base_rate: float
    n_rows: int
    n_flagged: int
    n_positives: int
