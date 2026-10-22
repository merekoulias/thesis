"""Constants and datatypes for the fairness audit package."""

from dataclasses import dataclass
from typing import Dict, List

from scipy.stats import norm

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
PTI_COL = "PTI"
PTI_Q_COL = "PTI_Q"
YHAT_COL = "yhat"

METRIC_NAMES: tuple = (
    "approval_rate",
    "fpr",
    "fnr",
    "tpr",
    "ppv",
    "default_among_approved",
)

OOF_COLUMNS: List[str] = ["SK_ID_CURR", "TARGET", "y_pred_proba", "fold"]


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
        return float(norm.ppf(1.0 - tail))


@dataclass
class ThresholdResult:
    """Group-blind base-rate threshold summary."""

    threshold: float
    base_rate: float
    n_rows: int
    n_flagged: int
    n_positives: int
