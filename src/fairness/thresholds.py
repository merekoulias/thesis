"""Group-blind base-rate decision threshold."""

import logging

import numpy as np

from src.fairness.types import ThresholdResult

logger = logging.getLogger(__name__)


def base_rate_threshold(score: np.ndarray, y: np.ndarray) -> ThresholdResult:
    """Choose ``t`` so that ``Ŷ = 1{score > t}`` flags the empirical base rate.

    Uses ``t = np.quantile(score, 1 - mean(y))`` (numpy default ``linear`` method).

    Args:
        score: Predicted scores in ``[0, 1]``.
        y: Binary labels (0/1).

    Returns:
        Threshold summary including ``n_flagged = (score > t).sum()``.
    """
    score_arr = np.asarray(score, dtype=float)
    y_arr = np.asarray(y)
    n_rows = y_arr.size
    base_rate = float(y_arr.mean()) if n_rows else float("nan")
    n_positives = int(y_arr.sum())
    if n_rows == 0:
        threshold = float("nan")
        n_flagged = 0
    else:
        threshold = float(np.quantile(score_arr, 1.0 - base_rate))
        n_flagged = int((score_arr > threshold).sum())
    if n_flagged != n_positives:
        logger.warning(
            "n_flagged (%s) != n_positives (%s) at base-rate threshold",
            n_flagged,
            n_positives,
        )
    return ThresholdResult(
        threshold=threshold,
        base_rate=base_rate,
        n_rows=n_rows,
        n_flagged=n_flagged,
        n_positives=n_positives,
    )


def apply_threshold(score: np.ndarray, threshold: float) -> np.ndarray:
    """Return ``yhat = (score > threshold).astype(np.int8)``.

    Args:
        score: Predicted scores in ``[0, 1]``.
        threshold: Decision threshold (``ThresholdResult.threshold``).

    Returns:
        ``int8`` array, 1 = flagged (credit refused), 0 = approved.
    """
    score_arr = np.asarray(score, dtype=float)
    return (score_arr > threshold).astype(np.int8)
