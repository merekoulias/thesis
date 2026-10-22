"""Confidence intervals, G-tests, Lum et al. summaries, and bootstrap helpers."""

import zlib
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.stats import chi2, norm
from sklearn.metrics import roc_auc_score


def z_value(confidence: float) -> float:
    """Normal quantile for a two-sided interval at ``confidence``.

    Args:
        confidence: Coverage level in (0, 1), e.g. 0.95.

    Returns:
        ``scipy.stats.norm.ppf(1 - (1 - confidence) / 2)``.
    """
    tail = (1.0 - confidence) / 2.0
    return float(norm.ppf(1.0 - tail))


def seed_for(base_seed: int, key: str) -> int:
    """Derive a deterministic 32-bit seed from a base seed and task key.

    Args:
        base_seed: Global audit seed from config.
        key: Task identifier (model, metric, group, …).

    Returns:
        Integer in ``[0, 2**32)``.
    """
    return (base_seed * 1_000_003 + zlib.crc32(key.encode("utf-8"))) % 2**32


def proportion_ci(events: int, n: int, z: float) -> Tuple[float, float, float]:
    """Plug-in binomial Wald interval for a proportion.

    Uses ``r = events / n`` and ``se = sqrt(r(1 - r) / n)``. The interval
    endpoints are clipped to ``[0, 1]``. When ``r`` is 0 or 1, ``se`` is 0
    and the interval collapses to ``r`` (plug-in method).

    Args:
        events: Count of successes.
        n: Denominator.
        z: Normal quantile (two-sided).

    Returns:
        ``(rate, ci_low, ci_high)``; ``(nan, nan, nan)`` if ``n == 0``.
    """
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    r = events / n
    se = float(np.sqrt(r * (1.0 - r) / n))
    ci_low = max(0.0, r - z * se)
    ci_high = min(1.0, r + z * se)
    return (r, ci_low, ci_high)


def diff_ci(
    r_a: float, n_a: int, r_b: float, n_b: int, z: float
) -> Tuple[float, float, float]:
    """Wald interval for the difference of two independent proportions.

    Args:
        r_a: Rate in group a.
        n_a: Size of group a.
        r_b: Rate in group b.
        n_b: Size of group b.
        z: Normal quantile (two-sided).

    Returns:
        ``(diff, ci_low, ci_high)`` with ``diff = r_a - r_b``; endpoints are
        not clipped to ``[0, 1]``.
    """
    d = r_a - r_b
    se = float(np.sqrt(r_a * (1.0 - r_a) / n_a + r_b * (1.0 - r_b) / n_b))
    return (d, d - z * se, d + z * se)


def ratio_ci(
    r_a: float, n_a: int, r_b: float, n_b: int, z: float
) -> Tuple[float, float, float]:
    """Delta-method interval for ``r_a / r_b`` on the log scale.

    Args:
        r_a: Rate in group a.
        n_a: Size of group a.
        r_b: Rate in group b.
        n_b: Size of group b.
        z: Normal quantile (two-sided).

    Returns:
        ``(ratio, ci_low, ci_high)``; all NaN if either rate is 0.
    """
    if r_a == 0.0 or r_b == 0.0:
        nan = float("nan")
        return (nan, nan, nan)
    q = r_a / r_b
    se_log = float(np.sqrt((1.0 - r_a) / (n_a * r_a) + (1.0 - r_b) / (n_b * r_b)))
    return (q, q * np.exp(-z * se_log), q * np.exp(z * se_log))


def _drop_zero_rows_cols(table: np.ndarray) -> np.ndarray:
    arr = np.asarray(table, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 0)
    row_ok = arr.sum(axis=1) > 0
    arr = arr[row_ok]
    if arr.size == 0:
        return arr.reshape(0, 0)
    col_ok = arr.sum(axis=0) > 0
    return arr[:, col_ok]


def g_test(table: np.ndarray) -> Tuple[float, int, float]:
    """Likelihood-ratio (G) test of independence (Hurlin et al.).

    All-zero rows and columns are dropped before computing expected counts.
    ``G = 2 sum O ln(O/E)`` over cells with ``O > 0``. If the residual
    degrees of freedom are 0, returns ``(0.0, 0, nan)``.

    Args:
        table: ``k × c`` array of non-negative counts.

    Returns:
        ``(G, df, p_value)``.
    """
    arr = _drop_zero_rows_cols(table)
    if arr.size == 0:
        return (0.0, 0, float("nan"))
    k, c = arr.shape
    if k <= 1 or c <= 1:
        return (0.0, 0, float("nan"))
    n_total = arr.sum()
    row_tot = arr.sum(axis=1, keepdims=True)
    col_tot = arr.sum(axis=0, keepdims=True)
    expected = row_tot @ col_tot / n_total
    observed = arr
    mask = observed > 0
    g_stat = float(
        2.0 * np.sum(observed[mask] * np.log(observed[mask] / expected[mask]))
    )
    df = (k - 1) * (c - 1)
    p_value = float(chi2.sf(g_stat, df))
    return (g_stat, df, p_value)


def stratified_g_test(tables: Sequence[np.ndarray]) -> Tuple[float, int, float]:
    """Sum G statistics and degrees of freedom across strata.

    Args:
        tables: One contingency table per stratum.

    Returns:
        ``(G_sum, df_sum, p_value)`` with ``p_value = chi2.sf(G_sum, df_sum)``.
    """
    g_sum = 0.0
    df_sum = 0
    for table in tables:
        g_stat, df, _ = g_test(table)
        if df == 0:
            continue
        g_sum += g_stat
        df_sum += df
    if df_sum == 0:
        return (0.0, 0, float("nan"))
    return (g_sum, df_sum, float(chi2.sf(g_sum, df_sum)))


def mean_ci(values: np.ndarray, z: float) -> Tuple[float, float, float]:
    """Wald interval for the mean of ``values``.

    Args:
        values: 1-D sample.
        z: Normal quantile (two-sided).

    Returns:
        ``(mean, ci_low, ci_high)``; ``(nan, nan, nan)`` if ``n == 0``.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        nan = float("nan")
        return (nan, nan, nan)
    m = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(n))
    return (m, m - z * se, m + z * se)


def worst_group_summary(
    n: Sequence[int],
    rate: Sequence[float],
    n_boot: int,
    seed: int,
    alpha: float,
) -> Dict[str, float]:
    """Max−min gap and Lum et al. between-group variance corrections.

    Replicates ``analysis.helpers.data_audit.worst_group_summary`` (binomial
    within-group bootstrap, single- and double-corrected variance).

    Args:
        n: Denominator of each group's rate.
        rate: Observed rate per group.
        n_boot: Bootstrap replicates.
        seed: RNG seed.
        alpha: One minus interval coverage (e.g. 0.05 for 95%).

    Returns:
        Dict with ``k_groups``, max−min gap and CI, variance summaries and
        standard deviations (see the data-audit helper for key names).
    """
    n_arr = np.asarray(n, dtype=float)
    y = np.asarray(rate, dtype=float)
    k = len(y)
    rng = np.random.default_rng(seed)

    var_uncorrected = float(np.var(y, ddof=1))
    sampling_var = y * (1.0 - y) / n_arr
    var_single = max(0.0, var_uncorrected - float(sampling_var.mean()))

    y_star = rng.binomial(n_arr.astype(int), y, size=(n_boot, k)) / n_arr
    maxmin_star = y_star.max(axis=1) - y_star.min(axis=1)
    var_star = np.var(y_star, axis=1, ddof=1)
    sampling_var_star = 2.0 * y_star * (1.0 - y_star) / n_arr - y_star * (
        1.0 - y_star
    ) / (n_arr**2)
    var_double_star = np.maximum(0.0, var_star - sampling_var_star.mean(axis=1))

    lo, hi = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    return {
        "k_groups": k,
        "max_min_gap": float(y.max() - y.min()),
        "max_min_ci_low": float(np.percentile(maxmin_star, lo)),
        "max_min_ci_high": float(np.percentile(maxmin_star, hi)),
        "var_uncorrected": var_uncorrected,
        "var_single_corrected": var_single,
        "var_double_corrected_ci_low": float(np.percentile(var_double_star, lo)),
        "var_double_corrected_ci_high": float(np.percentile(var_double_star, hi)),
        "sd_uncorrected": float(np.sqrt(var_uncorrected)),
        "sd_single_corrected": float(np.sqrt(var_single)),
        "sd_double_corrected_ci_low": float(
            np.sqrt(np.percentile(var_double_star, lo))
        ),
        "sd_double_corrected_ci_high": float(
            np.sqrt(np.percentile(var_double_star, hi))
        ),
    }


def bootstrap_auc_ci(
    y: np.ndarray,
    score: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float,
) -> Tuple[float, float, float]:
    """Percentile bootstrap interval for ROC AUC within one group.

    Args:
        y: Binary labels (0/1).
        score: Predicted scores in ``[0, 1]``.
        n_boot: Number of bootstrap draws.
        seed: RNG seed.
        alpha: One minus interval coverage.

    Returns:
        ``(point_auc, ci_low, ci_high)``. Replicates with a single class are
        skipped (stored as NaN in the bootstrap distribution).
    """
    y_arr = np.asarray(y)
    score_arr = np.asarray(score, dtype=float)
    n = y_arr.size
    point = float(roc_auc_score(y_arr, score_arr))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b = y_arr[idx]
        if len(np.unique(y_b)) < 2:
            boot[b] = np.nan
        else:
            boot[b] = roc_auc_score(y_b, score_arr[idx])
    lo_pct = 100.0 * alpha / 2.0
    hi_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_low = float(np.nanpercentile(boot, lo_pct))
    ci_high = float(np.nanpercentile(boot, hi_pct))
    return (point, ci_low, ci_high)
