"""Helpers for the pre-modelling bias audit (Phase 1, Step 5).

Used by ``analysis/notebooks/08_pre_modeling_bias_audit.ipynb``. Everything here
describes the labelled data (``TARGET``) or the engineered features, never the
predictions of a credit scorer. Interval methods follow ``docs/decisions.md``
("Fairness uncertainty method"): plug-in binomial intervals for rates, Wald
intervals for differences, log-scale (delta method) intervals for ratios, and
Lum et al.'s (2022) double-corrected bootstrap for summaries over many groups.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency, norm
from sklearn.metrics import roc_auc_score, roc_curve

from src.constants import SK_ID_CURR
from src.models.training import train_lgbm_cv
from src.models.types import OOF_PROBA_COL, TrainingData

Z_95 = float(norm.ppf(0.975))
MISSING_LABEL = "__MISSING__"


# ---------------------------------------------------------------------------
# Rates, gaps and ratios
# ---------------------------------------------------------------------------


def proportion_table(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    outcome_col: str,
    z: float = Z_95,
) -> pd.DataFrame:
    """Rate of a binary outcome per group with plug-in binomial intervals.

    Args:
        df: Rows with the grouping columns and a 0/1 outcome column.
        group_cols: Columns that define the groups.
        outcome_col: Binary outcome (1 = event).
        z: Normal quantile of the interval (1.96 for 95%).

    Returns:
        One row per group with ``n``, ``events``, ``non_events``, ``share``,
        ``rate``, ``se``, ``ci_low`` and ``ci_high``.
    """
    out = df.groupby(list(group_cols), observed=True)[outcome_col].agg(
        n="size", events="sum"
    )
    out["events"] = out["events"].astype(int)
    out["non_events"] = out["n"] - out["events"]
    out["share"] = out["n"] / out["n"].sum()
    out["rate"] = out["events"] / out["n"]
    out["se"] = np.sqrt(out["rate"] * (1 - out["rate"]) / out["n"])
    out["ci_low"] = (out["rate"] - z * out["se"]).clip(lower=0)
    out["ci_high"] = (out["rate"] + z * out["se"]).clip(upper=1)
    return out


def proportion_gap(
    p_a: float, n_a: int, p_b: float, n_b: int, z: float = Z_95
) -> Dict[str, float]:
    """Signed difference and ratio of two independent proportions.

    The difference uses a Wald interval with unpooled variance and a two-sided
    z-test with pooled variance (Hurlin et al. 2024, §4.1). The ratio ``p_a /
    p_b`` uses a delta-method interval on the log scale.

    Args:
        p_a: Proportion in group a.
        n_a: Size of group a.
        p_b: Proportion in group b.
        n_b: Size of group b.
        z: Normal quantile of the interval.

    Returns:
        Dict with ``diff``, ``diff_ci_low``, ``diff_ci_high``, ``p_value``,
        ``ratio``, ``ratio_ci_low`` and ``ratio_ci_high``.
    """
    diff = p_a - p_b
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    pooled = (p_a * n_a + p_b * n_b) / (n_a + n_b)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z_stat = diff / se_pooled if se_pooled > 0 else np.nan
    p_value = 2 * norm.sf(abs(z_stat)) if np.isfinite(z_stat) else np.nan

    ratio = p_a / p_b if p_b > 0 else np.nan
    if p_a > 0 and p_b > 0:
        se_log = np.sqrt((1 - p_a) / (n_a * p_a) + (1 - p_b) / (n_b * p_b))
        ratio_low = ratio * np.exp(-z * se_log)
        ratio_high = ratio * np.exp(z * se_log)
    else:
        ratio_low = ratio_high = np.nan

    return {
        "diff": diff,
        "diff_ci_low": diff - z * se_diff,
        "diff_ci_high": diff + z * se_diff,
        "p_value": p_value,
        "ratio": ratio,
        "ratio_ci_low": ratio_low,
        "ratio_ci_high": ratio_high,
    }


def pairwise_gaps(
    rates: pd.DataFrame,
    pairs: Sequence[Tuple[Any, Any]],
    rate_col: str = "rate",
    n_col: str = "n",
    z: float = Z_95,
) -> pd.DataFrame:
    """Signed gaps ``a − b`` and ratios ``a / b`` for chosen pairs of groups.

    Args:
        rates: Table indexed by group label with a rate and a size column
            (for example the output of :func:`proportion_table`).
        pairs: ``(a, b)`` index labels to compare.
        rate_col: Column with the proportion.
        n_col: Column with the denominator of the proportion.
        z: Normal quantile of the interval.

    Returns:
        One row per pair with both rates and the output of
        :func:`proportion_gap`.
    """
    rows = []
    for a, b in pairs:
        ra, rb = rates.loc[a], rates.loc[b]
        row = {
            "group_a": a,
            "group_b": b,
            "rate_a": ra[rate_col],
            "rate_b": rb[rate_col],
            "n_a": int(ra[n_col]),
            "n_b": int(rb[n_col]),
        }
        row.update(
            proportion_gap(ra[rate_col], ra[n_col], rb[rate_col], rb[n_col], z=z)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def worst_group_summary(
    n: Sequence[int],
    rate: Sequence[float],
    n_boot: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Max−min gap and between-group variance with Lum et al.'s corrections.

    The observed max−min gap and the uncorrected between-group variance are
    biased upwards (Lum et al. 2022, §4). The single-corrected variance
    subtracts the mean plug-in sampling variance (Eq. 2). For intervals, each
    group is resampled within itself with its size fixed (a binomial draw), and
    the double-corrected variance (§5.3) is computed on every bootstrap sample.

    Args:
        n: Denominator of each group's rate.
        rate: Observed rate of each group.
        n_boot: Number of bootstrap samples.
        seed: Random seed.
        alpha: One minus the interval coverage.

    Returns:
        Dict with the observed max−min gap and its percentile interval, and the
        uncorrected, single-corrected and double-corrected between-group
        variance (the last with a percentile interval). Standard deviations
        (square roots of the variances) are included for reading on the rate
        scale.
    """
    n_arr = np.asarray(n, dtype=float)
    y = np.asarray(rate, dtype=float)
    k = len(y)
    rng = np.random.default_rng(seed)

    var_uncorrected = float(np.var(y, ddof=1))
    sampling_var = y * (1 - y) / n_arr
    var_single = max(0.0, var_uncorrected - float(sampling_var.mean()))

    y_star = rng.binomial(n_arr.astype(int), y, size=(n_boot, k)) / n_arr
    maxmin_star = y_star.max(axis=1) - y_star.min(axis=1)
    var_star = np.var(y_star, axis=1, ddof=1)
    sampling_var_star = 2 * y_star * (1 - y_star) / n_arr - y_star * (1 - y_star) / (
        n_arr**2
    )
    var_double_star = np.maximum(0.0, var_star - sampling_var_star.mean(axis=1))

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
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


def cmh_test(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    strata_col: str,
    group_a: Any,
) -> Dict[str, float]:
    """Cochran-Mantel-Haenszel test of group vs outcome across strata.

    Tests whether ``group_a`` and the other group differ in the outcome after
    stratifying, and gives the Mantel-Haenszel common odds ratio (group a vs
    the rest). No continuity correction.

    Args:
        df: Rows with a two-level group column, a 0/1 outcome and the strata.
        group_col: Two-level group column.
        outcome_col: Binary outcome (1 = event).
        strata_col: Stratifying column.
        group_a: Level of ``group_col`` treated as the exposed group.

    Returns:
        Dict with ``k_strata``, ``mh_odds_ratio``, ``cmh_chi2`` and ``p_value``.
    """
    is_a = (df[group_col] == group_a).astype(int)
    frame = pd.DataFrame(
        {"a": is_a, "y": df[outcome_col].astype(int), "s": df[strata_col]}
    ).dropna()
    counts = (
        frame.groupby(["s", "a", "y"], observed=True)
        .size()
        .unstack(["a", "y"], fill_value=0)
        .reindex(columns=pd.MultiIndex.from_product([[0, 1], [0, 1]]), fill_value=0)
        .astype(float)
    )
    a, b = counts[(1, 1)], counts[(1, 0)]
    c, d = counts[(0, 1)], counts[(0, 0)]
    total = a + b + c + d

    mh_or = float((a * d / total).sum() / (b * c / total).sum())
    expected_a = (a + b) * (a + c) / total
    var_a = (a + b) * (c + d) * (a + c) * (b + d) / (total**2 * (total - 1))
    stat = float((a - expected_a).sum() ** 2 / var_a.sum())
    return {
        "k_strata": int(len(total)),
        "mh_odds_ratio": mh_or,
        "cmh_chi2": stat,
        "p_value": float(chi2.sf(stat, df=1)),
    }


# ---------------------------------------------------------------------------
# Association measures (proxies)
# ---------------------------------------------------------------------------


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V between two categorical series (no bias correction)."""
    table = pd.crosstab(x, y)
    stat = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return float(np.sqrt(stat / (n * k))) if k > 0 else np.nan


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Correlation ratio η of a numeric series over categories."""
    frame = pd.DataFrame({"cat": categories, "val": values}).dropna()
    grand_mean = frame["val"].mean()
    stats = frame.groupby("cat", observed=True)["val"].agg(["count", "mean"])
    between = (stats["count"] * (stats["mean"] - grand_mean) ** 2).sum()
    total = ((frame["val"] - grand_mean) ** 2).sum()
    return float(np.sqrt(between / total)) if total > 0 else np.nan


def discretize(series: pd.Series, n_bins: int = 10) -> pd.Series:
    """Categorical version of a column for Cramér's V.

    Numeric columns with more than ``n_bins`` distinct values are cut into
    quantile bins; other columns are used as they are. Missing values become
    their own category, so missingness counts as information.

    Args:
        series: Column to discretize.
        n_bins: Number of quantile bins for numeric columns.

    Returns:
        String categories aligned to ``series``.
    """
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > n_bins:
        binned = pd.qcut(series, q=n_bins, duplicates="drop").astype(str)
        return binned.where(series.notna(), MISSING_LABEL)
    return series.astype(object).where(series.notna(), MISSING_LABEL).astype(str)


def association_profile(
    df: pd.DataFrame,
    columns: Sequence[str],
    targets: Dict[str, pd.Series],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Cramér's V of each column with several categorical targets.

    Every column is discretized with :func:`discretize`, so categorical and
    numeric columns are on the same 0–1 scale (as in Hurlin et al. 2024,
    Fig. 1).

    Args:
        df: Data with the columns to profile.
        columns: Columns to profile.
        targets: Name → categorical series (aligned to ``df``) to associate
            each column with, e.g. gender, age band and ``TARGET``.
        n_bins: Number of quantile bins for numeric columns.

    Returns:
        One row per column with ``pct_missing`` and one ``v_<name>`` column per
        target.
    """
    rows = []
    for col in columns:
        binned = discretize(df[col], n_bins=n_bins)
        row = {"column": col, "pct_missing": 100 * df[col].isna().mean()}
        for name, target in targets.items():
            row[f"v_{name}"] = cramers_v(binned, target)
        rows.append(row)
    return pd.DataFrame(rows).set_index("column")


# ---------------------------------------------------------------------------
# Missingness
# ---------------------------------------------------------------------------


def missing_rate_by_group(indicators: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Share of missing values per column and group.

    Args:
        indicators: Boolean frame, True where a value is missing (or where a
            structural code such as a sentinel is present).
        groups: Group label per row, aligned to ``indicators``.

    Returns:
        Columns × groups table of missing shares, plus an ``overall`` column.
    """
    by_group = indicators.groupby(groups, observed=True).mean().T
    by_group["overall"] = indicators.mean()
    return by_group


def outcome_by_missingness(
    indicators: pd.DataFrame,
    outcome: pd.Series,
    groups: pd.Series,
) -> pd.DataFrame:
    """Outcome rate with and without a value, within each group.

    Args:
        indicators: Boolean frame, True where a value is missing.
        outcome: Binary outcome aligned to ``indicators``.
        groups: Group label per row, aligned to ``indicators``.

    Returns:
        Long table with one row per column and group: ``n_missing``,
        ``rate_missing``, ``rate_present`` and their signed gap
        ``missing − present`` with a Wald interval.
    """
    rows = []
    for col in indicators.columns:
        for group, idx in groups.groupby(groups, observed=True).groups.items():
            miss = indicators.loc[idx, col]
            y = outcome.loc[idx]
            n_miss, n_pres = int(miss.sum()), int((~miss).sum())
            if n_miss == 0 or n_pres == 0:
                continue
            p_miss, p_pres = float(y[miss].mean()), float(y[~miss].mean())
            gap = proportion_gap(p_miss, n_miss, p_pres, n_pres)
            rows.append(
                {
                    "column": col,
                    "group": group,
                    "n_missing": n_miss,
                    "pct_missing": 100 * n_miss / (n_miss + n_pres),
                    "rate_missing": p_miss,
                    "rate_present": p_pres,
                    "diff": gap["diff"],
                    "diff_ci_low": gap["diff_ci_low"],
                    "diff_ci_high": gap["diff_ci_high"],
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Joint predictability of the protected attribute (Feldman et al. 2015)
# ---------------------------------------------------------------------------


@dataclass
class PredictabilityResult:
    """Out-of-fold predictability of a binary protected attribute."""

    oof: pd.DataFrame
    roc_auc: float
    fold_roc_aucs: List[float]
    min_ber: float
    min_ber_threshold: float
    ber_at_prevalence: float
    importance_gain: Optional[pd.DataFrame]
    dropped_columns: List[str]


def balanced_error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced error rate: mean of the two class-conditional error rates."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    err_pos = float(np.mean(y_pred[y_true == 1] == 0))
    err_neg = float(np.mean(y_pred[y_true == 0] == 1))
    return 0.5 * (err_pos + err_neg)


def feldman_ber_threshold(beta: float, tau: float = 0.8) -> float:
    """BER threshold below which a data set admits disparate impact.

    Feldman et al. (2015), Thm. 4.1 and its general-τ form:
    ``ε = 1/2 − β(1/τ − 1)/2``, where β is the share of the protected group
    with the favourable outcome. For τ = 0.8 this is ``1/2 − β/8``.

    Args:
        beta: Favourable-outcome share of the protected group.
        tau: Disparate impact ratio threshold (0.8 = four-fifths rule).

    Returns:
        The BER threshold ε.
    """
    return 0.5 - beta * (1 / tau - 1) / 2


def protected_attribute_predictability(
    X: pd.DataFrame,
    attribute: pd.Series,
    ids: pd.Series,
    config: Dict[str, Any],
    positive_label: Any,
    drop_columns: Sequence[str],
) -> PredictabilityResult:
    """Predict a binary protected attribute from the other features.

    An auxiliary LightGBM classifier (not the credit scorer) is trained with
    the same stratified CV and hyperparameters as the credit model
    (``configs/model.yaml``), with the protected columns removed. Its
    out-of-fold balanced error rate (BER) is Feldman et al.'s (2015)
    predictability measure: the lower the BER, the more the attribute is
    encoded in the remaining features.

    Args:
        X: Engineered feature matrix (``artifacts/features/X_train.parquet``).
        attribute: Raw protected attribute aligned to ``X``.
        ids: ``SK_ID_CURR`` aligned to ``X``.
        config: Model configuration (seed, folds, LightGBM parameters).
        positive_label: Level of ``attribute`` coded as 1.
        drop_columns: Columns of ``X`` to remove (the protected columns).

    Returns:
        PredictabilityResult with OOF scores, AUC, BER and gain importance.
    """
    X_aux = X.drop(columns=list(drop_columns))
    y_aux = (attribute == positive_label).astype(int).rename("ATTRIBUTE")
    data = TrainingData(
        X=X_aux,
        y=y_aux.reset_index(drop=True),
        ids=ids.reset_index(drop=True),
        feature_names=list(X_aux.columns),
    )
    cv = train_lgbm_cv(data, config, return_models=True)

    proba = cv.oof_predictions[OOF_PROBA_COL].to_numpy()
    y_true = data.y.to_numpy()
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    ber_curve = 0.5 * (fpr + (1 - tpr))
    best = int(np.argmin(ber_curve))
    prevalence = float(y_true.mean())

    oof = cv.oof_predictions.rename(columns={"TARGET": "ATTRIBUTE"})
    return PredictabilityResult(
        oof=oof[[SK_ID_CURR, "ATTRIBUTE", OOF_PROBA_COL, "fold"]],
        roc_auc=float(roc_auc_score(y_true, proba)),
        fold_roc_aucs=cv.fold_roc_aucs,
        min_ber=float(ber_curve[best]),
        min_ber_threshold=float(thresholds[best]),
        ber_at_prevalence=balanced_error_rate(y_true, proba >= prevalence),
        importance_gain=cv.feature_importance_gain,
        dropped_columns=list(drop_columns),
    )
