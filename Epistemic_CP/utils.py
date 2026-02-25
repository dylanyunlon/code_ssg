"""
Epistemic CP Utilities
======================
Data splitting, evaluation metrics, and helper functions.

Metrics implemented:
  - coverage_rate: empirical coverage (fraction of true values inside intervals)
  - average_interval_length (AIL): mean width of prediction intervals
  - adaptive_interval_set_length (AISL): AIL normalized by local difficulty
  - interval_width_ratio: ratio of widths between two methods
  - partial_correlation (PCOR): correlation between interval width and epistemic uncertainty
  - conditional_coverage: coverage conditioned on epistemic uncertainty quantiles

All metrics are computed FROM DATA produced by running experiments — never hardcoded.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List


# =============================================================================
# Data Splitting
# =============================================================================

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.6,
    cal_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """
    Split data into train / calibration / test sets.

    Returns:
        (X_train, y_train, X_cal, y_cal, X_test, y_test)
    """
    assert abs(train_ratio + cal_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + cal_ratio + test_ratio}"

    n = len(X)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_cal = int(n * cal_ratio)

    train_idx = idx[:n_train]
    cal_idx = idx[n_train:n_train + n_cal]
    test_idx = idx[n_train + n_cal:]

    return (
        X[train_idx], y[train_idx],
        X[cal_idx], y[cal_idx],
        X[test_idx], y[test_idx],
    )


def split_data_indices(
    n: int,
    train_ratio: float = 0.6,
    cal_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return index arrays for train / calibration / test.

    Returns:
        (train_idx, cal_idx, test_idx)
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_cal = int(n * cal_ratio)

    return idx[:n_train], idx[n_train:n_train + n_cal], idx[n_train + n_cal:]


# =============================================================================
# Core Metrics
# =============================================================================

def coverage_rate(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Empirical coverage: fraction of y_true inside [lower, upper].

    Args:
        y_true: true values (n,)
        lower: lower bounds of prediction intervals (n,)
        upper: upper bounds of prediction intervals (n,)

    Returns:
        float in [0, 1]
    """
    y_true = np.asarray(y_true).ravel()
    lower = np.asarray(lower).ravel()
    upper = np.asarray(upper).ravel()
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def average_interval_length(
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Average Interval Length (AIL): mean width of prediction intervals.

    Args:
        lower: lower bounds (n,)
        upper: upper bounds (n,)

    Returns:
        float >= 0
    """
    widths = np.asarray(upper).ravel() - np.asarray(lower).ravel()
    return float(np.mean(widths))


def adaptive_interval_set_length(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    epistemic_uncertainty: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> float:
    """
    Adaptive Interval Set Length (AISL).

    AISL normalizes interval width by the local difficulty of the prediction.
    If epistemic_uncertainty is provided, bins are defined by uncertainty quantiles.
    Otherwise, bins are defined by y_true quantiles.

    AISL = mean over bins of (mean_width_in_bin / std_y_in_bin)

    This captures whether the model allocates wider intervals where there
    is genuinely more uncertainty, rewarding adaptive methods.

    Args:
        y_true: true values (n,)
        lower: lower bounds (n,)
        upper: upper bounds (n,)
        epistemic_uncertainty: optional uncertainty scores (n,)
        n_bins: number of quantile bins

    Returns:
        float (lower is better, conditioned on coverage being maintained)
    """
    y_true = np.asarray(y_true).ravel()
    widths = np.asarray(upper).ravel() - np.asarray(lower).ravel()

    # Define bin assignments
    if epistemic_uncertainty is not None:
        bin_values = np.asarray(epistemic_uncertainty).ravel()
    else:
        bin_values = y_true

    # Quantile-based binning
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(bin_values, quantiles)

    ratios = []
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (bin_values >= bin_edges[i]) & (bin_values < bin_edges[i + 1])
        else:
            mask = (bin_values >= bin_edges[i]) & (bin_values <= bin_edges[i + 1])

        if mask.sum() < 2:
            continue

        bin_width = np.mean(widths[mask])
        bin_std = np.std(y_true[mask])
        if bin_std > 1e-10:
            ratios.append(bin_width / bin_std)

    if not ratios:
        return float(np.mean(widths))

    return float(np.mean(ratios))


def interval_width_ratio(
    lower_a: np.ndarray, upper_a: np.ndarray,
    lower_b: np.ndarray, upper_b: np.ndarray,
) -> float:
    """
    Interval Width Ratio: mean(width_A) / mean(width_B).

    Used to compare two methods. Ratio < 1 means method A produces tighter intervals.

    Args:
        lower_a, upper_a: intervals from method A
        lower_b, upper_b: intervals from method B

    Returns:
        float > 0
    """
    width_a = np.mean(np.asarray(upper_a) - np.asarray(lower_a))
    width_b = np.mean(np.asarray(upper_b) - np.asarray(lower_b))
    if width_b < 1e-10:
        return float('inf')
    return float(width_a / width_b)


def partial_correlation(
    interval_widths: np.ndarray,
    epistemic_uncertainty: np.ndarray,
    y_residuals: Optional[np.ndarray] = None,
) -> float:
    """
    Partial Correlation (PCOR) between interval width and epistemic uncertainty.

    If y_residuals are provided, computes the partial correlation controlling
    for residual magnitude. Otherwise computes simple Pearson correlation.

    Higher PCOR indicates the model is correctly widening intervals where
    epistemic uncertainty is high.

    Args:
        interval_widths: width of each prediction interval (n,)
        epistemic_uncertainty: epistemic uncertainty estimate (n,)
        y_residuals: optional |y_true - y_pred| residuals (n,)

    Returns:
        float in [-1, 1]
    """
    w = np.asarray(interval_widths).ravel()
    u = np.asarray(epistemic_uncertainty).ravel()

    if y_residuals is None:
        # Simple Pearson correlation
        if np.std(w) < 1e-10 or np.std(u) < 1e-10:
            return 0.0
        return float(np.corrcoef(w, u)[0, 1])

    # Partial correlation: corr(w, u | r)
    r = np.asarray(y_residuals).ravel()

    # Regress w on r, get residuals
    if np.std(r) < 1e-10:
        w_resid = w - np.mean(w)
    else:
        beta_w = np.cov(w, r)[0, 1] / np.var(r)
        w_resid = w - (np.mean(w) + beta_w * (r - np.mean(r)))

    # Regress u on r, get residuals
    if np.std(r) < 1e-10:
        u_resid = u - np.mean(u)
    else:
        beta_u = np.cov(u, r)[0, 1] / np.var(r)
        u_resid = u - (np.mean(u) + beta_u * (r - np.mean(r)))

    if np.std(w_resid) < 1e-10 or np.std(u_resid) < 1e-10:
        return 0.0

    return float(np.corrcoef(w_resid, u_resid)[0, 1])


def conditional_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    condition_values: np.ndarray,
    n_bins: int = 5,
) -> Dict[str, float]:
    """
    Coverage conditioned on bins of condition_values (e.g., epistemic uncertainty).

    Returns:
        dict mapping bin label → coverage rate
    """
    y_true = np.asarray(y_true).ravel()
    lower = np.asarray(lower).ravel()
    upper = np.asarray(upper).ravel()
    cv = np.asarray(condition_values).ravel()

    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(cv, quantiles)

    result = {}
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (cv >= lo) & (cv < hi)
        else:
            mask = (cv >= lo) & (cv <= hi)

        if mask.sum() == 0:
            continue

        cov = coverage_rate(y_true[mask], lower[mask], upper[mask])
        label = f"Q{i+1}[{lo:.2f},{hi:.2f}]"
        result[label] = cov

    return result


def outlier_inlier_split(
    X: np.ndarray,
    epistemic_uncertainty: np.ndarray,
    threshold_quantile: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split indices into outliers (high epistemic uncertainty) and inliers.

    Args:
        X: features (used for alignment only)
        epistemic_uncertainty: uncertainty scores (n,)
        threshold_quantile: quantile above which points are 'outliers'

    Returns:
        (inlier_indices, outlier_indices)
    """
    eu = np.asarray(epistemic_uncertainty).ravel()
    threshold = np.quantile(eu, threshold_quantile)
    outlier_mask = eu >= threshold
    return np.where(~outlier_mask)[0], np.where(outlier_mask)[0]
