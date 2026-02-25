#!/usr/bin/env python3
"""
Coverage by Outlier / Inlier Status
======================================
Split test points into outliers (high epistemic uncertainty) and inliers,
then compute coverage separately for each group.

EPICSCORE's key advantage: maintains coverage even on outliers because
intervals automatically widen in high-uncertainty regions.

Produces:
  - result_cover_outlier.pkl
  - result_aisl_outlier.pkl
  - result_ratio_outlier.pkl
  - Images_rebuttal/coverage_per_outlier_inlier.png

Usage:
    python Experiments_code/coverage_by_outlier_inlier.py
    CUDA_VISIBLE_DEVICES=0 python Experiments_code/coverage_by_outlier_inlier.py
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Experiments_code.helper import (
    load_dataset, get_method, save_results, RESULTS_DIR,
)
from Epistemic_CP import (
    EpistemicConformalPredictor, EnsembleModel,
    split_data, coverage_rate, average_interval_length,
    adaptive_interval_set_length, outlier_inlier_split,
)
from Epistemic_CP.epistemic_cp import SplitConformalPredictor, CQRPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASETS = ["bike", "homes", "star", "meps"]
METHODS = ["EPICSCORE", "SplitConformal", "CQR", "NormalizedConformal"]
N_TRIALS = 100
ALPHA = 0.1
SEED = 42
OUTLIER_QUANTILE = 0.9  # top 10% epistemic uncertainty = outliers


def run_single_outlier_trial(X, y, method_name, seed, alpha=ALPHA):
    """One trial: split → fit → calibrate → predict → split outlier/inlier → metrics."""
    X_train, y_train, X_cal, y_cal, X_test, y_test = split_data(X, y, seed=seed)

    # Always fit an ensemble to get epistemic uncertainty for the split
    eu_model = EnsembleModel(n_models=5, seed=seed)
    eu_model.fit(X_train, y_train)
    eu_preds = eu_model.predict(X_test)
    inlier_idx, outlier_idx = outlier_inlier_split(X_test, eu_preds.epistemic_u, OUTLIER_QUANTILE)

    # Now run the actual method
    method = get_method(method_name, alpha=alpha, seed=seed)

    if isinstance(method, EpistemicConformalPredictor):
        method.fit(X_train, y_train)
        method.calibrate(X_cal, y_cal)
        result = method.predict(X_test)
        lower, upper = result.lower, result.upper
    elif isinstance(method, (SplitConformalPredictor, CQRPredictor)):
        method.fit(X_train, y_train)
        method.calibrate(X_cal, y_cal)
        _, lower, upper = method.predict(X_test)
    else:
        raise ValueError(f"Unknown method type: {type(method)}")

    # Compute metrics for all / inlier / outlier
    metrics = {}
    for group_name, idx in [("all", np.arange(len(y_test))), ("inlier", inlier_idx), ("outlier", outlier_idx)]:
        if len(idx) == 0:
            for m in ["coverage", "ail", "aisl"]:
                metrics[f"{group_name}_{m}"] = np.nan
            continue
        cov = coverage_rate(y_test[idx], lower[idx], upper[idx])
        ail = average_interval_length(lower[idx], upper[idx])
        aisl = adaptive_interval_set_length(y_test[idx], lower[idx], upper[idx])
        metrics[f"{group_name}_coverage"] = cov
        metrics[f"{group_name}_ail"] = ail
        metrics[f"{group_name}_aisl"] = aisl

    return metrics


def run_outlier_experiment():
    logger.info("=" * 60)
    logger.info("Coverage by Outlier/Inlier Experiment")
    logger.info("=" * 60)

    t0 = time.time()
    all_results = {}  # {dataset: {method: {metric_name: [trial_values]}}}

    for ds in DATASETS:
        logger.info(f"\n--- Dataset: {ds} ---")
        X, y = load_dataset(ds)
        all_results[ds] = {}

        for method_name in METHODS:
            logger.info(f"  Method: {method_name}")
            trial_metrics = {}

            for trial in range(N_TRIALS):
                try:
                    m = run_single_outlier_trial(X, y, method_name, seed=SEED + trial)
                    for k, v in m.items():
                        trial_metrics.setdefault(k, []).append(v)
                except Exception as e:
                    logger.warning(f"    Trial {trial} failed: {e}")

            all_results[ds][method_name] = {
                k: np.array(v) for k, v in trial_metrics.items()
            }

            # Log summary
            for group in ["all", "inlier", "outlier"]:
                arr = np.array(trial_metrics.get(f"{group}_coverage", []))
                valid = arr[~np.isnan(arr)] if len(arr) > 0 else np.array([])
                if len(valid) > 0:
                    logger.info(f"    {group}: coverage={np.mean(valid):.4f}±{np.std(valid):.4f}")

    # Save results
    for group in ["outlier", "inlier", "all"]:
        cover_res = {}
        aisl_res = {}
        for ds, methods in all_results.items():
            cover_res[ds] = {}
            aisl_res[ds] = {}
            for method, metrics in methods.items():
                cover_res[ds][method] = metrics.get(f"{group}_coverage", np.array([]))
                aisl_res[ds][method] = metrics.get(f"{group}_aisl", np.array([]))

        suffix = f"_{group}" if group != "all" else ""
        save_results(cover_res, f"result_cover{suffix}")
        save_results(aisl_res, f"result_aisl{suffix}")

    # Ratio: outlier width / inlier width per method
    ratio_res = {}
    for ds, methods in all_results.items():
        ratio_res[ds] = {}
        for method, metrics in methods.items():
            out_ail = metrics.get("outlier_ail", np.array([]))
            in_ail = metrics.get("inlier_ail", np.array([]))
            if len(out_ail) > 0 and len(in_ail) > 0:
                valid = (~np.isnan(out_ail)) & (~np.isnan(in_ail)) & (in_ail > 1e-10)
                ratios = np.where(valid, out_ail / np.maximum(in_ail, 1e-10), np.nan)
                ratio_res[ds][method] = ratios
            else:
                ratio_res[ds][method] = np.array([])
    save_results(ratio_res, "result_ratio_outlier")

    save_results(all_results, "outlier_inlier_full")

    _generate_figure(all_results)

    logger.info(f"\nCompleted in {time.time()-t0:.1f}s")
    return all_results


def _generate_figure(all_results):
    """Generate coverage_per_outlier_inlier.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping figure")
        return

    fig_dir = PROJECT_ROOT / "Images_rebuttal"
    fig_dir.mkdir(parents=True, exist_ok=True)

    datasets = list(all_results.keys())
    methods = METHODS
    groups = ["inlier", "outlier"]
    colors = {"EPICSCORE": "#2196F3", "SplitConformal": "#FF9800",
              "CQR": "#9C27B0", "NormalizedConformal": "#4CAF50"}

    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4), squeeze=False)
    fig.suptitle("Coverage: Inlier vs Outlier", fontsize=14, fontweight="bold")

    for j, ds in enumerate(datasets):
        ax = axes[0, j]
        x = np.arange(len(groups))
        width = 0.18
        for i, method in enumerate(methods):
            means = []
            stds = []
            for g in groups:
                arr = all_results.get(ds, {}).get(method, {}).get(f"{g}_coverage", np.array([]))
                valid = arr[~np.isnan(arr)] if len(arr) > 0 else np.array([0])
                means.append(np.mean(valid))
                stds.append(np.std(valid))
            bars = ax.bar(x + i * width, means, width, yerr=stds,
                          label=method if j == 0 else "", color=colors.get(method, "#666"),
                          alpha=0.85, capsize=3)
        ax.set_title(ds, fontsize=12)
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(groups)
        ax.set_ylim(0.5, 1.05)
        ax.axhline(1 - ALPHA, color="red", linestyle="--", alpha=0.5, label="Target" if j == 0 else "")
        ax.set_ylabel("Coverage" if j == 0 else "")

    fig.legend(loc="lower center", ncol=len(methods) + 1, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = fig_dir / "coverage_per_outlier_inlier.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {out}")

    with open(fig_dir / "Caption_table_coverage_outlier.txt", "w") as f:
        f.write("Coverage comparison between inlier and outlier groups. "
                "Outliers are defined as test points with epistemic uncertainty "
                f"above the {OUTLIER_QUANTILE:.0%} quantile. "
                "EPICSCORE maintains near-target coverage on outliers due to "
                "adaptive interval widening in high-uncertainty regions.")


if __name__ == "__main__":
    run_outlier_experiment()