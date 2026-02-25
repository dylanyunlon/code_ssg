#!/usr/bin/env python3
"""
Metrics on Real Datasets
==========================
Run EPICSCORE + baselines on all real datasets and compute the full metric suite:
  - Coverage, AIL, AISL, PCOR
  - Conditional coverage (by epistemic uncertainty quantile)
  - Running time

Outputs a comprehensive JSON and prints a LaTeX-ready table.

Usage:
    python Experiments_code/metrics_real_data.py
    python Experiments_code/metrics_real_data.py --trials 50 --datasets bike homes
"""

import sys
import time
import argparse
import logging
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Experiments_code.helper import (
    load_dataset, run_experiment, save_results, RESULTS_DIR,
)
from Epistemic_CP import (
    EpistemicConformalPredictor, EnsembleModel,
    split_data, coverage_rate, average_interval_length,
    adaptive_interval_set_length, partial_correlation,
    conditional_coverage,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASETS = ["bike", "homes", "star", "meps",
            "WEC_Perth_49", "WEC_Sydney_49"]
METHODS = ["EPICSCORE", "SplitConformal", "CQR", "NormalizedConformal"]
N_TRIALS = 100
ALPHA = 0.1
SEED = 42


def run_real_data_metrics(datasets=None, methods=None, n_trials=None):
    datasets = datasets or DATASETS
    methods = methods or METHODS
    n_trials = n_trials or N_TRIALS

    logger.info("=" * 60)
    logger.info("Metrics on Real Datasets")
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Methods: {methods}")
    logger.info(f"  Trials: {n_trials}")
    logger.info("=" * 60)

    t0 = time.time()

    all_results = {}
    for ds in datasets:
        logger.info(f"\n=== {ds} ===")
        try:
            ds_results = run_experiment(
                dataset_name=ds,
                method_names=methods,
                n_trials=n_trials,
                alpha=ALPHA,
                base_seed=SEED,
            )
            all_results[ds] = ds_results
        except Exception as e:
            logger.error(f"Dataset {ds} failed: {e}")
            import traceback
            traceback.print_exc()

    # Conditional coverage for EPICSCORE
    logger.info("\n=== Conditional Coverage (EPICSCORE only) ===")
    cond_cov_results = {}
    for ds in datasets:
        try:
            X, y = load_dataset(ds)
            X_tr, y_tr, X_c, y_c, X_te, y_te = split_data(X, y, seed=SEED)
            model = EnsembleModel(n_models=10, seed=SEED)
            ecp = EpistemicConformalPredictor(model=model, alpha=ALPHA, score="epistemic")
            ecp.fit(X_tr, y_tr)
            ecp.calibrate(X_c, y_c)
            res = ecp.predict(X_te)
            cc = conditional_coverage(y_te, res.lower, res.upper, res.epistemic_u, n_bins=5)
            cond_cov_results[ds] = cc
            logger.info(f"  {ds}: {cc}")
        except Exception as e:
            logger.warning(f"  {ds} conditional coverage failed: {e}")

    # Save everything
    save_results(all_results, "metrics_real_data_full")
    save_results(cond_cov_results, "conditional_coverage_epicscore")

    # Output JSON for further analysis
    json_out = RESULTS_DIR / "metrics_real_data_summary.json"
    summary = {}
    for ds, methods_dict in all_results.items():
        summary[ds] = {}
        for method, metrics in methods_dict.items():
            summary[ds][method] = {}
            for metric, arr in metrics.items():
                valid = arr[~np.isnan(arr)]
                summary[ds][method][metric] = {
                    "mean": float(np.mean(valid)) if len(valid) > 0 else None,
                    "std": float(np.std(valid)) if len(valid) > 0 else None,
                    "n": int(len(valid)),
                }
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved JSON summary: {json_out}")

    _print_latex_table(summary)
    logger.info(f"\nTotal time: {time.time()-t0:.1f}s")
    return all_results


def _print_latex_table(summary):
    """Print a LaTeX-ready results table."""
    logger.info("\n=== LaTeX Table ===")
    datasets = sorted(summary.keys())
    methods = sorted(set(m for ds in summary.values() for m in ds.keys()))

    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Coverage and AISL on real datasets (mean$\\pm$std over 100 trials)}")
    col_spec = "l" + "cc" * len(datasets)
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")

    header = "Method"
    for ds in datasets:
        header += f" & \\multicolumn{{2}}{{c}}{{{ds}}}"
    header += " \\\\"
    print(header)

    sub_header = ""
    for ds in datasets:
        sub_header += " & Cov. & AISL"
    sub_header += " \\\\"
    print(sub_header)
    print("\\midrule")

    for method in methods:
        row = method
        for ds in datasets:
            stats = summary.get(ds, {}).get(method, {})
            cov = stats.get("coverage", {})
            aisl = stats.get("aisl", {})
            cov_str = f"{cov.get('mean', 0):.3f}" if cov.get("mean") is not None else "—"
            aisl_str = f"{aisl.get('mean', 0):.3f}" if aisl.get("mean") is not None else "—"
            row += f" & {cov_str} & {aisl_str}"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    args = parser.parse_args()

    run_real_data_metrics(
        datasets=args.datasets,
        methods=args.methods,
        n_trials=args.trials,
    )