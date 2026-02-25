#!/usr/bin/env python3
"""
EPICSCORE Benchmarking Experiments
====================================
Main experiment: compare EPICSCORE against baselines across all datasets.

Produces pickle results for:
  - reg_result_cover.pkl     (coverage per method per dataset)
  - reg_result_il.pkl        (interval length per method per dataset)
  - reg_result_aisl.pkl      (AISL per method per dataset)
  - reg_result_pcor.pkl      (partial correlation per method per dataset)

ALL results come from running the actual experiments. No hardcoded numbers.

Usage:
    python Experiments_code/benchmarking_experiments.py
    python Experiments_code/benchmarking_experiments.py --trials 50 --datasets bike homes
    CUDA_VISIBLE_DEVICES=0 python Experiments_code/benchmarking_experiments.py  # H100 single GPU

Server config: single H100, /data/jiacheng/system/cache/temp/uai2026/code_ssg/
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Experiments_code.helper import (
    run_experiment,
    save_results,
    load_dataset,
    RESULTS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Experiment Configuration
# =============================================================================

DEFAULT_DATASETS = ["bike", "homes", "star", "meps"]
DEFAULT_METHODS = [
    "EPICSCORE",
    "SplitConformal",
    "CQR",
    "NormalizedConformal",
]
DEFAULT_ALPHA = 0.1
DEFAULT_TRIALS = 100
DEFAULT_SEED = 42


def run_benchmarking(
    datasets: list = None,
    methods: list = None,
    n_trials: int = DEFAULT_TRIALS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
):
    """
    Run full benchmarking: for each dataset × method, run n_trials,
    collect coverage / AIL / AISL / PCOR, save to pickle.
    """
    datasets = datasets or DEFAULT_DATASETS
    methods = methods or DEFAULT_METHODS

    logger.info("=" * 60)
    logger.info("EPICSCORE Benchmarking Experiments")
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Methods:  {methods}")
    logger.info(f"  Trials:   {n_trials}")
    logger.info(f"  Alpha:    {alpha}")
    logger.info(f"  Seed:     {seed}")
    logger.info("=" * 60)

    t_start = time.time()

    # Collect all results: {dataset: {method: {metric: array}}}
    all_results = {}
    for ds in datasets:
        logger.info(f"\n{'='*40} Dataset: {ds} {'='*40}")
        try:
            ds_results = run_experiment(
                dataset_name=ds,
                method_names=methods,
                n_trials=n_trials,
                alpha=alpha,
                base_seed=seed,
            )
            all_results[ds] = ds_results
        except Exception as e:
            logger.error(f"Dataset {ds} failed: {e}")
            import traceback
            traceback.print_exc()

    # Save per-metric result files (matching EPICSCORE paper output format)
    metrics_to_save = {
        "reg_result_cover": "coverage",
        "reg_result_il": "ail",
        "reg_result_aisl": "aisl",
        "reg_result_pcor": "pcor",
    }

    for result_name, metric_key in metrics_to_save.items():
        metric_results = {}
        for ds, methods_dict in all_results.items():
            metric_results[ds] = {}
            for method, metrics in methods_dict.items():
                metric_results[ds][method] = metrics.get(metric_key, np.array([]))
        save_results(metric_results, result_name)
        logger.info(f"Saved {result_name}.pkl")

    # Also save the full combined results
    save_results(all_results, "benchmarking_full")

    t_total = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking complete in {t_total:.1f}s ({t_total/60:.1f}min)")
    logger.info(f"Results saved to: {RESULTS_DIR}")

    # Print summary table
    _print_summary(all_results)

    return all_results


def _print_summary(all_results: dict):
    """Print a human-readable summary table."""
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY TABLE (mean ± std)")
    logger.info("=" * 80)

    datasets = list(all_results.keys())
    if not datasets:
        return

    methods = list(all_results[datasets[0]].keys())

    # Header
    header = f"{'Method':<25}"
    for ds in datasets:
        header += f" | {ds:>12}"
    logger.info(header)
    logger.info("-" * len(header))

    # Coverage
    logger.info("--- Coverage (target: 0.90) ---")
    for method in methods:
        row = f"  {method:<23}"
        for ds in datasets:
            arr = all_results.get(ds, {}).get(method, {}).get("coverage", np.array([]))
            if len(arr) > 0:
                valid = arr[~np.isnan(arr)]
                row += f" | {np.mean(valid):.3f}±{np.std(valid):.3f}"
            else:
                row += f" | {'N/A':>12}"
        logger.info(row)

    # AIL
    logger.info("--- Average Interval Length ---")
    for method in methods:
        row = f"  {method:<23}"
        for ds in datasets:
            arr = all_results.get(ds, {}).get(method, {}).get("ail", np.array([]))
            if len(arr) > 0:
                valid = arr[~np.isnan(arr)]
                row += f" | {np.mean(valid):.2f}±{np.std(valid):.2f}"
            else:
                row += f" | {'N/A':>12}"
        logger.info(row)

    # PCOR (only for EPICSCORE)
    logger.info("--- Partial Correlation (EPICSCORE only) ---")
    for method in methods:
        if "EPIC" not in method.upper():
            continue
        row = f"  {method:<23}"
        for ds in datasets:
            arr = all_results.get(ds, {}).get(method, {}).get("pcor", np.array([]))
            if len(arr) > 0:
                valid = arr[~np.isnan(arr)]
                row += f" | {np.mean(valid):.3f}±{np.std(valid):.3f}"
            else:
                row += f" | {'N/A':>12}"
        logger.info(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EPICSCORE Benchmarking")
    parser.add_argument("--trials", "-n", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--alpha", "-a", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--seed", "-s", type=int, default=DEFAULT_SEED)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    args = parser.parse_args()

    run_benchmarking(
        datasets=args.datasets,
        methods=args.methods,
        n_trials=args.trials,
        alpha=args.alpha,
        seed=args.seed,
    )