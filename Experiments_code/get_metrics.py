#!/usr/bin/env python3
"""
Compute and display metrics from saved experiment results.

Reads .pkl files produced by benchmarking_experiments.py and other scripts,
computes summary statistics, and produces markdown tables for the paper.

Outputs:
  - Images_rebuttal/table_coverage_outlier.md
  - Images_rebuttal/table_interval_width_ratio.md
  - Console summary tables

Usage:
    python Experiments_code/get_metrics.py
    python Experiments_code/get_metrics.py --results-dir Experiments_code/results
"""

import os
import sys
import argparse
import logging
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Experiments_code.helper import load_results, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGES_DIR = PROJECT_ROOT / "Images_rebuttal"


def compute_summary(results: dict, metric_name: str) -> dict:
    """
    From {dataset: {method: np.array(n_trials,)}},
    compute {dataset: {method: {"mean": float, "std": float, "median": float}}}.
    """
    summary = {}
    for ds, methods in results.items():
        summary[ds] = {}
        for method, values in methods.items():
            arr = np.asarray(values)
            valid = arr[~np.isnan(arr)]
            if len(valid) == 0:
                summary[ds][method] = {"mean": float("nan"), "std": 0, "median": float("nan")}
            else:
                summary[ds][method] = {
                    "mean": float(np.mean(valid)),
                    "std": float(np.std(valid)),
                    "median": float(np.median(valid)),
                }
    return summary


def format_table_md(summary: dict, metric_name: str, higher_better: bool = True) -> str:
    """Generate a markdown table from summary dict."""
    if not summary:
        return f"No results for {metric_name}\n"

    datasets = sorted(summary.keys())
    methods = sorted(set(m for ds in summary.values() for m in ds.keys()))

    lines = []
    lines.append(f"### {metric_name}\n")

    # Header
    header = "| Method |"
    sep = "|--------|"
    for ds in datasets:
        header += f" {ds} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    # Rows
    for method in methods:
        row = f"| {method} |"
        for ds in datasets:
            stats = summary.get(ds, {}).get(method, {})
            mean = stats.get("mean", float("nan"))
            std = stats.get("std", 0)
            if np.isnan(mean):
                row += " — |"
            else:
                row += f" {mean:.4f}±{std:.4f} |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def compute_width_ratio_table(results_il: dict, reference_method: str = "SplitConformal") -> str:
    """
    Compute interval width ratio relative to reference method.
    ratio = mean_width(method) / mean_width(reference)
    """
    if not results_il:
        return "No interval length results\n"

    datasets = sorted(results_il.keys())
    methods = sorted(set(m for ds in results_il.values() for m in ds.keys()))

    lines = ["### Interval Width Ratio (relative to SplitConformal)\n"]
    header = "| Method |"
    sep = "|--------|"
    for ds in datasets:
        header += f" {ds} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    for method in methods:
        row = f"| {method} |"
        for ds in datasets:
            ref = results_il.get(ds, {}).get(reference_method, np.array([]))
            cur = results_il.get(ds, {}).get(method, np.array([]))
            ref_valid = ref[~np.isnan(ref)] if len(ref) > 0 else np.array([])
            cur_valid = cur[~np.isnan(cur)] if len(cur) > 0 else np.array([])

            if len(ref_valid) == 0 or len(cur_valid) == 0 or np.mean(ref_valid) < 1e-10:
                row += " — |"
            else:
                ratio = np.mean(cur_valid) / np.mean(ref_valid)
                row += f" {ratio:.3f} |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def main(results_dir: Path = None):
    results_dir = results_dir or RESULTS_DIR
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading results from: {results_dir}")

    # Load available results
    metric_files = {
        "Coverage": "reg_result_cover",
        "Interval Length": "reg_result_il",
        "AISL": "reg_result_aisl",
        "PCOR": "reg_result_pcor",
    }

    all_tables = []
    results_il = None

    for metric_name, filename in metric_files.items():
        try:
            results = load_results(filename, results_dir)
            summary = compute_summary(results, metric_name)
            table_md = format_table_md(
                summary, metric_name,
                higher_better=(metric_name in ("Coverage", "PCOR")),
            )
            all_tables.append(table_md)
            print(table_md)

            if metric_name == "Interval Length":
                results_il = results

        except FileNotFoundError:
            logger.warning(f"Result file not found: {filename}.pkl — skipping {metric_name}")
            logger.warning(f"  Run benchmarking_experiments.py first to generate results.")

    # Width ratio table
    if results_il:
        ratio_md = compute_width_ratio_table(results_il)
        all_tables.append(ratio_md)
        print(ratio_md)

        with open(IMAGES_DIR / "table_interval_width_ratio.md", "w") as f:
            f.write(ratio_md)
        logger.info(f"Saved: {IMAGES_DIR / 'table_interval_width_ratio.md'}")

    # Outlier coverage table (from outlier results if available)
    try:
        outlier_cover = load_results("reg_result_cover_outlier", results_dir)
        outlier_summary = compute_summary(outlier_cover, "Outlier Coverage")
        outlier_md = format_table_md(outlier_summary, "Coverage by Outlier Status")
        all_tables.append(outlier_md)
        print(outlier_md)

        with open(IMAGES_DIR / "table_coverage_outlier.md", "w") as f:
            f.write(outlier_md)
        logger.info(f"Saved: {IMAGES_DIR / 'table_coverage_outlier.md'}")
    except FileNotFoundError:
        logger.info("No outlier coverage results yet — run coverage_by_outlier_reg.py first")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=None, type=Path)
    args = parser.parse_args()
    main(args.results_dir)