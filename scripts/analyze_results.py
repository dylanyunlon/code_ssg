"""
Experiment visualization and analysis.

Generates publication-quality figures inspired by:
1. GPS Figure 3: Abstention rate, coverage, set sizes across alpha levels
   with shaded standard deviation regions across 100 trials
2. EG-CFG benchmark comparison tables
3. Seed 2.0 style comprehensive evaluation charts

All plots include shaded confidence/std regions around curves.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)

# Style configuration for publication-quality figures
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Color palette
COLORS = {
    "gps_hs": "#2196F3",     # Blue
    "gps_l": "#4CAF50",      # Green
    "clm": "#FF9800",        # Orange
    "baseline": "#9E9E9E",   # Gray
    "ssg_hybrid": "#E91E63",  # Pink
    "ssg_exec": "#9C27B0",   # Purple
    "ssg_llm": "#00BCD4",    # Cyan
}


@dataclass
class ExperimentData:
    """Data for a single experiment configuration."""
    method_name: str
    alpha_levels: np.ndarray
    # Each metric: shape (n_alphas, n_trials)
    abstention_rates: np.ndarray
    coverages: np.ndarray
    set_sizes: np.ndarray
    samples_collected: np.ndarray


def plot_gps_figure3(
    experiments: List[ExperimentData],
    benchmark_name: str = "MBPP",
    output_path: str = "figures/gps_comparison.png",
    figsize: Tuple[int, int] = (18, 5),
):
    """
    Reproduce GPS Figure 3 style plot.

    Creates a 1x3 subplot showing:
    1. Abstention Rate vs Alpha
    2. Non-Abstention Coverage vs Alpha
    3. Set Size vs Alpha

    Each curve has a shaded region showing ±1 std across trials.
    The shaded region wraps around the curve (key visual feature).
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    metrics = [
        ("Abstention Rate", "abstention_rates", axes[0]),
        ("Non-Abstention Coverage", "coverages", axes[1]),
        ("Average Set Size", "set_sizes", axes[2]),
    ]

    for title, attr, ax in metrics:
        for exp in experiments:
            data = getattr(exp, attr)  # shape: (n_alphas, n_trials)
            mean = data.mean(axis=1)
            std = data.std(axis=1)

            color = COLORS.get(
                exp.method_name.lower().replace(" ", "_").replace("-", "_"),
                "#333333"
            )

            # Plot mean curve
            ax.plot(
                exp.alpha_levels, mean,
                label=exp.method_name,
                color=color,
                linewidth=2,
            )

            # Shaded std region (the key visual feature from GPS Figure 3)
            ax.fill_between(
                exp.alpha_levels,
                mean - std,
                mean + std,
                alpha=0.2,
                color=color,
            )

        ax.set_xlabel("α (Significance Level)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} — {benchmark_name}")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(exp.alpha_levels[0], exp.alpha_levels[-1])

    # Add base abstention rate line (dashed red)
    if experiments:
        base_rate = 0.15  # Example base model abstention rate
        axes[0].axhline(
            y=base_rate, color="red", linestyle="--", alpha=0.7,
            label=f"Base abstention ({base_rate:.0%})"
        )
        axes[0].legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved GPS Figure 3 plot to {output_path}")


def plot_benchmark_comparison(
    results: Dict[str, Dict[str, List[float]]],
    output_path: str = "figures/benchmark_comparison.png",
    figsize: Tuple[int, int] = (14, 8),
):
    """
    Plot benchmark comparison across methods with error bars.

    Results format:
    {
        "method_name": {
            "MBPP": [accuracy_trial_1, accuracy_trial_2, ...],
            "HumanEval": [...],
            ...
        }
    }
    """
    methods = list(results.keys())
    benchmarks = list(next(iter(results.values())).keys())
    n_methods = len(methods)
    n_benchmarks = len(benchmarks)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_benchmarks)
    width = 0.8 / n_methods

    for i, method in enumerate(methods):
        means = []
        stds = []
        for bench in benchmarks:
            trials = results[method].get(bench, [0])
            means.append(np.mean(trials))
            stds.append(np.std(trials))

        color = list(COLORS.values())[i % len(COLORS)]
        bars = ax.bar(
            x + i * width - 0.4 + width / 2,
            means,
            width,
            yerr=stds,
            label=method,
            color=color,
            alpha=0.85,
            capsize=3,
        )

    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Benchmark Comparison Across Methods")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=15, ha="right")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved benchmark comparison to {output_path}")


def plot_ssg_validation_curves(
    results: Dict[str, Dict[str, np.ndarray]],
    output_path: str = "figures/ssg_validation.png",
    figsize: Tuple[int, int] = (16, 10),
):
    """
    Plot SSG validation curves across benchmarks.

    For each benchmark (one row), shows:
    1. Validation pass rate over code lines
    2. Confidence distribution
    3. Method comparison with std shading

    This is the Seed 2.0 / GPS-inspired comprehensive eval figure.
    All plots have shaded std regions around the curves.
    """
    benchmarks = list(results.keys())
    n_benchmarks = len(benchmarks)

    fig, axes = plt.subplots(n_benchmarks, 3, figsize=figsize)
    if n_benchmarks == 1:
        axes = axes.reshape(1, -1)

    for row, benchmark in enumerate(benchmarks):
        data = results[benchmark]

        # Column 1: Pass rate curve with std shading
        ax1 = axes[row, 0]
        for method_name, method_data in data.items():
            # method_data shape: (n_tasks, n_trials) or similar
            if method_data.ndim == 2:
                mean = method_data.mean(axis=1)
                std = method_data.std(axis=1)
                x = np.arange(len(mean))
            else:
                mean = method_data
                std = np.zeros_like(mean)
                x = np.arange(len(mean))

            color = COLORS.get(
                method_name.lower().replace(" ", "_").replace("-", "_"),
                "#333333"
            )
            ax1.plot(x, mean, label=method_name, color=color, linewidth=1.5)
            ax1.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

        ax1.set_title(f"{benchmark} — Pass Rate")
        ax1.set_xlabel("Task Index")
        ax1.set_ylabel("Cumulative Pass Rate")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Column 2: Confidence distribution
        ax2 = axes[row, 1]
        for method_name, method_data in data.items():
            flat = method_data.flatten()
            color = COLORS.get(
                method_name.lower().replace(" ", "_").replace("-", "_"),
                "#333333"
            )
            ax2.hist(flat, bins=30, alpha=0.5, label=method_name, color=color)

        ax2.set_title(f"{benchmark} — Confidence Distribution")
        ax2.set_xlabel("Confidence Score")
        ax2.set_ylabel("Frequency")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Column 3: Accuracy comparison with std
        ax3 = axes[row, 2]
        method_names = list(data.keys())
        accuracies = [data[m].mean() for m in method_names]
        stds = [data[m].std() for m in method_names]
        colors = [
            COLORS.get(m.lower().replace(" ", "_").replace("-", "_"), "#333333")
            for m in method_names
        ]

        bars = ax3.bar(
            range(len(method_names)), accuracies,
            yerr=stds, color=colors, alpha=0.85, capsize=5
        )
        ax3.set_title(f"{benchmark} — Method Comparison")
        ax3.set_xticks(range(len(method_names)))
        ax3.set_xticklabels(method_names, rotation=15, fontsize=8)
        ax3.set_ylabel("Accuracy")
        ax3.grid(True, axis="y", alpha=0.3)

    plt.suptitle("SSG Validation Analysis Across Benchmarks", fontsize=16, y=1.01)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved SSG validation curves to {output_path}")


def generate_demo_figures(output_dir: str = "figures"):
    """Generate demo figures with synthetic data for visualization testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # ===== Demo 1: GPS Figure 3 =====
    alpha_levels = np.arange(0.05, 0.55, 0.05)
    n_trials = 100
    n_alphas = len(alpha_levels)

    def make_experiment(name, abs_base, cov_base, size_base, noise=0.05):
        abstention = np.clip(
            abs_base - alpha_levels[:, None] * 0.8 + np.random.randn(n_alphas, n_trials) * noise,
            0, 1,
        )
        coverage = np.clip(
            cov_base + alpha_levels[:, None] * 0.3 + np.random.randn(n_alphas, n_trials) * noise,
            0, 1,
        )
        sizes = np.clip(
            size_base - alpha_levels[:, None] * 5 + np.random.randn(n_alphas, n_trials) * noise * 10,
            1, 25,
        )
        samples = np.clip(
            size_base * 1.5 - alpha_levels[:, None] * 8 + np.random.randn(n_alphas, n_trials) * noise * 10,
            1, 25,
        )
        return ExperimentData(
            method_name=name,
            alpha_levels=alpha_levels,
            abstention_rates=abstention,
            coverages=coverage,
            set_sizes=sizes,
            samples_collected=samples,
        )

    experiments = [
        make_experiment("GPS HS", 0.3, 0.7, 12, 0.04),
        make_experiment("GPS L", 0.5, 0.6, 10, 0.06),
        make_experiment("CLM", 0.8, 0.5, 15, 0.08),
    ]

    plot_gps_figure3(
        experiments,
        benchmark_name="MBPP (100 trials)",
        output_path=str(output_dir / "gps_figure3_demo.png"),
    )

    # ===== Demo 2: Benchmark comparison =====
    benchmark_results = {
        "SSG-Hybrid": {
            "MBPP": np.random.normal(92, 2, 100).tolist(),
            "MBPP-ET": np.random.normal(70, 3, 100).tolist(),
            "HumanEval": np.random.normal(95, 2, 100).tolist(),
            "HumanEval-ET": np.random.normal(85, 3, 100).tolist(),
            "DS-1000": np.random.normal(65, 4, 100).tolist(),
            "CodeContests": np.random.normal(55, 5, 100).tolist(),
        },
        "Baseline": {
            "MBPP": np.random.normal(80, 3, 100).tolist(),
            "MBPP-ET": np.random.normal(60, 4, 100).tolist(),
            "HumanEval": np.random.normal(82, 3, 100).tolist(),
            "HumanEval-ET": np.random.normal(75, 4, 100).tolist(),
            "DS-1000": np.random.normal(40, 5, 100).tolist(),
            "CodeContests": np.random.normal(40, 6, 100).tolist(),
        },
        "SSG-Exec": {
            "MBPP": np.random.normal(88, 2, 100).tolist(),
            "MBPP-ET": np.random.normal(66, 3, 100).tolist(),
            "HumanEval": np.random.normal(90, 2, 100).tolist(),
            "HumanEval-ET": np.random.normal(80, 3, 100).tolist(),
            "DS-1000": np.random.normal(58, 4, 100).tolist(),
            "CodeContests": np.random.normal(48, 5, 100).tolist(),
        },
    }

    plot_benchmark_comparison(
        benchmark_results,
        output_path=str(output_dir / "benchmark_comparison_demo.png"),
    )

    # ===== Demo 3: SSG Validation Curves =====
    ssg_results = {}
    for bench_name in ["MBPP", "HumanEval", "DS-1000", "CodeContests"]:
        n_tasks = 50
        ssg_results[bench_name] = {
            "SSG-Hybrid": np.random.beta(8, 2, (n_tasks, 100)),
            "SSG-Exec": np.random.beta(6, 3, (n_tasks, 100)),
            "SSG-LLM": np.random.beta(7, 2.5, (n_tasks, 100)),
        }

    plot_ssg_validation_curves(
        ssg_results,
        output_path=str(output_dir / "ssg_validation_demo.png"),
    )

    logger.info(f"Generated all demo figures in {output_dir}")
    return [
        str(output_dir / "gps_figure3_demo.png"),
        str(output_dir / "benchmark_comparison_demo.png"),
        str(output_dir / "ssg_validation_demo.png"),
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_demo_figures()
