"""
Evaluation metrics for Code-SSG.

Includes:
- pass@k: Standard code generation metric
- RSR: Relative Success Rate (from EG-CFG)
- SSG metrics: Statement-level validation metrics
- Conformal prediction metrics: Coverage, set size, abstention
"""

import numpy as np
from typing import List, Optional, Dict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute pass@k metric.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k value

    Returns:
        Probability that at least one of k samples is correct.
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def relative_success_rate(accuracy: float, baseline_accuracy: float) -> float:
    """
    Compute Relative Success Rate (RSR) from EG-CFG.

    RSR = (accuracy - baseline) / (1 - baseline)
    Measures accuracy gain normalized to full success.
    """
    if baseline_accuracy >= 1.0:
        return 0.0
    return (accuracy - baseline_accuracy) / (1.0 - baseline_accuracy)


def ssg_pass_rate(validation_results: List[Dict]) -> float:
    """Compute SSG statement-level pass rate."""
    if not validation_results:
        return 0.0
    passed = sum(1 for r in validation_results if r.get("valid", False))
    return passed / len(validation_results)


def ssg_avg_confidence(validation_results: List[Dict]) -> float:
    """Compute average confidence across SSG validations."""
    confidences = [r.get("confidence", 0.0) for r in validation_results]
    return np.mean(confidences) if confidences else 0.0


def conformal_coverage(
    prediction_sets: List[Dict],
    ground_truth: List[bool],
) -> float:
    """Compute empirical coverage of conformal prediction sets."""
    if not prediction_sets:
        return 0.0
    covered = sum(1 for ps, gt in zip(prediction_sets, ground_truth) if gt)
    return covered / len(prediction_sets)


def conformal_set_size(prediction_sets: List[Dict]) -> Dict[str, float]:
    """Compute statistics on conformal prediction set sizes."""
    sizes = [len(ps.get("samples", [])) for ps in prediction_sets]
    return {
        "mean": float(np.mean(sizes)),
        "std": float(np.std(sizes)),
        "median": float(np.median(sizes)),
        "min": float(np.min(sizes)),
        "max": float(np.max(sizes)),
    }


def compute_all_metrics(
    results: List[Dict],
    baseline_accuracy: float = 0.0,
    n_samples: int = 1,
) -> Dict[str, float]:
    """Compute all metrics for a set of evaluation results."""
    passed = sum(1 for r in results if r.get("passed", False))
    total = len(results)
    accuracy = passed / total if total > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "pass@1": pass_at_k(n_samples, passed, 1),
        "rsr": relative_success_rate(accuracy, baseline_accuracy),
        "total_tasks": total,
        "passed_tasks": passed,
        "failed_tasks": total - passed,
    }

    # SSG metrics if available
    ssg_results = [r.get("ssg_report") for r in results if r.get("ssg_report")]
    if ssg_results:
        metrics["ssg_pass_rate"] = np.mean([
            r.get("pass_rate", 0) for r in ssg_results
        ])
        metrics["ssg_avg_confidence"] = np.mean([
            r.get("avg_confidence", 0) for r in ssg_results
        ])

    return metrics


def multi_trial_statistics(trial_metrics: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and std across multiple trials.

    Returns dict of metric_name → {mean, std, min, max}.
    """
    if not trial_metrics:
        return {}

    all_keys = set()
    for tm in trial_metrics:
        all_keys.update(tm.keys())

    stats = {}
    for key in all_keys:
        values = [tm.get(key, 0) for tm in trial_metrics if isinstance(tm.get(key, 0), (int, float))]
        if values:
            stats[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    return stats
