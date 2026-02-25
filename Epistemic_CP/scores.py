"""
Nonconformity Score Functions for EPICSCORE
=============================================
Each score function takes predictions and true values and returns a scalar
nonconformity score per sample. Higher scores → worse fit → wider intervals.

Score functions:
  1. Residual:     |y - mu|
  2. Normalized:   |y - mu| / sigma  (sigma = aleatoric uncertainty)
  3. Quantile:     max(q_lo - y, y - q_hi)  (for quantile regression)
  4. Epistemic:    |y - mu| / (sigma + lambda * epistemic_u)
                   Normalizes by TOTAL uncertainty (aleatoric + epistemic)

The epistemic score is the key contribution of EPICSCORE: by normalizing
the residual by epistemic uncertainty, the resulting conformal intervals
automatically widen in regions of high model uncertainty.
"""

import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass


# =============================================================================
# Score Function Type
# =============================================================================

@dataclass
class ScoreFunction:
    """
    Wraps a nonconformity score callable with metadata.

    Args:
        name: human-readable name
        fn: callable(y_true, predictions_dict) → scores array
        requires: list of keys required in predictions_dict
    """
    name: str
    fn: Callable[[np.ndarray, dict], np.ndarray]
    requires: list

    def __call__(self, y_true: np.ndarray, predictions: dict) -> np.ndarray:
        for key in self.requires:
            if key not in predictions:
                raise ValueError(
                    f"Score function '{self.name}' requires '{key}' "
                    f"in predictions dict. Got keys: {list(predictions.keys())}"
                )
        return self.fn(y_true, predictions)


# =============================================================================
# Score Implementations
# =============================================================================

def nonconformity_score_residual(
    y_true: np.ndarray,
    predictions: dict,
) -> np.ndarray:
    """
    Absolute residual score: |y - mu|.

    predictions must contain:
        'mu': point predictions (n,)

    Returns:
        scores (n,) — higher = worse fit
    """
    y = np.asarray(y_true).ravel()
    mu = np.asarray(predictions['mu']).ravel()
    return np.abs(y - mu)


def nonconformity_score_normalized(
    y_true: np.ndarray,
    predictions: dict,
) -> np.ndarray:
    """
    Normalized residual score: |y - mu| / sigma.

    Normalizes by the predicted aleatoric uncertainty (standard deviation).
    This makes the score scale-invariant across different input regions,
    leading to locally adaptive prediction intervals.

    predictions must contain:
        'mu': point predictions (n,)
        'sigma': predicted std deviation (n,)  [aleatoric uncertainty]

    Returns:
        scores (n,)
    """
    y = np.asarray(y_true).ravel()
    mu = np.asarray(predictions['mu']).ravel()
    sigma = np.asarray(predictions['sigma']).ravel()

    # Clamp sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)

    return np.abs(y - mu) / sigma


def nonconformity_score_quantile(
    y_true: np.ndarray,
    predictions: dict,
) -> np.ndarray:
    """
    Quantile regression score: max(q_lo - y, y - q_hi).

    Used with models that predict conditional quantiles (e.g., CQR).
    Score is 0 when y is inside [q_lo, q_hi], positive otherwise.

    predictions must contain:
        'q_lo': lower quantile predictions (n,)
        'q_hi': upper quantile predictions (n,)

    Returns:
        scores (n,)
    """
    y = np.asarray(y_true).ravel()
    q_lo = np.asarray(predictions['q_lo']).ravel()
    q_hi = np.asarray(predictions['q_hi']).ravel()

    return np.maximum(q_lo - y, y - q_hi)


def nonconformity_score_epistemic(
    y_true: np.ndarray,
    predictions: dict,
    lam: float = 1.0,
) -> np.ndarray:
    """
    EPICSCORE: Epistemic-aware nonconformity score.

        score = |y - mu| / (sigma + lambda * epistemic_u)

    Key insight: by including epistemic uncertainty in the denominator,
    regions where the model is uncertain (few training samples, distribution
    shift) get LOWER scores, which means the conformal quantile will produce
    WIDER intervals there. This is the core contribution of EPICSCORE.

    predictions must contain:
        'mu': point predictions (n,)
        'sigma': aleatoric std deviation (n,)
        'epistemic_u': epistemic uncertainty estimate (n,)

    Args:
        lam: weighting factor for epistemic uncertainty (default 1.0)

    Returns:
        scores (n,)
    """
    y = np.asarray(y_true).ravel()
    mu = np.asarray(predictions['mu']).ravel()
    sigma = np.asarray(predictions['sigma']).ravel()
    eu = np.asarray(predictions['epistemic_u']).ravel()

    # Total uncertainty = aleatoric + weighted epistemic
    total_u = np.maximum(sigma + lam * eu, 1e-8)

    return np.abs(y - mu) / total_u


# =============================================================================
# Pre-built ScoreFunction Objects (for convenience)
# =============================================================================

SCORE_RESIDUAL = ScoreFunction(
    name="residual",
    fn=nonconformity_score_residual,
    requires=["mu"],
)

SCORE_NORMALIZED = ScoreFunction(
    name="normalized",
    fn=nonconformity_score_normalized,
    requires=["mu", "sigma"],
)

SCORE_QUANTILE = ScoreFunction(
    name="quantile",
    fn=nonconformity_score_quantile,
    requires=["q_lo", "q_hi"],
)

SCORE_EPISTEMIC = ScoreFunction(
    name="epistemic",
    fn=nonconformity_score_epistemic,
    requires=["mu", "sigma", "epistemic_u"],
)

# Registry for lookup by name
SCORE_REGISTRY = {
    "residual": SCORE_RESIDUAL,
    "normalized": SCORE_NORMALIZED,
    "quantile": SCORE_QUANTILE,
    "epistemic": SCORE_EPISTEMIC,
}


def get_score_function(name: str) -> ScoreFunction:
    """Look up a score function by name."""
    if name not in SCORE_REGISTRY:
        raise ValueError(
            f"Unknown score function '{name}'. "
            f"Available: {list(SCORE_REGISTRY.keys())}"
        )
    return SCORE_REGISTRY[name]
