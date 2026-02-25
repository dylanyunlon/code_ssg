"""
Epistemic Conformal Prediction (EPICSCORE) — Core Predictor
==============================================================
Main class: EpistemicConformalPredictor

Pipeline:
  1. fit(X_train, y_train) — train the epistemic model
  2. calibrate(X_cal, y_cal) — compute conformal quantile on calibration set
  3. predict(X_test) — produce prediction intervals with coverage guarantees

The key difference from standard split conformal:
  Standard: interval = mu ± q  (uniform width)
  EPICSCORE: interval = mu ± q * (sigma + lambda * epistemic_u)  (adaptive width)

  By normalizing nonconformity scores by total uncertainty, regions with high
  epistemic uncertainty automatically get wider intervals, while well-covered
  regions get tighter intervals — all while maintaining marginal coverage.

References:
  - Lei et al. (2018) "Distribution-Free Predictive Inference for Regression"
  - Romano et al. (2019) "Conformalized Quantile Regression"
  - Barber et al. (2021) "Predictive Inference with the Jackknife+"
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .epistemic_models import EpistemicModel, ModelPredictions
from .scores import (
    ScoreFunction,
    SCORE_RESIDUAL,
    SCORE_NORMALIZED,
    SCORE_EPISTEMIC,
    SCORE_QUANTILE,
    get_score_function,
)
from .utils import coverage_rate, average_interval_length

logger = logging.getLogger(__name__)


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class CalibrationResult:
    """Stores calibration output."""
    quantile: float                # conformal quantile threshold
    n_calibration: int             # number of calibration points used
    alpha: float                   # target miscoverage rate
    empirical_coverage: float      # coverage on calibration set (should be ≈ 1-alpha)
    scores: np.ndarray             # all nonconformity scores on cal set
    score_function_name: str       # which score function was used


@dataclass
class ConformalResult:
    """Stores prediction results on test data."""
    mu: np.ndarray                 # point predictions (n,)
    lower: np.ndarray              # lower bounds (n,)
    upper: np.ndarray              # upper bounds (n,)
    sigma: np.ndarray              # aleatoric uncertainty (n,)
    epistemic_u: np.ndarray        # epistemic uncertainty (n,)
    widths: np.ndarray             # interval widths (n,)
    alpha: float                   # target miscoverage rate
    quantile: float                # conformal quantile used

    @property
    def mean_width(self) -> float:
        return float(np.mean(self.widths))

    def coverage(self, y_true: np.ndarray) -> float:
        return coverage_rate(y_true, self.lower, self.upper)


# =============================================================================
# Main Predictor
# =============================================================================

class EpistemicConformalPredictor:
    """
    EPICSCORE: Conformal prediction with epistemic uncertainty.

    Usage:
        model = EnsembleModel(base_model_class=RandomForestRegressor, n_models=10)
        ecp = EpistemicConformalPredictor(model=model, alpha=0.1)
        ecp.fit(X_train, y_train)
        ecp.calibrate(X_cal, y_cal)
        result = ecp.predict(X_test)
        print(f"Coverage: {result.coverage(y_test):.3f}")
        print(f"Mean width: {result.mean_width:.3f}")

    Args:
        model: an EpistemicModel instance
        alpha: target miscoverage rate (default 0.1 for 90% coverage)
        score: score function name or ScoreFunction instance
        lam: weighting factor for epistemic uncertainty in the score
        symmetry: if True, intervals are symmetric around mu
    """

    def __init__(
        self,
        model: EpistemicModel,
        alpha: float = 0.1,
        score: str = "epistemic",
        lam: float = 1.0,
        symmetry: bool = True,
    ):
        self.model = model
        self.alpha = alpha
        self.lam = lam
        self.symmetry = symmetry

        if isinstance(score, str):
            self.score_fn = get_score_function(score)
        elif isinstance(score, ScoreFunction):
            self.score_fn = score
        else:
            raise ValueError(f"Invalid score: {score}")

        self._calibration: Optional[CalibrationResult] = None
        self._fitted = False

    # ---- Public API ----

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EpistemicConformalPredictor':
        """
        Step 1: Train the epistemic model on training data.
        """
        logger.info(f"Fitting epistemic model on {len(X)} samples...")
        self.model.fit(X, y)
        self._fitted = True
        return self

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> CalibrationResult:
        """
        Step 2: Calibrate conformal quantile on held-out calibration data.

        Computes nonconformity scores on calibration set, then finds the
        (1-alpha)(1+1/n)-quantile as the conformal threshold.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .calibrate()")

        n = len(X_cal)
        logger.info(f"Calibrating on {n} samples (alpha={self.alpha})...")

        # Get model predictions with uncertainty
        preds = self.model.predict(X_cal)
        pred_dict = preds.to_dict()

        # Compute nonconformity scores
        scores = self.score_fn(y_cal, pred_dict)

        # Conformal quantile: ceil((n+1)(1-alpha)) / n
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        quantile = float(np.quantile(scores, q_level))

        # Check calibration coverage
        if self.score_fn.name == "epistemic":
            # For epistemic score: interval = mu ± quantile * (sigma + lam * eu)
            total_u = preds.sigma + self.lam * preds.epistemic_u
            lower = preds.mu - quantile * total_u
            upper = preds.mu + quantile * total_u
        elif self.score_fn.name == "normalized":
            lower = preds.mu - quantile * preds.sigma
            upper = preds.mu + quantile * preds.sigma
        elif self.score_fn.name == "quantile":
            q_lo = pred_dict.get('q_lo', preds.mu - preds.sigma)
            q_hi = pred_dict.get('q_hi', preds.mu + preds.sigma)
            lower = q_lo - quantile
            upper = q_hi + quantile
        else:  # residual
            lower = preds.mu - quantile
            upper = preds.mu + quantile

        cal_coverage = coverage_rate(y_cal, lower, upper)

        self._calibration = CalibrationResult(
            quantile=quantile,
            n_calibration=n,
            alpha=self.alpha,
            empirical_coverage=cal_coverage,
            scores=scores,
            score_function_name=self.score_fn.name,
        )

        logger.info(
            f"Calibration complete: quantile={quantile:.4f}, "
            f"cal_coverage={cal_coverage:.4f} (target={1-self.alpha:.2f})"
        )

        return self._calibration

    def predict(self, X_test: np.ndarray) -> ConformalResult:
        """
        Step 3: Produce prediction intervals on test data.

        Uses the calibrated quantile to construct intervals that adapt
        to the local epistemic uncertainty.
        """
        if self._calibration is None:
            raise RuntimeError("Call .calibrate() before .predict()")

        preds = self.model.predict(X_test)
        q = self._calibration.quantile

        # Build intervals based on score type
        if self.score_fn.name == "epistemic":
            total_u = preds.sigma + self.lam * preds.epistemic_u
            lower = preds.mu - q * total_u
            upper = preds.mu + q * total_u
        elif self.score_fn.name == "normalized":
            lower = preds.mu - q * preds.sigma
            upper = preds.mu + q * preds.sigma
        elif self.score_fn.name == "quantile":
            pred_dict = preds.to_dict()
            q_lo = pred_dict.get('q_lo', preds.mu - preds.sigma)
            q_hi = pred_dict.get('q_hi', preds.mu + preds.sigma)
            lower = q_lo - q
            upper = q_hi + q
        else:  # residual
            lower = preds.mu - q
            upper = preds.mu + q

        widths = upper - lower

        return ConformalResult(
            mu=preds.mu,
            lower=lower,
            upper=upper,
            sigma=preds.sigma,
            epistemic_u=preds.epistemic_u,
            widths=widths,
            alpha=self.alpha,
            quantile=q,
        )

    def fit_calibrate_predict(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_cal: np.ndarray, y_cal: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[ConformalResult, CalibrationResult]:
        """
        Convenience: run the full pipeline in one call.
        """
        self.fit(X_train, y_train)
        cal_result = self.calibrate(X_cal, y_cal)
        conf_result = self.predict(X_test)
        return conf_result, cal_result

    @property
    def calibration(self) -> Optional[CalibrationResult]:
        return self._calibration


# =============================================================================
# Baseline Methods (for comparison in experiments)
# =============================================================================

class SplitConformalPredictor:
    """
    Standard split conformal prediction (no epistemic awareness).

    Uses absolute residual as nonconformity score.
    Produces uniform-width intervals: mu ± q.
    """

    def __init__(self, base_model=None, alpha: float = 0.1, seed: int = 42):
        self.base_model = base_model
        self.alpha = alpha
        self.seed = seed
        self._quantile = None

    def _get_default_model(self):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, random_state=self.seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SplitConformalPredictor':
        if self.base_model is None:
            self.base_model = self._get_default_model()
        self.base_model.fit(X, y)
        return self

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> float:
        n = len(X_cal)
        preds = self.base_model.predict(X_cal)
        scores = np.abs(y_cal - preds)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self._quantile = float(np.quantile(scores, min(q_level, 1.0)))
        return self._quantile

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = self.base_model.predict(X_test)
        lower = mu - self._quantile
        upper = mu + self._quantile
        return mu, lower, upper


class CQRPredictor:
    """
    Conformalized Quantile Regression (CQR) — Romano et al. 2019.

    Uses a model that predicts conditional quantiles (q_lo, q_hi),
    then calibrates with the quantile nonconformity score.
    """

    def __init__(self, alpha: float = 0.1, seed: int = 42):
        self.alpha = alpha
        self.seed = seed
        self._quantile = None
        self._model_lo = None
        self._model_hi = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CQRPredictor':
        from sklearn.ensemble import GradientBoostingRegressor

        lo_q = self.alpha / 2
        hi_q = 1 - self.alpha / 2

        self._model_lo = GradientBoostingRegressor(
            loss='quantile', alpha=lo_q,
            n_estimators=200, max_depth=5, random_state=self.seed,
        )
        self._model_hi = GradientBoostingRegressor(
            loss='quantile', alpha=hi_q,
            n_estimators=200, max_depth=5, random_state=self.seed,
        )
        self._model_lo.fit(X, y)
        self._model_hi.fit(X, y)
        return self

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> float:
        n = len(X_cal)
        q_lo = self._model_lo.predict(X_cal)
        q_hi = self._model_hi.predict(X_cal)
        scores = np.maximum(q_lo - y_cal, y_cal - q_hi)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self._quantile = float(np.quantile(scores, min(q_level, 1.0)))
        return self._quantile

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_lo = self._model_lo.predict(X_test)
        q_hi = self._model_hi.predict(X_test)
        lower = q_lo - self._quantile
        upper = q_hi + self._quantile
        mu = (lower + upper) / 2
        return mu, lower, upper
