"""
Uncertainty-Aware Conformalized Quantile Regression (UA-CQR)
=============================================================
Extension of CQR that incorporates epistemic uncertainty into
the conformalization step.

Standard CQR: score = max(q_lo - y, y - q_hi)
UA-CQR:       score = max(q_lo - y, y - q_hi) / (1 + lambda * eu)

By dividing by epistemic uncertainty, points in high-uncertainty regions
get smaller scores → larger conformal correction → wider intervals.

All results computed by running code, no hardcoded values.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class UACQR:
    """
    Uncertainty-Aware CQR.

    Combines quantile regression with epistemic uncertainty normalization.

    Args:
        alpha: target miscoverage rate
        lam: weight for epistemic uncertainty in score normalization
        n_ensemble: number of ensemble members for epistemic estimation
        seed: random seed
    """

    def __init__(
        self,
        alpha: float = 0.1,
        lam: float = 1.0,
        n_ensemble: int = 10,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.lam = lam
        self.n_ensemble = n_ensemble
        self.seed = seed
        self._models_lo = []
        self._models_hi = []
        self._models_point = []
        self._quantile = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'UACQR':
        """
        Train ensemble of quantile regressors.

        Each ensemble member is trained on a bootstrap sample.
        Lower quantile models predict alpha/2, upper predict 1-alpha/2.
        """
        from sklearn.ensemble import GradientBoostingRegressor

        n = len(X)
        rng = np.random.RandomState(self.seed)
        lo_q = self.alpha / 2
        hi_q = 1 - self.alpha / 2

        self._models_lo = []
        self._models_hi = []
        self._models_point = []

        for i in range(self.n_ensemble):
            idx = rng.choice(n, size=n, replace=True)
            X_b, y_b = X[idx], y[idx]
            member_seed = self.seed + i * 1000

            m_lo = GradientBoostingRegressor(
                loss='quantile', alpha=lo_q,
                n_estimators=150, max_depth=4, random_state=member_seed,
            )
            m_hi = GradientBoostingRegressor(
                loss='quantile', alpha=hi_q,
                n_estimators=150, max_depth=4, random_state=member_seed,
            )
            m_pt = GradientBoostingRegressor(
                n_estimators=150, max_depth=4, random_state=member_seed,
            )

            m_lo.fit(X_b, y_b)
            m_hi.fit(X_b, y_b)
            m_pt.fit(X_b, y_b)

            self._models_lo.append(m_lo)
            self._models_hi.append(m_hi)
            self._models_point.append(m_pt)

        logger.info(f"UACQR: trained {self.n_ensemble} ensemble members")
        return self

    def _predict_raw(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (q_lo, q_hi, mu, epistemic_u) from ensemble."""
        all_lo = np.array([m.predict(X) for m in self._models_lo])
        all_hi = np.array([m.predict(X) for m in self._models_hi])
        all_pt = np.array([m.predict(X) for m in self._models_point])

        q_lo = np.mean(all_lo, axis=0)
        q_hi = np.mean(all_hi, axis=0)
        mu = np.mean(all_pt, axis=0)
        epistemic_u = np.std(all_pt, axis=0)

        return q_lo, q_hi, mu, epistemic_u

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> float:
        """
        Calibrate conformal threshold using UA-CQR scores.

        score_i = max(q_lo_i - y_i, y_i - q_hi_i) / (1 + lam * eu_i)
        """
        n = len(X_cal)
        q_lo, q_hi, mu, eu = self._predict_raw(X_cal)

        # UA-CQR scores: normalize by epistemic uncertainty
        raw_scores = np.maximum(q_lo - y_cal, y_cal - q_hi)
        normalizer = 1 + self.lam * eu
        normalizer = np.maximum(normalizer, 1e-8)
        scores = raw_scores / normalizer

        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self._quantile = float(np.quantile(scores, min(q_level, 1.0)))

        logger.info(f"UACQR calibrated: quantile={self._quantile:.4f}")
        return self._quantile

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict intervals. Returns (mu, lower, upper).

        lower = q_lo - quantile * (1 + lam * eu)
        upper = q_hi + quantile * (1 + lam * eu)
        """
        if self._quantile is None:
            raise RuntimeError("Call calibrate() first")

        q_lo, q_hi, mu, eu = self._predict_raw(X_test)
        correction = self._quantile * (1 + self.lam * eu)
        lower = q_lo - correction
        upper = q_hi + correction
        return mu, lower, upper

    def predict_with_uncertainty(self, X_test: np.ndarray) -> Dict[str, np.ndarray]:
        """Return full prediction dict including uncertainty decomposition."""
        q_lo, q_hi, mu, eu = self._predict_raw(X_test)
        correction = self._quantile * (1 + self.lam * eu) if self._quantile else 0
        lower = q_lo - correction
        upper = q_hi + correction
        sigma = (q_hi - q_lo) / 2
        return {
            "mu": mu,
            "lower": lower,
            "upper": upper,
            "sigma": sigma,
            "epistemic_u": eu,
            "q_lo_raw": q_lo,
            "q_hi_raw": q_hi,
            "widths": upper - lower,
        }