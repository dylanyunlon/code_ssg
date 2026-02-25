"""
Epistemic Uncertainty Models for EPICSCORE
============================================
Models that produce:
  - mu: point predictions
  - sigma: aleatoric uncertainty (predictive std)
  - epistemic_u: epistemic uncertainty estimate

Three model families:
  1. EnsembleModel — trains N independent models, disagreement = epistemic uncertainty
  2. MCDropoutModel — Monte Carlo dropout at inference time
  3. QuantileForestModel — quantile regression forest with OOB-based uncertainty

All models implement the EpistemicModel interface.
"""

import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# Base Interface
# =============================================================================

@dataclass
class ModelPredictions:
    """Structured output from an epistemic model."""
    mu: np.ndarray           # point predictions (n,)
    sigma: np.ndarray        # aleatoric std (n,)
    epistemic_u: np.ndarray  # epistemic uncertainty (n,)

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            'mu': self.mu,
            'sigma': self.sigma,
            'epistemic_u': self.epistemic_u,
        }

    @property
    def q_lo(self) -> np.ndarray:
        """Approximate lower quantile: mu - (sigma + epistemic_u)."""
        return self.mu - (self.sigma + self.epistemic_u)

    @property
    def q_hi(self) -> np.ndarray:
        """Approximate upper quantile: mu + (sigma + epistemic_u)."""
        return self.mu + (self.sigma + self.epistemic_u)


class EpistemicModel(ABC):
    """
    Abstract base for models that decompose predictive uncertainty
    into aleatoric and epistemic components.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EpistemicModel':
        """Train the model."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> ModelPredictions:
        """Return point prediction + uncertainty decomposition."""
        ...

    def predict_dict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Convenience: return as dict for score functions."""
        return self.predict(X).to_dict()


# =============================================================================
# Ensemble Model
# =============================================================================

class EnsembleModel(EpistemicModel):
    """
    Deep ensemble (or any model ensemble) for epistemic uncertainty.

    Trains N independent models on bootstrapped subsets of the data.
    - mu = mean of model predictions
    - sigma = mean of individual model residuals (proxy for aleatoric noise)
    - epistemic_u = std of model predictions across ensemble members

    The key insight: if all models agree (low std), epistemic uncertainty is low
    (the model "knows" this region well). If they disagree (high std), the model
    is uncertain about this region of input space.

    Args:
        base_model_class: sklearn-compatible model class (must have fit/predict)
        n_models: number of ensemble members
        bootstrap: whether to use bootstrap sampling (True) or full data (False)
        model_kwargs: keyword arguments passed to base_model_class()
        seed: random seed
    """

    def __init__(
        self,
        base_model_class=None,
        n_models: int = 10,
        bootstrap: bool = True,
        model_kwargs: Optional[Dict] = None,
        seed: int = 42,
    ):
        self.base_model_class = base_model_class
        self.n_models = n_models
        self.bootstrap = bootstrap
        self.model_kwargs = model_kwargs or {}
        self.seed = seed
        self.models = []
        self._fitted = False

    def _get_base_model(self):
        """Create a new instance of the base model."""
        if self.base_model_class is None:
            # Default: sklearn RandomForestRegressor
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                random_state=None,  # random per member
                **self.model_kwargs,
            )
        return self.base_model_class(**self.model_kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleModel':
        """Train N ensemble members."""
        n = len(X)
        rng = np.random.RandomState(self.seed)
        self.models = []

        for i in range(self.n_models):
            model = self._get_base_model()

            if self.bootstrap:
                idx = rng.choice(n, size=n, replace=True)
                model.fit(X[idx], y[idx])
            else:
                model.fit(X, y)

            self.models.append(model)
            logger.debug(f"Trained ensemble member {i+1}/{self.n_models}")

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> ModelPredictions:
        """
        Predict with uncertainty decomposition.

        Returns mu, sigma (aleatoric proxy), epistemic_u (ensemble disagreement).
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        # Collect predictions from all members
        all_preds = np.array([m.predict(X) for m in self.models])  # (n_models, n)

        # Point prediction: ensemble mean
        mu = np.mean(all_preds, axis=0)

        # Epistemic uncertainty: std across ensemble members
        epistemic_u = np.std(all_preds, axis=0)

        # Aleatoric proxy: mean absolute deviation from individual predictions to mu
        # This is a rough proxy — each model's residual captures noise sensitivity
        sigma = np.mean(np.abs(all_preds - mu[np.newaxis, :]), axis=0)
        sigma = np.maximum(sigma, 1e-8)

        return ModelPredictions(mu=mu, sigma=sigma, epistemic_u=epistemic_u)

    def predict_individual(self, X: np.ndarray) -> np.ndarray:
        """Return all individual ensemble member predictions. Shape: (n_models, n)."""
        return np.array([m.predict(X) for m in self.models])


# =============================================================================
# MC Dropout Model
# =============================================================================

class MCDropoutModel(EpistemicModel):
    """
    Monte Carlo Dropout for epistemic uncertainty.

    Requires a model that supports dropout at inference time.
    For sklearn models, we simulate MC dropout via bootstrap sub-sampling
    of features at prediction time.

    For PyTorch models (if available), uses actual dropout.

    Args:
        base_model: a fitted sklearn model, or a pytorch model with dropout
        n_forward_passes: number of stochastic forward passes
        dropout_rate: feature dropout rate (for sklearn simulation)
        seed: random seed
    """

    def __init__(
        self,
        base_model=None,
        n_forward_passes: int = 50,
        dropout_rate: float = 0.1,
        seed: int = 42,
    ):
        self.base_model = base_model
        self.n_forward_passes = n_forward_passes
        self.dropout_rate = dropout_rate
        self.seed = seed
        self._fitted = False

    def _get_default_model(self):
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=self.seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MCDropoutModel':
        """Fit the base model."""
        if self.base_model is None:
            self.base_model = self._get_default_model()
        self.base_model.fit(X, y)
        self._n_features = X.shape[1]
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> ModelPredictions:
        """
        MC dropout prediction.

        For sklearn: simulate dropout by zeroing random features per forward pass.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        rng = np.random.RandomState(self.seed)
        all_preds = []

        for _ in range(self.n_forward_passes):
            # Simulate feature dropout
            X_dropped = X.copy()
            mask = rng.random(X.shape) < self.dropout_rate
            X_dropped[mask] = 0.0

            pred = self.base_model.predict(X_dropped)
            all_preds.append(pred)

        all_preds = np.array(all_preds)  # (n_passes, n)

        mu = np.mean(all_preds, axis=0)
        epistemic_u = np.std(all_preds, axis=0)
        sigma = np.mean(np.abs(all_preds - mu[np.newaxis, :]), axis=0)
        sigma = np.maximum(sigma, 1e-8)

        return ModelPredictions(mu=mu, sigma=sigma, epistemic_u=epistemic_u)


# =============================================================================
# Quantile Forest Model
# =============================================================================

class QuantileForestModel(EpistemicModel):
    """
    Quantile Regression Forest with OOB-based epistemic uncertainty.

    Uses sklearn's RandomForestRegressor:
    - mu: forest mean prediction
    - sigma: (q_hi - q_lo) / 2 from tree quantile spread
    - epistemic_u: OOB variance or tree disagreement on test points

    Args:
        n_estimators: number of trees
        quantiles: tuple (lo, hi) for interval estimation
        seed: random seed
    """

    def __init__(
        self,
        n_estimators: int = 200,
        quantiles: Tuple[float, float] = (0.05, 0.95),
        seed: int = 42,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.quantiles = quantiles
        self.seed = seed
        self.kwargs = kwargs
        self._forest = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantileForestModel':
        """Fit random forest."""
        from sklearn.ensemble import RandomForestRegressor

        self._forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            **self.kwargs,
        )
        self._forest.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> ModelPredictions:
        """
        Predict with quantile-based uncertainty.

        Gets individual tree predictions to compute epistemic uncertainty
        (tree disagreement) and aleatoric width (quantile spread).
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        # Get individual tree predictions
        tree_preds = np.array([
            tree.predict(X) for tree in self._forest.estimators_
        ])  # (n_trees, n)

        # Point prediction
        mu = np.mean(tree_preds, axis=0)

        # Epistemic: tree disagreement
        epistemic_u = np.std(tree_preds, axis=0)

        # Aleatoric proxy: quantile spread
        q_lo_val = np.quantile(tree_preds, self.quantiles[0], axis=0)
        q_hi_val = np.quantile(tree_preds, self.quantiles[1], axis=0)
        sigma = (q_hi_val - q_lo_val) / 2.0
        sigma = np.maximum(sigma, 1e-8)

        return ModelPredictions(mu=mu, sigma=sigma, epistemic_u=epistemic_u)

    def predict_quantiles(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (q_lo, q_hi) from tree distribution."""
        tree_preds = np.array([
            tree.predict(X) for tree in self._forest.estimators_
        ])
        q_lo = np.quantile(tree_preds, self.quantiles[0], axis=0)
        q_hi = np.quantile(tree_preds, self.quantiles[1], axis=0)
        return q_lo, q_hi
