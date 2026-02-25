"""
Epistemic Conformal Prediction (EPICSCORE)
==========================================
Conformal prediction with epistemic uncertainty estimation.

Based on the EPICSCORE framework for providing prediction intervals
that account for both aleatoric and epistemic uncertainty.

Public API:
    - EpistemicConformalPredictor: main predictor class
    - EpistemicModel / EnsembleModel / MCDropoutModel: uncertainty models
    - nonconformity_score_* : score functions (residual, quantile, etc.)
    - split_data, calibrate, predict_intervals: pipeline functions

Usage:
    from Epistemic_CP import EpistemicConformalPredictor, EnsembleModel

    model = EnsembleModel(base_model_class=RandomForestRegressor, n_models=10)
    ecp = EpistemicConformalPredictor(model=model, alpha=0.1)
    ecp.fit(X_train, y_train)
    ecp.calibrate(X_cal, y_cal)
    intervals = ecp.predict(X_test)
"""

from .epistemic_cp import (
    EpistemicConformalPredictor,
    ConformalResult,
    CalibrationResult,
)
from .epistemic_models import (
    EpistemicModel,
    EnsembleModel,
    MCDropoutModel,
    QuantileForestModel,
)
from .scores import (
    nonconformity_score_residual,
    nonconformity_score_normalized,
    nonconformity_score_quantile,
    nonconformity_score_epistemic,
    ScoreFunction,
)
from .utils import (
    split_data,
    coverage_rate,
    average_interval_length,
    interval_width_ratio,
    adaptive_interval_set_length,
    partial_correlation,
)

__version__ = "0.1.0"
__all__ = [
    # Main predictor
    "EpistemicConformalPredictor",
    "ConformalResult",
    "CalibrationResult",
    # Models
    "EpistemicModel",
    "EnsembleModel",
    "MCDropoutModel",
    "QuantileForestModel",
    # Scores
    "nonconformity_score_residual",
    "nonconformity_score_normalized",
    "nonconformity_score_quantile",
    "nonconformity_score_epistemic",
    "ScoreFunction",
    # Utilities
    "split_data",
    "coverage_rate",
    "average_interval_length",
    "interval_width_ratio",
    "adaptive_interval_set_length",
    "partial_correlation",
]
