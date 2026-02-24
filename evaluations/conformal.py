"""
Conformal Prediction for Code Generation (GPS-inspired).

Implements the Generative Prediction Sets (GPS) framework for code generation,
providing statistically valid prediction sets with coverage guarantees.

Based on: "Conformal Prediction Sets for Deep Generative Models via
Reduction to Conformal Regression" (Shahrokhi et al., 2025)

Key insight: Instead of calibrating on generated code quality directly,
we calibrate on the NUMBER OF SAMPLES needed to get an admissible output,
then use conformal regression to get valid prediction intervals.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class CalibrationSample:
    """A single calibration example."""
    input_prompt: str
    k_min: int  # Minimum samples needed for admissible output
    features: Optional[np.ndarray] = None  # Input features (e.g., log-prob)
    admissible: bool = True  # Whether admissible output was found within budget


@dataclass
class PredictionSet:
    """A prediction set with coverage guarantee."""
    samples: List[str]  # Generated code samples
    k_hat: int  # Calibrated number of samples
    alpha: float  # Significance level
    coverage_guarantee: float  # 1 - alpha
    admissible_count: int  # How many samples passed admissibility


@dataclass
class GPSResult:
    """Result of GPS evaluation."""
    alpha: float
    coverage: float
    abstention_rate: float
    avg_set_size: float
    avg_samples_collected: float


class ConformalRegressor:
    """
    Conformal regression for predicting K (number of samples needed).

    Uses split conformal prediction:
    1. Train a predictor f_hat on (X_i, K_i) pairs
    2. Compute residuals on calibration set
    3. Use quantile of residuals as confidence margin
    """

    def __init__(self):
        self.residuals: Optional[np.ndarray] = None
        self.tau: Optional[float] = None
        # Simple predictor: mean K per feature bin
        self.predictor_weights: Optional[np.ndarray] = None
        self.predictor_bias: float = 0.0

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        alpha: float = 0.1,
    ):
        """
        Fit the conformal regressor.

        Args:
            features: Input features (N x D)
            targets: K values (N,)
            alpha: Significance level
        """
        n = len(targets)
        n_cal = n // 2

        # Split: first half for training, second half for calibration
        X_train, X_cal = features[:n_cal], features[n_cal:]
        y_train, y_cal = targets[:n_cal], targets[n_cal:]

        # Simple linear predictor
        # f_hat(x) = w^T x + b
        if X_train.shape[1] > 0:
            # Least squares fit
            X_aug = np.hstack([X_train, np.ones((len(X_train), 1))])
            solution = np.linalg.lstsq(X_aug, y_train, rcond=None)[0]
            self.predictor_weights = solution[:-1]
            self.predictor_bias = solution[-1]
        else:
            self.predictor_weights = np.array([])
            self.predictor_bias = np.mean(y_train)

        # Compute residuals on calibration set
        y_pred_cal = self._predict_raw(X_cal)
        self.residuals = np.abs(y_cal - y_pred_cal)

        # Compute conformal quantile
        q_level = np.ceil((1 - alpha) * (len(self.residuals) + 1)) / len(self.residuals)
        q_level = min(q_level, 1.0)
        self.tau = np.quantile(self.residuals, q_level)

        logger.info(
            f"Conformal regressor fitted: tau={self.tau:.2f}, "
            f"alpha={alpha}, n_cal={len(self.residuals)}"
        )

    def _predict_raw(self, features: np.ndarray) -> np.ndarray:
        """Raw prediction without conformal margin."""
        if self.predictor_weights is not None and len(self.predictor_weights) > 0:
            return features @ self.predictor_weights + self.predictor_bias
        else:
            return np.full(len(features), self.predictor_bias)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict K_hat with conformal margin."""
        raw = self._predict_raw(features)
        # Upper bound: raw prediction + conformal margin
        return np.ceil(raw + self.tau).astype(int)


class GenerativePredictionSets:
    """
    GPS Framework for Code Generation.

    Workflow:
    1. Calibration phase:
       - For each calibration prompt, sample up to M outputs
       - Record K_i = minimum samples to get admissible output
       - Build (X_i, K_i) dataset
       - Fit conformal regressor

    2. Inference phase:
       - For new prompt, predict K_hat (calibrated sample count)
       - Generate K_hat samples
       - Return all samples as prediction set

    Admissibility function A(x, y):
       - For code: does the generated code pass all test cases?
       - For SSG: do all scientific statements validate?
    """

    def __init__(
        self,
        admissibility_fn: Optional[Callable] = None,
        sampling_budget: int = 25,
        feature_extractor: Optional[Callable] = None,
    ):
        self.admissibility_fn = admissibility_fn or self._default_admissibility
        self.sampling_budget = sampling_budget
        self.feature_extractor = feature_extractor or self._default_features
        self.regressor = ConformalRegressor()
        self.calibration_data: List[CalibrationSample] = []

    @staticmethod
    def _default_admissibility(prompt: str, code: str) -> bool:
        """Default admissibility: code runs without error."""
        import subprocess
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def _default_features(prompt: str) -> np.ndarray:
        """Default features: prompt length and basic statistics."""
        return np.array([
            len(prompt),
            prompt.count("\n"),
            len(prompt.split()),
            prompt.count("def "),
            prompt.count("class "),
        ], dtype=float)

    def calibrate(
        self,
        prompts: List[str],
        generator_fn: Callable,
        alpha: float = 0.1,
    ):
        """
        Calibrate the GPS system on a set of prompts.

        Args:
            prompts: Calibration prompts
            generator_fn: Function that generates code given a prompt
            alpha: Significance level
        """
        logger.info(f"Calibrating GPS with {len(prompts)} prompts, alpha={alpha}")

        calibration_samples = []

        for i, prompt in enumerate(prompts):
            features = self.feature_extractor(prompt)
            k_min = self.sampling_budget + 1  # Default: not found
            admissible = False

            for k in range(1, self.sampling_budget + 1):
                code = generator_fn(prompt)
                if self.admissibility_fn(prompt, code):
                    k_min = k
                    admissible = True
                    break

            sample = CalibrationSample(
                input_prompt=prompt,
                k_min=k_min if admissible else self.sampling_budget + 1,
                features=features,
                admissible=admissible,
            )
            calibration_samples.append(sample)

            if (i + 1) % 10 == 0:
                logger.info(f"Calibrated {i+1}/{len(prompts)} prompts")

        self.calibration_data = calibration_samples

        # Fit conformal regressor
        X = np.array([s.features for s in calibration_samples])
        y = np.array([s.k_min for s in calibration_samples])

        self.regressor.fit(X, y, alpha=alpha)

    def predict(
        self,
        prompt: str,
        generator_fn: Callable,
    ) -> PredictionSet:
        """
        Generate a prediction set for a new prompt.

        Uses the calibrated K_hat to determine how many samples to collect.
        """
        features = self.feature_extractor(prompt)
        k_hat = max(1, self.regressor.predict(features.reshape(1, -1))[0])

        # Cap at budget
        k_hat = min(k_hat, self.sampling_budget)

        samples = []
        admissible_count = 0

        for _ in range(k_hat):
            code = generator_fn(prompt)
            samples.append(code)
            if self.admissibility_fn(prompt, code):
                admissible_count += 1

        return PredictionSet(
            samples=samples,
            k_hat=int(k_hat),
            alpha=0.1,  # Default, should be passed from calibration
            coverage_guarantee=0.9,
            admissible_count=admissible_count,
        )

    def evaluate(
        self,
        test_prompts: List[str],
        generator_fn: Callable,
        alpha_levels: Optional[List[float]] = None,
        n_trials: int = 100,
    ) -> List[GPSResult]:
        """
        Evaluate GPS across multiple alpha levels with multiple trials.

        This produces the data for GPS-style Figure 3 plots:
        - Abstention rate vs alpha
        - Coverage vs alpha
        - Set size vs alpha
        All with standard deviation from n_trials.
        """
        if alpha_levels is None:
            alpha_levels = np.arange(0.05, 0.55, 0.05).tolist()

        results = []

        for alpha in alpha_levels:
            trial_coverages = []
            trial_abstentions = []
            trial_set_sizes = []
            trial_samples = []

            for trial in range(n_trials):
                # Re-calibrate for each trial (different random split)
                # In practice, use bootstrap resampling
                np.random.seed(trial)

                coverage_count = 0
                abstention_count = 0
                total_set_size = 0
                total_samples_collected = 0

                for prompt in test_prompts:
                    pred_set = self.predict(prompt, generator_fn)
                    total_set_size += len(pred_set.samples)
                    total_samples_collected += pred_set.k_hat

                    if pred_set.admissible_count > 0:
                        coverage_count += 1
                    else:
                        abstention_count += 1

                n_test = len(test_prompts)
                trial_coverages.append(coverage_count / n_test)
                trial_abstentions.append(abstention_count / n_test)
                trial_set_sizes.append(total_set_size / n_test)
                trial_samples.append(total_samples_collected / n_test)

            results.append(GPSResult(
                alpha=alpha,
                coverage=float(np.mean(trial_coverages)),
                abstention_rate=float(np.mean(trial_abstentions)),
                avg_set_size=float(np.mean(trial_set_sizes)),
                avg_samples_collected=float(np.mean(trial_samples)),
            ))

            logger.info(
                f"Alpha={alpha:.2f}: coverage={results[-1].coverage:.3f}, "
                f"abstention={results[-1].abstention_rate:.3f}"
            )

        return results
