#!/usr/bin/env python3
"""
Coverage by Outlier Status — Regression Datasets
===================================================
Same analysis as coverage_by_outlier_inlier.py but specifically
for regression-type datasets, with prefix "reg_result_*".

Produces:
  - reg_result_cover_outlier.pkl
  - reg_result_aisl_outlier.pkl
  - reg_result_ratio_outlier.pkl

Usage:
    python Experiments_code/coverage_by_outlier_reg.py
"""

import sys
import logging
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Experiments_code.helper import load_dataset, get_method, save_results
from Epistemic_CP import (
    EpistemicConformalPredictor, EnsembleModel,
    split_data, coverage_rate, average_interval_length,
    adaptive_interval_set_length, outlier_inlier_split,
)
from Epistemic_CP.epistemic_cp import SplitConformalPredictor, CQRPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASETS = ["bike", "homes", "star", "meps"]
METHODS = ["EPICSCORE", "SplitConformal", "CQR", "NormalizedConformal"]
N_TRIALS = 100
ALPHA = 0.1
SEED = 42


def main():
    logger.info("Coverage by Outlier — Regression Datasets")

    reg_cover_outlier = {}
    reg_cover_inlier = {}
    reg_aisl_outlier = {}
    reg_ratio_outlier = {}

    for ds in DATASETS:
        logger.info(f"\n--- {ds} ---")
        X, y = load_dataset(ds)
        reg_cover_outlier[ds] = {}
        reg_cover_inlier[ds] = {}
        reg_aisl_outlier[ds] = {}
        reg_ratio_outlier[ds] = {}

        for method_name in METHODS:
            out_covs, in_covs, out_ails, in_ails = [], [], [], []

            for trial in range(N_TRIALS):
                try:
                    X_tr, y_tr, X_c, y_c, X_te, y_te = split_data(X, y, seed=SEED + trial)

                    eu_model = EnsembleModel(n_models=5, seed=SEED + trial)
                    eu_model.fit(X_tr, y_tr)
                    eu_preds = eu_model.predict(X_te)
                    in_idx, out_idx = outlier_inlier_split(X_te, eu_preds.epistemic_u, 0.9)

                    method = get_method(method_name, alpha=ALPHA, seed=SEED + trial)
                    if isinstance(method, EpistemicConformalPredictor):
                        method.fit(X_tr, y_tr)
                        method.calibrate(X_c, y_c)
                        res = method.predict(X_te)
                        lo, hi = res.lower, res.upper
                    else:
                        method.fit(X_tr, y_tr)
                        method.calibrate(X_c, y_c)
                        _, lo, hi = method.predict(X_te)

                    if len(out_idx) > 0:
                        out_covs.append(coverage_rate(y_te[out_idx], lo[out_idx], hi[out_idx]))
                        out_ails.append(average_interval_length(lo[out_idx], hi[out_idx]))
                    if len(in_idx) > 0:
                        in_covs.append(coverage_rate(y_te[in_idx], lo[in_idx], hi[in_idx]))
                        in_ails.append(average_interval_length(lo[in_idx], hi[in_idx]))
                except Exception:
                    pass

            reg_cover_outlier[ds][method_name] = np.array(out_covs)
            reg_cover_inlier[ds][method_name] = np.array(in_covs)
            reg_aisl_outlier[ds][method_name] = np.array(out_ails) if out_ails else np.array([])

            # Ratio
            if out_ails and in_ails:
                min_len = min(len(out_ails), len(in_ails))
                oa, ia = np.array(out_ails[:min_len]), np.array(in_ails[:min_len])
                reg_ratio_outlier[ds][method_name] = oa / np.maximum(ia, 1e-10)
            else:
                reg_ratio_outlier[ds][method_name] = np.array([])

            oc = np.array(out_covs)
            ic = np.array(in_covs)
            logger.info(f"  {method_name}: outlier_cov={np.mean(oc):.3f}±{np.std(oc):.3f}, "
                        f"inlier_cov={np.mean(ic):.3f}±{np.std(ic):.3f}")

    save_results(reg_cover_outlier, "reg_result_cover_outlier")
    save_results(reg_aisl_outlier, "reg_result_aisl_outlier")
    save_results(reg_ratio_outlier, "reg_result_ratio_outlier")
    logger.info("Done")


if __name__ == "__main__":
    main()