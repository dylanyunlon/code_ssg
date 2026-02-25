 tree
.
├── data
│   ├── data_scripts
│   │   ├── download_data.sh
│   │   ├── download.py
│   │   ├── process.py
│   │   └── utils.py
│   └── raw
│       ├── bike
│       │   └── bike_train.csv
│       ├── homes
│       │   └── kc_house_data.csv
│       ├── meps
│       │   └── meps_19_reg.csv
│       ├── star
│       │   └── STAR.csv
│       └── WEC
│           ├── WEC_Perth_100.csv
│           ├── WEC_Perth_49.csv
│           ├── WEC_Sydney_100.csv
│           └── WEC_Sydney_49.csv
├── demo_epic_quantile.ipynb
├── demo_epic_reg.ipynb
├── EPICSCORE_env.yml
├── Epistemic_CP
│   ├── epistemic_cp.py
│   ├── epistemic_models.py
│   ├── __init__.py
│   ├── scores.py
│   └── utils.py
├── Experiments_code
│   ├── benchmarking_experiments.py
│   ├── coverage_by_outlier_inlier_other_data.py
│   ├── coverage_by_outlier_inlier.py
│   ├── coverage_by_outlier_reg.py
│   ├── difused_prior_experiment.py
│   ├── get_metrics.py
│   ├── helper.py
│   ├── hpd_split_versus_bart_epicscore.py
│   ├── hpd_split_versus_epicscore.py
│   ├── metrics_real_data.py
│   ├── metrics_reg_data.py
│   ├── nn
│   │   ├── data
│   │   │   └── processed
│   │   │       ├── airfoil.csv
│   │   │       ├── bike_0.csv
│   │   │       ├── bike.csv
│   │   │       ├── cycle.csv
│   │   │       ├── electric.csv
│   │   │       ├── protein.csv
│   │   │       ├── star.csv
│   │   │       ├── winered.csv
│   │   │       └── winewhite.csv
│   │   ├── helper.py
│   │   ├── metrics_real_data.py
│   │   └── uacqr.py
│   ├── results
│   │   ├── reg_result_aisl_outlier.pkl
│   │   ├── reg_result_aisl.pkl
│   │   ├── reg_result_cover_outlier.pkl
│   │   ├── reg_result_cover.pkl
│   │   ├── reg_result_il.pkl
│   │   ├── reg_result_pcor.pkl
│   │   ├── reg_result_ratio_outlier.pkl
│   │   ├── result_aisl_outlier.pkl
│   │   ├── result_aisl.pkl
│   │   ├── result_cover_outlier.pkl
│   │   ├── result_cover.pkl
│   │   ├── result_il.pkl
│   │   ├── result_pcor.pkl
│   │   └── result_ratio_outlier.pkl
│   └── uacqr.py
├── Figures_and_tables
│   ├── all_results.ipynb
│   ├── all_results_outliers_inliers.ipynb
│   ├── figure_1_reg_split.ipynb
│   ├── figure_3_quantile_illustration.ipynb
│   └── image_experiments_figure_2.ipynb
├── Images_rebuttal
│   ├── AISL_versus_alpha.png
│   ├── Caption_AISL_versus_alpha.txt
│   ├── Caption_difused_versus_concentrated_priors.txt
│   ├── Caption_HPD_versus_epicscore.txt
│   ├── Caption_running_time_versus_n.txt
│   ├── Caption_table_coverage_outlier.txt
│   ├── Caption_table_interval_width_ratio.txt
│   ├── coverage_per_outlier_inlier.png
│   ├── difused_versus_concentrated_priors.png
│   ├── HPD_versus_epicscore.png
│   ├── running_time_versus_n.png
│   ├── table_coverage_outlier.md
│   └── table_interval_width_ratio.md
├── LICENCE.txt
├── MANIFEST.in
├── pyproject.toml
├── README.md
├── requirements.txt
├── setup.cfg
└── setup.py
