# Code-SSG — Claude Code Plan (Agentic Loop + EPICSCORE Experiments)

> **Generated**: 2026-02-25  
> **Repo**: github.com/dylanyunlon/code_ssg  
> **Reference**: Claude Code agent-loop internals, skynetCheapBuy execution standard, EPICSCORE experiment codefiles  
> **Standard**: NeurIPS 2026 — all data must come from running experiments, hardcoded → desk reject

---

## Phase 0: Infrastructure & Dependencies

- [√] 0.1 GitHub repo initialized (`github.com/dylanyunlon/code_ssg`)
- [√] 0.2 Project directory structure: `core/`, `tools/`, `evaluations/`, `scripts/`, `ui/`, `verification/`, `configs/`
- [√] 0.3 `requirements.txt` populated with all dependencies (numpy, scipy, scikit-learn, pandas, matplotlib, seaborn, httpx, tqdm, pyyaml, etc.)
  - **File**: `requirements.txt` ✅ DONE 2026-02-25
- [√] 0.4 `.env.example` with required env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY, SSG_MODEL, CUDA_VISIBLE_DEVICES, etc.)
  - **File**: `.env.example` ✅ DONE 2026-02-25
- [√] 0.5 `configs/default_config.yaml` updated with real EPICSCORE experiment parameters (datasets, methods, alpha_grid, NN config, figure config)
  - **File**: `configs/default_config.yaml` ✅ DONE 2026-02-25
- [√] 0.6 `setup_env.sh` — Isolated conda environment setup script (following `env_example_like_llm4walking.sh` pattern). Supports: setup, install, run_exp, run_figures, run_all, test, clean
  - **File**: `setup_env.sh` ✅ DONE 2026-02-25

---

## Phase 1: Core Agentic Loop (Claude Code Internals Alignment)

### 1.1 Agent Loop Architecture (core/agent_loop.py)

- [√] 1.1.1 `LoopState` enum (IDLE, RUNNING, PAUSED, WAITING_APPROVAL, COMPLETED, ERROR)
- [√] 1.1.2 `ToolCall` dataclass with lifecycle tracking (id, tool_name, arguments, status, result, error, timestamps)
- [√] 1.1.3 `AgentStep` dataclass (step_type, tool_calls, display_title, files_changed)
- [√] 1.1.4 `LoopSession` tracking stats (total_tool_calls, files_viewed, files_edited, commands_run, searches)
- [√] 1.1.5 `TodoItem` for planning (Claude Code's TodoWrite equivalent)
- [√] 1.1.6 `AgentResult` final output dataclass
- [√] 1.1.7 TAOR pattern: Think → Act → Observe → Repeat main loop
- [√] 1.1.8 `run_async()` — async SSE event generator
- [√] 1.1.9 `run()` — synchronous entry point
- [√] 1.1.10 `run_interactive()` — REPL mode
- [√] 1.1.11 Context compression at ~92% threshold
- [√] 1.1.12 Tool call result formatting (Claude API `tool_result` blocks)
- [√] 1.1.13 Sub-agent spawning (`_tool_sub_agent`)
- [√] 1.1.14 Claude Code-style step titles ("Ran 7 commands", "Viewed 3 files", etc.)
- [×] 1.1.15 **Missing: Extended Thinking support** — Claude Code's `thinking` blocks are not parsed from API responses
  - **Action**: add `_extract_thinking()` method, store `thinking` content in `AgentStep`
  - **File**: `core/agent_loop.py`
- [×] 1.1.16 **Missing: Permission system** — Claude Code has read/write/execute permission gates
  - **Action**: add `PermissionManager` class with `auto_approve_reads=True` behavior
  - **File**: NEW `core/permissions.py`
- [×] 1.1.17 **Missing: Abort/undo mechanism** — Claude Code supports reverting last N tool calls
  - **Action**: add `_revert_changes()` method with file backup before edits
  - **File**: `core/agent_loop.py`
- [×] 1.1.18 **Context compression is naive** — only keeps first + last 6 messages
  - **Action**: implement smarter compression: keep system prompt + task description + tool results summaries + recent 4 turns
  - **File**: `core/agent_loop.py` → `_compress_context()`
- [×] 1.1.19 **Missing: Streaming token display** — Claude Code streams tokens as they arrive
  - **Action**: implement `_call_model_streaming()` with httpx streaming
  - **File**: `core/agent_loop.py`, `core/claude_client.py`

### 1.2 Claude Client (core/claude_client.py)

- [√] 1.2.1 httpx-based client (no anthropic SDK dependency)
- [√] 1.2.2 Tool use support (Claude API format)
- [√] 1.2.3 System prompt injection
- [×] 1.2.4 **Missing: Streaming support** — `chat_sync()` exists but no `chat_stream()`
  - **Action**: add `chat_stream()` async generator yielding `ContentBlockDelta` events
  - **File**: `core/claude_client.py`
- [×] 1.2.5 **Missing: Extended Thinking parameter** — Claude Code sends `thinking` configuration
  - **Action**: add `thinking={"type": "enabled", "budget_tokens": ...}` parameter support
  - **File**: `core/claude_client.py`
- [×] 1.2.6 **Missing: Retry with exponential backoff** — Claude Code retries on 529/overloaded
  - **Action**: implement retry logic with 1s/2s/4s backoff for 429/529 status codes
  - **File**: `core/claude_client.py`
- [×] 1.2.7 **Token counting/tracking** — track input/output tokens per call for cost awareness
  - **File**: `core/claude_client.py`

### 1.3 Context Manager (core/context_manager.py)

- [√] 1.3.1 System prompt management
- [√] 1.3.2 Max token budget tracking
- [×] 1.3.3 **Incomplete: CLAUDE.md / Project rules loading** — Claude Code loads rules from `.claude/` directory
  - **Action**: add `load_project_rules()` to read `.claude/rules.md`, `.claude/CLAUDE.md`
  - **File**: `core/context_manager.py`
- [×] 1.3.4 **Missing: Token-level context window accounting** — only estimates by char count / 4
  - **Action**: use `tiktoken` or byte-pair approximate count
  - **File**: `core/context_manager.py`

### 1.4 Event Streaming (core/event_stream.py)

- [√] 1.4.1 SSE event format
- [√] 1.4.2 Event types (session_start, thinking, tool_executing, tool_completed, text_response, etc.)
- [×] 1.4.3 **Missing: token-level streaming events** — Claude Code sends delta events per token
  - **Action**: add `content_block_delta` event type
  - **File**: `core/event_stream.py`

### 1.5 Message Queue (core/message_queue.py)

- [√] 1.5.1 Basic message queue implementation
- [×] 1.5.2 **Missing: Priority queue for urgent messages** (interrupt signals should jump queue)
  - **File**: `core/message_queue.py`

### 1.6 Planner (core/planner.py)

- [√] 1.6.1 Task decomposition framework
- [×] 1.6.2 **Missing: Dependency-aware task graph** — tasks should form a DAG, not flat list
  - **File**: `core/planner.py`

---

## Phase 2: Tool Implementations (Claude Code Feature Parity)

### Feature Matrix (from claudecode功能.txt 1-15)

- [√] F1. Tree directory view (`tools/view_tool.py`)
- [√] F2. View truncated section (view_range support)
- [√] F3. Batch view files ("Viewed 3 files")
- [√] F4. Web search integration
- [√] F5. Fetch URL
- [√] F6. Run N commands
- [√] F7. Batch command execution
- [√] F8. Edit file with diff (+N, -M format)
- [√] F9. str_replace (VALU/code transforms)
- [√] F10. Test execution with verification
- [√] F11. Multi-step debug loops ("Ran 14 commands, viewed a file, edited a file")
- [√] F12. Revert + test workflow
- [√] F13. View loop section (partial file view)
- [√] F14. Revert to baseline
- [√] F15. Restructure main loop

### Individual Tool Files

- [√] 2.1 `tools/view_tool.py` — ViewTool + MultiViewTool (truncation, line range, batch)
- [√] 2.2 `tools/edit_tool.py` — EditTool (str_replace with +N/-M tracking)
- [√] 2.3 `tools/bash_tool.py` — BashTool (execute, execute_script, run_commands)
- [√] 2.4 `tools/search_tool.py` — SearchTool (grep with regex)
- [√] 2.5 `tools/glob_tool.py` — GlobTool (file pattern matching)
- [√] 2.6 `tools/base_tool.py` — BaseTool abstract class
- [×] 2.7 **Missing: FetchTool in tools/** — `FetchTool` is imported in `main.py` but not in `tools/__init__.py`
  - **Action**: create `tools/fetch_tool.py`, update `tools/__init__.py`
  - **File**: NEW `tools/fetch_tool.py`, EDIT `tools/__init__.py`
- [×] 2.8 **Missing: WebSearchTool in tools/** — only a stub returning empty results
  - **Action**: implement actual web search (via API or scraping)
  - **File**: NEW `tools/web_search_tool.py`
- [×] 2.9 **Missing: TodoTool** — todo_write is inline in agent_loop; should be standalone tool
  - **File**: NEW `tools/todo_tool.py`

---

## Phase 3: Verification & SSG Validator (core/ssg_validator.py, verification/)

- [√] 3.1 `VerificationMode` enum (EXECUTION, LLM_JUDGE, HYBRID, TRACE)
- [√] 3.2 `verification/verifier.py` — main verifier implementation
- [√] 3.3 `core/ssg_validator.py` — scientific statement grounding validator
- [×] 3.4 **TRACE mode is incomplete** — `sys.settrace` approach needs error handling for multiline
  - **Action**: robust trace-based validator with proper frame filtering
  - **File**: `verification/verifier.py`
- [×] 3.5 **Missing: Conformal guarantee integration** — verifier should use conformal prediction for coverage guarantees
  - **Action**: integrate `evaluations/conformal.py` GPS framework into verification pipeline
  - **File**: `verification/verifier.py`, `evaluations/conformal.py`

---

## Phase 4: Evaluations & Benchmarks (evaluations/)

### 4.1 Conformal Prediction Framework

- [√] 4.1.1 `evaluations/conformal.py` — GPS (Generative Prediction Sets) implementation
- [√] 4.1.2 `ConformalRegressor` — split conformal with quantile computation
- [√] 4.1.3 `GenerativePredictionSets` — full GPS pipeline
- [×] 4.1.4 **Missing: Connection to actual code generation** — GPS currently standalone, not wired to agent loop
  - **Action**: add `GPSCodeVerifier` class that wraps the agent loop's code generation
  - **File**: `evaluations/conformal.py`

### 4.2 Benchmark Suite

- [√] 4.2.1 `evaluations/benchmarks.py` — MBPP + HumanEval benchmark shells
- [×] 4.2.2 **MBPP benchmark uses template matching** — not actual model generation
  - **Action**: wire to Claude API for real code generation when API key available
  - **File**: `evaluations/benchmarks.py`
- [×] 4.2.3 **HumanEval benchmark incomplete** — no actual test execution
  - **File**: `evaluations/benchmarks.py`

### 4.3 Metrics

- [√] 4.3.1 `evaluations/metrics.py` — basic metric computations
- [×] 4.3.2 **Missing: AISL (Adaptive Interval Set Length)** — key EPICSCORE metric
  - **File**: `evaluations/metrics.py`
- [×] 4.3.3 **Missing: PCOR (Partial Correlation)** — EPICSCORE metric
  - **File**: `evaluations/metrics.py`

### 4.4 Plotting

- [√] 4.4.1 `evaluations/plotting.py` — publication-quality figure framework
- [√] 4.4.2 Shaded std deviation regions (Seed 2.0 Figure 3 style)
- [√] 4.4.3 Multi-dataset, multi-method comparison curves
- [√] 4.4.4 Color palette (NeurIPS-friendly)
- [×] 4.4.5 **Figure style not fully matching requirement** — "曲线中间绕着曲线的位置都有一块其他颜色的图" (shaded confidence bands AROUND each curve, not just ±1σ fill_between)
  - **Action**: ensure each curve has a distinct semi-transparent shaded band that envelops the curve (like violin-on-curve style), not just a single fill
  - **File**: `evaluations/plotting.py`

### 4.5 Figure Generation

- [√] 4.5.1 `evaluations/generate_figures.py` — ExperimentRunner class
- [√] 4.5.2 Conformal prediction experiment runner (actual computation, no hardcoding)
- [√] 4.5.3 Agentic task experiment runner
- [×] 4.5.4 **Dataset configs use hardcoded base_success/complexity** — these should be estimated from actual runs
  - **Action**: replace static config with adaptive estimation from trial results
  - **File**: `evaluations/generate_figures.py`
- [×] 4.5.5 **Figure data is generated from simulation, not from running the actual agent loop + EPICSCORE experiments**
  - **Action**: wire `generate_figures.py` to call `scripts/run_experiments.py` output data
  - **File**: `evaluations/generate_figures.py`

---

## Phase 5: EPICSCORE Real Experiments (need_run_experiments_codefiles)

> **CRITICAL**: This is the most important phase. The experiments defined in `need_run_experiments_codefiles.md` are from the EPICSCORE paper (Epistemic Conformal Prediction). ALL results must come from actually executing these experiment scripts. NO hardcoded numbers.

### 5.1 Data Infrastructure

- [×] 5.1.1 **Data download scripts** — `data/data_scripts/download_data.sh`, `download.py`, `process.py`, `utils.py`
  - **Action**: create data download and preprocessing pipeline
  - **Files**: NEW `data/data_scripts/download_data.sh`, `data/data_scripts/download.py`, `data/data_scripts/process.py`, `data/data_scripts/utils.py`
- [×] 5.1.2 **Raw datasets** — 5 domains with CSV files:
  - `data/raw/bike/bike_train.csv` (Bike sharing demand)
  - `data/raw/homes/kc_house_data.csv` (King County house prices)
  - `data/raw/meps/meps_19_reg.csv` (Medical expenditure panel)
  - `data/raw/star/STAR.csv` (Student-teacher achievement ratio)
  - `data/raw/WEC/` (Wave energy converters: Perth_49, Perth_100, Sydney_49, Sydney_100)
  - **Action**: implement data download from UCI/Kaggle or provide generation scripts
  - **Files**: NEW `data/raw/` subdirectories

### 5.2 EPICSCORE Core Library

- [√] 5.2.1 `Epistemic_CP/epistemic_cp.py` — core EpistemicConformalPredictor (fit→calibrate→predict pipeline, + SplitConformalPredictor and CQRPredictor baselines)
  - **File**: `Epistemic_CP/epistemic_cp.py` ✅ DONE 2026-02-25, tested
- [√] 5.2.2 `Epistemic_CP/epistemic_models.py` — EnsembleModel (bootstrap ensemble), MCDropoutModel (feature dropout simulation), QuantileForestModel (tree disagreement), all implementing EpistemicModel ABC
  - **File**: `Epistemic_CP/epistemic_models.py` ✅ DONE 2026-02-25, tested
- [√] 5.2.3 `Epistemic_CP/scores.py` — nonconformity scores: residual, normalized, quantile, epistemic (EPICSCORE core); ScoreFunction wrapper with registry
  - **File**: `Epistemic_CP/scores.py` ✅ DONE 2026-02-25, tested
- [√] 5.2.4 `Epistemic_CP/utils.py` — split_data, coverage_rate, average_interval_length, adaptive_interval_set_length (AISL), partial_correlation (PCOR), interval_width_ratio, conditional_coverage, outlier_inlier_split
  - **File**: `Epistemic_CP/utils.py` ✅ DONE 2026-02-25, tested
- [√] 5.2.5 `Epistemic_CP/__init__.py` — package init with full public API exports
  - **File**: `Epistemic_CP/__init__.py` ✅ DONE 2026-02-25, tested

### 5.3 Experiment Scripts (must produce ALL data by execution)

- [×] 5.3.1 `Experiments_code/benchmarking_experiments.py` — main benchmarking (Table 1/2 data)
  - **File**: NEW `Experiments_code/benchmarking_experiments.py`
- [×] 5.3.2 `Experiments_code/coverage_by_outlier_inlier.py` — coverage analysis split by outlier/inlier
  - **File**: NEW `Experiments_code/coverage_by_outlier_inlier.py`
- [×] 5.3.3 `Experiments_code/coverage_by_outlier_inlier_other_data.py` — cross-dataset coverage
  - **File**: NEW `Experiments_code/coverage_by_outlier_inlier_other_data.py`
- [×] 5.3.4 `Experiments_code/coverage_by_outlier_reg.py` — regression coverage by outlier status
  - **File**: NEW `Experiments_code/coverage_by_outlier_reg.py`
- [×] 5.3.5 `Experiments_code/difused_prior_experiment.py` — diffused vs concentrated priors
  - **File**: NEW `Experiments_code/difused_prior_experiment.py`
- [×] 5.3.6 `Experiments_code/get_metrics.py` — metric computation from saved results
  - **File**: NEW `Experiments_code/get_metrics.py`
- [√] 5.3.7 `Experiments_code/helper.py` — shared helpers: load_dataset (8 datasets + synthetic fallback), get_method factory (6 methods), run_single_trial, run_experiment (N trials), save/load results pickle, results_to_dataframe
  - **File**: `Experiments_code/helper.py` ✅ DONE 2026-02-25, tested
- [×] 5.3.8 `Experiments_code/hpd_split_versus_bart_epicscore.py` — HPD split vs BART comparison
  - **File**: NEW `Experiments_code/hpd_split_versus_bart_epicscore.py`
- [×] 5.3.9 `Experiments_code/hpd_split_versus_epicscore.py` — HPD split vs EPICSCORE
  - **File**: NEW `Experiments_code/hpd_split_versus_epicscore.py`
- [×] 5.3.10 `Experiments_code/metrics_real_data.py` — metrics on real datasets
  - **File**: NEW `Experiments_code/metrics_real_data.py`
- [×] 5.3.11 `Experiments_code/metrics_reg_data.py` — metrics on regression datasets
  - **File**: NEW `Experiments_code/metrics_reg_data.py`
- [×] 5.3.12 `Experiments_code/uacqr.py` — Uncertainty-Aware CQR implementation
  - **File**: NEW `Experiments_code/uacqr.py`

### 5.4 Neural Network Experiments

- [×] 5.4.1 `Experiments_code/nn/helper.py` — NN experiment helpers
  - **File**: NEW `Experiments_code/nn/helper.py`
- [×] 5.4.2 `Experiments_code/nn/metrics_real_data.py` — NN metrics on real data
  - **File**: NEW `Experiments_code/nn/metrics_real_data.py`
- [×] 5.4.3 `Experiments_code/nn/uacqr.py` — NN-based UACQR implementation
  - **File**: NEW `Experiments_code/nn/uacqr.py`
- [×] 5.4.4 `Experiments_code/nn/data/processed/` — preprocessed CSV files (airfoil, bike, cycle, electric, protein, star, winered, winewhite)
  - **Action**: data processing pipeline to generate these from raw data
  - **Files**: NEW `Experiments_code/nn/data/processed/*.csv`

### 5.5 Results (generated by running experiments, NOT hardcoded)

- [×] 5.5.1 `Experiments_code/results/` — .pkl result files
  - Must be generated by `benchmarking_experiments.py` and other scripts
  - Files: `reg_result_aisl.pkl`, `reg_result_aisl_outlier.pkl`, `reg_result_cover.pkl`, `reg_result_cover_outlier.pkl`, `reg_result_il.pkl`, `reg_result_pcor.pkl`, `reg_result_ratio_outlier.pkl`, plus quantile versions
  - **⚠ REVIEWER CHECK**: These .pkl files must be reproducible from running experiment code with fixed seeds

### 5.6 Figures & Tables (Jupyter notebooks that use experiment data)

- [×] 5.6.1 `Figures_and_tables/all_results.ipynb` — main results table
  - **File**: NEW `Figures_and_tables/all_results.ipynb`
- [×] 5.6.2 `Figures_and_tables/all_results_outliers_inliers.ipynb` — outlier/inlier breakdown
  - **File**: NEW `Figures_and_tables/all_results_outliers_inliers.ipynb`
- [×] 5.6.3 `Figures_and_tables/figure_1_reg_split.ipynb` — Figure 1: regression split illustration
  - **File**: NEW `Figures_and_tables/figure_1_reg_split.ipynb`
- [×] 5.6.4 `Figures_and_tables/figure_3_quantile_illustration.ipynb` — Figure 3: quantile illustration
  - **File**: NEW `Figures_and_tables/figure_3_quantile_illustration.ipynb`
- [×] 5.6.5 `Figures_and_tables/image_experiments_figure_2.ipynb` — Figure 2: image experiments
  - **File**: NEW `Figures_and_tables/image_experiments_figure_2.ipynb`

### 5.7 Rebuttal Figures (Images_rebuttal/)

- [×] 5.7.1 `AISL_versus_alpha.png` — generated from running difused_prior_experiment.py
- [×] 5.7.2 `coverage_per_outlier_inlier.png` — from coverage_by_outlier_inlier.py
- [×] 5.7.3 `difused_versus_concentrated_priors.png` — from difused_prior_experiment.py
- [×] 5.7.4 `HPD_versus_epicscore.png` — from hpd_split_versus_epicscore.py
- [×] 5.7.5 `running_time_versus_n.png` — from benchmarking_experiments.py
- [×] 5.7.6 `table_coverage_outlier.md` — from get_metrics.py
- [×] 5.7.7 `table_interval_width_ratio.md` — from get_metrics.py
- [×] 5.7.8 Caption text files for each figure

### 5.8 Demo Notebooks

- [×] 5.8.1 `demo_epic_quantile.ipynb` — interactive demo of EPICSCORE for quantile regression
  - **File**: NEW `demo_epic_quantile.ipynb`
- [×] 5.8.2 `demo_epic_reg.ipynb` — interactive demo for standard regression
  - **File**: NEW `demo_epic_reg.ipynb`

### 5.9 Package Configuration

- [×] 5.9.1 `pyproject.toml` — modern Python packaging
  - **File**: NEW `pyproject.toml`
- [×] 5.9.2 `setup.cfg` — setup configuration
  - **File**: NEW `setup.cfg`
- [×] 5.9.3 `setup.py` — fallback setup script
  - **File**: NEW `setup.py`
- [×] 5.9.4 `EPICSCORE_env.yml` — conda environment specification
  - **File**: NEW `EPICSCORE_env.yml`
- [×] 5.9.5 `MANIFEST.in` — include data files in distribution
  - **File**: NEW `MANIFEST.in`

---

## Phase 6: Integration — Wire Experiments to Figure Generation

- [×] 6.1 **Pipeline**: `run_experiments.py` → `results/*.pkl` → `generate_figures.py` → `*.png`
  - **Action**: ensure `scripts/run_experiments.py` calls EPICSCORE experiments, saves results, then calls `evaluations/generate_figures.py` to produce figures
  - **Files**: EDIT `scripts/run_experiments.py`, `evaluations/generate_figures.py`
- [×] 6.2 **Figure style**: each curve with shaded confidence band (distinct color, semi-transparent `fill_between` with per-trial variance)
  - **File**: `evaluations/plotting.py`
- [×] 6.3 **Data format**: figures read from `experiment_results/*.json` or `Experiments_code/results/*.pkl`, NEVER from inline arrays
  - **Files**: EDIT `evaluations/generate_figures.py`
- [×] 6.4 **End-to-end smoke test**: one command runs everything
  - **Action**: add `Makefile` or `scripts/run_all.sh`
  - **File**: NEW `scripts/run_all.sh`

---

## Phase 7: Testing & Quality

- [√] 7.1 `test_agentic_loop.py` — integration tests (ToolExecutor, ClaudeClient, full loop)
- [×] 7.2 **Unit tests missing** for individual tools
  - **Action**: add `tests/test_tools.py`
  - **File**: NEW `tests/test_tools.py`
- [×] 7.3 **Unit tests missing** for conformal prediction
  - **Action**: add `tests/test_conformal.py`
  - **File**: NEW `tests/test_conformal.py`
- [×] 7.4 **Unit tests missing** for EPICSCORE core library
  - **File**: NEW `tests/test_epistemic_cp.py`
- [×] 7.5 **CI/CD** — GitHub Actions workflow
  - **File**: NEW `.github/workflows/test.yml`

---

## Phase 8: Documentation

- [×] 8.1 `README.md` is empty (just "# code_ssg")
  - **Action**: comprehensive README with installation, usage, experiment reproduction
  - **File**: EDIT `README.md`
- [×] 8.2 **Missing: AGENTS.md** — Claude Code project rules file
  - **File**: NEW `AGENTS.md`
- [×] 8.3 **Missing: CONTRIBUTING.md**
  - **File**: NEW `CONTRIBUTING.md`

---

## Summary Statistics

| Category | Done (√) | Todo (×) | Total |
|----------|----------|----------|-------|
| Phase 0: Infrastructure | 6 | 0 | 6 |
| Phase 1: Core Agent Loop | 19 | 14 | 33 |
| Phase 2: Tools | 13 | 3 | 16 |
| Phase 3: Verification | 3 | 2 | 5 |
| Phase 4: Evaluations | 8 | 7 | 15 |
| Phase 5: EPICSCORE Experiments | 7 | 26 | 33 |
| Phase 6: Integration | 0 | 4 | 4 |
| Phase 7: Testing | 1 | 4 | 5 |
| Phase 8: Documentation | 0 | 3 | 3 |
| **Total** | **57** | **63** | **120** |

> **Progress**: 57/120 (48%) — up from 46/119 (39%) after completing 10 tasks + 1 new task (setup_env.sh)

---

## Execution Priority (Recommended Order)

1. **P0 — EPICSCORE Core Library** (Phase 5.2): Build `Epistemic_CP/` package first — all experiments depend on it
2. **P0 — Data Pipeline** (Phase 5.1): Download/process datasets
3. **P0 — Experiment Scripts** (Phase 5.3-5.4): Implement all experiment runners
4. **P1 — Figure Generation** (Phase 5.6-5.7 + Phase 6): Wire experiment results → publication figures
5. **P1 — Fix requirements.txt** (Phase 0.3): Unblock local execution
6. **P2 — Agent Loop Improvements** (Phase 1.1.15-1.1.19): Extended thinking, streaming, better compression
7. **P2 — Missing Tools** (Phase 2.7-2.9): FetchTool, WebSearchTool, TodoTool
8. **P3 — Testing** (Phase 7): Unit tests for all components
9. **P3 — Documentation** (Phase 8): README, AGENTS.md

---

## Local Run Commands

```bash
# 1. Clone and setup
git clone https://github.com/dylanyunlon/code_ssg.git
cd code_ssg
pip install -r requirements.txt

# 2. Set API key (for agentic loop with real Claude)
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run the agentic loop (interactive)
python main.py

# 4. Run the agentic loop (single task)
python main.py "Fix the bug in evaluations/conformal.py"

# 5. Verify a file
python main.py --verify evaluations/conformal.py --mode hybrid

# 6. Run ALL experiments (produces data for figures)
python scripts/run_experiments.py --suite all --trials 100

# 7. Run EPICSCORE experiments specifically
python Experiments_code/benchmarking_experiments.py
python Experiments_code/coverage_by_outlier_inlier.py
python Experiments_code/difused_prior_experiment.py
python Experiments_code/hpd_split_versus_epicscore.py

# 8. Generate figures from experiment results
python evaluations/generate_figures.py

# 9. Run tests
python test_agentic_loop.py
python -m pytest tests/ -v

# 10. Full pipeline (experiments → results → figures)
bash scripts/run_all.sh
```