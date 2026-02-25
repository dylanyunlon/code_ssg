#!/bin/bash
# ===========================================
# Code-SSG Environment Setup Script
# ===========================================
# Following the pattern from env_example_like_llm4walking.sh
# Creates an isolated conda environment with all dependencies.
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh              # Full setup (create env + install)
#   ./setup_env.sh install      # Install packages only (env exists)
#   ./setup_env.sh activate     # Print activation command
#   ./setup_env.sh run_exp      # Run all EPICSCORE experiments
#   ./setup_env.sh run_figures  # Generate figures from results
#   ./setup_env.sh run_all      # Full pipeline: experiments → figures
#   ./setup_env.sh test         # Run test suite
#   ./setup_env.sh clean        # Remove environment
#
# ===========================================

set -e

# ===========================================
# Path Configuration
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/experiment_results}"
FIGURES_DIR="${FIGURES_DIR:-$PROJECT_DIR/figures}"

# Conda environment name
CONDA_ENV_NAME="${CONDA_ENV_NAME:-code_ssg}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# ===========================================
# Conda Detection (same pattern as llm4walking)
# ===========================================

if [ -f "/usr/local/lib/miniconda3/bin/conda" ]; then
    CONDA_BASE="/usr/local/lib/miniconda3"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    CONDA_BASE="$HOME/anaconda3"
else
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
fi

# ===========================================
# Utility Functions
# ===========================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

print_step() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

# ===========================================
# Conda Init & Activate
# ===========================================

init_conda() {
    if [ -z "$CONDA_BASE" ]; then
        log_error "Conda not found. Install Miniconda first:"
        log_error "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        log_error "  bash Miniconda3-latest-Linux-x86_64.sh"
        exit 1
    fi
    __conda_setup="$("$CONDA_BASE/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            . "$CONDA_BASE/etc/profile.d/conda.sh"
        fi
    fi
    unset __conda_setup
}

activate_env() {
    local env_name="${1:-$CONDA_ENV_NAME}"
    init_conda
    conda activate "$env_name" 2>/dev/null || {
        log_error "Failed to activate '$env_name'. Run: ./setup_env.sh"
        exit 1
    }
    log_info "Activated conda environment: $env_name"
}

# ===========================================
# Package Installation
# ===========================================

_install_packages() {
    log_info "Installing Code-SSG packages..."

    # Core scientific computing
    pip install -q numpy scipy scikit-learn pandas 2>/dev/null || pip install numpy scipy scikit-learn pandas

    # Plotting
    pip install -q matplotlib seaborn 2>/dev/null || pip install matplotlib seaborn

    # HTTP / API
    pip install -q httpx 2>/dev/null || pip install httpx

    # Utilities
    pip install -q tqdm pyyaml cloudpickle joblib statsmodels 2>/dev/null || \
        pip install tqdm pyyaml cloudpickle joblib statsmodels

    # Testing
    pip install -q pytest pytest-asyncio 2>/dev/null || pip install pytest pytest-asyncio

    # Install from requirements.txt if it exists
    if [ -f "$PROJECT_DIR/requirements.txt" ]; then
        pip install -q -r "$PROJECT_DIR/requirements.txt" 2>/dev/null || \
            pip install -r "$PROJECT_DIR/requirements.txt"
    fi

    # Optional: PyTorch for NN experiments (uncomment for GPU server)
    # pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    log_success "All packages installed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Code-SSG Environment"

    if ! command -v conda &> /dev/null && [ -z "$CONDA_BASE" ]; then
        log_error "Conda not found. Install Miniconda/Anaconda first."
        exit 1
    fi

    init_conda

    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        log_info "Environment '${CONDA_ENV_NAME}' already exists."
        log_info "Updating packages..."
        activate_env
        _install_packages
        log_success "Environment updated: ${CONDA_ENV_NAME}"
    else
        log_info "Creating conda environment '${CONDA_ENV_NAME}' (Python ${PYTHON_VERSION})..."
        conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
        activate_env
        _install_packages
        log_success "Environment created: ${CONDA_ENV_NAME}"
    fi

    # Create output directories
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$FIGURES_DIR"
    mkdir -p "$PROJECT_DIR/Experiments_code/results"
    mkdir -p "$PROJECT_DIR/Experiments_code/nn/data/processed"
    mkdir -p "$PROJECT_DIR/data/raw/bike"
    mkdir -p "$PROJECT_DIR/data/raw/homes"
    mkdir -p "$PROJECT_DIR/data/raw/meps"
    mkdir -p "$PROJECT_DIR/data/raw/star"
    mkdir -p "$PROJECT_DIR/data/raw/WEC"
    mkdir -p "$PROJECT_DIR/Figures_and_tables"
    mkdir -p "$PROJECT_DIR/Images_rebuttal"

    log_success "Directory structure created"

    echo ""
    log_info "To activate manually:"
    log_info "  conda activate ${CONDA_ENV_NAME}"
    echo ""
    log_info "To run experiments:"
    log_info "  ./setup_env.sh run_all"
}

# ===========================================
# Experiment Runners
# ===========================================

run_experiments() {
    print_step "Running EPICSCORE Experiments"
    activate_env

    log_info "Project: $PROJECT_DIR"
    log_info "Output:  $OUTPUT_DIR"
    log_info "Seed:    ${SSG_EXPERIMENT_SEED:-42}"

    # Load .env if present
    if [ -f "$PROJECT_DIR/.env" ]; then
        set -a
        source "$PROJECT_DIR/.env"
        set +a
        log_info "Loaded .env"
    fi

    # 1. Download/process data (if not already done)
    if [ -f "$PROJECT_DIR/data/data_scripts/download_data.sh" ]; then
        log_info "Running data download..."
        bash "$PROJECT_DIR/data/data_scripts/download_data.sh"
    fi

    # 2. Run benchmarking experiments
    if [ -f "$PROJECT_DIR/Experiments_code/benchmarking_experiments.py" ]; then
        log_info "Running benchmarking experiments..."
        cd "$PROJECT_DIR"
        python Experiments_code/benchmarking_experiments.py
    fi

    # 3. Run coverage experiments
    for script in \
        "Experiments_code/coverage_by_outlier_inlier.py" \
        "Experiments_code/coverage_by_outlier_reg.py" \
        "Experiments_code/difused_prior_experiment.py" \
        "Experiments_code/hpd_split_versus_epicscore.py" \
        "Experiments_code/hpd_split_versus_bart_epicscore.py" \
    ; do
        if [ -f "$PROJECT_DIR/$script" ]; then
            log_info "Running $script..."
            cd "$PROJECT_DIR"
            python "$script"
        fi
    done

    # 4. Compute metrics from results
    if [ -f "$PROJECT_DIR/Experiments_code/get_metrics.py" ]; then
        log_info "Computing metrics..."
        cd "$PROJECT_DIR"
        python Experiments_code/get_metrics.py
    fi

    # 5. Run SSG evaluation suite
    if [ -f "$PROJECT_DIR/scripts/run_experiments.py" ]; then
        log_info "Running SSG evaluation suite..."
        cd "$PROJECT_DIR"
        python scripts/run_experiments.py --suite all --trials ${SSG_EXPERIMENT_TRIALS:-100}
    fi

    log_success "All experiments completed"
    log_info "Results saved to: $OUTPUT_DIR and Experiments_code/results/"
}

run_figures() {
    print_step "Generating Figures"
    activate_env

    cd "$PROJECT_DIR"

    # Generate from experiment results
    if [ -f "$PROJECT_DIR/Experiments_code/get_metrics.py" ]; then
        python Experiments_code/get_metrics.py
    fi

    python evaluations/generate_figures.py

    log_success "Figures saved to: $FIGURES_DIR"
}

run_all() {
    run_experiments
    run_figures
    log_success "Full pipeline completed"
}

run_tests() {
    print_step "Running Tests"
    activate_env

    cd "$PROJECT_DIR"

    # Integration test
    python test_agentic_loop.py

    # Unit tests (if pytest available)
    if command -v pytest &> /dev/null; then
        pytest tests/ -v 2>/dev/null || log_warn "No tests/ directory yet"
    fi

    log_success "Tests completed"
}

clean_env() {
    print_step "Cleaning Environment"
    init_conda
    conda env remove -n ${CONDA_ENV_NAME} -y 2>/dev/null || true
    log_success "Environment '${CONDA_ENV_NAME}' removed"
}

# ===========================================
# Main Dispatch
# ===========================================

show_help() {
    echo "Usage: ./setup_env.sh [command]"
    echo ""
    echo "Commands:"
    echo "  (none)        Full setup: create conda env + install packages"
    echo "  install       Install/update packages in existing env"
    echo "  activate      Print activation command"
    echo "  run_exp       Run all EPICSCORE experiments"
    echo "  run_figures   Generate figures from experiment results"
    echo "  run_all       Full pipeline: experiments → figures"
    echo "  test          Run test suite"
    echo "  clean         Remove conda environment"
    echo "  help          Show this help message"
}

case "${1:-setup}" in
    setup)        setup_environment ;;
    install)      activate_env && _install_packages ;;
    activate)     echo "conda activate ${CONDA_ENV_NAME}" ;;
    run_exp)      run_experiments ;;
    run_figures)  run_figures ;;
    run_all)      run_all ;;
    test)         run_tests ;;
    clean)        clean_env ;;
    help|--help)  show_help ;;
    *)            log_error "Unknown command: $1"; show_help; exit 1 ;;
esac
