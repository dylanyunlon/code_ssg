#!/bin/bash
# ===========================================
# LLM4CCPO - CCPO强化学习MathCoder2训练与评测流水线 (v7.0 - 全模型支持版)
# ===========================================
#
# v7.0 新增功能:
#   1. 支持所有四个MathCoder2模型 (deepseek, mistral, codellama, llama3)
#   2. 批量训练所有模型: ./llm4ccpo.sh all_models
#   3. 批量评测所有模型: ./llm4ccpo.sh eval_all_models
#   4. 汇总对比报告: ./llm4ccpo.sh compare_results
#   5. 继承MathCoder2的prompts配置
#
# 历史修复:
#   v6.1 - DeepSeek评测修复 (pebble等依赖)
#   v6   - lm_eval兼容性修复 (FutureWarning)
#   v5   - MMLU评测修复
#
# CCPO核心思想 (Code-Consistency Preference Optimization):
#   Step 1: 从大模型推理过程中提取答案 (模型可能算错，存在幻觉)
#   Step 2: 将推理步骤发送给代码验证服务，获取代码执行结果
#   Step 3: 对比模型答案和代码执行结果:
#       - 模型正确 + 代码正确 → 高ranking分 (推理步骤准确)
#       - 模型错误 + 代码正确 → 次级ranking (代码验证了推理过程正确)
#       - 都错误 → 低ranking
#   Step 4: 构建偏好对进行CCPO训练
#
# 使用示例:
#   # 训练单个模型 (100样本快速测试)
#   LIMIT_SAMPLES=100 ./llm4ccpo.sh full
#
#   # 训练所有四个模型
#   LIMIT_SAMPLES=100 ./llm4ccpo.sh all_models
#
#   # 评测所有模型并生成对比报告
#   ./llm4ccpo.sh eval_all_models
#
# ===========================================

set -e

# ===========================================
# 路径配置
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 自动检测PROJECT_DIR
if [ -f "$SCRIPT_DIR/ccpo/run_ccpo.py" ] || [ -f "$SCRIPT_DIR/scripts/generate.py" ]; then
    PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
else
    PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/Code-Consistency-Preference-Optimization}"
fi

DATA_DIR="${DATA_DIR:-$PROJECT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/checkpoints}"
TRAIN_CONFIG_DIR="${TRAIN_CONFIG_DIR:-$PROJECT_DIR}"

# HuggingFace缓存
HF_CACHE_DIR="${HF_HOME:-/data/jiacheng/system/cache/temp/huggingface}"
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"

# Conda环境配置
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ccpo}"
CONDA_ENV_MAMMOTH="${CONDA_ENV_MAMMOTH:-mammoth}"
CONDA_ENV_LMEVAL="${CONDA_ENV_LMEVAL:-lm-eval}"
CONDA_ENV_DEEPSEEK="${CONDA_ENV_DEEPSEEK:-deepseek}"
SOURCE_ENV="${SOURCE_ENV:-base}"

# Conda路径（自动检测）
if [ -f "/usr/local/lib/miniconda3/bin/conda" ]; then
    CONDA_BASE="/usr/local/lib/miniconda3"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    CONDA_BASE="$HOME/anaconda3"
else
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "/usr/local/lib/miniconda3")
fi

# ===========================================
# 模型配置 - 支持所有四个MathCoder2模型
# ===========================================

declare -A MODEL_PATHS=(
    ["deepseek"]="/data/jiacheng/system/cache/temp/huggingface/MathGenie/MathCoder2-DeepSeekMath-7B"
    ["mistral"]="/data/jiacheng/system/cache/temp/huggingface/MathGenie/MathCoder2-Mistral-7B"
    ["codellama"]="/data/jiacheng/system/cache/temp/huggingface/MathGenie/MathCoder2-CodeLlama-7B"
    ["llama3"]="/data/jiacheng/system/cache/temp/huggingface/MathGenie/MathCoder2-Llama-3-8B"
    ["metallama3"]="/data/jiacheng/system/cache/temp/huggingface/meta-llama/Meta-Llama-3-8B-Instruct"
)

declare -A MODEL_TEMPLATES=(
    ["deepseek"]="deepseek"
    ["mistral"]="mistral"
    ["codellama"]="codellama"
    ["llama3"]="llama3"
    ["metallama3"]="llama3"
)

declare -A MODEL_SIZES=(
    ["deepseek"]="7b"
    ["mistral"]="7b"
    ["codellama"]="7b"
    ["llama3"]="8b"
    ["metallama3"]="8b"
)

# MathCoder2的四个主要模型
MATHCODER2_MODELS=("deepseek" "mistral" "codellama" "llama3")

# 默认模型
MODEL_KEY="${MODEL_KEY:-deepseek}"
BASE_MODEL="${BASE_MODEL:-${MODEL_PATHS[$MODEL_KEY]}}"
TEMPLATE="${TEMPLATE:-${MODEL_TEMPLATES[$MODEL_KEY]}}"
MODEL_SIZE="${MODEL_SIZE:-${MODEL_SIZES[$MODEL_KEY]:-7b}}"

# CCPO输出 - 模型名称只带ccpo
CCPO_OUTPUT_NAME="${CCPO_OUTPUT_NAME:-ccpo_${MODEL_KEY}_${MODEL_SIZE}}"
CCPO_OUTPUT_DIR="${CCPO_OUTPUT_DIR:-$OUTPUT_DIR/$CCPO_OUTPUT_NAME}"

# ===========================================
# 训练参数
# ===========================================

LIMIT_SAMPLES="${LIMIT_SAMPLES:-100}"
PAIRS="${PAIRS:-5}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
MAX_LEN="${MAX_LEN:-2048}"

LEARNING_RATE="${LEARNING_RATE:-1.0e-5}"
BETA="${BETA:-0.05}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-500}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
API_PORT="${API_PORT:-8555}"

# 代码验证服务配置 (CCPO核心)
VERIFICATION_BASE_URL="${VERIFICATION_BASE_URL:-https://8.163.12.28:17432}"
VERIFICATION_USERNAME="${VERIFICATION_USERNAME:-newuser}"
VERIFICATION_PASSWORD="${VERIFICATION_PASSWORD:-newPass123}"
VERIFICATION_BATCH_SIZE="${VERIFICATION_BATCH_SIZE:-5}"
VERIFICATION_SAMPLE_RATE="${VERIFICATION_SAMPLE_RATE:-0.1}"
ENABLE_CODE_VERIFICATION="${ENABLE_CODE_VERIFICATION:-true}"

USE_FEWSHOT="${USE_FEWSHOT:-true}"

# 数据路径
GENERATED_DIR="${GENERATED_DIR:-$DATA_DIR/generated_ccpo_${MODEL_KEY}}"
RANKING_DIR="${RANKING_DIR:-$DATA_DIR/ranking_ccpo_${MODEL_KEY}}"
PROCESSED_DIR="${PROCESSED_DIR:-$DATA_DIR/processed_data_ccpo_${MODEL_KEY}}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$DATA_DIR/eval_results_ccpo_${MODEL_KEY}}"
PROMPTS_DATASET="${PROMPTS_DATASET:-dylansss/ccpo_math_dataset}"
DATA_SIZE="${DATA_SIZE:-$LIMIT_SAMPLES}"

# MathCoder2评测配置
MATHCODER2_TEST_DIR="${MATHCODER2_TEST_DIR:-$PROJECT_DIR/MathCoder2/test}"
MATHCODER2_SCRIPTS_DIR="${MATHCODER2_SCRIPTS_DIR:-$MATHCODER2_TEST_DIR/scripts}"
MAMMOTH_EVAL_DIR="${MAMMOTH_EVAL_DIR:-$MATHCODER2_TEST_DIR/MAmmoTH/math_eval}"
DEEPSEEK_EVAL_DIR="${DEEPSEEK_EVAL_DIR:-$MATHCODER2_TEST_DIR/DeepSeek-Math/evaluation}"

# ===========================================
# 工具函数
# ===========================================

print_header() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  LLM4CCPO - CCPO强化学习MathCoder2训练流水线 (v7.0)           ║"
    echo "║  Code-Consistency Preference Optimization                      ║"
    echo "║  [v7.0] 支持所有四个MathCoder2模型批量训练与评测              ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
}

print_step() {
    echo ""
    echo "┌──────────────────────────────────────────────────────────────┐"
    echo "│  $1"
    echo "└──────────────────────────────────────────────────────────────┘"
    echo ""
}

check_dir() { mkdir -p "$1"; }
log_info() { echo "[INFO] $(date '+%H:%M:%S') $1"; }
log_error() { echo "[ERROR] $(date '+%H:%M:%S') $1" >&2; }
log_warn() { echo "[WARN] $(date '+%H:%M:%S') $1" >&2; }
log_success() { echo "[✓] $(date '+%H:%M:%S') $1"; }

# Conda环境管理
init_conda() {
    __conda_setup="$("$CONDA_BASE/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            . "$CONDA_BASE/etc/profile.d/conda.sh"
        else
            export PATH="$CONDA_BASE/bin:$PATH"
        fi
    fi
    unset __conda_setup
}

activate_env() {
    local env_name="${1:-$CONDA_ENV_NAME}"
    init_conda
    conda activate "$env_name" 2>/dev/null || {
        log_warn "Conda environment '$env_name' not found, using current env"
        return 0
    }
    log_info "Activated conda environment: $env_name"
}

check_server() {
    local port="${1:-$API_PORT}"
    curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1
}

wait_for_server() {
    local port="${1:-$API_PORT}"
    local max_wait="${2:-180}"
    local waited=0
    log_info "Waiting for server on port $port..."
    while [ $waited -lt $max_wait ]; do
        if check_server "$port"; then
            log_success "Server ready on port $port"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        echo -n "."
    done
    echo ""
    log_error "Server timeout (${max_wait}s)"
    return 1
}

# ===========================================
# 模型配置更新函数 (v7.0新增)
# ===========================================

# 切换当前工作模型
switch_model() {
    local new_model_key="$1"
    
    if [[ -z "${MODEL_PATHS[$new_model_key]}" ]]; then
        log_error "Unknown model key: $new_model_key"
        log_info "Available models: ${!MODEL_PATHS[*]}"
        return 1
    fi
    
    MODEL_KEY="$new_model_key"
    BASE_MODEL="${MODEL_PATHS[$MODEL_KEY]}"
    TEMPLATE="${MODEL_TEMPLATES[$MODEL_KEY]}"
    MODEL_SIZE="${MODEL_SIZES[$MODEL_KEY]:-7b}"
    
    CCPO_OUTPUT_NAME="ccpo_${MODEL_KEY}_${MODEL_SIZE}"
    CCPO_OUTPUT_DIR="$OUTPUT_DIR/$CCPO_OUTPUT_NAME"
    
    GENERATED_DIR="$DATA_DIR/generated_ccpo_${MODEL_KEY}"
    RANKING_DIR="$DATA_DIR/ranking_ccpo_${MODEL_KEY}"
    PROCESSED_DIR="$DATA_DIR/processed_data_ccpo_${MODEL_KEY}"
    EVAL_OUTPUT_DIR="$DATA_DIR/eval_results_ccpo_${MODEL_KEY}"
    
    log_info "Switched to model: $MODEL_KEY"
    log_info "  Base Model: $BASE_MODEL"
    log_info "  Output: $CCPO_OUTPUT_DIR"
}

# ===========================================
# 模型路径解析
# ===========================================

get_hf_model_local_path() {
    local model_id="$1"
    local cache_model_id=$(echo "$model_id" | sed 's/\//-_-/g')
    local hub_path="$HF_CACHE_DIR/hub/models--${cache_model_id//-_-/--}"
    if [ -d "$hub_path/snapshots" ]; then
        local latest=$(ls -t "$hub_path/snapshots" 2>/dev/null | head -1)
        if [ -n "$latest" ] && [ -f "$hub_path/snapshots/$latest/config.json" ]; then
            echo "$hub_path/snapshots/$latest"
            return 0
        fi
    fi
    return 1
}

validate_model_path() {
    local model_path="$1"
    [ -z "$model_path" ] && return 1
    if [[ "$model_path" == /* ]]; then
        [ -f "$model_path/config.json" ] && python3 -c "import json; json.load(open('$model_path/config.json'))" 2>/dev/null && return 0
        return 1
    fi
    local local_path=$(get_hf_model_local_path "$model_path")
    [ -n "$local_path" ] && [ -f "$local_path/config.json" ] && return 0
    return 1
}

resolve_model_path() {
    local model_key_or_path="$1"
    [[ -n "${MODEL_PATHS[$model_key_or_path]}" ]] && { echo "${MODEL_PATHS[$model_key_or_path]}"; return 0; }
    [[ "$model_key_or_path" == /* ]] && [[ -f "$model_key_or_path/config.json" ]] && { echo "$model_key_or_path"; return 0; }
    local hf_local=$(get_hf_model_local_path "$model_key_or_path")
    [ -n "$hf_local" ] && { echo "$hf_local"; return 0; }
    echo "$model_key_or_path"
}

get_model_template() {
    local model_path="$1"
    for key in "${!MODEL_PATHS[@]}"; do
        [[ "$model_path" == *"${MODEL_PATHS[$key]}"* ]] || [[ "$model_path" == *"$key"* ]] && { echo "${MODEL_TEMPLATES[$key]}"; return 0; }
    done
    local lower_path=$(echo "$model_path" | tr '[:upper:]' '[:lower:]')
    [[ "$lower_path" == *"deepseek"* ]] && { echo "deepseek"; return; }
    [[ "$lower_path" == *"mistral"* ]] && { echo "mistral"; return; }
    [[ "$lower_path" == *"codellama"* ]] && { echo "codellama"; return; }
    [[ "$lower_path" == *"llama"* ]] && { echo "llama3"; return; }
    echo "default"
}

# ===========================================
# 环境配置
# ===========================================

setup_environment() {
    print_step "Setting up Environment"
    
    if ! command -v conda &> /dev/null; then
        log_error "Conda not found. Install Miniconda/Anaconda first."
        exit 1
    fi
    
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        log_info "Environment '${CONDA_ENV_NAME}' exists."
        read -p "Update packages? (Y/n): " choice
        if [[ "$choice" =~ ^[Nn]$ ]]; then
            log_info "Recreating environment..."
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            activate_env
            _install_ccpo_packages
            log_success "Environment updated"
            return 0
        fi
    fi
    
    log_info "Creating conda environment '${CONDA_ENV_NAME}'..."
    if conda env list | grep -q "^${SOURCE_ENV} "; then
        conda create --name ${CONDA_ENV_NAME} --clone ${SOURCE_ENV} -y
    else
        conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    fi
    
    activate_env
    _install_ccpo_packages
    log_success "Environment setup complete: ${CONDA_ENV_NAME}"
}

_install_ccpo_packages() {
    log_info "Installing CCPO packages..."
    pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || true
    pip install -q transformers datasets accelerate peft trl vllm 2>/dev/null || true
    pip install -q pyyaml numpy pandas scipy sentencepiece protobuf 2>/dev/null || true
    pip install -q openai tenacity tqdm huggingface_hub 2>/dev/null || true
    [ -f "$PROJECT_DIR/requirements.txt" ] && pip install -q -r "$PROJECT_DIR/requirements.txt" 2>/dev/null || true
    log_success "Packages installed"
}

prepare_dirs() {
    print_step "Preparing Directories"
    
    check_dir "$OUTPUT_DIR"
    check_dir "$OUTPUT_DIR/logs"
    check_dir "$GENERATED_DIR/iter1"
    check_dir "$RANKING_DIR/generated/iter1"
    check_dir "$PROCESSED_DIR/iter1"
    check_dir "$EVAL_OUTPUT_DIR"
    check_dir "$CCPO_OUTPUT_DIR"
    check_dir "$TRAIN_CONFIG_DIR"
    
    log_info "Project:     $PROJECT_DIR"
    log_info "Output:      $OUTPUT_DIR"
    log_info "CCPO Model:  $CCPO_OUTPUT_DIR (名称: $CCPO_OUTPUT_NAME)"
    log_info "HF Cache:    $HF_CACHE_DIR"
    log_success "Directories prepared"
}

download_base_model() {
    print_step "Downloading/Verifying Base Model"
    
    local model_path=$(resolve_model_path "$BASE_MODEL")
    
    if [[ "$model_path" == /* ]] && [ -f "$model_path/config.json" ]; then
        log_success "Model already cached: $model_path"
        return 0
    fi
    
    log_info "Downloading model: $BASE_MODEL"
    activate_env
    
    python3 << EOF
import os, sys
from huggingface_hub import snapshot_download
import json

model_id = "$BASE_MODEL"
cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

try:
    local_path = snapshot_download(
        repo_id=model_id, 
        cache_dir=cache_dir, 
        resume_download=True
    )
    print(f"Model downloaded: {local_path}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF
    
    [ $? -ne 0 ] && { log_error "Failed to download model"; exit 1; }
    log_success "Model ready: $BASE_MODEL"
}

# ===========================================
# Step 1: 数据生成 (继承MathCoder2 Prompts)
# ===========================================

generate_training_data() {
    print_step "Step 1: Generating Training Data (MathCoder2 Few-Shot Prompts)"
    activate_env
    cd "$PROJECT_DIR"
    
    local limit="${1:-$LIMIT_SAMPLES}"
    local pairs="${2:-$PAIRS}"
    
    log_info "╔═══════════════════════════════════════════════════════════════╗"
    log_info "║  CCPO数据生成 - 继承MathCoder2 Prompts                        ║"
    log_info "╠═══════════════════════════════════════════════════════════════╣"
    log_info "║  Base Model: $BASE_MODEL"
    log_info "║  Model Key:  $MODEL_KEY"
    log_info "║  Samples:    $limit"
    log_info "║  Pairs:      $pairs"
    log_info "║  Output:     $GENERATED_DIR/iter1"
    log_info "║  Few-Shot:   ${USE_FEWSHOT:-true}"
    log_info "╚═══════════════════════════════════════════════════════════════╝"
    
    check_dir "$GENERATED_DIR/iter1"
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
    
    # 优先使用MathCoder2的prompt格式
    if [ "${USE_FEWSHOT:-true}" = "true" ] && [ -f "scripts/generate_with_fewshot.py" ]; then
        log_info "Using generate_with_fewshot.py (MathCoder2 style few-shot prompts)"
        log_info "Prompts来源: $PROMPTS_DATASET"
        
        python scripts/generate_with_fewshot.py \
            --model "$BASE_MODEL" \
            --output_dir "$GENERATED_DIR/iter1" \
            --prompts "$PROMPTS_DATASET" \
            --maxlen $MAX_LEN \
            --pairs $pairs \
            --world_size $WORLD_SIZE \
            --limit_samples $limit \
            --temperature $TEMPERATURE \
            --top_p $TOP_P
    elif [ -f "scripts/generate.py" ]; then
        log_warn "Using generate.py (zero-shot mode)"
        python scripts/generate.py \
            --model "$BASE_MODEL" \
            --output_dir "$GENERATED_DIR/iter1" \
            --prompts "$PROMPTS_DATASET" \
            --maxlen $MAX_LEN \
            --pairs $pairs \
            --world_size $WORLD_SIZE \
            --limit_samples $limit \
            --temperature $TEMPERATURE \
            --top_p $TOP_P
    else
        log_error "No generation script found!"
        exit 1
    fi
    
    local file_count=$(ls -1 "$GENERATED_DIR/iter1"/*.json 2>/dev/null | wc -l)
    log_success "Generated $file_count files in $GENERATED_DIR/iter1"
    cd "$SCRIPT_DIR"
}

# ===========================================
# Step 2: CCPO核心 - 代码一致性排名
# ===========================================

convert_training_data() {
    print_step "Step 2: CCPO Code Verification & Ranking"
    activate_env
    cd "$PROJECT_DIR"
    
    log_info "╔═══════════════════════════════════════════════════════════════╗"
    log_info "║  CCPO 代码一致性验证流程                                      ║"
    log_info "╠═══════════════════════════════════════════════════════════════╣"
    log_info "║  Input: $GENERATED_DIR/iter1"
    log_info "║  Server: $VERIFICATION_BASE_URL"
    log_info "║                                                               ║"
    log_info "║  CCPO核心逻辑:                                                ║"
    log_info "║  1. 从大模型推理中提取答案 (可能有幻觉)                       ║"
    log_info "║  2. 发送推理步骤到代码验证服务                                ║"
    log_info "║  3. 服务端根据推理步骤生成代码并运行                          ║"
    log_info "║  4. 对比: 模型答案 vs 代码执行结果 vs Ground Truth            ║"
    log_info "║                                                               ║"
    log_info "║  Ranking规则:                                                 ║"
    log_info "║  - 推理正确+答案正确 → 高分 (推理步骤准确)                    ║"
    log_info "║  - 推理正确+答案错误 → 次级 (代码验证推理过程)                ║"
    log_info "║  - 都错误 → 低分                                              ║"
    log_info "╚═══════════════════════════════════════════════════════════════╝"
    
    mkdir -p "$RANKING_DIR/generated/iter1"
    mkdir -p "$PROCESSED_DIR/iter1"
    
    local output_subdir="ccpo_${MODEL_KEY}_iter1"
    local gen_link="$PROJECT_DIR/generated/$output_subdir"
    
    rm -f "$gen_link"
    mkdir -p "$PROJECT_DIR/generated"
    mkdir -p "$PROJECT_DIR/ranking"
    ln -sf "$GENERATED_DIR/iter1" "$gen_link"
    
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
    
    if [ -f "scripts/code_verified_rank.py" ]; then
        log_info "Running code_verified_rank.py with code verification..."
        
        local force_restart_flag=""
        [ "${FORCE_RESTART:-false}" = "true" ] && force_restart_flag="--force_restart"
        
        python scripts/code_verified_rank.py \
            --output_dir "generated/$output_subdir" \
            --prompts "$PROMPTS_DATASET" \
            --model "$BASE_MODEL" \
            --pairs $PAIRS \
            --gpu 0 --numgpu 1 --data_frac 0 \
            --verification_url "$VERIFICATION_BASE_URL" \
            --verification_username "$VERIFICATION_USERNAME" \
            --verification_password "$VERIFICATION_PASSWORD" \
            --max_concurrent "${VERIFICATION_BATCH_SIZE:-5}" \
            --verification_sample_rate "${VERIFICATION_SAMPLE_RATE:-0.1}" \
            --enable_answer_scoring --debug_v2 $force_restart_flag
        
        local rank_output="$PROJECT_DIR/ranking/generated/$output_subdir"
        [ -d "$rank_output" ] && cp -r "$rank_output"/* "$RANKING_DIR/generated/iter1/" 2>/dev/null || true
    elif [ -f "scripts/rank.py" ]; then
        log_warn "code_verified_rank.py not found, using rank.py (no code verification)"
        python scripts/rank.py \
            --output_dir "$output_subdir" \
            --prompts "$PROMPTS_DATASET" \
            --model "$BASE_MODEL" \
            --pairs $PAIRS \
            --gpu 0 --numgpu 1
    else
        log_error "No ranking script found!"
        exit 1
    fi
    
    # 构建CCPO训练数据
    log_info "Building CCPO training data from ranking results..."
    export GENERATED_DIR="$GENERATED_DIR"
    export RANKING_DIR="$RANKING_DIR"
    export PROCESSED_DIR="$PROCESSED_DIR"
    export PAIRS="$PAIRS"
    
    _build_ccpo_training_data
    
    local train_file="$PROCESSED_DIR/iter1/train_prefs.jsonl"
    if [ -f "$train_file" ]; then
        local sample_count=$(wc -l < "$train_file")
        log_success "Training data: $sample_count samples"
    else
        log_error "No training data generated"
        exit 1
    fi
    
    rm -f "$gen_link"
    cd "$SCRIPT_DIR"
}

_build_ccpo_training_data() {
    # 从排名分数(.npy)和生成数据构建CCPO偏好训练数据
    # CCPO核心: 使用代码一致性作为偏好信号
    python3 << 'BUILDSCRIPT'
import json
import os
import numpy as np
from pathlib import Path

# 路径配置
generated_dir = os.environ.get('GENERATED_DIR', '') + '/iter1'
ranking_dir = os.environ.get('RANKING_DIR', '') + '/generated/iter1'
output_dir = os.environ.get('PROCESSED_DIR', '') + '/iter1'
pairs = int(os.environ.get('PAIRS', '5'))

print(f"📂 构建CCPO训练数据 (代码一致性偏好优化)")
print(f"   生成数据目录: {generated_dir}")
print(f"   排名结果目录: {ranking_dir}")
print(f"   输出目录: {output_dir}")
print(f"")
print(f"   CCPO核心逻辑:")
print(f"   - 从模型推理中提取答案 (可能有幻觉)")
print(f"   - 从代码执行中提取答案 (实际计算结果)")
print(f"   - 代码正确 vs 模型错误 → 构建偏好对")

# 1. 加载元数据
metadata_file = os.path.join(generated_dir, 'metadata_0.json')
if not os.path.exists(metadata_file):
    print(f"❌ 元数据文件不存在: {metadata_file}")
    exit(1)

with open(metadata_file, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

data_items = metadata.get('data_items', [])
print(f"✅ 加载元数据: {len(data_items)} 个样本")

# 2. 加载所有响应
all_responses = []
for i in range(pairs):
    resp_file = os.path.join(generated_dir, f'responses_{i}.json')
    if os.path.exists(resp_file):
        with open(resp_file, 'r', encoding='utf-8') as f:
            all_responses.append(json.load(f))
        print(f"✅ 加载 responses_{i}.json: {len(all_responses[-1])} 条")

if not all_responses:
    print("❌ 没有找到响应文件")
    exit(1)

# 3. 加载排名分数 (基于代码一致性)
ranks = None
for npy_file in Path(ranking_dir).glob('*.npy'):
    ranks = np.load(npy_file, allow_pickle=True)
    print(f"✅ 加载排名分数: {npy_file.name}, shape: {ranks.shape if hasattr(ranks, 'shape') else 'list'}")
    break

# 4. 构建偏好数据 (CCPO核心)
train_data = []
num_samples = len(data_items)

# 统计信息
stats = {
    'total': 0,
    'code_correct_model_wrong': 0,  # 代码正确，模型错误 (最有价值)
    'both_correct': 0,               # 都正确
    'both_wrong': 0,                 # 都错误
    'model_correct_code_wrong': 0,   # 模型正确，代码错误 (罕见)
}

for idx in range(num_samples):
    # 获取问题和答案
    item = data_items[idx]
    prompt = item.get('prompt', item.get('question', ''))
    ground_truth = item.get('answer', '')
    
    # 获取该样本的所有响应
    responses = []
    for p in range(len(all_responses)):
        if idx < len(all_responses[p]):
            responses.append(all_responses[p][idx])
    
    if len(responses) < 2:
        continue
    
    # 获取排名分数 (代码一致性分数)
    if ranks is not None and idx < len(ranks):
        sample_ranks = ranks[idx]
        if hasattr(sample_ranks, 'tolist'):
            sample_ranks = sample_ranks.tolist()
        elif not isinstance(sample_ranks, list):
            sample_ranks = list(sample_ranks)
    else:
        # 没有排名分数，使用默认顺序
        sample_ranks = list(range(len(responses)))
    
    # 确保排名分数与响应数量匹配
    if len(sample_ranks) < len(responses):
        sample_ranks.extend([0] * (len(responses) - len(sample_ranks)))
    sample_ranks = sample_ranks[:len(responses)]
    
    # 根据排名选择chosen和rejected
    # CCPO: 分数越高表示代码一致性越好 (代码执行正确)
    sorted_indices = sorted(range(len(sample_ranks)), key=lambda i: sample_ranks[i], reverse=True)
    
    best_idx = sorted_indices[0]
    worst_idx = sorted_indices[-1]
    
    chosen_response = responses[best_idx]
    rejected_response = responses[worst_idx]
    
    best_score = float(sample_ranks[best_idx])
    worst_score = float(sample_ranks[worst_idx])
    score_diff = best_score - worst_score
    
    # 统计CCPO对比情况
    stats['total'] += 1
    if best_score > 50 and worst_score < 50:
        stats['code_correct_model_wrong'] += 1
    elif best_score > 50 and worst_score > 50:
        stats['both_correct'] += 1
    elif best_score < 50 and worst_score < 50:
        stats['both_wrong'] += 1
    else:
        stats['model_correct_code_wrong'] += 1
    
    # 计算偏好概率
    if score_diff > 50:
        chosen_prob = 0.95
    elif score_diff > 30:
        chosen_prob = 0.85
    elif score_diff > 10:
        chosen_prob = 0.75
    elif score_diff > 0:
        chosen_prob = 0.65
    else:
        chosen_prob = 0.55
    
    # 构建对话格式
    train_data.append({
        'chosen': [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen_response}
        ],
        'rejected': [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected_response}
        ],
        'chosen_probs': chosen_prob,
        'chosen_probs_win': chosen_prob,
        'chosen_probs_lose': 1 - chosen_prob,
        'chosen_score': best_score,
        'rejected_score': worst_score,
        'score_difference': score_diff,
        'original_index': idx,
        'ccpo_quality': 'high' if score_diff > 30 else ('medium' if score_diff > 10 else 'low')
    })

# 5. 保存训练数据
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'train_prefs.jsonl')

with open(output_file, 'w', encoding='utf-8') as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\n🎉 CCPO训练数据构建完成!")
print(f"   总样本数: {len(train_data)}")
print(f"   输出文件: {output_file}")

# 显示CCPO统计
print(f"\n📊 CCPO代码一致性统计:")
print(f"   代码正确+模型错误: {stats['code_correct_model_wrong']} (最有价值的偏好对)")
print(f"   都正确: {stats['both_correct']}")
print(f"   都错误: {stats['both_wrong']}")
print(f"   模型正确+代码错误: {stats['model_correct_code_wrong']} (罕见)")

# 显示样本统计
if train_data:
    avg_chosen_prob = sum(d['chosen_probs'] for d in train_data) / len(train_data)
    high_quality = sum(1 for d in train_data if d['ccpo_quality'] == 'high')
    medium_quality = sum(1 for d in train_data if d['ccpo_quality'] == 'medium')
    low_quality = sum(1 for d in train_data if d['ccpo_quality'] == 'low')
    
    print(f"\n   平均chosen概率: {avg_chosen_prob:.3f}")
    print(f"   质量分布: high={high_quality}, medium={medium_quality}, low={low_quality}")
BUILDSCRIPT
}

# ===========================================
# Step 3: CCPO训练
# ===========================================

prepare_train_config() {
    local model_path="$1"
    local output_dir="$2"
    local config_file="$TRAIN_CONFIG_DIR/config_ccpo_auto_${MODEL_KEY}.yaml"
    local train_data_file="$PROCESSED_DIR/iter1/train_prefs.jsonl"
    [ ! -f "$train_data_file" ] && train_data_file="$PROCESSED_DIR/iter1/train_prefs.parquet"
    
    echo "[INFO] Creating training config: $config_file" >&2
    
    # 获取模型对应的chat_template
    local chat_template_config=""
    chat_template_config=$(_get_chat_template_config "$MODEL_KEY")
    
    cat > "$config_file" << EOF
model_name_or_path: $model_path
torch_dtype: auto
dataset_mixer:
  "$train_data_file": 1.0
dataset_splits:
- train
preprocessing_num_workers: 4
bf16: true
fp16: false
beta: $BETA
loss_type: ccpo
do_eval: false
eval_strategy: 'no'
gradient_accumulation_steps: $GRAD_ACCUM
gradient_checkpointing: true
learning_rate: $LEARNING_RATE
logging_steps: 10
max_length: $MAX_LENGTH
max_prompt_length: $MAX_PROMPT_LENGTH
num_train_epochs: $NUM_EPOCHS
optim: adamw_torch
output_dir: $output_dir
per_device_train_batch_size: $BATCH_SIZE
save_steps: $SAVE_STEPS
save_total_limit: 3
seed: 42
warmup_steps: $WARMUP_STEPS
remove_unused_columns: false
dataloader_num_workers: 2
report_to: null
logging_first_step: true
use_peft: false
dataloader_pin_memory: false
$chat_template_config
EOF
    
    echo "$config_file"
}

# 获取模型对应的chat_template配置
_get_chat_template_config() {
    local model_key="$1"
    
    # 不同模型的chat_template
    case "$model_key" in
        deepseek)
            # DeepSeekMath使用简单的User/Assistant格式
            echo 'chat_template: "{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}User: {{ message['\''content'\''] }}\n{% elif message['\''role'\''] == '\''assistant'\'' %}Assistant: {{ message['\''content'\''] }}{% if not loop.last %}\n{% endif %}{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"'
            ;;
        mistral)
            # Mistral使用标准的[INST]格式
            echo 'chat_template: "{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}[INST] {{ message['\''content'\''] }} [/INST]{% elif message['\''role'\''] == '\''assistant'\'' %}{{ message['\''content'\''] }}{% if not loop.last %}</s>{% endif %}{% endif %}{% endfor %}"'
            ;;
        codellama)
            # CodeLlama使用与Llama2类似的格式
            echo 'chat_template: "{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}[INST] {{ message['\''content'\''] }} [/INST]{% elif message['\''role'\''] == '\''assistant'\'' %}{{ message['\''content'\''] }}{% if not loop.last %}</s>{% endif %}{% endif %}{% endfor %}"'
            ;;
        llama3|metallama3)
            # Llama3使用更新的格式
            echo 'chat_template: "{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}<|start_header_id|>user<|end_header_id|>\n\n{{ message['\''content'\''] }}<|eot_id|>{% elif message['\''role'\''] == '\''assistant'\'' %}<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['\''content'\''] }}{% if not loop.last %}<|eot_id|>{% endif %}{% endif %}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"'
            ;;
        *)
            # 默认使用简单格式
            echo 'chat_template: "{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}User: {{ message['\''content'\''] }}\n{% elif message['\''role'\''] == '\''assistant'\'' %}Assistant: {{ message['\''content'\''] }}{% if not loop.last %}\n{% endif %}{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"'
            ;;
    esac
}

# 确保tokenizer有chat_template设置
_ensure_chat_template() {
    local model_path="$1"
    local tokenizer_config="$model_path/tokenizer_config.json"
    
    if [ ! -f "$tokenizer_config" ]; then
        log_warn "tokenizer_config.json not found at $model_path"
        return 1
    fi
    
    # 检查是否已经有chat_template
    local has_template=$(python3 -c "
import json
try:
    with open('$tokenizer_config', 'r') as f:
        config = json.load(f)
    if config.get('chat_template'):
        print('yes')
    else:
        print('no')
except:
    print('error')
" 2>/dev/null)
    
    if [ "$has_template" = "yes" ]; then
        log_info "Tokenizer already has chat_template"
        return 0
    fi
    
    log_info "Setting chat_template for $MODEL_KEY model..."
    
    # 根据模型类型设置chat_template
    python3 << PYEOF
import json
import os

model_key = "$MODEL_KEY"
tokenizer_config_path = "$tokenizer_config"

# 定义各模型的chat_template
chat_templates = {
    "deepseek": "{% for message in messages %}{% if message['role'] == 'user' %}User: {{ message['content'] }}\n{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% if not loop.last %}\n{% endif %}{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}",
    
    "mistral": "{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% if not loop.last %}</s>{% endif %}{% endif %}{% endfor %}",
    
    "codellama": "{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% if not loop.last %}</s>{% endif %}{% endif %}{% endfor %}",
    
    "llama3": "{% for message in messages %}{% if message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}{% if not loop.last %}<|eot_id|>{% endif %}{% endif %}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}",
    
    "metallama3": "{% for message in messages %}{% if message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}{% if not loop.last %}<|eot_id|>{% endif %}{% endif %}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
}

# 默认template
default_template = "{% for message in messages %}{% if message['role'] == 'user' %}User: {{ message['content'] }}\n{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% if not loop.last %}\n{% endif %}{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"

try:
    with open(tokenizer_config_path, 'r') as f:
        config = json.load(f)
    
    # 获取对应的chat_template
    template = chat_templates.get(model_key, default_template)
    config['chat_template'] = template
    
    # 备份原始文件
    backup_path = tokenizer_config_path + '.backup'
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy(tokenizer_config_path, backup_path)
        print(f"✓ Backed up original config to {backup_path}")
    
    # 保存更新后的配置
    with open(tokenizer_config_path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ chat_template set for {model_key} model")
    
except Exception as e:
    print(f"✗ Error setting chat_template: {e}")
    exit(1)
PYEOF
    
    log_success "chat_template configured for $MODEL_KEY"
}

run_training() {
    print_step "Step 3: Training CCPO Model"
    activate_env
    cd "$PROJECT_DIR"
    
    local config="${1:-}"
    local model_path=$(resolve_model_path "$BASE_MODEL")
    local output_dir="$CCPO_OUTPUT_DIR"
    
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
    check_dir "$OUTPUT_DIR/logs"
    
    log_info "Base Model: $BASE_MODEL"
    log_info "Output Model: $CCPO_OUTPUT_NAME"
    log_info "Output Dir: $output_dir"
    
    if ! validate_model_path "$model_path"; then
        log_warn "Model not found locally, downloading..."
        download_base_model
        model_path=$(resolve_model_path "$BASE_MODEL")
    fi
    log_success "Model validated: $model_path"
    
    # 检查并设置tokenizer的chat_template
    _ensure_chat_template "$model_path"
    
    local train_file="$PROCESSED_DIR/iter1/train_prefs.jsonl"
    [ ! -f "$train_file" ] && train_file="$PROCESSED_DIR/iter1/train_prefs.parquet"
    [ ! -f "$train_file" ] && { log_error "Training data missing at $train_file"; exit 1; }
    
    log_info "Training data: $train_file"
    
    [ -z "$config" ] || [ ! -f "$config" ] && config=$(prepare_train_config "$model_path" "$output_dir")
    
    if [ ! -f "$config" ]; then
        log_error "Config file not found: $config"
        exit 1
    fi
    
    log_info "Using config: $config"
    
    local log_file="$OUTPUT_DIR/logs/train_ccpo_${MODEL_KEY}_$(date +%Y%m%d_%H%M%S).log"
    log_info "Training log: $log_file"
    log_info "Starting CCPO training..."
    
    python ccpo/run_ccpo.py "$config" 2>&1 | tee "$log_file"
    local train_status=${PIPESTATUS[0]}
    
    cd "$SCRIPT_DIR"
    
    if [ $train_status -eq 0 ]; then
        log_success "Training complete!"
        log_success "Model saved to: $output_dir"
        _update_model_config "$output_dir"
    else
        log_error "Training failed with status $train_status"
        exit $train_status
    fi
}

_update_model_config() {
    local model_dir="$1"
    
    [ -f "$model_dir/config.json" ] && python3 << EOF
import json
try:
    with open("$model_dir/config.json", 'r') as f:
        config = json.load(f)
    config['ccpo_trained'] = True
    config['ccpo_model_name'] = "$CCPO_OUTPUT_NAME"
    config['ccpo_base_model'] = "$BASE_MODEL"
    config['ccpo_model_key'] = "$MODEL_KEY"
    config['ccpo_beta'] = $BETA
    config['ccpo_learning_rate'] = "$LEARNING_RATE"
    with open("$model_dir/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print("✓ Model config updated with CCPO metadata")
except Exception as e:
    print(f"Warning: Could not update config: {e}")
EOF
    
    local link_path="$OUTPUT_DIR/ccpo_latest"
    rm -f "$link_path"
    ln -sf "$model_dir" "$link_path"
    log_info "Created symlink: $link_path -> $model_dir"
}

# ===========================================
# 服务功能
# ===========================================

serve_model() {
    print_step "Starting Model Server"
    activate_env
    
    local model="${1:-$CCPO_OUTPUT_DIR}"
    local port="${2:-$API_PORT}"
    
    if check_server "$port"; then
        log_info "Server already running on port $port"
        return 0
    fi
    
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
    log_info "Model: $model"
    log_info "Port: $port"
    
    python -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --port "$port" \
        --trust-remote-code \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.92 &
    
    echo $! > /tmp/llm4ccpo_server.pid
    wait_for_server "$port" 180
}

stop_server() {
    print_step "Stopping Server"
    
    [ -f /tmp/llm4ccpo_server.pid ] && {
        kill $(cat /tmp/llm4ccpo_server.pid) 2>/dev/null
        rm -f /tmp/llm4ccpo_server.pid
    }
    
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    log_success "Server stopped"
}

test_server() {
    local port="${1:-$API_PORT}"
    
    if ! check_server "$port"; then
        log_error "Server not running on port $port"
        exit 1
    fi
    
    log_success "Server running on port $port"
    curl -s "http://localhost:$port/v1/models" | python -m json.tool 2>/dev/null || echo "Response parsing failed"
}

# ===========================================
# Step 4: 评测 (MathCoder2风格)
# ===========================================

run_eval() {
    print_step "Step 4: Running MathCoder2 Evaluation"
    
    local model_path="${1:-$CCPO_OUTPUT_DIR}"
    local eval_output="${2:-$EVAL_OUTPUT_DIR}"
    
    log_info "╔═══════════════════════════════════════════════════════════════╗"
    log_info "║  MathCoder2 评测流程                                           ║"
    log_info "╠═══════════════════════════════════════════════════════════════╣"
    log_info "║  评测模型: $model_path"
    log_info "║  输出目录: $eval_output"
    log_info "║                                                                ║"
    log_info "║  评测基准:                                                     ║"
    log_info "║    - GSM8K (小学数学应用题)                                    ║"
    log_info "║    - MATH (竞赛级数学)                                         ║"
    log_info "║    - SAT/OCW (SAT数学和MIT课程)                                ║"
    log_info "║    - MMLU-Math (多任务评测)                                    ║"
    log_info "╚═══════════════════════════════════════════════════════════════╝"
    
    check_dir "$eval_output"
    
    if [ ! -f "$model_path/config.json" ]; then
        log_error "Model not found at $model_path"
        exit 1
    fi
    
    export TOKENIZERS_PARALLELISM=true
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
    
    local eval_script_dir="$MATHCODER2_SCRIPTS_DIR"
    
    if [ ! -d "$eval_script_dir" ]; then
        log_error "MathCoder2 evaluation scripts not found at: $eval_script_dir"
        log_error "Please ensure MathCoder2/test/scripts/ exists"
        exit 1
    fi
    
    log_info "Using MathCoder2 evaluation scripts from: $eval_script_dir"
    
    # ==========================================
    # GSM8K 评测
    # ==========================================
    log_info ""
    log_info "📊 [1/5] Evaluating on GSM8K..."
    _run_mammoth_eval_direct "gsm8k" "$model_path" "$eval_output"
    _ensure_mammoth_metrics "$eval_output/gsm8k_mammoth" "gsm8k"
    
    # ==========================================
    # MATH 评测
    # ==========================================
    log_info ""
    log_info "📊 [2/5] Evaluating on MATH..."
    _run_mammoth_eval_direct "math" "$model_path" "$eval_output"
    _ensure_mammoth_metrics "$eval_output/math_mammoth" "math"
    
    # ==========================================
    # MMLU 多任务评测
    # ==========================================
    log_info ""
    log_info "📊 [3/5] Running MMLU multi-task evaluation..."
    _run_lm_eval_direct "$model_path" "$eval_output"
    
    # ==========================================
    # DeepSeekMath 评测 (SAT + OCW)
    # ==========================================
    log_info ""
    log_info "📊 [4/5] Running DeepSeek-Math evaluation (SAT + OCW)..."
    _run_deepseek_eval_direct "math_sat" "$model_path" "$eval_output"
    _run_deepseek_eval_direct "OCWCourses" "$model_path" "$eval_output"
    
    # ==========================================
    # 汇总评测结果
    # ==========================================
    log_info ""
    log_info "📈 [5/5] Summarizing evaluation results..."
    _summarize_eval_results "$eval_output"
    
    _show_eval_summary "$eval_output"
    
    log_success "Evaluation complete!"
    log_info "Results saved to: $eval_output"
}

_run_mammoth_eval_direct() {
    local dataset="$1"
    local model_path="$2"
    local eval_output="$3"
    local output_dir="${eval_output}/${dataset}_mammoth"
    
    mkdir -p "$output_dir"
    
    local mammoth_dir="$MATHCODER2_TEST_DIR/MAmmoTH/math_eval"
    
    if [ ! -d "$mammoth_dir" ]; then
        log_warn "MAmmoTH directory not found: $mammoth_dir"
        return 1
    fi
    
    if [ ! -f "$mammoth_dir/run_open.py" ]; then
        log_warn "run_open.py not found in $mammoth_dir"
        return 1
    fi
    
    log_info "Running MAmmoTH evaluation for $dataset"
    log_info "  Directory: $mammoth_dir"
    log_info "  Model: $model_path"
    log_info "  Output: $output_dir/result.jsonl"
    
    cd "$mammoth_dir"
    
    python run_open.py \
        --model "$model_path" \
        --shots 4 \
        --dataset "$dataset" \
        --form short \
        --output "$output_dir/result.jsonl" 2>&1 || {
        log_warn "$dataset evaluation failed"
        cd "$PROJECT_DIR"
        return 1
    }
    
    cd "$PROJECT_DIR"
    log_success "$dataset evaluation completed"
}

# lm_eval兼容性修复
_fix_lm_eval_compatibility() {
    log_info "Checking lm_eval and transformers compatibility..."
    
    local check_result
    check_result=$(PYTHONWARNINGS=ignore python3 -W ignore -c "
import sys
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

result = 'OK'
try:
    from lm_eval import evaluator
except AttributeError as e:
    if 'Qwen2AudioForConditionalGeneration' in str(e):
        result = 'NEED_FIX_TRANSFORMERS'
    else:
        result = 'OTHER_ERROR'
except ImportError as e:
    result = 'IMPORT_ERROR'
except Exception as e:
    result = 'UNKNOWN_ERROR'

print(result)
" 2>/dev/null)
    
    check_result=$(echo "$check_result" | grep -E '^(OK|NEED_FIX_TRANSFORMERS|OTHER_ERROR|IMPORT_ERROR|UNKNOWN_ERROR)$' | head -1)
    
    if [ -z "$check_result" ]; then
        log_info "Primary check returned empty, trying simple detection..."
        if PYTHONWARNINGS=ignore python3 -W ignore -c "import lm_eval" 2>/dev/null; then
            check_result="OK"
        else
            check_result="IMPORT_ERROR"
        fi
    fi
    
    log_info "Compatibility check result: $check_result"
    
    case "$check_result" in
        "OK")
            log_success "lm_eval and transformers are compatible"
            return 0
            ;;
        "NEED_FIX_TRANSFORMERS")
            log_warn "Detected lm_eval/transformers incompatibility"
            log_info "Trying: pip install -U transformers"
            pip install -U transformers -q 2>&1 | tail -3 || true
            
            local verify_result
            verify_result=$(PYTHONWARNINGS=ignore python3 -W ignore -c "
import warnings
warnings.filterwarnings('ignore')
try:
    from lm_eval import evaluator
    print('FIXED')
except:
    print('STILL_BROKEN')
" 2>/dev/null | grep -E '^(FIXED|STILL_BROKEN)$' | head -1)
            
            if [ "$verify_result" = "FIXED" ]; then
                log_success "transformers upgraded successfully"
                return 0
            fi
            
            log_warn "Standard upgrade didn't work, trying GitHub version..."
            pip install -U git+https://github.com/huggingface/transformers -q 2>&1 | tail -3 || true
            
            verify_result=$(PYTHONWARNINGS=ignore python3 -W ignore -c "
import warnings
warnings.filterwarnings('ignore')
try:
    from lm_eval import evaluator
    print('FIXED')
except:
    print('STILL_BROKEN')
" 2>/dev/null | grep -E '^(FIXED|STILL_BROKEN)$' | head -1)
            
            if [ "$verify_result" = "FIXED" ]; then
                log_success "transformers (GitHub) installed successfully"
                return 0
            fi
            
            log_warn "Trying to downgrade lm_eval to compatible version..."
            pip install lm_eval==0.4.2 -q 2>&1 | tail -3 || true
            return 0
            ;;
        "IMPORT_ERROR")
            log_warn "lm_eval not installed, installing..."
            pip install lm-eval -q 2>&1 | tail -3 || true
            return 0
            ;;
        *)
            log_warn "Unexpected check result: $check_result"
            if PYTHONWARNINGS=ignore python3 -W ignore -c "import lm_eval" 2>/dev/null; then
                log_success "lm_eval import successful despite warnings"
                return 0
            fi
            pip install lm-eval -q 2>&1 | tail -3 || true
            return 0
            ;;
    esac
}

_create_mmlu_placeholder() {
    local output_dir="$1"
    mkdir -p "$output_dir"
    
    cat > "$output_dir/mmlu_placeholder.json" << EOF
{
    "status": "evaluation_failed",
    "reason": "lm_eval_compatibility_issue",
    "note": "MMLU evaluation was skipped"
}
EOF
    log_info "Created MMLU placeholder result file"
}

_run_lm_eval_direct() {
    local model_path="$1"
    local eval_output="$2"
    local output_dir="${eval_output}/lm_eval"
    
    mkdir -p "$output_dir"
    
    if ! command -v lm_eval &> /dev/null; then
        log_warn "lm_eval not found in current environment"
        log_info "Installing lm-eval..."
        pip install lm-eval -q 2>&1 | tail -3 || true
    fi
    
    log_info "Running lm_eval compatibility check and fix..."
    _fix_lm_eval_compatibility
    
    if ! command -v lm_eval &> /dev/null; then
        log_warn "lm_eval still not available after installation"
        _create_mmlu_placeholder "$output_dir"
        return 1
    fi
    
    log_info "Running lm_eval for MMLU"
    log_info "  Model: $model_path"
    log_info "  Output: $output_dir"
    
    export PYTHONWARNINGS=ignore
    export TRANSFORMERS_VERBOSITY=error
    export TF_CPP_MIN_LOG_LEVEL=3
    
    lm_eval --model hf \
        --model_args pretrained="$model_path" \
        --tasks mmlu \
        --device "cuda:$CUDA_DEVICE" \
        --batch_size 8 \
        --output_path "$output_dir" \
        --log_samples 2>&1 || {
        log_warn "MMLU evaluation failed"
        _create_mmlu_placeholder "$output_dir"
        return 1
    }
    
    log_success "MMLU evaluation completed"
}

_run_deepseek_eval_direct() {
    local dataset="$1"
    local model_path="$2"
    local eval_output="$3"
    local output_dir="${eval_output}/${dataset}"
    
    mkdir -p "$output_dir"
    
    local deepseek_eval_dir="$MATHCODER2_TEST_DIR/DeepSeek-Math/evaluation"
    local test_data_dir="$MATHCODER2_TEST_DIR/DeepSeek-Math/test_data/$dataset"
    
    if [ ! -d "$deepseek_eval_dir" ]; then
        log_warn "DeepSeek evaluation directory not found: $deepseek_eval_dir"
        return 1
    fi
    
    if [ ! -f "$deepseek_eval_dir/infer/run_cot_eval.py" ]; then
        log_warn "run_cot_eval.py not found"
        return 1
    fi
    
    if [ ! -d "$test_data_dir" ]; then
        log_warn "Test data not found: $test_data_dir"
        return 1
    fi
    
    log_info "Running DeepSeek-Math evaluation for $dataset"
    log_info "  Data: $test_data_dir"
    log_info "  Model: $model_path"
    log_info "  Output: $output_dir"
    
    _install_deepseek_eval_deps "$deepseek_eval_dir"
    
    local few_shot_prompt=""
    local answer_extraction_fn=""
    local eval_fn=""
    
    if [ "$dataset" = "math_sat" ]; then
        few_shot_prompt="CoTSATPrompt"
        answer_extraction_fn="extract_sat_few_shot_answer"
        eval_fn="eval_math_sat"
    elif [ "$dataset" = "OCWCourses" ]; then
        few_shot_prompt="OCWCoursesPrompt"
        answer_extraction_fn="extract_ocwcourses_few_shot_answer"
        eval_fn="eval_ocwcourses"
    fi
    
    cd "$deepseek_eval_dir"
    
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    
    python infer/run_cot_eval.py \
        --data_dir "$test_data_dir" \
        --save_dir "$output_dir" \
        --model_name_or_path "$model_path" \
        --tokenizer_name_or_path "$model_path" \
        --eval_batch_size 16 \
        --temperature 0.0 \
        --prompt_format few_shot \
        --few_shot_prompt "$few_shot_prompt" \
        --answer_extraction_fn "$answer_extraction_fn" \
        --eval_fn "$eval_fn" \
        --use_vllm 2>&1 || {
        log_warn "$dataset evaluation failed"
        cd "$PROJECT_DIR"
        return 1
    }
    
    cd "$PROJECT_DIR"
    log_success "$dataset evaluation completed"
}

_install_deepseek_eval_deps() {
    local deepseek_eval_dir="$1"
    
    if ! python3 -c "import pebble" 2>/dev/null; then
        log_info "Installing DeepSeek-Math evaluation dependencies..."
        pip install pebble -q 2>&1 | tail -2 || true
    fi
    
    local missing_deps=""
    python3 -c "import sympy" 2>/dev/null || missing_deps="$missing_deps sympy"
    python3 -c "import latex2sympy2" 2>/dev/null || missing_deps="$missing_deps latex2sympy2"
    python3 -c "import antlr4" 2>/dev/null || missing_deps="$missing_deps antlr4-python3-runtime"
    
    if [ -n "$missing_deps" ]; then
        log_info "Installing additional dependencies:$missing_deps"
        pip install $missing_deps -q 2>&1 | tail -3 || true
    fi
    
    if [ -f "$deepseek_eval_dir/requirements.txt" ]; then
        log_info "Installing from DeepSeek requirements.txt..."
        pip install -r "$deepseek_eval_dir/requirements.txt" -q 2>&1 | tail -3 || true
    fi
}

_ensure_mammoth_metrics() {
    local output_dir="$1"
    local dataset="$2"
    
    mkdir -p "$output_dir"
    
    if [ -f "$output_dir/result_metrics.json" ]; then
        log_info "$dataset: result_metrics.json already exists"
        return 0
    fi
    
    if [ -f "$output_dir/result.jsonl" ]; then
        log_info "$dataset: Computing metrics from result.jsonl..."
        
        python3 << PYEOF
import json
import os
import re

output_dir = "$output_dir"
dataset = "$dataset"
result_file = os.path.join(output_dir, "result.jsonl")
metrics_file = os.path.join(output_dir, "result_metrics.json")

correct = 0
total = 0

def normalize_answer(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r'[\$,]', '', s)
    numbers = re.findall(r'-?\d+\.?\d*', s)
    if numbers:
        return numbers[-1]
    return s

with open(result_file, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            total += 1
            
            is_correct = False
            
            if item.get('correct', False) or item.get('is_correct', False):
                is_correct = True
            elif 'score' in item and (item['score'] == 1 or item['score'] == True):
                is_correct = True
            elif 'prediction' in item and 'answer' in item:
                pred = normalize_answer(item.get('prediction', ''))
                ans = normalize_answer(item.get('answer', ''))
                if pred and ans and pred == ans:
                    is_correct = True
            elif 'output' in item and 'answer' in item:
                pred = normalize_answer(item.get('output', ''))
                ans = normalize_answer(item.get('answer', ''))
                if pred and ans and pred == ans:
                    is_correct = True
            elif 'model_output' in item and 'ground_truth' in item:
                pred = normalize_answer(item.get('model_output', ''))
                ans = normalize_answer(item.get('ground_truth', ''))
                if pred and ans and pred == ans:
                    is_correct = True
            
            if is_correct:
                correct += 1
        except json.JSONDecodeError:
            continue

acc = correct / total if total > 0 else 0.0

metrics = {
    'dataset': dataset,
    'total': total,
    'correct': correct,
    'acc': acc,
    'accuracy': acc
}

with open(metrics_file, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2)

print(f"✅ {dataset} Accuracy: {acc:.4f} ({correct}/{total})")
PYEOF
        log_success "$dataset: result_metrics.json generated"
    else
        log_warn "$dataset: No result.jsonl found in $output_dir"
        cat > "$output_dir/result_metrics.json" << EOF
{
  "dataset": "$dataset",
  "total": 0,
  "correct": 0,
  "acc": 0.0,
  "accuracy": 0.0,
  "note": "evaluation failed or skipped"
}
EOF
        log_info "$dataset: Placeholder result_metrics.json created"
    fi
}

_summarize_eval_results() {
    local eval_output="$1"
    local summary_file="${eval_output}/summary.txt"
    local metrics_file="${eval_output}/result_metrics.json"
    
    log_info "Summarizing evaluation results..."
    
    python3 << PYEOF
import json
import os
from glob import glob

eval_output = "$eval_output"
summary_file = "$summary_file"
metrics_file = "$metrics_file"

results = {}

# MAmmoTH结果 (GSM8K, MATH)
for dataset in ["gsm8k", "math"]:
    mf = os.path.join(eval_output, f"{dataset}_mammoth", "result_metrics.json")
    if os.path.exists(mf):
        try:
            with open(mf, 'r') as f:
                data = json.load(f)
            results[dataset] = data.get('acc', data.get('accuracy', 0))
        except Exception as e:
            print(f"Warning: Failed to read {mf}: {e}")
            results[dataset] = None
    else:
        results[dataset] = None

# DeepSeek结果 (SAT, OCW)
for dataset, dirname in [("sat", "math_sat"), ("ocw", "OCWCourses")]:
    for metrics_name in ["metrics.json", "result_metrics.json", "eval_results.json"]:
        mf = os.path.join(eval_output, dirname, metrics_name)
        if os.path.exists(mf):
            try:
                with open(mf, 'r') as f:
                    data = json.load(f)
                results[dataset] = data.get('accuracy', data.get('acc', 0))
                break
            except Exception as e:
                print(f"Warning: Failed to read {mf}: {e}")
    if dataset not in results:
        results[dataset] = None

# MMLU结果
mmlu_dir = os.path.join(eval_output, "lm_eval")
mmlu_files = glob(os.path.join(mmlu_dir, "results*.json"))
if mmlu_files:
    try:
        with open(mmlu_files[0], 'r') as f:
            data = json.load(f)
        
        math_subjects = [
            "mmlu_high_school_statistics",
            "mmlu_high_school_mathematics", 
            "mmlu_elementary_mathematics",
            "mmlu_college_mathematics"
        ]
        
        total_num, total_score = 0, 0
        for subject in math_subjects:
            if subject in data.get("results", {}):
                n = data.get("n-samples", {}).get(subject, {}).get("effective", 0)
                acc = data["results"][subject].get("acc,none", 0)
                if n > 0:
                    total_num += n
                    total_score += n * acc
        
        results["mmlu_math"] = total_score / total_num if total_num > 0 else None
    except Exception as e:
        print(f"Warning: Failed to read MMLU results: {e}")
        results["mmlu_math"] = None
else:
    results["mmlu_math"] = None

# 生成摘要文本
text = "=" * 60 + "\n"
text += "  MathCoder2 Evaluation Results\n"
text += "=" * 60 + "\n\n"

for name, label in [("gsm8k", "GSM8K"), ("math", "MATH"), ("sat", "SAT"), ("ocw", "OCW"), ("mmlu_math", "MMLU-Math")]:
    val = results.get(name)
    if val is not None:
        text += f"  {label:<12}: {val:.4f} ({val*100:.1f}%)\n"
    else:
        text += f"  {label:<12}: N/A (evaluation failed or skipped)\n"

text += "\n" + "=" * 60 + "\n"

# Markdown表格
md_text = "\n## Results Table\n|Dataset|Accuracy|\n|---|---|\n"
for name, label in [("gsm8k", "GSM8K"), ("math", "MATH"), ("sat", "SAT"), ("ocw", "OCW"), ("mmlu_math", "MMLU-Math")]:
    val = results.get(name)
    md_text += f"|{label}|{val:.4f}|\n" if val else f"|{label}|N/A|\n"

# 保存摘要
with open(summary_file, 'w') as f:
    f.write(text)
    f.write("\n" + md_text)

# 保存JSON格式
with open(metrics_file, 'w') as f:
    json.dump(results, f, indent=2)

print(text)
print(f"\nSummary saved to: {summary_file}")
print(f"Metrics saved to: {metrics_file}")
PYEOF
}

_show_eval_summary() {
    local eval_output="$1"
    
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  评测结果摘要                                                  ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    
    if [ -f "$eval_output/summary.txt" ]; then
        echo ""
        cat "$eval_output/summary.txt"
    fi
    
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  MathCoder2 论文报告参考结果 (greedy decoding)                ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  Model                    │ GSM8K  │ MATH   │ CMATH  │ SAT   ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  MathCoder2-DeepSeekMath  │ 83.9%  │ 48.9%  │ 79.6%  │ 81.3% ║"
    echo "║  MathCoder2-Llama-3-8B    │ 80.2%  │ 45.2%  │ 68.5%  │ 75.0% ║"
    echo "║  MathCoder2-Mistral-7B    │ 71.8%  │ 32.7%  │ 60.9%  │ 65.6% ║"
    echo "║  MathCoder2-CodeLlama-7B  │ 67.4%  │ 28.9%  │ 54.9%  │ 59.4% ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
}

# ===========================================
# v7.0 新增: 批量处理所有四个模型
# ===========================================

# 训练所有四个MathCoder2模型
train_all_models() {
    print_step "Training All MathCoder2 Models with CCPO"
    local limit="${1:-$LIMIT_SAMPLES}"
    local start_time=$(date +%s)
    
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  CCPO 批量训练所有MathCoder2模型                               ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  模型列表: ${MATHCODER2_MODELS[*]}"
    echo "║  训练样本: $limit"
    echo "║  训练轮数: $NUM_EPOCHS"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    
    local success_models=()
    local failed_models=()
    
    for model_key in "${MATHCODER2_MODELS[@]}"; do
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "  Training model: $model_key"
        echo "═══════════════════════════════════════════════════════════════"
        
        switch_model "$model_key"
        
        if [ -f "$CCPO_OUTPUT_DIR/config.json" ]; then
            log_info "Model $model_key already trained, skipping..."
            log_info "To retrain, remove: $CCPO_OUTPUT_DIR"
            success_models+=("$model_key (skipped)")
            continue
        fi
        
        if run_full_train_single "$limit"; then
            success_models+=("$model_key")
        else
            failed_models+=("$model_key")
        fi
    done
    
    local duration=$(($(date +%s) - start_time))
    
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  批量训练完成                                                  ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  总耗时: $((duration/60))m $((duration%60))s"
    echo "║  成功: ${#success_models[@]} (${success_models[*]})"
    echo "║  失败: ${#failed_models[@]} (${failed_models[*]:-无})"
    echo "╚═══════════════════════════════════════════════════════════════╝"
}

# 单个模型的完整训练流程 (内部使用)
run_full_train_single() {
    local limit="${1:-$LIMIT_SAMPLES}"
    
    prepare_dirs || return 1
    download_base_model || return 1
    generate_training_data "$limit" || return 1
    convert_training_data || return 1
    run_training || return 1
    
    return 0
}

# 评测所有训练好的模型
eval_all_models() {
    print_step "Evaluating All CCPO Trained Models"
    local start_time=$(date +%s)
    
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  评测所有CCPO训练模型                                          ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  模型列表: ${MATHCODER2_MODELS[*]}"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    
    local success_models=()
    local failed_models=()
    
    for model_key in "${MATHCODER2_MODELS[@]}"; do
        switch_model "$model_key"
        
        if [ ! -f "$CCPO_OUTPUT_DIR/config.json" ]; then
            log_warn "Model $model_key not found at $CCPO_OUTPUT_DIR, skipping..."
            failed_models+=("$model_key (not trained)")
            continue
        fi
        
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "  Evaluating model: $model_key"
        echo "═══════════════════════════════════════════════════════════════"
        
        if run_eval "$CCPO_OUTPUT_DIR" "$EVAL_OUTPUT_DIR"; then
            success_models+=("$model_key")
        else
            failed_models+=("$model_key")
        fi
    done
    
    local duration=$(($(date +%s) - start_time))
    
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  批量评测完成                                                  ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  总耗时: $((duration/60))m $((duration%60))s"
    echo "║  成功: ${#success_models[@]} (${success_models[*]})"
    echo "║  失败: ${#failed_models[@]} (${failed_models[*]:-无})"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    
    # 生成对比报告
    compare_results
}

# 生成所有模型的对比报告
compare_results() {
    print_step "Generating Comparison Report for All Models"
    
    local report_file="$DATA_DIR/ccpo_comparison_report.md"
    local json_report="$DATA_DIR/ccpo_comparison_report.json"
    
    python3 << PYEOF
import json
import os
from datetime import datetime

data_dir = "$DATA_DIR"
mathcoder2_models = ["deepseek", "mistral", "codellama", "llama3"]
model_sizes = {"deepseek": "7b", "mistral": "7b", "codellama": "7b", "llama3": "8b"}

# 论文参考结果
paper_results = {
    "deepseek": {"gsm8k": 0.839, "math": 0.489, "sat": 0.813},
    "mistral": {"gsm8k": 0.718, "math": 0.327, "sat": 0.656},
    "codellama": {"gsm8k": 0.674, "math": 0.289, "sat": 0.594},
    "llama3": {"gsm8k": 0.802, "math": 0.452, "sat": 0.750}
}

all_results = {}

for model_key in mathcoder2_models:
    model_size = model_sizes[model_key]
    eval_dir = os.path.join(data_dir, f"eval_results_ccpo_{model_key}")
    metrics_file = os.path.join(eval_dir, "result_metrics.json")
    
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                results = json.load(f)
            all_results[model_key] = results
        except:
            all_results[model_key] = None
    else:
        all_results[model_key] = None

# 生成Markdown报告
report = f"""# CCPO训练结果对比报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 模型评测结果

| Model | GSM8K | MATH | SAT | OCW | MMLU-Math |
|-------|-------|------|-----|-----|-----------|
"""

for model_key in mathcoder2_models:
    model_name = f"CCPO-{model_key.capitalize()}"
    results = all_results.get(model_key)
    
    if results:
        gsm8k = f"{results.get('gsm8k', 0)*100:.1f}%" if results.get('gsm8k') else "N/A"
        math = f"{results.get('math', 0)*100:.1f}%" if results.get('math') else "N/A"
        sat = f"{results.get('sat', 0)*100:.1f}%" if results.get('sat') else "N/A"
        ocw = f"{results.get('ocw', 0)*100:.1f}%" if results.get('ocw') else "N/A"
        mmlu = f"{results.get('mmlu_math', 0)*100:.1f}%" if results.get('mmlu_math') else "N/A"
    else:
        gsm8k = math = sat = ocw = mmlu = "未评测"
    
    report += f"| {model_name} | {gsm8k} | {math} | {sat} | {ocw} | {mmlu} |\n"

report += """
## MathCoder2论文参考结果 (greedy decoding)

| Model | GSM8K | MATH | SAT |
|-------|-------|------|-----|
| MathCoder2-DeepSeekMath | 83.9% | 48.9% | 81.3% |
| MathCoder2-Llama-3-8B | 80.2% | 45.2% | 75.0% |
| MathCoder2-Mistral-7B | 71.8% | 32.7% | 65.6% |
| MathCoder2-CodeLlama-7B | 67.4% | 28.9% | 59.4% |

## 与基准对比

"""

for model_key in mathcoder2_models:
    results = all_results.get(model_key)
    paper = paper_results.get(model_key, {})
    
    if results and paper:
        report += f"### {model_key.capitalize()}\n"
        
        for metric in ["gsm8k", "math", "sat"]:
            ccpo_val = results.get(metric)
            paper_val = paper.get(metric)
            
            if ccpo_val and paper_val:
                diff = (ccpo_val - paper_val) * 100
                sign = "+" if diff >= 0 else ""
                report += f"- {metric.upper()}: CCPO {ccpo_val*100:.1f}% vs Paper {paper_val*100:.1f}% ({sign}{diff:.1f}%)\n"
        
        report += "\n"

# 保存报告
with open("$report_file", 'w') as f:
    f.write(report)

# 保存JSON
with open("$json_report", 'w') as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "ccpo_results": all_results,
        "paper_reference": paper_results
    }, f, indent=2)

print(report)
print(f"\n报告已保存到:")
print(f"  - Markdown: $report_file")
print(f"  - JSON: $json_report")
PYEOF
}

# 完整流水线 - 训练所有模型 + 评测
run_all_models_pipeline() {
    print_step "Complete Pipeline: Train and Evaluate All Models"
    local limit="${1:-$LIMIT_SAMPLES}"
    local start_time=$(date +%s)
    
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  CCPO 完整流水线 - 训练并评测所有MathCoder2模型               ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  模型列表: ${MATHCODER2_MODELS[*]}"
    echo "║  训练样本: $limit"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    
    train_all_models "$limit"
    eval_all_models
    
    local duration=$(($(date +%s) - start_time))
    
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  完整流水线完成                                                ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  总耗时: $((duration/3600))h $((duration%3600/60))m $((duration%60))s"
    echo "║  对比报告: $DATA_DIR/ccpo_comparison_report.md"
    echo "╚═══════════════════════════════════════════════════════════════╝"
}

# ===========================================
# 流水线命令 (单模型)
# ===========================================

run_full_train() {
    print_step "Full Training Pipeline"
    local limit="${1:-$LIMIT_SAMPLES}"
    local start_time=$(date +%s)
    
    log_info "Starting full training pipeline"
    log_info "Model: $MODEL_KEY"
    log_info "Output: $CCPO_OUTPUT_NAME"
    log_info "Samples: $limit"
    
    prepare_dirs
    download_base_model
    generate_training_data "$limit"
    convert_training_data
    run_training
    
    local duration=$(($(date +%s) - start_time))
    log_success "Full training complete in $((duration/60))m $((duration%60))s"
}

run_full_eval() {
    print_step "Full Evaluation Pipeline"
    local model_path="${1:-$CCPO_OUTPUT_DIR}"
    run_eval "$model_path"
    log_success "Full evaluation complete!"
}

run_full_pipeline() {
    print_step "Complete CCPO Pipeline (Train + Eval)"
    local limit="${1:-$LIMIT_SAMPLES}"
    local start_time=$(date +%s)
    
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  CCPO Pipeline Configuration                                   ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  Model Key:      $MODEL_KEY"
    echo "║  Base Model:     $(basename $BASE_MODEL)"
    echo "║  Output Name:    $CCPO_OUTPUT_NAME"
    echo "║  Samples:        $limit"
    echo "║  Pairs:          $PAIRS"
    echo "║  Epochs:         $NUM_EPOCHS"
    echo "║  Learning Rate:  $LEARNING_RATE"
    echo "║  Beta:           $BETA"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "CCPO核心流程:"
    echo "  1. 数据生成 (MathCoder2 few-shot prompts)"
    echo "  2. 代码验证排名 (提取模型答案 vs 代码执行结果)"
    echo "  3. CCPO训练 (代码一致性偏好优化)"
    echo "  4. MathCoder2评测 (GSM8K, MATH, SAT, OCW, MMLU)"
    echo ""
    
    run_full_train "$limit"
    run_full_eval "$CCPO_OUTPUT_DIR"
    
    local duration=$(($(date +%s) - start_time))
    
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  Pipeline Complete!                                            ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  Total Time:     $((duration/60))m $((duration%60))s"
    echo "║  Model Output:   $CCPO_OUTPUT_NAME"
    echo "║  Model Path:     $CCPO_OUTPUT_DIR"
    echo "║  Eval Results:   $EVAL_OUTPUT_DIR"
    echo "╚═══════════════════════════════════════════════════════════════╝"
}

train_only() {
    print_step "Training Only (Skip Data Generation)"
    local config="${1:-}"
    local train_file="$PROCESSED_DIR/iter1/train_prefs.jsonl"
    [ ! -f "$train_file" ] && train_file="$PROCESSED_DIR/iter1/train_prefs.parquet"
    
    if [ ! -f "$train_file" ]; then
        log_error "Training data not found!"
        log_info "Expected at: $PROCESSED_DIR/iter1/train_prefs.jsonl"
        log_info "Run 'generate' and 'convert' first, or use 'full_train'"
        exit 1
    fi
    
    log_info "Using existing training data: $train_file"
    run_training "$config"
}

eval_only() {
    print_step "Evaluation Only"
    local model_path="${1:-$CCPO_OUTPUT_DIR}"
    run_eval "$model_path"
}

eval_base_model() {
    print_step "Evaluating Base Model (for comparison)"
    local model_path="${1:-$BASE_MODEL}"
    local eval_output="${2:-$DATA_DIR/eval_results_base_${MODEL_KEY}}"
    log_info "Evaluating base model for comparison"
    log_info "Model: $model_path"
    log_info "Output: $eval_output"
    run_eval "$model_path" "$eval_output"
    log_success "Base model evaluation complete!"
}

# ===========================================
# 状态显示
# ===========================================

show_data_status() {
    print_step "Data Status"
    
    echo "Generated Data: $GENERATED_DIR"
    if [ -d "$GENERATED_DIR/iter1" ]; then
        local file_count=$(ls -1 "$GENERATED_DIR/iter1"/*.json 2>/dev/null | wc -l)
        echo "  Files: $file_count"
        ls -la "$GENERATED_DIR/iter1"/*.json 2>/dev/null | head -5 || true
    else
        echo "  (not generated)"
    fi
    
    echo ""
    echo "Processed Data: $PROCESSED_DIR"
    if [ -d "$PROCESSED_DIR/iter1" ]; then
        ls -la "$PROCESSED_DIR/iter1"/* 2>/dev/null | head -3 || echo "  (empty)"
    else
        echo "  (not processed)"
    fi
    
    echo ""
    echo "Trained Model: $CCPO_OUTPUT_DIR"
    if [ -f "$CCPO_OUTPUT_DIR/config.json" ]; then
        echo "  ✓ Model trained"
        ls -la "$CCPO_OUTPUT_DIR"/*.safetensors 2>/dev/null | head -3 || true
    else
        echo "  ✗ Not trained"
    fi
    
    echo ""
    echo "Evaluation: $EVAL_OUTPUT_DIR"
    if [ -d "$EVAL_OUTPUT_DIR" ]; then
        ls -la "$EVAL_OUTPUT_DIR"/* 2>/dev/null | head -3 || echo "  (empty)"
    else
        echo "  (not evaluated)"
    fi
}

show_all_models_status() {
    print_step "All Models Status"
    
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  所有MathCoder2模型的CCPO训练状态                              ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    
    printf "%-12s %-8s %-10s %-10s %-10s\n" "Model" "Size" "Trained" "Generated" "Evaluated"
    echo "─────────────────────────────────────────────────────────────────"
    
    for model_key in "${MATHCODER2_MODELS[@]}"; do
        local size="${MODEL_SIZES[$model_key]}"
        local ccpo_dir="$OUTPUT_DIR/ccpo_${model_key}_${size}"
        local gen_dir="$DATA_DIR/generated_ccpo_${model_key}"
        local eval_dir="$DATA_DIR/eval_results_ccpo_${model_key}"
        
        local trained="✗"
        local generated="✗"
        local evaluated="✗"
        
        [ -f "$ccpo_dir/config.json" ] && trained="✓"
        [ -d "$gen_dir/iter1" ] && generated="✓"
        [ -f "$eval_dir/result_metrics.json" ] && evaluated="✓"
        
        printf "%-12s %-8s %-10s %-10s %-10s\n" "$model_key" "$size" "$trained" "$generated" "$evaluated"
    done
    
    echo ""
}

show_config() {
    print_step "Current Configuration"
    
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Model Settings"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  MODEL_KEY:          $MODEL_KEY"
    echo "  BASE_MODEL:         $BASE_MODEL"
    echo "  CCPO_OUTPUT_NAME:   $CCPO_OUTPUT_NAME"
    echo "  CCPO_OUTPUT_DIR:    $CCPO_OUTPUT_DIR"
    echo ""
    echo "  Available Models:"
    for key in "${!MODEL_PATHS[@]}"; do
        echo "    - $key: ${MODEL_PATHS[$key]}"
    done
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Training Settings"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  LIMIT_SAMPLES:      $LIMIT_SAMPLES"
    echo "  PAIRS:              $PAIRS"
    echo "  LEARNING_RATE:      $LEARNING_RATE"
    echo "  BETA:               $BETA"
    echo "  NUM_EPOCHS:         $NUM_EPOCHS"
    echo "  BATCH_SIZE:         $BATCH_SIZE"
    echo "  GRAD_ACCUM:         $GRAD_ACCUM"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Code Verification (CCPO核心)"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  VERIFICATION_URL:   $VERIFICATION_BASE_URL"
    echo "  ENABLE_CODE_VERIFY: $ENABLE_CODE_VERIFICATION"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Environment"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  CONDA_ENV:          $CONDA_ENV_NAME"
    echo "  CUDA_DEVICE:        $CUDA_DEVICE"
    echo "  HF_CACHE:           $HF_CACHE_DIR"
    echo "  PROJECT_DIR:        $PROJECT_DIR"
    echo "═══════════════════════════════════════════════════════════════"
}

quick_fix() {
    print_step "Quick Fix"
    activate_env
    _install_ccpo_packages
    
    log_info "Fixing lm_eval compatibility..."
    _fix_lm_eval_compatibility
    
    prepare_dirs
    log_success "Quick fix complete"
}

fix_lm_eval() {
    print_step "Fix lm_eval Compatibility"
    activate_env
    _fix_lm_eval_compatibility
    log_success "lm_eval compatibility fix complete"
}

clean_data() {
    print_step "Clean Data"
    
    echo "This will delete:"
    echo "  - Generated data: $GENERATED_DIR"
    echo "  - Ranking data: $RANKING_DIR"
    echo "  - Processed data: $PROCESSED_DIR"
    echo "  - Trained model: $CCPO_OUTPUT_DIR"
    echo ""
    
    read -p "Are you sure? (y/N): " resp
    [[ ! "$resp" =~ ^[Yy]$ ]] && { log_info "Cancelled"; return; }
    
    rm -rf "$GENERATED_DIR" "$RANKING_DIR" "$PROCESSED_DIR" "$CCPO_OUTPUT_DIR"
    log_success "Data cleaned"
}

rename_model() {
    print_step "Rename Model Directory"
    
    local src="${1:-}"
    local dst="${2:-}"
    
    if [ -z "$src" ] || [ -z "$dst" ]; then
        log_info "Usage: ./llm4ccpo.sh rename <source_dir> <dest_dir>"
        return 1
    fi
    
    if [ ! -d "$src" ]; then
        log_error "Source directory not found: $src"
        return 1
    fi
    
    log_info "Renaming: $src -> $dst"
    mv "$src" "$dst"
    
    if [ -f "$dst/config.json" ]; then
        local new_name=$(basename "$dst")
        python3 << EOF
import json
try:
    with open("$dst/config.json", 'r') as f:
        config = json.load(f)
    config['ccpo_model_name'] = "$new_name"
    with open("$dst/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Updated model name to: $new_name")
except Exception as e:
    print(f"Warning: Could not update config: {e}")
EOF
    fi
    
    log_success "Model renamed to: $dst"
}

# ===========================================
# 帮助
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CCPO - CCPO强化学习MathCoder2训练流水线 (v7.0 - 全模型支持版)
======================================================================

[v7.0新增功能]:
  - 支持所有四个MathCoder2模型 (deepseek, mistral, codellama, llama3)
  - 批量训练: ./llm4ccpo.sh all_models
  - 批量评测: ./llm4ccpo.sh eval_all_models  
  - 对比报告: ./llm4ccpo.sh compare_results

CCPO核心思想 (Code-Consistency Preference Optimization):
  Step 1: 从大模型推理过程中提取答案 (模型可能算错，存在幻觉)
  Step 2: 将推理步骤发送给代码验证服务，获取代码执行结果
  Step 3: 对比模型答案和代码执行结果:
      示例: 模型答案='50', 代码执行='3', 标准答案='3'
      判断: 模型错误 ❌, 代码正确 ✓
  Step 4: 构建偏好对: 代码一致的响应为chosen, 不一致的为rejected

Usage: ./llm4ccpo.sh <command> [args...]

BATCH COMMANDS (v7.0新增 - 处理所有模型):
  all_models [n]      训练所有四个MathCoder2模型 (n samples)
  eval_all_models     评测所有已训练的模型
  compare_results     生成所有模型的对比报告
  all_status          显示所有模型的状态

SINGLE MODEL COMMANDS:
  full [n]            完整流水线: Train + Eval (当前模型)
  full_train [n]      完整训练: Generate → Convert → Train
  full_eval [model]   完整评测

SETUP COMMANDS:
  setup               设置conda环境
  prepare             创建目录
  config              显示当前配置
  fix                 快速修复包依赖
  fix_lm_eval         修复lm_eval兼容性
  clean               清除生成数据
  status              显示数据状态

DATA COMMANDS:
  generate [n]        生成训练数据 (n samples)
  convert             转换和排名训练数据 (代码验证)

TRAINING COMMANDS:
  train [config]      运行CCPO训练
  train_only          仅训练 (使用现有数据)

EVALUATION COMMANDS:
  eval [model]        运行完整MathCoder2评测
  eval_only [model]   仅评测
  eval_base           评测基础模型 (用于对比)

SUPPORTED MODELS (MODEL_KEY):
  deepseek    MathCoder2-DeepSeekMath-7B (default)
  mistral     MathCoder2-Mistral-7B
  codellama   MathCoder2-CodeLlama-7B
  llama3      MathCoder2-Llama-3-8B

ENVIRONMENT VARIABLES:
  LIMIT_SAMPLES       训练样本数 (default: 100)
  MODEL_KEY           模型选择 (default: deepseek)
  LEARNING_RATE       学习率 (default: 1.0e-5)
  BETA                CCPO beta参数 (default: 0.05)
  NUM_EPOCHS          训练轮数 (default: 10)
  CUDA_DEVICE         GPU设备ID (default: 0)
  FORCE_RESTART       强制重新验证所有样本 (default: false)

EXAMPLES:
  # 快速测试 - 单模型100样本
  LIMIT_SAMPLES=100 ./llm4ccpo.sh full

  # 训练所有四个模型
  LIMIT_SAMPLES=100 ./llm4ccpo.sh all_models

  # 评测所有模型并生成报告
  ./llm4ccpo.sh eval_all_models

  # 使用Mistral模型
  MODEL_KEY=mistral ./llm4ccpo.sh full

  # 强制重新验证
  FORCE_RESTART=true LIMIT_SAMPLES=100 ./llm4ccpo.sh full

MATHCODER2 PAPER REFERENCE RESULTS (greedy decoding):
  Model                    | GSM8K  | MATH   | SAT
  MathCoder2-DeepSeekMath  | 83.9%  | 48.9%  | 81.3%
  MathCoder2-Llama-3-8B    | 80.2%  | 45.2%  | 75.0%
  MathCoder2-Mistral-7B    | 71.8%  | 32.7%  | 65.6%
  MathCoder2-CodeLlama-7B  | 67.4%  | 28.9%  | 59.4%

EOF
}

# ===========================================
# 主入口
# ===========================================

main() {
    print_header
    
    local cmd="${1:-help}"
    shift 2>/dev/null || true
    
    case $cmd in
        # Setup
        setup)              setup_environment ;;
        prepare)            prepare_dirs ;;
        config)             show_config ;;
        fix)                quick_fix ;;
        fix_lm_eval)        fix_lm_eval ;;
        clean)              clean_data ;;
        status)             show_data_status ;;
        all_status)         show_all_models_status ;;
        
        # Model management
        download_base)      download_base_model ;;
        rename)             rename_model "$@" ;;
        switch)             switch_model "$@" ;;
        
        # Data generation
        generate|gen)       generate_training_data "$@" ;;
        convert)            convert_training_data ;;
        
        # Training
        train)              run_training "$@" ;;
        train_only)         train_only "$@" ;;
        
        # Serving
        serve)              serve_model "$@" ;;
        stop)               stop_server ;;
        test_server)        test_server "$@" ;;
        
        # Evaluation
        eval)               run_eval "$@" ;;
        eval_only)          eval_only "$@" ;;
        eval_base)          eval_base_model "$@" ;;
        
        # Single model pipeline
        full_train)         run_full_train "$@" ;;
        full_eval)          run_full_eval "$@" ;;
        full|all)           run_full_pipeline "$@" ;;
        
        # v7.0 Batch commands
        all_models)         run_all_models_pipeline "$@" ;;
        train_all)          train_all_models "$@" ;;
        eval_all_models)    eval_all_models ;;
        compare|compare_results) compare_results ;;
        
        # Help
        help|--help|-h)     show_help ;;
        
        *)
            log_error "Unknown command: $cmd"
            show_help
            exit 1
            ;;
    esac
}

main "$@"