#!/bin/bash
# ==============================================================================
# OPTIMIZATION RUN SCRIPT
# Runs main.py and generates a run summary.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# ==============================================================================
# DEFAULT CONSTANTS (Mapped to directory tree)
# ==============================================================================
MAIN_SCRIPT="./src/main.py"
SUMMARY_SCRIPT="./src/summarize.py"

SCENARIO_CONFIG="./scenario/multigpu.json"
GA_CONFIG="./scenario/ga1.json"
MODEL_FILE="./src/release_2026_09_13/variable_die_domain_model.pt"
OUTPUT_BASE_DIR="./runs"
SEED=42
RANDOM_SEED=false
SUMMARY_ONLY=false
TARGET_RUN_DIR=""

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
print_help() {
    echo "Usage: ./run.sh [OPTIONS]"
    echo ""
    echo "Runs Optimization Pipeline with specified arguments. Uses defaults if none are provided."
    echo ""
    echo "Configuration Options:"
    echo "  --scenario FILE    Path to scenario file (default: $SCENARIO_CONFIG)"
    echo "  --ga-config FILE   Path to GA config file (default: $GA_CONFIG)"
    echo "  --model FILE       Path to ML model file (default: $MODEL_FILE)"
    echo "  --output-dir DIR   Base directory for run outputs (default: $OUTPUT_BASE_DIR)"
    echo "  --seed INT         Random seed for optimization (default: $SEED)"
    echo "  --random-seed      Generate and use a random integer seed"
    echo "  --summary-only     Refresh summary.json across all runs in output-dir"
    echo "  --run-dir DIR      Target a specific run directory (optional with --summary-only)"
    echo "  -h, --help         Show this help message and exit"
    echo ""
    echo "Examples:"
    echo "  ./run.sh --seed 123"
    echo "  ./run.sh --summary-only"
    echo "  ./run.sh --summary-only --run-dir ./runs/run_20260913_120000"
}

format_time() {
    local T=$1
    local H=$((T / 3600))
    local M=$(((T % 3600) / 60))
    local S=$((T % 60))
    printf "%02dh:%02dm:%02ds\n" $H $M $S
}

# ==============================================================================
# PARSE COMMAND-LINE FLAGS
# ==============================================================================
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --scenario) SCENARIO_CONFIG="$2"; shift ;;
        --ga-config) GA_CONFIG="$2"; shift ;;
        --model) MODEL_FILE="$2"; shift ;;
        --output-dir) OUTPUT_BASE_DIR="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --random-seed) RANDOM_SEED=true ;;
        --summary-only) SUMMARY_ONLY=true ;;
        --run-dir) TARGET_RUN_DIR="$2"; shift ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# ==============================================================================
# RE-RUN SUMMARY ONLY MODE (BATCH REFRESH)
# ==============================================================================
if [ "$SUMMARY_ONLY" = true ]; then
    SEARCH_DIR="${TARGET_RUN_DIR:-$OUTPUT_BASE_DIR}"

    if [ ! -d "$SEARCH_DIR" ]; then
        echo "Error: Directory '$SEARCH_DIR' does not exist."
        exit 1
    fi

    # Locate all run.h5 files within the search directory
    mapfile -t H5_FILES < <(find "$SEARCH_DIR" -type f -name "run.h5")

    if [ ${#H5_FILES[@]} -eq 0 ]; then
        echo "Error: No 'run.h5' files found under '$SEARCH_DIR'."
        exit 1
    fi

    echo "========================================"
    echo " Refreshing Summaries across Runs"
    echo " Search Directory:  $SEARCH_DIR"
    echo " Total Runs Found:  ${#H5_FILES[@]}"
    echo "========================================"

    START_OPT=$(date +%s)
    COUNT=0

    for H5_OUT in "${H5_FILES[@]}"; do
        COUNT=$((COUNT + 1))
        RUN_DIR=$(dirname "$H5_OUT")
        SUMMARY_OUT="${RUN_DIR}/summary.json"

        echo "[$COUNT/${#H5_FILES[@]}] Processing: $RUN_DIR"
        python3 "$SUMMARY_SCRIPT" --input "$H5_OUT" --output "$SUMMARY_OUT"
    done

    END_OPT=$(date +%s)
    TIME_OPT=$((END_OPT - START_OPT))

    echo "========================================"
    echo " Batch Summary Update Complete."
    echo " Refreshed: ${#H5_FILES[@]} summary file(s)"
    echo " Total Runtime: $(format_time $TIME_OPT)"
    echo "========================================"
    exit 0
fi

# ==============================================================================
# FULL PIPELINE SETUP & EXECUTION
# ==============================================================================
if [ "$RANDOM_SEED" = true ]; then
    SEED=$((1 + RANDOM % 100000))
fi

# Define run ID (use Slurm Job ID if available, otherwise timestamp)
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OUTPUT_BASE_DIR}/run_${RUN_ID}"

# Define output files
H5_OUT="${RUN_DIR}/run.h5"
SUMMARY_OUT="${RUN_DIR}/summary.json"

mkdir -p "$RUN_DIR"

# Copy and rename configuration files to the run directory
cp "$SCENARIO_CONFIG" "${RUN_DIR}/chiplets.json"
cp "$GA_CONFIG" "${RUN_DIR}/gaconfig.json"

echo "========================================"
echo " Starting Optimization Pipeline"
echo " Run Directory:    $RUN_DIR"
echo " Random Seed:      $SEED"
echo "========================================"

START_OPT=$(date +%s)

# 1. Execute Optimizer
echo "[1/2] Running main optimizer..."
python3 "$MAIN_SCRIPT" \
    "$SCENARIO_CONFIG" \
    "$GA_CONFIG" \
    "$MODEL_FILE" \
    "$SEED" \
    "$H5_OUT"

# 2. Generate Summary
echo "[2/2] Extracting summary metrics..."
python3 "$SUMMARY_SCRIPT" --input "$H5_OUT" --output "$SUMMARY_OUT"

END_OPT=$(date +%s)
TIME_OPT=$((END_OPT - START_OPT))

echo "========================================"
echo " Pipeline Complete."
echo " Total Runtime: $(format_time $TIME_OPT)"
echo " Outputs saved to: $RUN_DIR"
echo "========================================"
