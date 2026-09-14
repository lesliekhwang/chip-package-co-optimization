#!/bin/bash
# ==============================================================================
# UNIFIED PIPELINE SCRIPT
# Runs Optimization (main.py), Analysis (analyze.py), and Animation (animate.py)
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# ==============================================================================
# DEFAULT CONSTANTS (Override these via command-line flags)
# ==============================================================================
# Script Paths
MAIN_SCRIPT="./src/main.py"
ANZ_SCRIPT="./src/analyze.py"
ANI_SCRIPT="./src/animate.py"

# File & Directory Paths
SCENARIO_FILE="./scenario/demo1.toml"
OUTPUT_PATH="output"
HISTORY_FNAME="history.parquet"
MODEL_PATH="./src/release_2026_08_31/variable_die_full.pt"

# Pipeline parameters
POP_SIZE=64
N_GENS=128
SEED=42
MUT_X=0.2
MUT_Y=0.2
MUT_WS=0.2
WS_MAX=5.0
GRID_SIZE=100
DPI=72
INTERVAL_MS=200
EXPORT_CSV="true"

# Pipeline execution flags (default to false, enabled via args)
RUN_OPT=false
RUN_ANZ=false
RUN_ANI=false

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
print_help() {
    echo "Usage: ./run.sh [STAGES] [OPTIONS]"
    echo ""
    echo "Pipeline Stages (select one or more):"
    echo "  --opt                Run Optimization (main.py)"
    echo "  --anz                Run Analysis (analyze.py)"
    echo "  --ani                Run Animation (animate.py)"
    echo "  --all                Run all 3 stages in sequence"
    echo ""
    echo "Configuration Options:"
    echo "  --scenario FILE      Path to scenario file (default: $SCENARIO_FILE)"
    echo "  --output DIR         Output directory (default: $OUTPUT_PATH)"
    echo "  --history FILE       History Parquet filename (default: $HISTORY_FNAME)"
    echo "  --model FILE         Path to ML model for analysis (default: $MODEL_PATH)"
    echo "  --pop-size INT       Population size for optimization (default: $POP_SIZE)"
    echo "  --n-gens INT         Number of generations (default: $N_GENS)"
    echo "  --seed INT           Random seed (default: $SEED)"
    echo "  --mut-x FLOAT        Mutation rate X (default: $MUT_X)"
    echo "  --mut-y FLOAT        Mutation rate Y (default: $MUT_Y)"
    echo "  --mut-ws FLOAT       Mutation rate Whitespace (default: $MUT_WS)"
    echo "  --ws-max FLOAT       Max Whitespace value (default: $WS_MAX)"
    echo "  --grid-size INT      Grid size for analysis/model (default: $GRID_SIZE)"
    echo "  --dpi INT            DPI for plots/video (default: $DPI)"
    echo "  --interval INT       Animation frame interval in ms (default: $INTERVAL_MS)"
    echo "  --export-csv BOOL    Export CSV during analysis (default: $EXPORT_CSV)"
    echo "  --print-defaults     Print the current default constants and exit"
    echo "  -h, --help           Show this help message and exit"
    echo ""
    echo "Example: ./run.sh --all --output my_run --dpi 150"
}

print_defaults() {
    echo "========================================"
    echo " DEFAULT CONSTANTS"
    echo "========================================"
    echo " MAIN_SCRIPT     = $MAIN_SCRIPT"
    echo " ANZ_SCRIPT      = $ANZ_SCRIPT"
    echo " ANI_SCRIPT      = $ANI_SCRIPT"
    echo " SCENARIO_FILE   = $SCENARIO_FILE"
    echo " OUTPUT_PATH     = $OUTPUT_PATH"
    echo " HISTORY_FNAME   = $HISTORY_FNAME"
    echo " MODEL_PATH      = $MODEL_PATH"
    echo " POP_SIZE        = $POP_SIZE"
    echo " N_GENS          = $N_GENS"
    echo " SEED            = $SEED"
    echo " MUT_X           = $MUT_X"
    echo " MUT_Y           = $MUT_Y"
    echo " MUT_WS          = $MUT_WS"
    echo " WS_MAX          = $WS_MAX"
    echo " GRID_SIZE       = $GRID_SIZE"
    echo " DPI             = $DPI"
    echo " INTERVAL_MS     = $INTERVAL_MS"
    echo " EXPORT_CSV      = $EXPORT_CSV"
    echo "========================================"
}

format_time() {
    local T=$1
    local H=$((T / 3600))
    local M=$(((T % 3600) / 60))
    local S=$((T % 60))
    printf "%02dh:%02dm:%02ds" $H $M $S
}

# ==============================================================================
# PARSE COMMAND-LINE FLAGS
# ==============================================================================
if [[ "$#" -eq 0 ]]; then
    print_help
    exit 0
fi

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --opt) RUN_OPT=true ;;
        --anz) RUN_ANZ=true ;;
        --ani) RUN_ANI=true ;;
        --all) RUN_OPT=true; RUN_ANZ=true; RUN_ANI=true ;;
        --scenario) SCENARIO_FILE="$2"; shift ;;
        --output) OUTPUT_PATH="$2"; shift ;;
        --history) HISTORY_FNAME="$2"; shift ;;
        --model) MODEL_PATH="$2"; shift ;;
        --pop-size) POP_SIZE="$2"; shift ;;
        --n-gens) N_GENS="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --mut-x) MUT_X="$2"; shift ;;
        --mut-y) MUT_Y="$2"; shift ;;
        --mut-ws) MUT_WS="$2"; shift ;;
        --ws-max) WS_MAX="$2"; shift ;;
        --grid-size) GRID_SIZE="$2"; shift ;;
        --dpi) DPI="$2"; shift ;;
        --interval) INTERVAL_MS="$2"; shift ;;
        --export-csv) EXPORT_CSV="$2"; shift ;;
        --print-defaults) print_defaults; exit 0 ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ "$RUN_OPT" = false ] && [ "$RUN_ANZ" = false ] && [ "$RUN_ANI" = false ]; then
    echo "Error: No stages selected. Use --opt, --anz, --ani, or --all."
    exit 1
fi

# ==============================================================================
# SETUP
# ==============================================================================
mkdir -p "$OUTPUT_PATH"
PARQUET_FILE="${OUTPUT_PATH}/${HISTORY_FNAME}"

# Initialize timers
TIME_OPT=0
TIME_ANZ=0
TIME_ANI=0
TOTAL_START_TIME=$(date +%s)

echo "========================================"
echo " Starting Pipeline Execution"
echo " Output directory: $OUTPUT_PATH"
echo "========================================"

# ==============================================================================
# 1. OPTIMIZATION (main.py - 12 args + script_name)
# ==============================================================================
if [ "$RUN_OPT" = true ]; then
    echo ""
    echo ">>> RUNNING OPTIMIZATION STAGE"
    START_OPT=$(date +%s)
    
    python3 "$MAIN_SCRIPT" "$SCENARIO_FILE" "$OUTPUT_PATH" "$POP_SIZE" "$N_GENS" "$SEED" "$HISTORY_FNAME" "$MUT_X" "$MUT_Y" "$MUT_WS" "$WS_MAX" "$MODEL_PATH" "$GRID_SIZE"
    
    END_OPT=$(date +%s)
    TIME_OPT=$((END_OPT - START_OPT))
    echo ">>> Optimization completed in $(format_time $TIME_OPT)."
fi

# ==============================================================================
# 2. ANALYSIS (analyze.py - 7 args + script_name)
# ==============================================================================
if [ "$RUN_ANZ" = true ]; then
    echo ""
    echo ">>> RUNNING ANALYSIS STAGE"
    START_ANZ=$(date +%s)
    
    python3 "$ANZ_SCRIPT" "$SCENARIO_FILE" "$OUTPUT_PATH" "$HISTORY_FNAME" "$MODEL_PATH" "$GRID_SIZE" "$DPI" "$EXPORT_CSV"
    
    END_ANZ=$(date +%s)
    TIME_ANZ=$((END_ANZ - START_ANZ))
    echo ">>> Analysis completed in $(format_time $TIME_ANZ)."
fi

# ==============================================================================
# 3. ANIMATION (animate.py - 4 args + script_name)
# ==============================================================================
if [ "$RUN_ANI" = true ]; then
    echo ""
    echo ">>> RUNNING ANIMATION STAGE"
    START_ANI=$(date +%s)
    
    python3 "$ANI_SCRIPT" "$SCENARIO_FILE" "${PARQUET_FILE}" "$INTERVAL_MS" "$DPI"
    
    END_ANI=$(date +%s)
    TIME_ANI=$((END_ANI - START_ANI))
    echo ">>> Animation completed in $(format_time $TIME_ANI)."
fi

# ==============================================================================
# TIMING REPORT GENERATION
# ==============================================================================
TOTAL_END_TIME=$(date +%s)
TOTAL_TIME=$((TOTAL_END_TIME - TOTAL_START_TIME))

REPORT_FILE="${OUTPUT_PATH}/timing_report.txt"

echo "========================================" > "$REPORT_FILE"
echo " TIMING REPORT" >> "$REPORT_FILE"
echo " $(date)" >> "$REPORT_FILE"
echo "========================================" >> "$REPORT_FILE"
if [ "$RUN_OPT" = true ]; then
    echo " Optimization time : $(format_time $TIME_OPT)" >> "$REPORT_FILE"
else
    echo " Optimization time : Skipped" >> "$REPORT_FILE"
fi

if [ "$RUN_ANZ" = true ]; then
    echo " Analysis time     : $(format_time $TIME_ANZ)" >> "$REPORT_FILE"
else
    echo " Analysis time     : Skipped" >> "$REPORT_FILE"
fi

if [ "$RUN_ANI" = true ]; then
    echo " Animation time    : $(format_time $TIME_ANI)" >> "$REPORT_FILE"
else
    echo " Animation time    : Skipped" >> "$REPORT_FILE"
fi
echo "----------------------------------------" >> "$REPORT_FILE"
echo " Total runtime     : $(format_time $TOTAL_TIME)" >> "$REPORT_FILE"
echo "========================================" >> "$REPORT_FILE"

echo ""
echo "Pipeline complete! Check $REPORT_FILE for execution times."
cat "$REPORT_FILE"
