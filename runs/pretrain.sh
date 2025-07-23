#!/bin/bash

# =============================================================================
# ReconMAE Pretraining Script
# =============================================================================
# This script handles pretraining of the ReconMAE model across different 
# scenarios and configurations.

# =============================================================================
# Configuration Section
# =============================================================================

# Data configurations
DATA_BASE_DIR="./data/ps2_p72"
SCENARIOS=("uma")  # Scenarios to pretrain
SPEED="5"
SNR_RANGE="10to25"  # Training SNR range

# Model configurations  
MODEL_NAME="ReconMAE"  # Options: ReconMAE, ReconMAE_MainOnly
EXPERIMENT_NAME="siso_1_umi_block_1_ps2_p72"
MASKING_TYPE="discrete"  # Options: discrete, contiguous, fixed, random_symbols, fix_length

# Training configurations
EPOCHS=80
LEARNING_RATE=0.001
TRAIN_BATCH_SIZE=64
EVAL_BATCH_SIZE=64
TRAIN_SPLIT=0.9
AUG_FACTOR=1
SEED=0

# Input/output configurations
MAIN_INPUT_TYPE="low"
AUX_INPUT_TYPE="low"
OUTPUT_DIR="./model_output"
EARLY_STOPPING=true

# Logging
VERBOSE=1
LOG_DIR="./runs/logs"
mkdir -p $LOG_DIR

# =============================================================================
# Functions
# =============================================================================

print_header() {
    echo "============================================================================="
    echo "$1"
    echo "============================================================================="
}

print_section() {
    echo "-----------------------------------------------------------------------------"
    echo "$1"
    echo "-----------------------------------------------------------------------------"
}


pretrain_scenario() {
    local scenario=$1
    local data_path="${DATA_BASE_DIR}/${scenario}/snr${SNR_RANGE}_speed${SPEED}"
    local model_suffix=$([ "$MODEL_NAME" = "ReconMAE_MainOnly" ] && echo "MainOnly" || echo "Full")
    local weights_name="${scenario}_${model_suffix}"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="${LOG_DIR}/pretrain_${scenario}_${model_suffix}_${timestamp}.log"
    
    print_section "Pretraining $MODEL_NAME on Scenario: $scenario"
    
    # Check if data path exists
    if [ ! -d "$data_path" ]; then
        echo "⚠️  Warning: Data path $data_path does not exist, skipping..."
        return 1
    fi
    
    echo "📁 Data path: $data_path"
    echo "🤖 Model: $MODEL_NAME"
    echo "💾 Weights name: $weights_name"
    echo "📝 Log file: $log_file"
    echo ""
    
    # Build command
    local cmd="python3 cebed/pretrain_recon_mae.py \
        --data_path \"$data_path\" \
        --model_name \"$MODEL_NAME\" \
        --experiment_name \"$EXPERIMENT_NAME\" \
        --masking_type \"$MASKING_TYPE\" \
        --epochs $EPOCHS \
        --learning_rate $LEARNING_RATE \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --eval_batch_size $EVAL_BATCH_SIZE \
        --train_split $TRAIN_SPLIT \
        --main_input_type \"$MAIN_INPUT_TYPE\" \
        --aux_input_type \"$AUX_INPUT_TYPE\" \
        --aug_factor $AUG_FACTOR \
        --seed $SEED \
        --output_dir \"$OUTPUT_DIR\" \
        --weights_name \"$weights_name\" \
        --verbose $VERBOSE"
    
    # Add early stopping flag
    if [ "$EARLY_STOPPING" = true ]; then
        cmd="$cmd --early_stopping"
    else
        cmd="$cmd --no_early_stopping"
    fi
    
    echo "🚀 Starting pretraining..."
    echo "Command: $cmd"
    echo ""
    
    # Execute command and log output
    if eval $cmd 2>&1 | tee "$log_file"; then
        echo "✅ Pretraining completed for scenario $scenario ($MODEL_NAME)"
        echo "📄 Log saved to: $log_file"
        return 0
    else
        echo "❌ Pretraining failed for scenario $scenario ($MODEL_NAME)"
        echo "📄 Check log file: $log_file"
        return 1
    fi
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    print_header "ReconMAE Pretraining Pipeline"
    
    echo "Configuration:"
    echo "  Data base directory: $DATA_BASE_DIR"
    echo "  Scenarios: ${SCENARIOS[*]}"
    echo "  Model: $MODEL_NAME"
    echo "  Masking type: $MASKING_TYPE"
    echo "  Epochs: $EPOCHS"
    echo "  Learning rate: $LEARNING_RATE"
    echo "  Batch sizes: train=$TRAIN_BATCH_SIZE, eval=$EVAL_BATCH_SIZE"
    echo "  Output directory: $OUTPUT_DIR"
    echo ""
    
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Track results
    local successful_scenarios=()
    local failed_scenarios=()
    local total_scenarios=${#SCENARIOS[@]}
    local current=0
    
    # Pretrain each scenario
    for scenario in "${SCENARIOS[@]}"; do
        current=$((current + 1))
        echo ""
        print_section "Progress: $current/$total_scenarios"
        
        if pretrain_scenario "$scenario"; then
            successful_scenarios+=("$scenario")
        else
            failed_scenarios+=("$scenario")
        fi
        
        echo ""
        echo "💤 Waiting 5 seconds before next scenario..."
        sleep 5
    done
    
    # Print summary
    echo ""
    print_header "Pretraining Summary"
    
    echo "Model: $MODEL_NAME"
    echo "Total scenarios: $total_scenarios"
    echo "Successful: ${#successful_scenarios[@]}"
    echo "Failed: ${#failed_scenarios[@]}"
    echo ""
    
    if [ ${#successful_scenarios[@]} -gt 0 ]; then
        echo "✅ Successfully pretrained scenarios:"
        for scenario in "${successful_scenarios[@]}"; do
            echo "  - $scenario"
        done
        echo ""
    fi
    
    if [ ${#failed_scenarios[@]} -gt 0 ]; then
        echo "❌ Failed scenarios:"
        for scenario in "${failed_scenarios[@]}"; do
            echo "  - $scenario"
        done
        echo ""
        echo "⚠️  Please check the log files for error details."
        return 1
    else
        echo "🎉 All pretraining completed successfully!"
        return 0
    fi
}

# =============================================================================
# Command Line Options
# =============================================================================

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -s, --scenario SCENARIO Pretrain only specific scenario (e.g., rt1)"
    echo "  -e, --epochs EPOCHS     Number of training epochs (default: $EPOCHS)"
    echo "  -lr, --learning-rate LR Learning rate (default: $LEARNING_RATE)"
    echo "  -b, --batch-size SIZE   Training batch size (default: $TRAIN_BATCH_SIZE)"
    echo "  -m, --masking TYPE      Masking type (default: $MASKING_TYPE)"
    echo "  --model MODEL           Model name: ReconMAE or ReconMAE_MainOnly (default: $MODEL_NAME)"
    echo "  --no-early-stopping     Disable early stopping"
    echo "  --dry-run               Show commands without executing"
    echo ""
    echo "Examples:"
    echo "  $0                      # Pretrain all scenarios with ReconMAE"
    echo "  $0 --model ReconMAE_MainOnly  # Pretrain with main-only model"
    echo "  $0 -s rt1              # Pretrain only rt1 scenario"
    echo "  $0 -e 100 -lr 0.0005   # Custom epochs and learning rate"
    echo "  $0 --dry-run            # Show what would be executed"
}

# Parse command line arguments
DRY_RUN=false
SINGLE_SCENARIO=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--scenario)
            SINGLE_SCENARIO="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -lr|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -b|--batch-size)
            TRAIN_BATCH_SIZE="$2"
            EVAL_BATCH_SIZE="$2"
            shift 2
            ;;
        -m|--masking)
            MASKING_TYPE="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            if [[ "$MODEL_NAME" != "ReconMAE" && "$MODEL_NAME" != "ReconMAE_MainOnly" ]]; then
                echo "❌ Invalid model name: $MODEL_NAME"
                echo "Valid options: ReconMAE, ReconMAE_MainOnly"
                exit 1
            fi
            shift 2
            ;;
        --no-early-stopping)
            EARLY_STOPPING=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use -h or --help for usage information."
            exit 1
            ;;
    esac
done

# Handle single scenario option
if [ -n "$SINGLE_SCENARIO" ]; then
    SCENARIOS=("$SINGLE_SCENARIO")
fi

# Handle dry run
if [ "$DRY_RUN" = true ]; then
    print_header "DRY RUN - Commands that would be executed"
    for scenario in "${SCENARIOS[@]}"; do
        data_path="${DATA_BASE_DIR}/${scenario}/snr${SNR_RANGE}_speed${SPEED}"
        model_suffix=$([ "$MODEL_NAME" = "ReconMAE_MainOnly" ] && echo "MainOnly" || echo "Full")
        weights_name="${scenario}_${model_suffix}"
        echo ""
        echo "Scenario: $scenario"
        echo "Model: $MODEL_NAME"
        echo "Data path: $data_path"
        echo "Weights name: $weights_name"
        echo "Command: python3 pretrain_recon_mae.py --model_name \"$MODEL_NAME\" --data_path \"$data_path\" ..."
    done
    exit 0
fi

# Execute main function
main
exit_code=$?
exit $exit_code
