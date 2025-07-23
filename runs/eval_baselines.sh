#!/bin/bash

# Model configurations - paths to trained baseline models
HA02_MODEL="offline_train_output/bs64_lr0.001_snr10to25_HA02_rt4_3_10"
CHANNELNET_MODEL="offline_train_output/bs64_lr0.001_snr10to25_ChannelNet_rt4_3_10"
REESNET_MODEL="offline_train_output/bs64_lr0.001_snr10to25_ReEsNet_rt4_3_10"

# Environment configurations
PRETRAIN_SCENARIO="rt4"  # Scenario for pretraining
SCENARIOS=("rt3")
SPEED="5"
DATA_BASE_DIR="./data/ps2_p72"

# Evaluation configurations
EVAL_SNR_MIN=0
EVAL_SNR_MAX=20
EVAL_SNR_STEP=5
EVAL_BATCH_SIZE=64
EVAL_SPLIT=0.1

# Other configurations
SEED=43
OUTPUT_DIR="experiment_results/classic_eval"
EXPERIMENT_NAME="siso_1_umi_block_1_ps2_p72"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Base command
BASE_CMD="python cebed/classic_evaluator.py"

# Common arguments
COMMON_ARGS="--eval_batch_size ${EVAL_BATCH_SIZE} \
             --eval_split ${EVAL_SPLIT} \
             --eval_snr_min ${EVAL_SNR_MIN} \
             --eval_snr_max ${EVAL_SNR_MAX} \
             --eval_snr_step ${EVAL_SNR_STEP} \
             --seed ${SEED} \
             --output_dir ${OUTPUT_DIR} \
             --experiment_name ${EXPERIMENT_NAME}"

# Function to run evaluation for a given model
run_evaluation() {
    local MODEL_PATH=$1
    local MODEL_NAME=$2
    local SCENARIO=$3
    
    
    WANDB_NAME="${PRETRAIN_SCENARIO}_on_${SCENARIO}_${MODEL_NAME}_speed${SPEED}"
    
    echo "Evaluating $MODEL_NAME on $SCENARIO scenario..."
    
    $BASE_CMD \
        --trained_model_dir "$MODEL_PATH" \
        --model_name "$MODEL_NAME" \
        --eval_base_dir "$DATA_BASE_DIR" \
        --scenario "$SCENARIO" \
        --speed "$SPEED" \
        --wandb_name "$WANDB_NAME" \
        $COMMON_ARGS \
        2>&1 | tee "${OUTPUT_DIR}/${MODEL_NAME}_${SCENARIO}_evaluation.log"
    
    echo "Finished evaluating $MODEL_NAME on $SCENARIO"
    echo "----------------------------------------"
    sleep 5
}

# Main execution
echo "Starting baseline model evaluation..."
echo "Models to evaluate: HA02, ChannelNet, ReEsNet"
echo "Scenarios: ${SCENARIOS[*]}"
echo "SNR range: ${EVAL_SNR_MIN} to ${EVAL_SNR_MAX} (step: ${EVAL_SNR_STEP})"
echo "Speed: ${SPEED}"
echo "========================================"

# Run evaluations for each model and scenario combination
for SCENARIO in "${SCENARIOS[@]}"; do
    echo "Starting evaluations for scenario: $SCENARIO"
    
    # Evaluate HA02
    run_evaluation "$HA02_MODEL" "HA02" "$SCENARIO"
    
    # Evaluate ChannelNet
    run_evaluation "$CHANNELNET_MODEL" "ChannelNet" "$SCENARIO"
    
    # Evaluate ReEsNet
    run_evaluation "$REESNET_MODEL" "ReEsNet" "$SCENARIO"
    
    echo "Completed all model evaluations for scenario: $SCENARIO"
    echo "========================================"
done

echo "All baseline evaluations completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "Check individual log files for detailed outputs."
