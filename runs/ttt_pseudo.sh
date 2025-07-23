#!/bin/bash

# Model configurations
MODEL1="./models/rt0.h5"

MODEL2="./models/rt1.h5"

MODEL3="./models/rt2.h5"

MODEL4="./models/rt3.h5"

MODEL5="./models/rt4.h5"

MODEL6="./models/uma_channel.h5"

# Environment configurations
SCENARIOS=("umi")
PRETRAIN_SCENARIO="uma" # When change this, make sure you change the model below as well! line 107
PRETRAIN_SPEED="5"
SPEED="30"
DATA_BASE_DIR="./data/ps2_p72"
TRAIN_SNR_MIN=10
TRAIN_SNR_MAX=10
TRAIN_SNR_STEP=1
EVAL_SNR_MIN=0
EVAL_SNR_MAX=20
EVAL_SNR_STEP=5
EPOCHS=1
LEARNING_RATE=5e-4

# Dataset configurations
TRAIN_BATCH_SIZE=64
EVAL_BATCH_SIZE=64
TTT_SPLIT=0.5  # Train-Test-Train split, only for training dataset size
EVAL_SPLIT=0.5  # test split, only for pre/post-evaluation dataset size
AUG_TIMES=1 # can only be 1 for pseudo labeling

# Method configurations
DET_METHOD="lmmse" 
OUTPUT="symbol"
MASKING_TYPE="discrete"

# Results directory
RESULTS_DIR="experiment_results/pseudo"
mkdir -p $RESULTS_DIR

# Base command
BASE_CMD="python cebed/online_ttt_v4.py"

# Common arguments
COMMON_ARGS="--eval_dataset_name RandomMask \
             --train_batch_size ${TRAIN_BATCH_SIZE} \
             --eval_batch_size ${EVAL_BATCH_SIZE} \
             --ttt_split ${TTT_SPLIT} \
             --eval_split ${EVAL_SPLIT} \
             --epochs ${EPOCHS} \
             --main_input_type low \
             --aux_input_type low \
             --supervised 0 \
             --ssl 1 \
             --learning_rate ${LEARNING_RATE} \
             --det_method ${DET_METHOD} \
             --output ${OUTPUT} \
             --aug_times ${AUG_TIMES} \
             --masking_type ${MASKING_TYPE} \
             --seed 43"

# Function to run experiments for a given model and training SNR
run_experiments() {
    local MODEL=$1
    local TRAIN_SNR=$2
    local SCENARIO=$3
    local MODEL_NAME=$(basename "$MODEL" .h5)
    local TRAIN_SNR_END=$((TRAIN_SNR+10))
    
    # Construct training data directory
    TRAIN_DATA_DIR="./data_TTTtrain/ps2_p72/${SCENARIO}/snr${TRAIN_SNR}to${TRAIN_SNR_END}_speed${SPEED}"
    EXPERIMENT_NAME="pseudo_${PRETRAIN_SCENARIO}_on_${SCENARIO}_aug${AUG_TIMES}_TTTSplit${TTT_SPLIT}"

    # Create model-specific output directory
    MODEL_RESULTS_DIR="${RESULTS_DIR}/${PRETRAIN_SCENARIO}_on_${SCENARIO}/aug${AUG_TIMES}/TTTSplit${TTT_SPLIT}"
    mkdir -p "$MODEL_RESULTS_DIR"
    
    echo "Running experiment for model $MODEL_NAME with training SNR ${TRAIN_SNR} on scenario ${SCENARIO}"
    
    $BASE_CMD \
        --trained_model_dir "$MODEL" \
        --train_data_dir "$TRAIN_DATA_DIR" \
        --train_snr "$TRAIN_SNR" \
        --eval_base_dir "$DATA_BASE_DIR" \
        --scenario "$SCENARIO" \
        --speed "$SPEED" \
        --eval_snr_min "$EVAL_SNR_MIN" \
        --eval_snr_max "$EVAL_SNR_MAX" \
        --eval_snr_step "$EVAL_SNR_STEP" \
        --output_dir "$MODEL_RESULTS_DIR" \
        --wandb_name "$EXPERIMENT_NAME" \
        $COMMON_ARGS \
        2>&1 | tee "${MODEL_RESULTS_DIR}/experiment.log"
    
    sleep 5
}

# Run experiments for each model across training SNR range and scenarios
for MODEL in "$MODEL6"; do
    echo "Running experiments for $(basename "$MODEL")..."
    for SCENARIO in "${SCENARIOS[@]}"; do
        echo "Running experiments for scenario: $SCENARIO"
        for TRAIN_SNR in $(seq $TRAIN_SNR_MIN $TRAIN_SNR_STEP $TRAIN_SNR_MAX); do
            run_experiments "$MODEL" "$TRAIN_SNR" "$SCENARIO"
        done
    done
done

echo "All experiments completed. Results saved in $RESULTS_DIR"