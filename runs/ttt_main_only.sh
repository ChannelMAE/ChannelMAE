#!/bin/bash

# Model configurations
MODEL1="./models/rt0_MainOnly.h5"

MODEL2="./models/uma_MainOnly.h5"

MODEL3="./models/umi_MainOnly_1MB.h5" # 11 decoders

MODEL4="./models/umi_MainOnly_2en.h5" # 2 encoders and 2 decoders

MODEL5="./models/rt4_MainOnly.h5"

# Environment configurations
SCENARIOS=("umi")  # Scenarios to run experiments on
PRETRAIN_SCENARIO="uma" # When change this, make sure you change the model below as well!
PRETRAIN_SPEED="5"
SPEED="30"
DATA_BASE_DIR="./data/ps2_p72"
TRAIN_DATA_DIR="./data_TTTtrain/ps2_p72"
TRAIN_SNR_MIN=10
TRAIN_SNR_MAX=10
TRAIN_SNR_STEP=1
EVAL_SNR_MIN=0
EVAL_SNR_MAX=20
EVAL_SNR_STEP=5
EPOCHS=1
LEARNING_RATE=5e-4

# Dataset configurations
TRAIN_BATCH_SIZE_VALUES=(32)
EVAL_BATCH_SIZE=64
TTT_SPLIT_VALUES=(0.5)  # Train-Test-Train split, only for training dataset size
AUG_TIMES_VALUES=(1) # can only be 1 for pseudo labeling
EVAL_SPLIT=0.5  # test split, only for pre/post-evaluation dataset size

# Method configurations
DET_METHOD="lmmse" 
OUTPUT="symbol"
MASKING_TYPE="discrete" # If you change this, make sure you change the model config in corresponding yaml file as well!

# Results directory
RESULTS_DIR="experiment_results/v3_main_only"
mkdir -p $RESULTS_DIR

# Base command
BASE_CMD="python cebed/online_main_only.py"

# Common arguments - reorganized and complete
COMMON_ARGS="--eval_dataset_name RandomMask \
             --eval_batch_size ${EVAL_BATCH_SIZE} \
             --eval_split ${EVAL_SPLIT} \
             --epochs ${EPOCHS} \
             --main_input_type low \
             --aux_input_type low \
             --supervised 0 \
             --ssl 1 \
             --learning_rate ${LEARNING_RATE} \
             --det_method ${DET_METHOD} \
             --output ${OUTPUT} \
             --masking_type ${MASKING_TYPE} \
             --seed 43 \
             --pretrain_scenario ${PRETRAIN_SCENARIO} \
             --pretrain_speed ${PRETRAIN_SPEED}"

# Function to run experiments for a given model and training SNR
run_experiments() {
    local MODEL=$1
    local TRAIN_SNR=$2
    local TTT_SPLIT=$3
    local AUG_TIMES=$4
    local SCENARIO=$5
    local TRAIN_BATCH_SIZE=$6
    local MODEL_NAME=$(basename "$MODEL" .h5)
    local TRAIN_SNR_END=$((TRAIN_SNR+10))
    
    # Construct training data directory
    TRAIN_DATA_DIR="${TRAIN_DATA_DIR}/${SCENARIO}/snr${TRAIN_SNR}to${TRAIN_SNR_END}_speed${SPEED}"
    
    # Updated experiment name
    EXPERIMENT_NAME="pseudo_mainONLY_${PRETRAIN_SCENARIO}_on_${SCENARIO}_TTTSplit${TTT_SPLIT}_batch${TRAIN_BATCH_SIZE}"
    
    # Create model-specific output directory
    MODEL_RESULTS_DIR="${RESULTS_DIR}/${PRETRAIN_SCENARIO}_on_${SCENARIO}/aug${AUG_TIMES}/TTTSplit${TTT_SPLIT}/batch${TRAIN_BATCH_SIZE}"
    mkdir -p "$MODEL_RESULTS_DIR"
    
    echo "Running experiment for model $MODEL_NAME with training SNR ${TRAIN_SNR}, TTT_SPLIT ${TTT_SPLIT}, AUG_TIMES ${AUG_TIMES}, BATCH_SIZE ${TRAIN_BATCH_SIZE}"
    
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
        --ttt_split "${TTT_SPLIT}" \
        --aug_times "${AUG_TIMES}" \
        --train_batch_size "${TRAIN_BATCH_SIZE}" \
        $COMMON_ARGS \
        2>&1 | tee "${MODEL_RESULTS_DIR}/experiment.log"
    
    sleep 5
}

# Run experiments for each model across all combinations
for MODEL in "$MODEL2"; do
    echo "Running experiments for $(basename "$MODEL")..."
    for TRAIN_SNR in $(seq $TRAIN_SNR_MIN $TRAIN_SNR_STEP $TRAIN_SNR_MAX); do
        for TTT_SPLIT in "${TTT_SPLIT_VALUES[@]}"; do
            for AUG_TIMES in "${AUG_TIMES_VALUES[@]}"; do
                for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZE_VALUES[@]}"; do
                    for SCENARIO in "${SCENARIOS[@]}"; do
                        run_experiments "$MODEL" "$TRAIN_SNR" "$TTT_SPLIT" "$AUG_TIMES" "$SCENARIO" "$TRAIN_BATCH_SIZE"
                    done
                done
            done
        done
    done
done

echo "All experiments completed. Results saved in $RESULTS_DIR"
