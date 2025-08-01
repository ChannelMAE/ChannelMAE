#!/bin/bash

# Model configurations
MODEL1="./3gpp_models/dncnn.h5"
MNAME=$(basename $MODEL1 .h5)


# Environment configurations
SCENARIOS=("umi")
PRETRAIN_SCENARIO="uma"
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
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=64
ADAPT_SPLIT_VALUES=(0.5)
# AUG_TIMES_VALUES=(3)
EVAL_SPLIT=0.5

# Method configurations
DET_METHOD="lmmse" 
OUTPUT="symbol"
MASKING_TYPE="discrete" # If you change this, make sure you change the model config in corresponding yaml file as well!

# Results directory
RESULTS_DIR="experiment_results/$MNAME"
mkdir -p $RESULTS_DIR

# Base command
BASE_CMD="python cebed/online_adapt_$MNAME.py"

# Common arguments - reorganized and complete
COMMON_ARGS="--eval_dataset_name Denoise \
             --train_batch_size ${TRAIN_BATCH_SIZE} \
             --eval_batch_size ${EVAL_BATCH_SIZE} \
             --eval_split ${EVAL_SPLIT} \
             --epochs ${EPOCHS} \
             --main_input_type low \
             --aux_input_type raw \
             --learning_rate ${LEARNING_RATE} \
             --output ${OUTPUT} \
             --seed 43 \
             --pretrain_scenario ${PRETRAIN_SCENARIO} \
             --pretrain_speed ${PRETRAIN_SPEED}"


    

# Function to run experiments for a given model and training SNR
run_experiments() {

    local MODEL=$1
    local TRAIN_SNR=$2
    local ADAPT_SPLIT=$3
    local SCENARIO=$4
    local MODEL_NAME=$(basename "$MODEL" .h5)
    local TRAIN_SNR_END=$((TRAIN_SNR+10))


    # Construct training data directory
    TRAIN_DATA_DIR="./data_TTTtrain/ps2_p72/${SCENARIO}/snr${TRAIN_SNR}to${TRAIN_SNR_END}_speed${SPEED}"
    EXPERIMENT_NAME="${MODEL_NAME}_${PRETRAIN_SCENARIO}_on_${SCENARIO}_AdaptSplit${ADAPT_SPLIT}"
    
    # Create model-specific output directory with ADAPT_SPLIT
    MODEL_RESULTS_DIR="${RESULTS_DIR}/${PRETRAIN_SCENARIO}_on_${SCENARIO}/AdaptSplit${ADAPT_SPLIT}"
    mkdir -p "$MODEL_RESULTS_DIR"
    
    echo "Running experiment for model $MODEL_NAME with training SNR ${TRAIN_SNR}, ADAPT_SPLIT ${ADAPT_SPLIT}"
    
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
        --adapt_split "$ADAPT_SPLIT" \
        $COMMON_ARGS \
        2>&1 | tee "${MODEL_RESULTS_DIR}/experiment.log"
    
    sleep 5
}

# Run experiments for each model across all combinations
for MODEL in "$MODEL1"; do
    echo "Running experiments for $(basename "$MODEL")..."
    for TRAIN_SNR in $(seq $TRAIN_SNR_MIN $TRAIN_SNR_STEP $TRAIN_SNR_MAX); do
        for ADAPT_SPLIT in "${ADAPT_SPLIT_VALUES[@]}"; do
            # for AUG_TIMES in "${AUG_TIMES_VALUES[@]}"; do
            for SCENARIO in "${SCENARIOS[@]}"; do
                run_experiments "$MODEL" "$TRAIN_SNR" "$ADAPT_SPLIT" "$SCENARIO"
            done
            # done
        done
    done
done

echo "All experiments completed. Results saved in $RESULTS_DIR"
