#!/bin/bash

# Model configurations
MODEL1="./models/rt0.h5"

MODEL2="./models/rt1.h5"

MODEL3="./models/rt2.h5"

MODEL12="./models/rt2_new.h5"

MODEL4="./models/rt3.h5"

MODEL5="./models/rt4.h5"

MODEL13="./models/rt4_new.h5"

MODEL6="./models/uma_symbol.h5"

MODEL14="./models/uma_channel.h5" # This one used for most experiments

MODEL7="./models/umi.h5"

MODEL8="./models/umi_1MB.h5" # 9 main decoders

MODEL9="./models/umi_2en.h5" # 2 encoders

MODEL10="./models/umi_2en_3_10.h5" # 2 encoders

MODEL11="./models/umi_3_10.h5" # 2 encoders

MODEL15="./models/uma_full_attn.h5"

MODEL16="./models/uma_attn_1_1.h5"

MODEL17="./models/uma_attn_2_2.h5"

MODEL18="./models/uma_channel_skip_con.h5"

MODEL19="./models/uma_1_1.h5"

MODEL20="./models/uma_2_2.h5"

MODEL21="./models/uma_2en.h5"

MODEL22="./models/uma_4_2.h5"

MODEL23="./models/03142.h5"


# Environment configurations
SCENARIOS=("umi")  # Scenarios to run experiments on
PRETRAIN_SCENARIO="uma" # When change this, make sure you change the model below as well! line 107
PRETRAIN_SPEED="5"
SPEED="30"
DATA_BASE_DIR="./data/ps2_p72"
TRAIN_DATA_BASE="./data_TTTtrain/ps2_p72"
TRAIN_SNR_MIN=10
TRAIN_SNR_MAX=10
TRAIN_SNR_STEP=5
EVAL_SNR_MIN=0
EVAL_SNR_MAX=20
EVAL_SNR_STEP=5
EPOCHS=1
LEARNING_RATE=5e-4

# Dataset configurations
TRAIN_BATCH_SIZE_VALUES=(32)
EVAL_BATCH_SIZE=64
TTT_SPLIT_VALUES=(0.5)
AUG_TIMES_VALUES=(5)
EVAL_SPLIT=0.5

# Method configurations
DET_METHOD="lmmse"  # Options: k-best, lmmse, ep, mmse-pic
OUTPUT="symbol"  # Options: symbol, bit
MASKING_TYPE="discrete" # If you change this, make sure you change the model config in corresponding yaml file as well!
SSL=1  # Set SSL flag (1 for SSL, 0 for supervised)

# Continual TTT configuration
CONTINUAL=0  # Set to 1 to enable continual TTT

# Results directory
RESULTS_DIR="experiment_results/ttt"
mkdir -p $RESULTS_DIR

# Base command
BASE_CMD="python cebed/online_ttt_v3.py"

# Common arguments - reorganized and complete
COMMON_ARGS="--eval_dataset_name RandomMask \
             --eval_batch_size ${EVAL_BATCH_SIZE} \
             --eval_split ${EVAL_SPLIT} \
             --epochs ${EPOCHS} \
             --main_input_type low \
             --aux_input_type low \
             --supervised $((1-SSL)) \
             --ssl ${SSL} \
             --learning_rate ${LEARNING_RATE} \
             --det_method ${DET_METHOD} \
             --output ${OUTPUT} \
             --masking_type ${MASKING_TYPE} \
             --seed 43 \
             --pretrain_scenario ${PRETRAIN_SCENARIO} \
             --pretrain_speed ${PRETRAIN_SPEED}"

# Function to convert array to comma-separated string
array_to_csv() {
    local IFS=','
    echo "$*"
}

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
    TRAIN_DATA_DIR="${TRAIN_DATA_BASE}/${SCENARIO}/snr${TRAIN_SNR}to${TRAIN_SNR_END}_speed${SPEED}"
    EXPERIMENT_NAME="${PRETRAIN_SCENARIO}_on_${SCENARIO}_aug${AUG_TIMES}_TTTSplit${TTT_SPLIT}_batch${TRAIN_BATCH_SIZE}"
    # EXPERIMENT_NAME="supervised_${PRETRAIN_SCENARIO}_on_${SCENARIO}_aug${AUG_TIMES}_TTTSplit${TTT_SPLIT}_batch${TRAIN_BATCH_SIZE}"
    
    # Create model-specific output directory with TTT_SPLIT and AUG_TIMES
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

# Function to run continual TTT experiments
run_continual_experiments() {
    local MODEL=$1
    local TRAIN_SNR=$2
    local TTT_SPLIT=$3
    local AUG_TIMES=$4
    local TRAIN_BATCH_SIZE=$5
    local MODEL_NAME=$(basename "$MODEL" .h5)
    local SCENARIOS_CSV=$(array_to_csv "${SCENARIOS[@]}")
    
    EXPERIMENT_NAME="${PRETRAIN_SCENARIO}_continual_aug${AUG_TIMES}_TTTSplit${TTT_SPLIT}_batch${TRAIN_BATCH_SIZE}"
    # EXPERIMENT_NAME="super_${PRETRAIN_SCENARIO}_continual_aug${AUG_TIMES}_TTTSplit${TTT_SPLIT}"
    
    # Create model-specific output directory for continual TTT
    MODEL_RESULTS_DIR="${RESULTS_DIR}/${PRETRAIN_SCENARIO}_continual/aug${AUG_TIMES}/TTTSplit${TTT_SPLIT}/batch${TRAIN_BATCH_SIZE}"
    mkdir -p "$MODEL_RESULTS_DIR"
    
    echo "Running continual TTT experiment for model $MODEL_NAME with scenarios: ${SCENARIOS_CSV}, BATCH_SIZE ${TRAIN_BATCH_SIZE}"
    
    $BASE_CMD \
        --trained_model_dir "$MODEL" \
        --train_snr "$TRAIN_SNR" \
        --eval_base_dir "$DATA_BASE_DIR" \
        --speed "$SPEED" \
        --output_dir "$MODEL_RESULTS_DIR" \
        --wandb_name "$EXPERIMENT_NAME" \
        --ttt_split "${TTT_SPLIT}" \
        --aug_times "${AUG_TIMES}" \
        --train_batch_size "${TRAIN_BATCH_SIZE}" \
        --continual 1 \
        --scenarios "${SCENARIOS_CSV}" \
        $COMMON_ARGS \
        2>&1 | tee "${MODEL_RESULTS_DIR}/continual_experiment.log"
    
    sleep 5
}

# Run experiments for each model across all combinations
for MODEL in "$MODEL14"; do
    echo "Running experiments for $(basename "$MODEL")..."
    for TRAIN_SNR in $(seq $TRAIN_SNR_MIN $TRAIN_SNR_STEP $TRAIN_SNR_MAX); do
        for TTT_SPLIT in "${TTT_SPLIT_VALUES[@]}"; do
            for AUG_TIMES in "${AUG_TIMES_VALUES[@]}"; do
                for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZE_VALUES[@]}"; do
                    if [ "$CONTINUAL" -eq 1 ]; then
                        # Run continual TTT experiment
                        run_continual_experiments "$MODEL" "$TRAIN_SNR" "$TTT_SPLIT" "$AUG_TIMES" "$TRAIN_BATCH_SIZE"
                    else
                        # Run original single scenario experiments
                        for SCENARIO in "${SCENARIOS[@]}"; do
                            run_experiments "$MODEL" "$TRAIN_SNR" "$TTT_SPLIT" "$AUG_TIMES" "$SCENARIO" "$TRAIN_BATCH_SIZE"
                        done
                    fi
                done
            done
        done
    done
done
