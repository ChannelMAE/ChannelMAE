#!/bin/bash

# Define array of SNR ranges
# snr_ranges=(
#     # "100 101 1"  # Noiseless, single SNR
#     "0 25 5"     # Noisy, multi SNR
#     "20 21 1"    # Noisy, single high SNR
#     "0 1 1"      # Noisy, single low SNR
# )
# # scenarios=("rma" "uma" "umi" "Rayleigh")
# # scenario="rma" # or "uma", "umi", "Rayleigh"
# scenarios=("umi")
# ue_speed=5
# size=15000
# seed=0

# # Loop through each SNR range
# for snr_range in "${snr_ranges[@]}"; do
#     # Split the SNR range into start and end
#     start_snr=$(echo $snr_range | cut -d' ' -f1)
#     end_snr=$(echo $snr_range | cut -d' ' -f2)
#     num_snr=$(echo $snr_range | cut -d' ' -f3)

#     for scenario in "${scenarios[@]}"; do
#         # Call the Python script with the current SNR range
#         python ./scripts/generate_datasets_from_sionna.py \
#             --output_dir "./data" \
#             --num_domains $num_snr \
#             --start_ds $start_snr \
#             --end_ds $end_snr \
#             --ue_speed $ue_speed \
#             --size $size \
#             --seed $seed \
#             --scenario $scenario
#     done
# done


# ---------------------------------------------------------------------------- #
snr_ranges=(
    # "100 101 1"  # Noiseless, single SNR
    # "0 25 5"     # All SNRs, noisy
    # "20 21 1"    # Noisy, single high SNR
    # "5 6 1"
    # "10 11 1"
    # "0 1 1"
    # "5 6 1"
    # "10 11 1"
    # "15 16 1"
    # "20 21 1"
    # "-5 -4 1"
    # "-4 -3 1"
    # "-3 -2 1"
    # "-2 -1 1"
    # "10 25 5"   # Pretrain
    # "10 20 2"   # TTT Train
    # "5 15 2"
    "0 1 1"     # Eval
    "5 6 1"     # Eval
    "10 11 1"   # Eval
    "15 16 1"   # Eval
    "20 21 1"   # Eval
)

scenarios=("umi")  # Scenarios to generate data for
ue_speed=("30")
output_dir="./data_TTTtrain"  # data or data_TTTtrain
size=20000
seed=0

# Loop through each SNR range
for snr_range in "${snr_ranges[@]}"; do
    # Split the SNR range into start and end
    start_snr=$(echo $snr_range | cut -d' ' -f1)
    end_snr=$(echo $snr_range | cut -d' ' -f2)
    num_snr=$(echo $snr_range | cut -d' ' -f3)

    for scenario in "${scenarios[@]}"; do
        for ue_speed in "${ue_speed[@]}"; do
            # Call the Python script with the current SNR range
            python ./scripts/generate_datasets_from_sionna.py \
                --output_dir $output_dir \
                --num_domains $num_snr \
                --start_ds $start_snr \
                --end_ds $end_snr \
                --ue_speed $ue_speed \
                --size $size \
                --seed $seed \
                --scenario $scenario \
                --encode
        done
    done
done

