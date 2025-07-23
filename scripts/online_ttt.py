"""
Evaluate a model trained with CeBed
Either in-distribution or out-of-distribution evaluation
"""

import sys
from pathlib import Path
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)

import os
import argparse
import numpy as np

from cebed.ttt import Evaluator, EvalConfig
from cebed.utils import read_metadata, set_random_seed

def main(args):

    saved_config = read_metadata(os.path.join(args.trained_model_dir, "config.yaml"))  # saved_config must be a dict
    eval_config = EvalConfig(**vars(args))

    # initialize evaluator
    set_random_seed(eval_config.seed)
    evaluator = Evaluator(eval_config)
    evaluator.setup()

    ## load pre-trained model into the evaluator
    evaluator.load_model_from_path(os.path.join(args.trained_model_dir, "cp.ckpt"))

    ## Evaluate the model before adaptation
    saved_config = read_metadata(os.path.join(args.trained_model_dir, "config.yaml"))  # saved_config must be a dict    
    start_snr = saved_config["start_ds"]
    end_snr = saved_config["end_ds"]
    step = (end_snr - start_snr) / saved_config["num_domains"]
    snr_range = np.arange(saved_config["start_ds"], saved_config["end_ds"], step) 
    evaluator.evaluate_channel_est(snr_range=snr_range, baselines=["LS","LMMSE", "ALMMSE"], save=True)
    evaluator.save()

    ## Online TTT in the new environment for multiple epochs
    # TODO: work on ttt_evaluator.py
    evaluator.online_ttt()

    ### load TTT-adapted model
    evaluator.load_model_from_path("./eval_output/rma/ttt/cp.ckpt")
    evaluator.evaluate_channel_est(snr_range=[20],baselines=[], save=True)
    evaluator.save()


if __name__ == "__main__":
    # The following arguments are exactly the same as the dataclass EvalConfig in cebed/TTT/trainer.py
    parser = argparse.ArgumentParser(description="Channel Estimation")

    ########################## General args ###################################
    parser.add_argument("--trained_model_dir", type=str, default="./pretrain_output/siso_1_umi_block_1_ps2_p72/0/DenoiseSingleBranch")
    parser.add_argument("--eval_data_dir", type=str, default="./data/ps2_p72/speed0") # related to OOD test dataset
    parser.add_argument("--eval_scenario", type=str, default="rma", choices=["rma", "umi", "uma", "Rayleigh"])
    parser.add_argument("--eval_dataset_name", type=str, default="Denoise")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--train_split", type=float, default=0.001)

    parser.add_argument("--main_input_type", type=str, default="low", choices=["low", "raw"])
    parser.add_argument("--aux_input_type", type=str, default="raw", choices=["low", "raw"])
    parser.add_argument("--aug_noise_std", type=float, default=0.5)

    # parser.add_argument("--baselines", type=str, nargs="+", default=[])
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="eval_output")
    parser.add_argument("--ttt_period", type=int, default=1)
    parser.add_argument("--ttt_active", action="store_true", help="If TTT should be used") # in this way, we can deal with bool arguments

    main(parser.parse_args(sys.argv[1:]))
