"""
Evaluator for classic channel estimation models (HA02, ReEsNet, ChannelNet)
Performs cross-SNR evaluation on different test datasets
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import sys
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)

import swanlab as wandb
wandb.login()
from typing import List, Dict
import pandas as pd
from dataclasses import dataclass, asdict
from collections import defaultdict

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tqdm import tqdm
import numpy as np
import cebed.datasets_with_ssl as cds
import cebed.models as cm

from cebed.utils import write_metadata, read_metadata
from cebed.utils_eval import mse
from cebed.utils import set_random_seed
from types import MethodType

@dataclass
class EvalConfig:
    # model-related
    trained_model_dir: str = ""
    model_name: str = "HA02"  # "HA02", "ChannelNet", or "ReEsNet"
    
    # dataset-related
    eval_base_dir: str = "./data/ps2_p72"
    scenario: str = "umi"
    speed: str = "30"
    eval_batch_size: int = 64
    main_input_type: str = "low"
    
    # evaluation configs
    eval_split: float = 0.5
    eval_snr_min: int = -2
    eval_snr_max: int = 2
    eval_snr_step: int = 1
    
    # other configs
    seed: int = 43
    verbose: int = 1
    output_dir: str = "experiment_results/classic_eval"
    wandb_name: str = ""
    experiment_name: str = "siso_1_umi_block_1_ps2_p72"

class Evaluator:
    """
    Evaluate trained classic channel estimation models across different SNRs
    """
    def __init__(self, config: EvalConfig):
        self.config = config
        self.model_dir = self.config.trained_model_dir
        self.model_name = self.config.model_name

        # Set up logging directory
        self.log_dir = os.path.join(
            self.config.output_dir,
            f"{self.model_name}_{self.config.scenario}_speed{self.config.speed}"
        )
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize model and dataset objects
        self.model = None
        self.dataset = None

    def setup(self):
        """Setup the evaluator"""
        # Create initial dataset (will be updated for each SNR during evaluation)
        dataset_class = cds.get_dataset_class("Classic")  # Using Classic dataset for evaluation
        eval_data_dir = os.path.join(
            self.config.eval_base_dir,
            self.config.scenario,
            f"snr0to1_speed{self.config.speed}"
        )
        self.dataset = dataset_class(
            eval_data_dir,
            train_split=self.config.eval_split,
            main_input_type=self.config.main_input_type,
            seed=self.config.seed
        )

        # Create model
        model_hparams = cm.get_model_hparams(
            self.config.model_name,
            self.config.experiment_name
        )

        model_class = cm.get_model_class(self.config.model_name)
        if "output_dim" not in model_hparams:
            raise ValueError("output_dim is not in model_hparams")

        self.model = model_class(model_hparams)
        
        # Build model
        input_shape_main, _ = self.dataset.get_input_shape()
        self.model.build(
            tf.TensorShape([None, input_shape_main[0], input_shape_main[1], input_shape_main[2]])
        )

        # Load trained weights from checkpoint
        checkpoint_path = os.path.join(self.model_dir, "cp.ckpt")
        if os.path.exists(checkpoint_path + ".index"):  # Check if checkpoint exists
            self.model.load_weights(checkpoint_path)
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    def evaluate(self):
        """
        Evaluate the model across different SNRs
        """
        print(f"Starting evaluation of {self.model_name} model...")
        
        # Initialize results tracking
        all_results = pd.DataFrame(columns=["snr", "mse", "method", "scenario", "speed"])
        
        # Evaluate across SNR range
        for snr in range(self.config.eval_snr_min, self.config.eval_snr_max + 1, self.config.eval_snr_step):
            print(f"\nEvaluating SNR {snr}...")
            
            # Construct eval data directory for current SNR
            next_snr = snr + 1
            eval_data_dir = os.path.join(
                self.config.eval_base_dir,
                self.config.scenario,
                f"snr{snr}to{next_snr}_speed{self.config.speed}"
            )
            
            # Update dataset with current eval directory
            dataset_class = cds.get_dataset_class("Classic")
            self.dataset = dataset_class(
                eval_data_dir,
                train_split=self.config.eval_split,
                main_input_type=self.config.main_input_type,
                seed=self.config.seed
            )

            # Get evaluation data loader
            eval_loader = self.dataset.get_eval_loader(
                self.config.eval_batch_size,
                "test",
                "main"
            )

            # Evaluate model
            model_mse = []
            for h_ls, h_true in eval_loader:
                h_pred = self.model(h_ls)
                batch_mse = mse(h_true, h_pred).numpy()
                model_mse.append(batch_mse)

            # Calculate average MSE
            avg_model_mse = np.mean(model_mse)
            
            # Log results to wandb
            wandb.log({
                f'{self.model_name}_mse': avg_model_mse,
            })
            
            # Add model results to DataFrame
            all_results.loc[len(all_results)] = [
                snr, avg_model_mse, self.model_name,
                self.config.scenario, self.config.speed
            ]
            
            print(f"\nSNR {snr} Results:")
            print(f"{self.model_name} MSE: {avg_model_mse:.6f}")

        # Save final results
        results_file = os.path.join(self.log_dir, "evaluation_results.csv")
        all_results.to_csv(results_file, index=False)
        print(f"\nResults saved to {results_file}")

def main(args):
    wandb_config = {**vars(args)}
    run = wandb.init(
        project='INFOCOM2026', 
        config=wandb_config,
        name=args.wandb_name
    )

    eval_config = EvalConfig(**vars(args))
    set_random_seed(eval_config.seed)
    
    evaluator = Evaluator(eval_config)
    evaluator.setup()
    evaluator.evaluate()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Classic Channel Estimation Models")

    # Model related
    parser.add_argument("--trained_model_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True, 
                       choices=["HA02", "ChannelNet", "ReEsNet"])
    
    # Evaluation specific
    parser.add_argument("--eval_base_dir", type=str, default="./data_0_12/ps2_p72")
    parser.add_argument("--scenario", type=str, default="umi")
    parser.add_argument("--speed", type=str, default="30")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_split", type=float, default=0.5)
    parser.add_argument("--eval_snr_min", type=int, default=0)
    parser.add_argument("--eval_snr_max", type=int, default=20)
    parser.add_argument("--eval_snr_step", type=int, default=5)
    
    # Other configs
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output_dir", type=str, default="experiment_results/classic_eval")
    parser.add_argument("--wandb_name", type=str, default="umi_[3,10]_on_[0,12]_speed30_HA02")
    parser.add_argument("--experiment_name", type=str, 
                       default="siso_1_umi_block_1_ps2_p72")
    
    main(parser.parse_args(sys.argv[1:]))
