'''
Follow the basic logic of online_adapt_v3.py
1. eval in offline env
2. eval in online env
3. train in online env (using ha03 or dncnn)
4. eval in online env

other baseline: eval ddce
'''

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import sys
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)

import swanlab as wandb
wandb.login()
import pandas as pd
from dataclasses import dataclass, asdict
from collections import defaultdict
import time
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tqdm import tqdm
import numpy as np
import cebed.datasets_with_ssl as cds
import cebed.models as cm # Two-branch Model
from types import MethodType
from cebed.baselines import evaluate_baselines,evaluate_lmmse,evaluate_ls,evaluate_almmse,evaluate_ddce
from cebed.utils_eval import mse, generate_aux_data_online, get_ser,map_indices_to_symbols,real_to_complex_batch
from cebed.utils import set_random_seed
import argparse

# from sionna.phy.ofdm import LinearDetector, KBestDetector, EPDetector, MMSEPICDetector
# from sionna.phy.fec.ldpc import LDPC5GDecoder
from sionna.phy.channel import ApplyOFDMChannel


@dataclass
class EvalConfig:
    # model-related
    trained_model_dir: str = ""
    
    # training configs
    train_snr: int = 20
    train_data_dir: str = "" 
    train_batch_size: int = 64
    adapt_split: float = 0.5
    epochs: int = 5
    learning_rate: float = 5e-4

    # evaluation configs
    eval_split: float = 0.5
    eval_snr_min: int = -2 
    eval_snr_max: int = 2
    eval_snr_step: int = 1
    eval_base_dir: str = "./data/ps2_p72"  # Base directory for eval datasets
    scenario: str = "umi"     # Add scenario config
    speed: str = "30"         # Add speed config
    pretrain_scenario: str = "uma"  # Add this line for pre-training scenario
    pretrain_speed: str = "30"     # Add this line for pre-training scenario

    # dataset-related
    eval_dataset_name: str = "LabelPilot"
    eval_batch_size: int = 64
    main_input_type: str = "low"
    aux_input_type: str = "low"
    # aug_noise_std: float = 0.0

    # other configs
    seed: int = 43
    verbose: int = 1
    output_dir: str = "experiment_results/v3"
    wandb_name: str = ""


class Evaluator:
    """
    Evaluate the trained model and the related baselines
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.model_dir = self.config.trained_model_dir

        # read the config.yaml file from self.model_dir
        # saved_train_config = read_metadata(os.path.join(self.model_dir, "config.yaml"))
        self.train_exp_name = "siso_1_umi_block_1_ps2_p72"
        self.model_name = "HA03" # fix mask for main, random mask for aux
        # self.model_name = saved_train_config["model_name"]
        
        # set log dir
        os.makedirs(self.config.output_dir, exist_ok=True)
        # self.log_dir = os.path.join(self.config.output_dir,
        #                             self.mode,
        #                             self.config.eval_data_dir.split('/')[-1]
        #                             )

        self.log_dir = os.path.join(self.config.output_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        # set the model and dataset objects
        self.model = None
        self.dataset = None
        self.test_loader = None 

        # initialize baselines functions
        self.evaluate_ls = MethodType(evaluate_ls, self)
        self.evaluate_lmmse = MethodType(evaluate_lmmse, self)
        self.evaluate_almmse = MethodType(evaluate_almmse, self)
        self.evaluate_baselines = MethodType(evaluate_baselines, self)
        self.evaluate_ddce = MethodType(evaluate_ddce, self)

        # Add channel application objects
        self.apply_noiseless_channel = ApplyOFDMChannel(add_awgn=False, dtype=tf.as_dtype(tf.complex64))
        self.apply_noisy_channel = ApplyOFDMChannel(add_awgn=True, dtype=tf.as_dtype(tf.complex64))


    def setup(self):
        """Setup the evaluator"""
        # -------------------- Bind with data and create datasets -------------------- #
        # create dataset
        dataset_class = cds.get_dataset_class(self.config.eval_dataset_name)
        assert self.config.eval_dataset_name == "LabelPilot" # MUST be RandomMask for adapt-task
        # Construct data directory in online_adapt(), here is just for initialization
        self.dataset = dataset_class(
            self.config.train_data_dir,
            train_split=self.config.eval_split,
            main_input_type=self.config.main_input_type,
            aux_input_type=self.config.aux_input_type,
            # sym_error_rate=self.config.sym_error_rate,
            seed=self.config.seed,
            # aug_factor=1,
            # masking_type=self.config.masking_type
        )
        
        # ------------------------------- Create model ------------------------------- #
        # get model hyper-parameters from .yaml file
        assert self.model_name == "HA03"
        model_hparams = cm.get_model_hparams(
           self.model_name, self.train_exp_name
        )

        # get the class of model
        model_class = cm.get_model_class(self.model_name)
        if "output_dim" not in model_hparams:
            model_hparams["output_dim"] = self.dataset.output_shape
            # output_shape: [num_symbol, num_subcarrier, 2] for one sample in siso case

        # initialize the model
        self.model = model_class(model_hparams)
        
        # initial inputs to the encoder
        # input_shape_main, input_shape_aux = self.dataset.get_input_shape()
        # self.model.set_mask(self.dataset.env.get_mask())

        # TODO: hard code the input shape
        input = tf.ones([1,2,72,2])
        self.model(input)

        # ---------------------------------------------------------------------------- #
      
    def online_adapt(self) -> None:
        '''
        Main adapt pipeline: Load model -> Pre-adapt eval -> Training -> Post-adapt eval
        '''
        # Load model and compile
        self.model.load_weights(self.model_dir)
        self.model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(self.config.learning_rate),
            loss="mse"
        )
    
        # Pre-adapt evaluations with baselines
        print("Performing pre-adapt evaluation...")
        self._evaluate_pre_adapt(
            self.config.pretrain_scenario, 
            self.config.pretrain_speed, 
            "offline"
        )
        self._evaluate_pre_adapt(
            self.config.scenario, 
            self.config.speed, 
            "online"
        )
        
        # Prepare training data
        print("Preparing training data for adapt...")
        total_samples = self._prepare_training_data()

        # Perform adapt training
        print("Performing adapt training...")
        self._perform_adapt_training(total_samples)
        
        # Post-adapt evaluations
        print("Performing post-adapt evaluation...")
        self._evaluate_post_adapt(
            self.config.pretrain_scenario, 
            self.config.pretrain_speed, 
            "offline"
        )
        self._evaluate_post_adapt(
            self.config.scenario, 
            self.config.speed, 
            "online"
        )

    def _evaluate_pre_adapt(self, scenario: str, speed: str, scenario_name: str):
        """Evaluate model performance before adapt training"""
        print(f"\nEvaluating on {scenario_name} {scenario} environment before adapt...")
        
        dataset_class = cds.get_dataset_class(self.config.eval_dataset_name)
        
        # Evaluate baselines
        if scenario == "rt1":
            baselines = []
        else:
            # baselines = ["ALMMSE", "LMMSE", "LS", "DDCE"]
            baselines = []

        # Evaluate baselines
        print(f"{scenario_name} {scenario} Baselines: {baselines}...")

        snr_range = range(self.config.eval_snr_min, self.config.eval_snr_max + 1, self.config.eval_snr_step)
        for snr in tqdm(snr_range, desc=f"Pre-adapt {scenario_name} SNR Evaluation"):
            print(f"Evaluating SNR {snr} dB...")
            # Construct eval data directory for current SNR
            next_snr = snr + 1
            eval_data_dir = os.path.join(
                self.config.eval_base_dir,
                scenario,
                f"snr{snr}to{next_snr}_speed{speed}"
            )
            test_mse = defaultdict(list)
            mses_before = pd.DataFrame(columns=["snr", "mse", "method", "seed"])

            
            # Update dataset with eval directory
            self.dataset = dataset_class(
                eval_data_dir,
                train_split=self.config.eval_split,
                main_input_type=self.config.main_input_type,
                aux_input_type=self.config.aux_input_type,
                # sym_error_rate=self.config.sym_error_rate,
                seed=self.config.seed,
                # aug_factor=1,
                # masking_type=self.config.masking_type
            )

            num_steps = (
                len(self.dataset.test_indices[0]) + self.config.eval_batch_size - 1
            ) // self.config.eval_batch_size

            for step in tqdm(range(num_steps), desc=f"Baseline Evaluation SNR {snr}", leave=False):
                start, end = step * self.config.eval_batch_size, min(
                    (step + 1) * self.config.eval_batch_size,
                    len(self.dataset.test_indices[0]),
                )
                
                tx_symbols = self.dataset.x_samples[0][self.dataset.test_indices[0]][start:end]
                rx_symbols = self.dataset.y_samples[0][self.dataset.test_indices[0]][start:end]
                h_true = self.dataset.test_y_main[0][start:end]
                
                noiseless_signals = self.apply_noiseless_channel(tx_symbols, h_true)
                baseline_mses = self.evaluate_baselines(
                    rx_symbols, 
                    noiseless_signals, 
                    h_true, 
                    snr, 
                    baselines
                )
                
                for baseline in baselines:
                    test_mse[baseline].append(baseline_mses[baseline])
        
            # Log baseline results
            for baseline in baselines:
                avg_baseline_mse = np.mean(test_mse[baseline])
                wandb.log({
                    f'{scenario_name}_{baseline}_mse': avg_baseline_mse
                })
                mses_before.loc[len(mses_before)] = [snr, avg_baseline_mse, baseline, self.config.seed]

             # ----------------------- Evaluate the pretrained model ---------------------- #
            self.dataset = dataset_class(
                eval_data_dir,
                train_split=self.config.eval_split,
                main_input_type=self.config.main_input_type,
                aux_input_type=self.config.aux_input_type,
                # sym_error_rate=self.config.sym_error_rate,
                seed=self.config.seed,
                # aug_factor=1,
                # masking_type=self.config.masking_type
            )

            num_steps = (
                len(self.dataset.test_indices[0]) + self.config.eval_batch_size - 1
            ) // self.config.eval_batch_size

            self.eval_loader = self.dataset.get_eval_loader(self.config.eval_batch_size, "test", "both")
            eval_data_iterator = iter(self.eval_loader)

            eval_channel_mse = []
            eval_aux_mse = []
            
            for step in tqdm(range(num_steps), desc=f"Model Evaluation SNR {snr}", leave=False):
                (_, h_true), (h_ls_pilot, h_label_pilot) = next(eval_data_iterator)
                
                h_pred_label_pilot = self.model(h_ls_pilot)
                # h_pred_label_pilot shape: [batch, 2, 72, 2]
                # bilinear interpolation to the model output
                h_pred = tf.image.resize(h_pred_label_pilot, [14, 72], method=tf.image.ResizeMethod.BILINEAR)
                
                # Cast tensors to same dtype to avoid dtype mismatch
                h_true = tf.cast(h_true, tf.float32)
                h_pred = tf.cast(h_pred, tf.float32)
                
                step_channel_mse = mse(h_true, h_pred).numpy()
                eval_channel_mse.append(step_channel_mse)

            # Calculate and log average MSE
            avg_channel_mse = np.mean(eval_channel_mse)
            
            wandb.log({
                f'pre_adapt_{scenario_name}_channel_mse': avg_channel_mse,
                # f'pre_adapt_{scenario_name}_aux_mse': avg_aux_mse
            })
            
            # Save results
            mses_before.loc[len(mses_before)] = [snr, avg_channel_mse, "Model", self.config.seed]
            results_file = os.path.join(self.log_dir, f"pre_adapt_{scenario_name}_{scenario}_results_snr_{snr}.csv")
            mses_before.to_csv(results_file, index=False)

            print(f"\nPre-adapt {scenario_name} Scenario (SNR {snr}) Evaluation Results:")
            print(f"Channel Estimation MSE: {avg_channel_mse:.6f}")
            # print(f"Auxiliary Task MSE: {avg_aux_mse:.6f}")
            print(f"Results saved to {results_file}")
    
    
    def _prepare_training_data(self):
        """Prepare and merge training data from multiple domains"""
        dataset_class = cds.get_dataset_class(self.config.eval_dataset_name)
        
        # Initialize dataset with training data directory
        self.dataset = dataset_class(
            self.config.train_data_dir,
            train_split=self.config.adapt_split,
            main_input_type=self.config.main_input_type,
            aux_input_type=self.config.aux_input_type,
            # sym_error_rate=self.config.sym_error_rate,
            seed=self.config.seed,
            # aug_factor=1,
            # masking_type=self.config.masking_type
        )
        
        # Preprocess for multi-snr training
        test_x_list, test_y_list = [], []
        test_mx_list, test_my_list = [], []
        test_ax_list, test_ay_list = [], []
        
        for ds in range(self.dataset.num_domains):
            idxs = self.dataset.test_indices[ds]
            test_x_list.append(self.dataset.x_samples[ds][idxs])
            test_y_list.append(self.dataset.y_samples[ds][idxs])
            test_mx_list.append(self.dataset.test_x_main[ds])
            test_my_list.append(self.dataset.test_y_main[ds])
            test_ax_list.append(self.dataset.test_x_aux[ds])
            test_ay_list.append(self.dataset.test_y_aux[ds])

        # Merge and shuffle data
        x_merged = np.concatenate(test_x_list, axis=0)
        y_merged = np.concatenate(test_y_list, axis=0)
        mx_merged = np.concatenate(test_mx_list, axis=0)
        my_merged = np.concatenate(test_my_list, axis=0)
        ax_merged = np.concatenate(test_ax_list, axis=0)
        ay_merged = np.concatenate(test_ay_list, axis=0)

        perm = np.arange(len(x_merged))
        np.random.shuffle(perm)
        x_merged, y_merged = x_merged[perm], y_merged[perm]
        mx_merged, my_merged = mx_merged[perm], my_merged[perm]
        ax_merged, ay_merged = ax_merged[perm], ay_merged[perm]

        # Update dataset with merged data
        self.dataset.x_samples = [x_merged]
        self.dataset.y_samples = [y_merged]
        self.dataset.test_x_main = [mx_merged]
        self.dataset.test_y_main = [my_merged]
        self.dataset.test_x_aux = [ax_merged]
        self.dataset.test_y_aux = [ay_merged]
        self.dataset.test_indices = [np.arange(len(x_merged))]

        self.train_loader = self.dataset.get_eval_loader(
            self.config.train_batch_size,
            "test",
            "both"
        )
        
        for domain_idx in range(len(self.dataset.test_indices)):
            print(f"Domain {domain_idx}: {len(self.dataset.test_indices[domain_idx])} samples")
            
        return len(self.dataset.test_indices[0])
    

    def _perform_adapt_training(self, total_samples):
        """Perform the actual adapt training process"""
        # Noise computation can be on CPU
        noise_lin = tf.pow(10.0, -self.config.train_snr / 10.0)
        
        domain_idx = 0
        num_steps = (total_samples + self.config.train_batch_size - 1) // self.config.train_batch_size
        
        training_step_counter = 0

        # Pre-adaptation evaluation
        before_adapt_test_data = iter(self.train_loader)
        before_adapt_channel_mse = []

        print(f"Starting pre-adaptation evaluation with {num_steps} steps...")
        for step in tqdm(range(num_steps), desc="Pre-adaptation Evaluation", leave=False):
            (_, h_true), (h_ls_pilot, h_label_pilot) = next(before_adapt_test_data)
            h_pred_label_pilot = self.model(h_ls_pilot)
            h_pred = tf.image.resize(h_pred_label_pilot, [14, 72], method=tf.image.ResizeMethod.BILINEAR)
            
            # Cast tensors to same dtype to avoid dtype mismatch
            h_true = tf.cast(h_true, tf.float32)
            h_pred = tf.cast(h_pred, tf.float32)
            
            step_channel_mse = mse(h_true, h_pred).numpy()
            before_adapt_channel_mse.append(step_channel_mse)

        pre_adapt_avg_channel = np.mean(before_adapt_channel_mse)

        print(f"Pre-adaptation - Channel MSE: {pre_adapt_avg_channel:.6f}")

        wandb.log({
            'pre_adapt_channel_mse': pre_adapt_avg_channel  
        })
        
        # Training epochs
        for epoch in tqdm(range(self.config.epochs), desc="Adaptation Training Epochs"):
            print(f"\n=== Starting Adaptation Epoch {epoch+1}/{self.config.epochs} ===")
            channel_mse = []
            label_pilot_mse = []
            batch_data = iter(self.train_loader)
            
            for step in tqdm(range(num_steps), desc=f"Epoch {epoch+1} Training Steps", leave=False):
                start, end = step * self.config.train_batch_size, min(
                    (step + 1) * self.config.train_batch_size,
                    total_samples,
                )
                actual_batch_size = end - start

                # Get batch data
                (_, h_true), (h_ls_pilot, h_label_pilot) = next(batch_data)

                with tf.GradientTape() as tape:
                    h_pred_label_pilot = self.model(h_ls_pilot)
                    h_pred = tf.image.resize(h_pred_label_pilot, [14, 72], method=tf.image.ResizeMethod.BILINEAR)
                    main_loss = self.model.compiled_loss(h_true, h_pred)
                    train_loss = self.model.compiled_loss(h_label_pilot, h_pred_label_pilot)
                
                # Cast tensors to same dtype to avoid dtype mismatch
                h_true = tf.cast(h_true, tf.float32)
                h_pred = tf.cast(h_pred, tf.float32)
                h_label_pilot = tf.cast(h_label_pilot, tf.float32)
                h_pred_label_pilot = tf.cast(h_pred_label_pilot, tf.float32)
                    
                step_main_mse = mse(h_true, h_pred).numpy()
                step_label_pilot_mse = mse(h_label_pilot, h_pred_label_pilot).numpy()
                channel_mse.append(step_main_mse)
                label_pilot_mse.append(step_label_pilot_mse)

                grads = tape.gradient(train_loss, self.model.trainable_weights)
                self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))


                # Log step-wise metrics to wandb
                wandb.log({
                    'step': training_step_counter,
                    'adapt_step_channel_mse': step_main_mse,
                    'adapt_step_train_loss': step_label_pilot_mse,
                })

            # Log epoch metrics
            avg_channel_mse = np.mean(channel_mse)
            avg_label_pilot_mse = np.mean(label_pilot_mse)
            
            # Log to wandb with clear output format distinction
            wandb.log({
                'adapt_epoch_channel_mse': avg_channel_mse,
                'adapt_epoch_train_loss': avg_label_pilot_mse,
            })
            
            print(f"Epoch {epoch+1} Summary:")
            print(f"Average Channel MSE: {avg_channel_mse:.6f}")
            print(f"Average Auxiliary MSE: {avg_label_pilot_mse:.6f}")
            print("-" * 50)


    def _evaluate_post_adapt(self, scenario: str, speed: str, scenario_name: str):
        """Evaluate model performance after adapt training"""
        print(f"\nEvaluating on {scenario_name} {scenario} environment after adapt...")
        
        dataset_class = cds.get_dataset_class(self.config.eval_dataset_name)
        
        snr_range = range(self.config.eval_snr_min, self.config.eval_snr_max + 1, self.config.eval_snr_step)
        for snr in tqdm(snr_range, desc=f"Post-adapt {scenario_name} SNR Evaluation"):
            print(f"Evaluating SNR {snr} dB...")
            # Construct eval data directory
            next_snr = snr + 1
            eval_data_dir = os.path.join(
                self.config.eval_base_dir,
                scenario,
                f"snr{snr}to{next_snr}_speed{speed}"
            )
            
            # Update dataset
            self.dataset = dataset_class(
                eval_data_dir,
                train_split=self.config.eval_split,
                main_input_type=self.config.main_input_type,
                aux_input_type=self.config.aux_input_type,
                # sym_error_rate=self.config.sym_error_rate,
                seed=self.config.seed,
                # aug_factor=1,
                # masking_type=self.config.masking_type
            )

            num_steps = (
                len(self.dataset.test_indices[0]) + self.config.eval_batch_size - 1
            ) // self.config.eval_batch_size

            # Initialize tracking
            mses = pd.DataFrame(columns=["snr", "mse", "method", "seed"])
            
            # Evaluate the model - NO device forcing
            self.eval_loader = self.dataset.get_eval_loader(self.config.eval_batch_size, "test", "both")
            eval_data_iterator = iter(self.eval_loader)

            eval_channel_mse = []
            eval_aux_mse = []
            
            for step in tqdm(range(num_steps), desc=f"Post-adapt Model Evaluation SNR {snr}", leave=False):
                (_, h_true), (h_ls_pilot, h_label_pilot) = next(eval_data_iterator)
                
                h_pred_label_pilot = self.model(h_ls_pilot)
                # h_pred_label_pilot shape: [batch, 2, 72, 2]
                # bilinear interpolation to the model output
                h_pred = tf.image.resize(h_pred_label_pilot, [14, 72], method=tf.image.ResizeMethod.BILINEAR)
                
                # Cast tensors to same dtype to avoid dtype mismatch
                h_true = tf.cast(h_true, tf.float32)
                h_pred = tf.cast(h_pred, tf.float32)
                
                step_channel_mse = mse(h_true, h_pred).numpy()
                eval_channel_mse.append(step_channel_mse)

            # Calculate and log results
            avg_eval_channel_mse = np.mean(eval_channel_mse)
            
            wandb.log({
                f'post_adapt_{scenario_name}_channel_mse': avg_eval_channel_mse,
                # f'post_adapt_{scenario_name}_aux_mse': avg_eval_aux_mse,
            })
            
            # Save results
            mses.loc[len(mses)] = [snr, avg_eval_channel_mse, "Model", self.config.seed]
            results_file = os.path.join(self.log_dir, f"post_adapt_{scenario_name}_{scenario}_results_snr_{snr}.csv")
            mses.to_csv(results_file, index=False)
            
            print(f"SNR {snr} Evaluation Results:")
            print(f"Channel Estimation MSE: {avg_eval_channel_mse:.6f}")
            # print(f"Auxiliary Task MSE: {avg_eval_aux_mse:.6f}")
            print(f"Results saved to {results_file}")


def main(args):
    wandb_config = {**vars(args)}
    run = wandb.init(workspace="ttt4wireless",
                    project='INFOCOM2026', 
                     config=wandb_config,
                     name=args.wandb_name)
    
    eval_config = EvalConfig(
        **vars(args)
    )

    # initialize evaluator
    set_random_seed(eval_config.seed)
    evaluator = Evaluator(eval_config)
    evaluator.setup()

    # load trained model into the evaluator
    evaluator.online_adapt()


if __name__ == "__main__":
    print("Running online adapt channel estimation script...")
    parser = argparse.ArgumentParser(description="Online adapt Channel Estimation")

    # Model related
    parser.add_argument("--trained_model_dir", type=str, default="train_output/HA03_bs64_lr0.001_baseline/ha03.h5")
    
    # Dataset related
    parser.add_argument("--eval_dataset_name", type=str, default="LabelPilot")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--main_input_type", type=str, default="low", choices=["low", "raw"])
    parser.add_argument("--aux_input_type", type=str, default="low", choices=["low", "raw"])
    
    # Training configs
    parser.add_argument("--adapt_split", type=float, default=0.75)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    
    # Other configs
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output_dir", type=str, default="experiment_results/ha03")
    parser.add_argument("--wandb_name", type=str, default="online_adapt_v3")
    
    # Training specific
    parser.add_argument("--train_data_dir", type=str, default="./data/ps2_p72/rt1/snr20to21_speed5")
    parser.add_argument("--train_snr", type=int, default=20)
    
    # Evaluation specific
    parser.add_argument("--eval_split", type=float, default=0.5)
    parser.add_argument("--eval_base_dir", type=str, default="./data/ps2_p72")
    parser.add_argument("--scenario", type=str, default="rt1")
    parser.add_argument("--speed", type=str, default="5")
    parser.add_argument("--eval_snr_min", type=int, default=15)
    parser.add_argument("--eval_snr_max", type=int, default=15)
    parser.add_argument("--eval_snr_step", type=int, default=1)

    # Add pre-training scenario argument
    parser.add_argument("--pretrain_scenario", type=str, default="rt0",
                       help="Scenario used during pre-training (for forgetting evaluation)")
    parser.add_argument("--pretrain_speed", type=str, default="5",
                       help="Speed of the pre-training scenario")

    with tf.device('/CPU'):
        print("Forcing CPU for evaluation, unless APPLE fixes MPS support")
        main(parser.parse_args(sys.argv[1:]))
