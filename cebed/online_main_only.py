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
import cebed.models_with_ssl as cm # Two-branch Model
from types import MethodType
from cebed.baselines import evaluate_baselines,evaluate_lmmse,evaluate_ls,evaluate_almmse, evaluate_ddce
from cebed.utils_eval import mse, generate_aux_data_online, get_ser,map_indices_to_symbols,real_to_complex_batch
from cebed.utils import set_random_seed
import argparse

from sionna.phy.ofdm import LinearDetector, KBestDetector, EPDetector, MMSEPICDetector
from sionna.phy.fec.ldpc import LDPC5GDecoder
from sionna.phy.channel import ApplyOFDMChannel


@dataclass
class EvalConfig:
    # model-related
    trained_model_dir: str = ""
    
    # training configs
    train_snr: int = 20
    train_data_dir: str = "" 
    train_batch_size: int = 64
    ttt_split: float = 0.5
    epochs: int = 5
    supervised: int = 1  # Changed default to supervised mode
    ssl: int = 0        # Changed default to disable SSL
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
    pretrain_speed: str = "5"     # Add this line for pre-training scenario

    # dataset-related
    eval_dataset_name: str = "RandomMask"
    eval_batch_size: int = 64
    main_input_type: str = "low"
    aux_input_type: str = "low"
    aug_noise_std: float = 0.0
    sym_error_rate: float = 0.0

    # method configs
    det_method: str = "lmmse"
    output: str = "symbol"
    aug_times: int = 2
    masking_type: str = "discrete"

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
        self.model_name = "ReconMAE_MainOnly"
        # self.model_name = saved_train_config["model_name"]
        
        # set log dir
        os.makedirs(self.config.output_dir, exist_ok=True)
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
        dataset_class = cds.get_dataset_class(self.config.eval_dataset_name)
        assert self.config.eval_dataset_name == "RandomMask" # MUST be RandomMask for ttt-task
        # Construct data directory in online_adapt(), here is just for initialization
        self.dataset = dataset_class(
            self.config.train_data_dir,
            train_split=self.config.eval_split,
            main_input_type=self.config.main_input_type,
            aux_input_type=self.config.aux_input_type,
            sym_error_rate=self.config.sym_error_rate,
            seed=self.config.seed,
            aug_factor=1,
            masking_type=self.config.masking_type
        )
        
        # ------------------------------- Create model ------------------------------- #
        assert self.model_name == "ReconMAE_MainOnly"
        model_hparams = cm.get_model_hparams(
           self.model_name, self.train_exp_name
        )

        model_class = cm.get_model_class(self.model_name)
        if "output_dim" not in model_hparams:
            model_hparams["output_dim"] = self.dataset.output_shape

        # initialize the model
        self.model = model_class(model_hparams)
        
        # initial inputs to the encoder
        input_shape_main, input_shape_aux = self.dataset.get_input_shape()
        self.model.set_mask(self.dataset.env.get_mask())

        # build the model
        main_low_input = tf.ones([1, input_shape_main[0], input_shape_main[1], input_shape_main[2]])
        aux_low_input = tf.ones([1, input_shape_aux[0], input_shape_aux[1], input_shape_aux[2]])
        example_mask = self.dataset.env.get_mask()
        example_mask = tf.squeeze(example_mask)
        example_mask = tf.expand_dims(example_mask, axis=0) # [batch, 14, 72]
        self.model((main_low_input, (aux_low_input, example_mask)))

        # ---------------------------------------------------------------------------- #
        
        # NOTE:
        self.num_tx = 1
        self.num_bits_per_symbol = 2
        self.k = self.dataset.env.k
        self.n = self.dataset.env.n
        
        ## If output is symbol, then no FEC is used and hard decision are output
        self.output = self.config.output # "symbol" or "bit"
        self.hard_out = (self.output == "symbol")
        self.coded = (self.output == "bit")
        
        rg = self.dataset.env.rg
        sm = self.dataset.env.sm
        
        # Detection
        if self.config.det_method == "lmmse":
            self.detector = LinearDetector("lmmse", self.output, "app", rg, sm, constellation_type="qam", num_bits_per_symbol=self.num_bits_per_symbol, hard_out=self.hard_out)
        elif self.config.det_method == 'k-best':
            self.detector = KBestDetector(self.output, self.num_tx, 64, rg, sm, constellation_type="qam", num_bits_per_symbol=self.num_bits_per_symbol, hard_out=self.hard_out)
        elif self.config.det_method == "ep":
            self.detector = EPDetector(self.output, rg, sm, self.num_bits_per_symbol, l=10, hard_out=self.hard_out)
        elif self.config.det_method == 'mmse-pic':
            self.detector = MMSEPICDetector(self.output, "app", rg, sm, num_iter=4, constellation_type="qam", num_bits_per_symbol=self.num_bits_per_symbol, hard_out=self.hard_out)
        if self.coded:
            self.channel_encoder = self.dataset.env.channel_encoder
            self.channel_decoder = LDPC5GDecoder(self.channel_encoder, hard_out=True, num_iter=10)
        
        self.aug_times = self.config.aug_times
        self.mapper = self.dataset.env.mapper

    def online_adapt(self) -> None:
        '''
        Main TTT pipeline: Load model -> Pre-TTT eval -> Training -> Post-TTT eval
        '''
        # Load model and compile
        self.model.load_weights(self.model_dir)
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(self.config.learning_rate),
            loss="mse"
        )

        # Pre-TTT evaluations with baselines
        print("Performing pre-ttt evaluation...")
        self._evaluate_pre_ttt(
            self.config.pretrain_scenario, 
            self.config.pretrain_speed, 
            "offline"
        )
        self._evaluate_pre_ttt(
            self.config.scenario, 
            self.config.speed, 
            "online"
        )
        
        # Prepare training data
        print("Preparing training data for TTT...")
        total_samples = self._prepare_training_data()

        # Perform TTT training
        print("Performing TTT training...")
        self._perform_ttt_training(total_samples)
        
        # Post-TTT evaluations
        print("Performing post-ttt evaluation...")
        self._evaluate_post_ttt(
            self.config.pretrain_scenario, 
            self.config.pretrain_speed, 
            "offline"
        )
        self._evaluate_post_ttt(
            self.config.scenario, 
            self.config.speed, 
            "online"
        )

    def _evaluate_pre_ttt(self, scenario: str, speed: str, scenario_name: str):
        """Evaluate model performance before TTT training"""
        print(f"\nEvaluating on {scenario_name} {scenario} environment before TTT...")
        
        dataset_class = cds.get_dataset_class(self.config.eval_dataset_name)
        
        # Evaluate baselines
        if scenario in ["rt0", "rt1", "rt2", "rt3", "rt4", "uma", "umi"]:
            baselines = []
        else:
            baselines = ["ALMMSE", "LMMSE", "LS", "DDCE"]

        snr_range = range(self.config.eval_snr_min, self.config.eval_snr_max + 1, self.config.eval_snr_step)
        for snr in tqdm(snr_range, desc=f"Pre-TTT {scenario_name} SNR Evaluation"):
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
                sym_error_rate=self.config.sym_error_rate,
                seed=self.config.seed,
                aug_factor=1,
                masking_type=self.config.masking_type
            )

            num_steps = (
                len(self.dataset.test_indices[0]) + self.config.eval_batch_size - 1
            ) // self.config.eval_batch_size

            # Evaluate baselines
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

            # Evaluate model
            self.eval_loader = self.dataset.get_eval_loader(self.config.eval_batch_size, "test", "both")
            eval_data_iterator = iter(self.eval_loader)

            eval_channel_mse = []
            
            for step in tqdm(range(num_steps), desc=f"Model Evaluation SNR {snr}", leave=False):
                (h_ls, h_true), (x_aux, mask, y_aux) = next(eval_data_iterator)
                
                h_pred = self.model((h_ls, (x_aux, mask)))
                
                step_channel_mse = mse(h_true, h_pred).numpy()
                eval_channel_mse.append(step_channel_mse)

            # Calculate and log average MSE
            avg_channel_mse = np.mean(eval_channel_mse)
            
            wandb.log({
                f'pre_ttt_{scenario_name}_channel_mse': avg_channel_mse
            })
            
            # Save results
            mses_before.loc[len(mses_before)] = [snr, avg_channel_mse, "Model", self.config.seed]
            results_file = os.path.join(self.log_dir, f"pre_ttt_{scenario_name}_{scenario}_results_snr_{snr}.csv")
            mses_before.to_csv(results_file, index=False)

            print(f"\nPre-TTT {scenario_name} Scenario (SNR {snr}) Evaluation Results Saved to {results_file}")

    def _prepare_training_data(self):
        """Prepare and merge training data from multiple domains"""
        dataset_class = cds.get_dataset_class(self.config.eval_dataset_name)
        
        # Initialize dataset with training data directory
        self.dataset = dataset_class(
            self.config.train_data_dir,
            train_split=self.config.ttt_split,
            main_input_type=self.config.main_input_type,
            aux_input_type=self.config.aux_input_type,
            sym_error_rate=self.config.sym_error_rate,
            seed=self.config.seed,
            aug_factor=1,
            masking_type=self.config.masking_type
        )
        
        # Preprocess for multi-snr training
        test_x_list, test_y_list = [], []
        test_mx_list, test_my_list = [], []
        test_ax1_list, test_ax2_list, test_ay_list = [], [], []
        
        for ds in range(self.dataset.num_domains):
            idxs = self.dataset.test_indices[ds]
            test_x_list.append(self.dataset.x_samples[ds][idxs])
            test_y_list.append(self.dataset.y_samples[ds][idxs])
            test_mx_list.append(self.dataset.test_x_main[ds])
            test_my_list.append(self.dataset.test_y_main[ds])
            test_ax1_list.append(self.dataset.test_x1_aux[ds])
            test_ax2_list.append(self.dataset.test_x2_aux[ds])
            test_ay_list.append(self.dataset.test_y_aux[ds])

        # Merge and shuffle data
        x_merged = np.concatenate(test_x_list, axis=0)
        y_merged = np.concatenate(test_y_list, axis=0)
        mx_merged = np.concatenate(test_mx_list, axis=0)
        my_merged = np.concatenate(test_my_list, axis=0)
        ax1_merged = np.concatenate(test_ax1_list, axis=0)
        ax2_merged = np.concatenate(test_ax2_list, axis=0)
        ay_merged = np.concatenate(test_ay_list, axis=0)

        perm = np.arange(len(x_merged))
        np.random.shuffle(perm)
        x_merged, y_merged = x_merged[perm], y_merged[perm]
        mx_merged, my_merged = mx_merged[perm], my_merged[perm]
        ax1_merged, ax2_merged, ay_merged = ax1_merged[perm], ax2_merged[perm], ay_merged[perm]

        # Update dataset with merged data
        self.dataset.x_samples = [x_merged]
        self.dataset.y_samples = [y_merged]
        self.dataset.test_x_main = [mx_merged]
        self.dataset.test_y_main = [my_merged]
        self.dataset.test_x1_aux = [ax1_merged]
        self.dataset.test_x2_aux = [ax2_merged]
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

    def _perform_detection_and_recovery(self, h_pred, tx_symbols, rx_symbols, noise_lin, actual_batch_size):
        """Perform symbol detection and recovery"""
        h_hat = real_to_complex_batch(h_pred) 
        h_hat = tf.reshape(h_hat, [h_pred.shape[0], 1, 1, 1, 1, h_pred.shape[1], h_pred.shape[2]])
        
        # Make sure h_hat matches rx_symbols batch size
        if h_hat.shape[0] != actual_batch_size:
            h_hat = h_hat[:actual_batch_size]

        # Define bits_shape for symbol output
        bits_shape = [h_pred.shape[0], 1, 1, self.n] 
        err_var = 0.0

        # Detection
        if self.config.det_method == "mmse-pic":
            if self.output == "bit":
                prior_shape = bits_shape
            elif self.output == "symbol":
                prior_shape = tf.concat([tf.shape(tx_symbols), [self.num_bits_per_symbol]], axis=0)
            prior = tf.zeros(prior_shape)
            det_out = self.detector(rx_symbols, h_hat, prior, err_var, noise_lin)
        else:
            det_out = self.detector(rx_symbols, h_hat, err_var, noise_lin)

        # Recovery based on output format
        if self.output == "bit":
            llr = tf.reshape(det_out, bits_shape)
            b_hat = self.channel_decoder(llr)
            c_hat = self.channel_encoder(b_hat)
            recov_tx_data_sym = self.mapper(c_hat) 
            recov_tx_symbols = self.dataset.env.rg_mapper(recov_tx_data_sym)
            recovered_symbols = recov_tx_symbols
        elif self.output == "symbol":
            est_tx_data_sym = map_indices_to_symbols(det_out)
            est_tx_symbols = self.dataset.env.rg_mapper(est_tx_data_sym)
            recovered_symbols = est_tx_symbols

        # Compute Symbol Error Rate
        ser = get_ser(tx_symbols, recovered_symbols)
        
        return recovered_symbols, ser

    def _perform_ttt_training(self, total_samples):
        """Perform the actual TTT training process"""
        # Noise computation can be on CPU
        noise_lin = tf.pow(10.0, -self.config.train_snr / 10.0)
        
        domain_idx = 0
        num_steps = (total_samples + self.config.train_batch_size - 1) // self.config.train_batch_size
        
        # Pre-adaptation evaluation
        before_adapt_test_data = iter(self.train_loader)
        before_adapt_channel_mse = []

        print(f"Starting pre-adaptation evaluation with {num_steps} steps...")
        for step in tqdm(range(num_steps), desc="Pre-adaptation Evaluation", leave=False):
            (h_ls, h_true), (x_aux, mask, y_aux) = next(before_adapt_test_data)
            h_pred = self.model((h_ls, (x_aux, mask)))
            step_channel_mse = mse(h_true, h_pred).numpy()
            before_adapt_channel_mse.append(step_channel_mse)

        pre_adapt_avg_channel = np.mean(before_adapt_channel_mse)

        print(f"Pre-adaptation - Channel MSE: {pre_adapt_avg_channel:.6f}")

        wandb.log({
            'ttt/epoch_channel_mse': pre_adapt_avg_channel
        })

        # Training epochs
        for epoch in tqdm(range(self.config.epochs), desc="TTT Training Epochs"):
            print(f"\n=== Starting TTT Epoch {epoch+1}/{self.config.epochs} ===")
            channel_mse = []
            epoch_sers = []
            batch_data = iter(self.train_loader)
            
            for step in tqdm(range(num_steps), desc=f"Epoch {epoch+1} Training Steps", leave=False):
                start, end = step * self.config.train_batch_size, min(
                    (step + 1) * self.config.train_batch_size,
                    total_samples,
                )
                actual_batch_size = end - start

                # Get batch data
                (h_ls, h_true), _ = next(batch_data)
                h_pred = self.model.main_branch(h_ls)
                
                # Get symbols for detection
                tx_symbols = self.dataset.x_samples[domain_idx][self.dataset.test_indices[domain_idx]][start:end]
                rx_symbols = self.dataset.y_samples[domain_idx][self.dataset.test_indices[domain_idx]][start:end] 

                # Perform detection and recovery
                recovered_symbols, ser = self._perform_detection_and_recovery(
                    h_pred, tx_symbols, rx_symbols, noise_lin, actual_batch_size
                )
                epoch_sers.append(ser)

                # Generate auxiliary data
                (aug_x_aux, aug_mask, aug_y_aux) = generate_aux_data_online(
                    rx_symbols, recovered_symbols, self.aug_times, self.config.masking_type
                )

                # TTT training step
                with tf.GradientTape() as tape:
                    h_pred = self.model((h_ls, (aug_x_aux, aug_mask)))
                    main_loss = self.model.compiled_loss(h_true, h_pred)
                    pseudo_loss = self.model.compiled_loss(aug_y_aux, h_pred)

                step_main_mse = mse(h_true, h_pred).numpy()
                channel_mse.append(step_main_mse)

                # Apply gradients based on training mode
                if self.config.ssl:
                    self.model.encoder.trainable = True
                    self.model.main_decoder.trainable = False
                    grads = tape.gradient(pseudo_loss, self.model.trainable_weights)
                    self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                elif self.config.supervised:
                    self.model.encoder.trainable = True
                    self.model.main_decoder.trainable = False
                    grads = tape.gradient(main_loss, self.model.trainable_weights)
                    self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                # Log step-wise metrics to wandb
                wandb.log({
                    'ttt/step_channel_mse': step_main_mse,
                    'ttt/step_ser': ser,
                })

            # Log epoch metrics
            avg_ser = np.mean(epoch_sers)
            avg_channel_mse = np.mean(channel_mse)
            
            wandb.log({
                'ttt/epoch_channel_mse': avg_channel_mse,
                'ttt/epoch_avg_ser': avg_ser,
            })
            
            print(f"Epoch {epoch+1} Summary (Output Format: {self.config.output}):")
            print(f"Average Channel MSE: {avg_channel_mse:.6f}")
            print(f"Average Symbol Error Rate: {avg_ser:.6f}")
            print("-" * 50)
        
    def _evaluate_post_ttt(self, scenario: str, speed: str, scenario_name: str):
        """Evaluate model performance after TTT training"""
        print(f"\nEvaluating on {scenario_name} {scenario} environment after TTT...")
        
        dataset_class = cds.get_dataset_class(self.config.eval_dataset_name)
        
        snr_range = range(self.config.eval_snr_min, self.config.eval_snr_max + 1, self.config.eval_snr_step)
        for snr in tqdm(snr_range, desc=f"Post-TTT {scenario_name} SNR Evaluation"):
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
                sym_error_rate=self.config.sym_error_rate,
                seed=self.config.seed,
                aug_factor=1,
                masking_type=self.config.masking_type
            )

            num_steps = (
                len(self.dataset.test_indices[0]) + self.config.eval_batch_size - 1
            ) // self.config.eval_batch_size

            # Initialize tracking
            mses = pd.DataFrame(columns=["snr", "mse", "method", "seed"])
            
            # Evaluate the model
            self.eval_loader = self.dataset.get_eval_loader(self.config.eval_batch_size, "test", "both")
            eval_data_iterator = iter(self.eval_loader)

            eval_channel_mse = []
            
            for step in tqdm(range(num_steps), desc=f"Post-TTT Model Evaluation SNR {snr}", leave=False):
                (h_ls, h_true), (x_aux, mask, y_aux) = next(eval_data_iterator)
                
                h_pred = self.model((h_ls, (x_aux, mask)))
                
                step_channel_mse = mse(h_true, h_pred).numpy()
                eval_channel_mse.append(step_channel_mse)

            # Calculate and log results
            avg_eval_channel_mse = np.mean(eval_channel_mse)
            
            wandb.log({
                f'post_ttt_{scenario_name}_channel_mse': avg_eval_channel_mse
            })
            
            # Save results
            mses.loc[len(mses)] = [snr, avg_eval_channel_mse, "Model", self.config.seed]
            results_file = os.path.join(self.log_dir, f"post_ttt_{scenario_name}_{scenario}_results_snr_{snr}.csv")
            mses.to_csv(results_file, index=False)
            
            print(f"SNR {snr} Evaluation Results:")
            print(f"Channel Estimation MSE: {avg_eval_channel_mse:.6f}")
            print(f"Results saved to {results_file}")


def main(args):
    wandb_config = {**vars(args)}
    run = wandb.init(project='INFOCOM2026', 
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
    print("Running online main-only TTT channel estimation script...")
    parser = argparse.ArgumentParser(description="Online Main-Only TTT Channel Estimation")

    # Model related
    parser.add_argument("--trained_model_dir", type=str, default="./models/mainOnly1_weights_ds_discrete.h5")
    
    # Dataset related
    parser.add_argument("--eval_dataset_name", type=str, default="RandomMask")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--main_input_type", type=str, default="low", choices=["low", "raw"])
    parser.add_argument("--aux_input_type", type=str, default="low", choices=["low", "raw"])
    
    # Training configs
    parser.add_argument("--ttt_split", type=float, default=0.75)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--supervised", type=int, default=1)
    parser.add_argument("--ssl", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    
    # Method configs
    parser.add_argument("--det_method", type=str, default="lmmse", choices=["lmmse", "k-best", "ep", "mmse-pic"])
    parser.add_argument("--output", type=str, default="symbol", choices=["symbol", "bit"])
    parser.add_argument("--aug_times", type=int, default=1)
    parser.add_argument("--masking_type", type=str, default="discrete", 
                       choices=["discrete", "random_symbols", "fixed", "fix_length"])
    
    # Other configs
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output_dir", type=str, default="experiment_results/v3")
    parser.add_argument("--wandb_name", type=str, default="online_main_only")
    
    # Training specific
    parser.add_argument("--train_data_dir", type=str, default="./data/ps2_p72/umi/snr10to20_speed30")
    parser.add_argument("--train_snr", type=int, default=10)
    
    # Evaluation specific
    parser.add_argument("--eval_split", type=float, default=0.5)
    parser.add_argument("--eval_base_dir", type=str, default="./data/ps2_p72")
    parser.add_argument("--scenario", type=str, default="umi")
    parser.add_argument("--speed", type=str, default="30")
    parser.add_argument("--eval_snr_min", type=int, default=0)
    parser.add_argument("--eval_snr_max", type=int, default=20)
    parser.add_argument("--eval_snr_step", type=int, default=5)

    # Add pre-training scenario argument
    parser.add_argument("--pretrain_scenario", type=str, default="uma",
                       help="Scenario used during pre-training (for forgetting evaluation)")
    parser.add_argument("--pretrain_speed", type=str, default="5",
                       help="Speed of the pre-training scenario")

    main(parser.parse_args(sys.argv[1:]))