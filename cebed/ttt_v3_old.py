"""
This file evaluates the online TTT performance of a two-branch model with shared encoder
        -- main_decoder
- Encoder 
        -- aux_decoder

Compared with online_ttt_v1.py, this file adds two key modules: 
    1. Transmitted symbol recovery 
        we do not use the true transmitted symbols in aux-task datasets; 
        instead, we recover them for aux-task from detection pipeline;
        there are two different recovery methods: on the bit level or on the symbol level, controlled by "--output"
    2. Online aux-task data augmentation
        Realized by function "generate_aux_data_online(...)"
        If one wants to change random masking scheme in online aux-data generation, modify inside "generate_aux_data_online(...)"
        Augmentation time controlled by "self.aug_times" [TODO: please put this as an input argument in parser]
    Other Notes:
    -> in this file, "--sym_error_rate" always be set as 0. [If one wants directly generate datasets with SER > 0, please refer to online_ttt_v1.py]
        This file only uses detection pipeline to model SER.
    -> "--aug_noise_std" should be set as 0 unless one wants to vary the noise level in data augmentation.
    -> Information related to "supervise" & "ssl" can be found in online_TTT_v1.py
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
from copy import deepcopy
import time  # Add time import for measurements

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tqdm import tqdm
import numpy as np
import cebed.datasets_with_ssl as cds
import cebed.models_with_ssl as cm # Two-branch Model

from types import MethodType
from cebed.baselines import evaluate_baselines,evaluate_lmmse,evaluate_ls,evaluate_almmse
from cebed.utils import unflatten_last_dim, write_metadata, read_metadata
from cebed.datasets_with_ssl.utils import postprocess

from cebed.utils_eval import mse, generate_aux_data_online, get_ser,map_indices_to_symbols,real_to_complex_batch
from cebed.utils import set_random_seed
from cebed.datasets_with_ssl.base_multi_inputs import combine_datasets
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
    # supervised: int = 1  # Changed default to supervised mode
    # ssl: int = 0        # Changed default to disable SSL
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
        self.model_name = "ReconMAE" # fix mask for main, random mask for aux
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

        # Add channel application objects
        self.apply_noiseless_channel = ApplyOFDMChannel(add_awgn=False, dtype=tf.as_dtype(tf.complex64))
        self.apply_noisy_channel = ApplyOFDMChannel(add_awgn=True, dtype=tf.as_dtype(tf.complex64))


    
    def setup(self):
        """Setup the evaluator"""
        # -------------------- Bind with data and create datasets -------------------- #
        # create dataset
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
        # get model hyper-parameters from .yaml file
        assert self.model_name == "ReconMAE"
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
            self.detector = MMSEPICDetector(self.output, rg, sm, 'app', num_iter=4, constellation_type="qam", num_bits_per_symbol=self.num_bits_per_symbol, hard_out=self.hard_out)
        if self.coded:
            self.channel_encoder = self.dataset.env.channel_encoder
            self.channel_decoder = LDPC5GDecoder(self.channel_encoder, hard_out=True, num_iter=10)
        
        self.aug_times = self.config.aug_times
        self.mapper = self.dataset.env.mapper
        # ---------------------------------------------------------------------------- #
      
    

    def online_adapt(self) -> None:
        '''
        Train on self.config.train_snr first, then do evals on eval_snr_range
        '''
        # Load model and compile
        self.model.load_weights(self.model_dir)
        # checkpoint_path = os.path.join(self.model_dir, "cp.ckpt")
        # if os.path.exists(checkpoint_path + ".index"):  # Check if checkpoint exists
        #     self.model.load_weights(checkpoint_path)
        # else:
        #     raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
            
        dataset_class = cds.get_dataset_class(self.config.eval_dataset_name)
        self.model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(self.config.learning_rate),
            loss="mse"
        )
        
        # Before training evaluation for all SNRs
        print("Performing before-training evaluation...")

        # Pre-TTT id evaluation for all SNRs + Baselines
        print(f"\nEvaluating on pretrain {self.config.pretrain_scenario} environment before TTT...")
        for snr in range(self.config.eval_snr_min, self.config.eval_snr_max + 1, self.config.eval_snr_step):
            # Construct eval data directory for pre-training scenario
            next_snr = snr + 1
            pretrain_eval_data_dir = os.path.join(
                self.config.eval_base_dir,
                self.config.pretrain_scenario,  # Use pre-training scenario
                f"snr{snr}to{next_snr}_speed{self.config.pretrain_speed}"
            )
            
            # Update dataset with pre-training scenario eval directory
            dataset_class = cds.get_dataset_class(self.config.eval_dataset_name)
            self.dataset = dataset_class(
                pretrain_eval_data_dir,
                train_split=self.config.eval_split,
                main_input_type=self.config.main_input_type,
                aux_input_type=self.config.aux_input_type,
                sym_error_rate=self.config.sym_error_rate,
                seed=self.config.seed,
                aug_factor=1,
                masking_type=self.config.masking_type
            )

            # Initialize baseline tracking for pre-training scenario
            pretrain_mses_before = pd.DataFrame(columns=["snr", "mse", "method", "seed"])
            baselines = []  # Add baselines to evaluate
            test_mse = defaultdict(list)
            
            num_steps = (
                len(self.dataset.test_indices[0]) + self.config.eval_batch_size - 1
            ) // self.config.eval_batch_size
            
            # Evaluate baselines on pre-training scenario
            for step in range(num_steps):
                print(f"Pre-TTT Pre-training Scenario Baseline Step {step}/{num_steps}")
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
            
            # Log pre-TTT pre-training scenario baseline results
            for baseline in baselines:
                avg_baseline_mse = np.mean(test_mse[baseline])
                wandb.log({
                    'eval_snr': snr,
                    f'offline_{baseline}_mse': avg_baseline_mse
                })
                pretrain_mses_before.loc[len(pretrain_mses_before)] = [snr, avg_baseline_mse, baseline, self.config.seed]

            # Evaluate model on pre-training scenario before TTT
            self.eval_loader = self.dataset.get_eval_loader(self.config.eval_batch_size, "test", "both")
            eval_data_iterator = iter(self.eval_loader)

            pretrain_eval_channel_mse = []
            pretrain_eval_aux_mse = []
            
            for step in range(num_steps):
                print(f"Pre-TTT Pre-training Scenario Model Evaluation Step {step}/{num_steps}")
                (h_ls, h_true), (x_aux, mask, y_aux) = next(eval_data_iterator)
                
                h_pred, aux_pred = self.model((h_ls, (x_aux, mask)))
                
                step_channel_mse = mse(h_true, h_pred).numpy()
                step_aux_mse = mse(y_aux, aux_pred).numpy()
                
                pretrain_eval_channel_mse.append(step_channel_mse)
                pretrain_eval_aux_mse.append(step_aux_mse)

            # Calculate and log average MSE for pre-training scenario before TTT
            avg_pretrain_channel_mse = np.mean(pretrain_eval_channel_mse)
            avg_pretrain_aux_mse = np.mean(pretrain_eval_aux_mse)
            
            wandb.log({
                'eval_snr': snr,
                'pre_ttt_offline_channel_mse': avg_pretrain_channel_mse,
                'pre_ttt_offline_aux_mse': avg_pretrain_aux_mse
            })
            
            # Add model results to the pre-training scenario mses DataFrame
            pretrain_mses_before.loc[len(pretrain_mses_before)] = [snr, avg_pretrain_channel_mse, "Model", self.config.seed]
            
            print(f"\nPre-TTT Pre-training Scenario (SNR {snr}) Evaluation Results:")
            print(f"Channel Estimation MSE: {avg_pretrain_channel_mse:.6f}")
            print(f"Auxiliary Task MSE: {avg_pretrain_aux_mse:.6f}")
            
            # Save pre-TTT pre-training scenario results to CSV for this SNR
            pretrain_results_file = os.path.join(self.log_dir, f"pre_ttt_{self.config.pretrain_scenario}_results_snr_{snr}.csv")
            pretrain_mses_before.to_csv(pretrain_results_file, index=False)
            print(f"Pre-TTT pre-training scenario results saved to {pretrain_results_file}")

        # Pre-TTT ood evaluation for all SNRs
        print(f"\nEvaluating on online {self.config.scenario} environment before TTT...")
        for snr in range(self.config.eval_snr_min, self.config.eval_snr_max + 1, self.config.eval_snr_step):
            # Construct eval data directory for current SNR
            next_snr = snr + 1
            eval_data_dir = os.path.join(
                self.config.eval_base_dir,
                self.config.scenario,
                f"snr{snr}to{next_snr}_speed{self.config.speed}"
            )
            
            # Construct dataset with current eval directory
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

            self.eval_loader = self.dataset.get_eval_loader(self.config.eval_batch_size, "test", "both")
            eval_data_iterator = iter(self.eval_loader)
            num_steps = (
                len(self.dataset.test_indices[0]) + self.config.eval_batch_size - 1
            ) // self.config.eval_batch_size

            before_train_channel_mse = []
            before_train_aux_mse = []
            
            for step in range(num_steps):
                (h_ls, h_true), (x_aux, mask, y_aux) = next(eval_data_iterator)
                h_pred, aux_pred = self.model((h_ls, (x_aux, mask)))
                step_channel_mse = mse(h_true, h_pred).numpy()
                step_aux_mse = mse(y_aux, aux_pred).numpy()
                before_train_channel_mse.append(step_channel_mse)
                before_train_aux_mse.append(step_aux_mse)

            # Log before-training results for this SNR
            avg_before_channel_mse = np.mean(before_train_channel_mse)
            avg_before_aux_mse = np.mean(before_train_aux_mse)
            
            wandb.log({
                'eval_snr': snr,
                'pre_ttt_online_channel_mse': avg_before_channel_mse,
                'pre_ttt_online_aux_mse': avg_before_aux_mse,
            })
            
            print(f"Before Training - SNR {snr}:")
            print(f"Channel Estimation MSE: {avg_before_channel_mse:.6f}")
            print(f"Auxiliary Task MSE: {avg_before_aux_mse:.6f}")

            # Save before-training results to CSV
            results_file = os.path.join(self.log_dir, f"pre_ttt_{self.config.scenario}_results_snr_{snr}.csv")
            online_mses_before = pd.DataFrame({
                'snr': [snr],
                'channel_mse': [avg_before_channel_mse],
                'method': ['Model'],
                'seed': [self.config.seed]
            })
            online_mses_before.to_csv(results_file, index=False)

        noise_lin = tf.pow(10.0, -self.config.train_snr / 10.0)

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
            
        domain_idx = 0  # Now we only have one domain
        num_steps = (
            len(self.dataset.test_indices[domain_idx]) + self.config.train_batch_size - 1
        ) // self.config.train_batch_size


        # Pre-TTT ood evaluation for TTT SNR
        before_adapt_test_data = iter(self.train_loader)
        before_adapt_channel_mse = []
        before_adapt_aux_mse = []

        for step in range(num_steps):
            
            (h_ls, h_true),(x_aux, mask, y_aux) = next(before_adapt_test_data)
            # NOTE: here x_aux is the true transmitted symbols (coded or not) without any error (SER must be 0)
            h_pred, aux_pred = self.model((h_ls, (x_aux, mask)))
            before_adapt_channel_mse.append(mse(h_true, h_pred).numpy())
            before_adapt_aux_mse.append(mse(y_aux, aux_pred).numpy())

        wandb.log({'epoch': 0,
                    'ttt_epoch_channel_mse': np.mean(before_adapt_channel_mse),
                    'ttt_epoch_aux_loss': np.mean(before_adapt_aux_mse)})

        # Re-initialize / reset any step logging variable here
        training_step_counter = 0

        # Training phase
        print(f"Starting training phase at SNR {self.config.train_snr}...")
        
        # Initialize time tracking variables
        total_channel_coding_time = 0.0
        total_aux_data_gen_time = 0.0
        total_ttt_time = 0.0
        step_count = 0
        
        for epoch in range(self.config.epochs):
            channel_mse = []
            aux_mse = []
            epoch_sers = []
            batch_data = iter(self.train_loader)
            
            for step in range(num_steps):
                step_start_time = time.time()
                start, end = step * self.config.train_batch_size, min(
                    (step + 1) * self.config.train_batch_size,
                    len(self.dataset.test_indices[0]),
                )
                actual_batch_size = end - start

                ### test the channel estimation task on the same test batch
                (h_ls, h_true), _ = next(batch_data)

                ### through the channelMAE to get the predicted H
                h_pred = self.model.main_branch(h_ls)
                h_hat = real_to_complex_batch(h_pred) 
                h_hat = tf.reshape(h_hat,[h_pred.shape[0], 1, 1, 1, 1, h_pred.shape[1], h_pred.shape[2]]) # [batch, 1, 1, 1, 1, 14, 72]

                # Make sure h_hat matches rx_symbols batch size
                if h_hat.shape[0] != actual_batch_size:
                    h_hat = h_hat[:actual_batch_size]

                ### generate the augmented data for aux-task
                # true transmitted symbols (via ldpc encoder)
                tx_symbols = self.dataset.x_samples[domain_idx][self.dataset.test_indices[domain_idx]][start:end]
                rx_symbols = self.dataset.y_samples[domain_idx][self.dataset.test_indices[domain_idx]][start:end] 

                # define the bits_shape when self.output = "symbol"
                bits_shape = [h_pred.shape[0], 1, 1, self.n] 

                # NOTE: the error variance of channel estimation & we do not know the exact value for NN estimator
                err_var = 0.0

                # ------------------------- Generate detected/recovered TX Symbol ------------------------- #
                channel_coding_start_time = time.time()
                
                # Detection: we can implement different detectors or the neural detector
                if self.config.det_method == "mmse-pic":
                    if self.output == "bit":
                        prior_shape = bits_shape
                    elif self.output == "symbol":
                        prior_shape = tf.concat([tf.shape(tx_symbols), [self.num_bits_per_symbol]], axis=0)
                    prior = tf.zeros(prior_shape)
                    det_out = self.detector(rx_symbols,h_hat,prior,err_var,noise_lin)
                else:
                    det_out = self.detector(rx_symbols,h_hat,err_var,noise_lin)

                # (Decoding) and output
                if self.output == "bit":
                    ### Use coded bits (--> re-encode --> re-mapper) for online TTT 
                    ### NOTE: the test dataset must be generated using ldpc encoder
                    llr = tf.reshape(det_out, bits_shape)
                    b_hat = self.channel_decoder(llr) # NOTE used for coded BER
                    c_hat = self.channel_encoder(b_hat)
                    recov_tx_data_sym = self.mapper(c_hat) 
                    recov_tx_symbols = self.dataset.env.rg_mapper(recov_tx_data_sym)

                    channel_coding_end_time = time.time()
                    channel_coding_time = channel_coding_end_time - channel_coding_start_time
                    total_channel_coding_time += channel_coding_time

                    # compute the SER here
                    ser = get_ser(tx_symbols, recov_tx_symbols)
                    epoch_sers.append(ser)
                    print("Step SER: ", ser)

                    # Measure auxiliary data generation time
                    aux_data_gen_start_time = time.time()
                    (aug_x_aux, aug_mask, aug_y_aux) = generate_aux_data_online(rx_symbols, recov_tx_symbols, self.aug_times, self.config.masking_type)
                    aux_data_gen_end_time = time.time()
                    aux_data_gen_time = aux_data_gen_end_time - aux_data_gen_start_time
                    total_aux_data_gen_time += aux_data_gen_time

                elif self.output == "symbol":
                    # the output of the detector is symbol indices (tf.int)
                    est_tx_data_sym = map_indices_to_symbols(det_out) # det_out (tf.int)[batch, 1, 1, num_data_symbols]
                    est_tx_symbols = self.dataset.env.rg_mapper(est_tx_data_sym)

                    channel_coding_end_time = time.time()
                    channel_coding_time = channel_coding_end_time - channel_coding_start_time
                    # Note: For "symbol" format, no actual channel coding is performed, so this measures detection time
                    total_channel_coding_time += channel_coding_time

                    # compute the SER here: compare est_tx_symbols with tx_symbols
                    ser = get_ser(tx_symbols, est_tx_symbols)
                    epoch_sers.append(ser)
                    print("Step SER: ", ser)

                    ### Use uncoded symbols for online TTT
                    # Measure auxiliary data generation time
                    aux_data_gen_start_time = time.time()
                    (aug_x_aux, aug_mask, aug_y_aux) = generate_aux_data_online(rx_symbols, est_tx_symbols, self.aug_times, self.config.masking_type)
                    aux_data_gen_end_time = time.time()
                    aux_data_gen_time = aux_data_gen_end_time - aux_data_gen_start_time
                    total_aux_data_gen_time += aux_data_gen_time

                # ---------------------------------------------------------------------------- #
                # Measure TTT process time (training step)
                ttt_start_time = time.time()
                
                # use the online generated data to train the aux task
                # generate computational graphs
                with tf.GradientTape() as tape:
                    # feed the augmented data to the aux_branch
                    online_aux_pred = self.model.aux_branch(aug_x_aux, aug_mask)
                    h_pred = self.model.main_branch(h_ls)
                    main_loss = self.model.compiled_loss(h_true, h_pred)

                    # TODO: only consider the loss for the masked part
                    aux_loss = self.model.compiled_loss(aug_y_aux, online_aux_pred)

                step_main_mse = mse(h_true, h_pred).numpy()
                step_aux_mse = mse(aug_y_aux, online_aux_pred).numpy()

                channel_mse.append(step_main_mse)
                aux_mse.append(step_aux_mse)

                # if self.config.ssl:
                self.model.encoder.trainable = True
                self.model.main_decoder.trainable = False 
                self.model.aux_decoder.trainable = False
                grads = tape.gradient(aux_loss, self.model.trainable_weights)
                self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                    
                # elif self.config.supervised:
                #     self.model.encoder.trainable = True
                #     self.model.main_decoder.trainable = True
                #     self.model.aux_decoder.trainable = True
                #     grads = tape.gradient(main_loss, self.model.trainable_weights)
                #     self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                ttt_end_time = time.time()
                ttt_time = ttt_end_time - ttt_start_time
                total_ttt_time += ttt_time
                
                step_total_time = time.time() - step_start_time
                step_count += 1

                # Log time consumption metrics
                wandb.log({
                    'batch_step': training_step_counter,
                    'ttt_channel_est_mse': main_loss,
                    'ttt_ssl_train_loss': aux_loss,
                    'ttt_step_ser': ser,
                    'step_channel_coding_time': channel_coding_time,
                    'step_aux_data_gen_time': aux_data_gen_time,
                    'step_ttt_time': ttt_time,
                    'step_total_time': step_total_time,
                    'output_format': self.output
                })

                # Increment the training step counter
                training_step_counter += 1

            avg_ser = np.mean(epoch_sers)
            avg_channel_mse = np.mean(channel_mse)
            avg_aux_mse = np.mean(aux_mse)
            
            # Calculate average times for this epoch
            avg_channel_coding_time = total_channel_coding_time / step_count if step_count > 0 else 0
            avg_aux_data_gen_time = total_aux_data_gen_time / step_count if step_count > 0 else 0
            avg_ttt_time = total_ttt_time / step_count if step_count > 0 else 0
            
            wandb.log({
                'epoch': epoch+1,
                'ttt_epoch_channel_mse': avg_channel_mse,
                'ttt_epoch_aux_loss': avg_aux_mse,
                'ttt_epoch_avg_ser': avg_ser,
                'epoch_avg_channel_coding_time': avg_channel_coding_time,
                'epoch_avg_aux_data_gen_time': avg_aux_data_gen_time,
                'epoch_avg_ttt_time': avg_ttt_time,
                'epoch_total_channel_coding_time': total_channel_coding_time,
                'epoch_total_aux_data_gen_time': total_aux_data_gen_time,
                'epoch_total_ttt_time': total_ttt_time
            })
            
            print(f"Epoch {epoch+1} Summary:")
            print(f"Average Channel MSE: {avg_channel_mse:.6f}")
            print(f"Average Auxiliary MSE: {avg_aux_mse:.6f}")
            print(f"Average Symbol Error Rate: {avg_ser:.6f}")
            print(f"Average Channel Coding Time: {avg_channel_coding_time:.4f}s")
            print(f"Average Aux Data Gen Time: {avg_aux_data_gen_time:.4f}s") 
            print(f"Average TTT Time: {avg_ttt_time:.4f}s")
            print(f"Total Channel Coding Time: {total_channel_coding_time:.4f}s")
            print(f"Total Aux Data Gen Time: {total_aux_data_gen_time:.4f}s")
            print(f"Total TTT Time: {total_ttt_time:.4f}s")
            print("-" * 50)

        # Log final time consumption summary
        final_avg_channel_coding_time = total_channel_coding_time / step_count if step_count > 0 else 0
        final_avg_aux_data_gen_time = total_aux_data_gen_time / step_count if step_count > 0 else 0
        final_avg_ttt_time = total_ttt_time / step_count if step_count > 0 else 0
        
        wandb.log({
            'final_avg_channel_coding_time': final_avg_channel_coding_time,
            'final_avg_aux_data_gen_time': final_avg_aux_data_gen_time,
            'final_avg_ttt_time': final_avg_ttt_time,
            'final_total_channel_coding_time': total_channel_coding_time,
            'final_total_aux_data_gen_time': total_aux_data_gen_time,
            'final_total_ttt_time': total_ttt_time,
            'total_training_steps': step_count,
            'output_format': self.output
        })

        print(f"\nFinal Time Consumption Summary:")
        print(f"Output Format: {self.output}")
        print(f"Total Training Steps: {step_count}")
        print(f"Average Channel Coding Time per Step: {final_avg_channel_coding_time:.4f}s")
        print(f"Average Aux Data Generation Time per Step: {final_avg_aux_data_gen_time:.4f}s")
        print(f"Average TTT Time per Step: {final_avg_ttt_time:.4f}s")
        print(f"Total Channel Coding Time: {total_channel_coding_time:.4f}s")
        print(f"Total Aux Data Generation Time: {total_aux_data_gen_time:.4f}s")
        print(f"Total TTT Time: {total_ttt_time:.4f}s")


        # Post-TTT ood evaluation for all SNRs + Baselines
        print(f"\nEvaluating on {self.config.scenario} environment after TTT...")
        for snr in range(self.config.eval_snr_min, self.config.eval_snr_max + 1, self.config.eval_snr_step):
                
            # Construct eval data directory for current SNR
            next_snr = snr + 1
            eval_data_dir = os.path.join(
                self.config.eval_base_dir,
                self.config.scenario,
                f"snr{snr}to{next_snr}_speed{self.config.speed}"
            )
            
            # Update dataset with current eval directory
            dataset_class = cds.get_dataset_class(self.config.eval_dataset_name)
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

            # Initialize baseline tracking
            mses = pd.DataFrame(columns=["snr", "mse", "method", "seed"])
            baselines = [] # Add baselines to evaluate
            test_mse = defaultdict(list)
            
            # Iterate over all test data for baseline evaluation
            for step in range(num_steps):
                print(f"Baseline Step {step}/{num_steps}")
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
            
            # Average and log baseline results
            for baseline in baselines:
                avg_baseline_mse = np.mean(test_mse[baseline])
                wandb.log({
                    'eval_snr': snr,
                    f'online_{baseline}_mse': avg_baseline_mse
                    })
                mses.loc[len(mses)] = [snr, avg_baseline_mse, baseline, self.config.seed]

            # Evaluate the model on various SNRs
            self.eval_loader = self.dataset.get_eval_loader(self.config.eval_batch_size, "test", "both")
            eval_data_iterator = iter(self.eval_loader)
            num_steps = (
                len(self.dataset.test_indices[0]) + self.config.eval_batch_size - 1
            ) // self.config.eval_batch_size

            eval_channel_mse = []
            eval_aux_mse = []
            
            for step in range(num_steps):
                print(f"Model Evaluation Step {step}/{num_steps}")
                (h_ls, h_true), (x_aux, mask, y_aux) = next(eval_data_iterator)
                
                # Get predictions from both branches
                h_pred, aux_pred = self.model((h_ls, (x_aux, mask)))
                
                # Calculate MSE for both tasks
                step_channel_mse = mse(h_true, h_pred).numpy()
                step_aux_mse = mse(y_aux, aux_pred).numpy()
                
                eval_channel_mse.append(step_channel_mse)
                eval_aux_mse.append(step_aux_mse)

            # Calculate average MSE for this SNR
            avg_eval_channel_mse = np.mean(eval_channel_mse)
            avg_eval_aux_mse = np.mean(eval_aux_mse)
            
            # Log evaluation results
            wandb.log({
                'eval_snr': snr,
                'post_ttt_online_channel_mse': avg_eval_channel_mse,
                'post_ttt_online_aux_mse': avg_eval_aux_mse,
            })
            
            # Add model results to the mses DataFrame
            mses.loc[len(mses)] = [snr, avg_eval_channel_mse, "Model", self.config.seed]
            
            print(f"SNR {snr} Evaluation Results:")
            print(f"Channel Estimation MSE: {avg_eval_channel_mse:.6f}")
            print(f"Auxiliary Task MSE: {avg_eval_aux_mse:.6f}")
            
            # Save results to CSV for this SNR
            results_file = os.path.join(self.log_dir, f"post_ttt_{self.config.scenario}_results_snr_{snr}.csv")
            mses.to_csv(results_file, index=False)
            print(f"Results saved to {results_file}")

        # Post-TTT id evaluation for all SNRs
        print(f"\nEvaluating on {self.config.pretrain_scenario} environment to test forgetting...")
        for snr in range(self.config.eval_snr_min, self.config.eval_snr_max + 1, self.config.eval_snr_step):
            # Construct eval data directory for pre-training scenario
            next_snr = snr + 1
            pretrain_eval_data_dir = os.path.join(
                self.config.eval_base_dir,
                self.config.pretrain_scenario,  # Use pre-training scenario
                f"snr{snr}to{next_snr}_speed{self.config.pretrain_speed}"
            )
            
            # Update dataset with pre-training scenario eval directory
            dataset_class = cds.get_dataset_class(self.config.eval_dataset_name)
            self.dataset = dataset_class(
                pretrain_eval_data_dir,
                train_split=self.config.eval_split,
                main_input_type=self.config.main_input_type,
                aux_input_type=self.config.aux_input_type,
                sym_error_rate=self.config.sym_error_rate,
                seed=self.config.seed,
                aug_factor=1,
                masking_type=self.config.masking_type
            )

            # Initialize baseline tracking for pre-training scenario
            pretrain_mses = pd.DataFrame(columns=["snr", "mse", "method", "seed"])
            test_mse = defaultdict(list)
            
            num_steps = (
                len(self.dataset.test_indices[0]) + self.config.eval_batch_size - 1
            ) // self.config.eval_batch_size

            # Evaluate TTTed model on pre-training scenario
            self.eval_loader = self.dataset.get_eval_loader(self.config.eval_batch_size, "test", "both")
            eval_data_iterator = iter(self.eval_loader)

            pretrain_eval_channel_mse = []
            pretrain_eval_aux_mse = []
            
            for step in range(num_steps):
                print(f"Pre-training Scenario Model Evaluation Step {step}/{num_steps}")
                (h_ls, h_true), (x_aux, mask, y_aux) = next(eval_data_iterator)
                
                h_pred, aux_pred = self.model((h_ls, (x_aux, mask)))
                
                step_channel_mse = mse(h_true, h_pred).numpy()
                step_aux_mse = mse(y_aux, aux_pred).numpy()
                
                pretrain_eval_channel_mse.append(step_channel_mse)
                pretrain_eval_aux_mse.append(step_aux_mse)

            # Calculate and log average MSE for pre-training scenario
            avg_pretrain_channel_mse = np.mean(pretrain_eval_channel_mse)
            avg_pretrain_aux_mse = np.mean(pretrain_eval_aux_mse)
            
            wandb.log({
                'eval_snr': snr,
                'post_ttt_offline_channel_mse': avg_pretrain_channel_mse,
                'post_ttt_offline_aux_mse': avg_pretrain_aux_mse
            })
            
            # Add model results to the pre-training scenario mses DataFrame
            pretrain_mses.loc[len(pretrain_mses)] = [snr, avg_pretrain_channel_mse, "Model", self.config.seed]
            
            print(f"\nPre-training Scenario (SNR {snr}) Evaluation Results:")
            print(f"Channel Estimation MSE: {avg_pretrain_channel_mse:.6f}")
            print(f"Auxiliary Task MSE: {avg_pretrain_aux_mse:.6f}")
            
            # Save pre-training scenario results to CSV for this SNR
            pretrain_results_file = os.path.join(self.log_dir, f"post_ttt_{self.config.pretrain_scenario}_results_snr_{snr}.csv")
            pretrain_mses.to_csv(pretrain_results_file, index=False)
            print(f"Pre-training scenario results saved to {pretrain_results_file}")


def main(args):
    print("Starting online TTT channel estimation...")
    wandb_config = {**vars(args)}
    run = wandb.init(project='INFOCOM2026', 
                     config=wandb_config,
                     name=args.wandb_name,
                     workspace="ttt4wireless")
    
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
    print("Running online TTT channel estimation script...")
    parser = argparse.ArgumentParser(description="Online TTT Channel Estimation")

    # Model related
    parser.add_argument("--trained_model_dir", type=str, default="./models/rt0.h5")
    
    # Dataset related
    parser.add_argument("--eval_dataset_name", type=str, default="RandomMask")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--main_input_type", type=str, default="low", choices=["low", "raw"])
    parser.add_argument("--aux_input_type", type=str, default="low", choices=["low", "raw"])
    
    # Training configs
    parser.add_argument("--ttt_split", type=float, default=0.75)
    parser.add_argument("--epochs", type=int, default=5)
    ## NOTE: in ttt files, we only do SSL mode; to do SL (true labels or pseudo labels) for the main task, run online_sl_main_xxx.py
    # parser.add_argument("--supervised", type=int, default=0) 
    # parser.add_argument("--ssl", type=int, default=1) 
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    
    # Method configs
    parser.add_argument("--det_method", type=str, default="lmmse", choices=["lmmse", "k-best", "ep", "mmse-pic"])
    parser.add_argument("--output", type=str, default="symbol", choices=["symbol", "bit"])
    parser.add_argument("--aug_times", type=int, default=3)
    parser.add_argument("--masking_type", type=str, default="discrete", 
                       choices=["discrete", "random_symbols", "fixed", "fix_length"])
    
    # Other configs
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output_dir", type=str, default="experiment_results/v3")
    parser.add_argument("--wandb_name", type=str, default="online_ttt_v3",)
    
    # Training specific
    parser.add_argument("--train_data_dir", type=str, default="./data/ps2_p72/rt1/snr10to20_speed5")
    parser.add_argument("--train_snr", type=int, default=20)
    
    # Evaluation specific
    parser.add_argument("--eval_split", type=float, default=0.5)
    parser.add_argument("--eval_base_dir", type=str, default="./data/ps2_p72")
    parser.add_argument("--scenario", type=str, default="rt1")
    parser.add_argument("--speed", type=str, default="5")
    parser.add_argument("--eval_snr_min", type=int, default=0)
    parser.add_argument("--eval_snr_max", type=int, default=25)
    parser.add_argument("--eval_snr_step", type=int, default=5)

    # Add pre-training scenario argument
    parser.add_argument("--pretrain_scenario", type=str, default="rt0",
                       help="Scenario used during pre-training (for forgetting evaluation)")
    parser.add_argument("--pretrain_speed", type=str, default="5",
                       help="Speed of the pre-training scenario")
    
    with tf.device('/GPU'):
        main(parser.parse_args(sys.argv[1:]))
