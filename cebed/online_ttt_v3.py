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
        Augmentation time controlled by "self.aug_times"
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

# Add FLOPs calculation imports
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph


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
    pretrain_speed: str = "30"     # Add this line for pre-training scenario

    # continual TTT configs
    continual: int = 0  # Flag to enable continual TTT
    scenarios: str = ""  # Comma-separated scenarios for continual TTT

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
    calculate_recovery_flops: int = 0


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
        self.evaluate_ddce = MethodType(evaluate_ddce, self)

        # Add channel application objects
        self.apply_noiseless_channel = ApplyOFDMChannel(add_awgn=False, dtype=tf.as_dtype(tf.complex64))
        self.apply_noisy_channel = ApplyOFDMChannel(add_awgn=True, dtype=tf.as_dtype(tf.complex64))
        
        # Initialize FLOPs tracking
        self.recovery_flops = {
            'detection': 0,
            'channel_decoding': 0,
            'channel_encoding': 0,
            'mapping': 0,
            'rg_mapping': 0,
            'symbol_mapping': 0,
            'total_recovery': 0
        }

    def calculate_operation_flops(self, operation_func, *args, **kwargs):
        """Calculate FLOPs for a specific operation"""
        try:
            @tf.function
            def wrapped_operation(*args, **kwargs):
                return operation_func(*args, **kwargs)
            
            # Create concrete function
            concrete_func = wrapped_operation.get_concrete_function(*args, **kwargs)
            
            # Convert to constants for FLOPs calculation
            frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
            
            # Calculate FLOPs with suppressed output
            with tf.Graph().as_default() as graph:
                tf.graph_util.import_graph_def(graph_def, name='')
                
                run_meta = tf.compat.v1.RunMetadata()
                opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
                opts['output'] = 'none'  # Suppress stdout output
                
                # Redirect stdout to suppress verbose output
                from io import StringIO
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                try:
                    flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
                    return flops.total_float_ops if flops else 0
                finally:
                    sys.stdout = old_stdout
                    
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not calculate FLOPs for operation: {e}")
            return 0
        

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
            self.detector = MMSEPICDetector(self.output, "app", rg, sm, num_iter=4, constellation_type="qam", num_bits_per_symbol=self.num_bits_per_symbol, hard_out=self.hard_out)
        if self.coded:
            self.channel_encoder = self.dataset.env.channel_encoder
            self.channel_decoder = LDPC5GDecoder(self.channel_encoder, hard_out=True, num_iter=10)
        
        self.aug_times = self.config.aug_times
        self.mapper = self.dataset.env.mapper
        # ---------------------------------------------------------------------------- #
      
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
    
        if self.config.continual:
            # Continual TTT across multiple scenarios
            self._perform_continual_ttt()
        else:
            # Pre-TTT evaluations with baselines
            # print("Performing pre-ttt evaluation...")
            # self._evaluate_pre_ttt(
            #     self.config.pretrain_scenario, 
            #     self.config.pretrain_speed, 
            #     "offline"
            # )
            # self._evaluate_pre_ttt(
            #     self.config.scenario, 
            #     self.config.speed, 
            #     "online"
            # )
            
            # Prepare training data
            print("Preparing training data for TTT...")
            total_samples = self._prepare_training_data()

            # Perform TTT training
            print("Performing TTT training...")
            self._perform_ttt_training(total_samples)
            
            # Post-TTT evaluations
            # print("Performing post-ttt evaluation...")
            # self._evaluate_post_ttt(
            #     self.config.pretrain_scenario, 
            #     self.config.pretrain_speed, 
            #     "offline"
            # )
            # self._evaluate_post_ttt(
            #     self.config.scenario, 
            #     self.config.speed, 
            #     "online"
            # )

    def _perform_continual_ttt(self):
        """Perform continual TTT across multiple scenarios"""
        scenarios_list = [s.strip() for s in self.config.scenarios.split(',') if s.strip()]
        print(f"Starting continual TTT across scenarios: {scenarios_list}")
        
        # Create directory for saving continual models
        continual_models_dir = os.path.join(self.log_dir, "continual_models")
        os.makedirs(continual_models_dir, exist_ok=True)
        
        for i, scenario in enumerate(scenarios_list):
            print(f"\n=== Continual TTT Step {i+1}/{len(scenarios_list)}: {scenario} ===")
            
            # Update current scenario
            current_scenario = scenario
            current_speed = self.config.speed
            
            # For continual TTT, use train_snr as both train and eval SNR
            eval_snr = self.config.train_snr
            next_snr = eval_snr + 1
            
            # Construct training data directory
            train_data_dir = os.path.join(
                "./data_TTTtrain/ps2_p72",
                current_scenario,
                f"snr{eval_snr}to{next_snr}_speed{current_speed}"
            )
            
            # Update config for current scenario
            self.config.train_data_dir = train_data_dir
            self.config.scenario = current_scenario
            
            print(f"Training on scenario {current_scenario} with SNR {eval_snr}")
            
            # Prepare training data for current scenario
            total_samples = self._prepare_training_data()
            
            # Perform TTT training
            self._perform_ttt_training(total_samples, is_continual=True, step_scenario=current_scenario, continual_step=i+1)
            
            # Evaluate on current scenario at training SNR only
            print(f"Evaluating on scenario {current_scenario} at SNR {eval_snr}")
            self._evaluate_continual_single_snr(current_scenario, current_speed, eval_snr, i+1)
            
        # Save final continually-adapted model
        final_model_path = os.path.join(
            continual_models_dir, 
            f"final_continual_adapted_model_{len(scenarios_list)}_scenarios.h5"
        )
        self.model.save_weights(final_model_path)
        print(f"\nFinal continually-adapted model saved to: {final_model_path}")
        
        print("Continual TTT completed!")

    def _evaluate_continual_single_snr(self, scenario: str, speed: str, snr: int, step: int):
        """Evaluate model at single SNR for continual TTT"""
        dataset_class = cds.get_dataset_class(self.config.eval_dataset_name)
        
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

        # Evaluate the model
        self.eval_loader = self.dataset.get_eval_loader(self.config.eval_batch_size, "test", "both")
        eval_data_iterator = iter(self.eval_loader)

        eval_channel_mse = []
        eval_aux_mse = []
        
        for step_idx in tqdm(range(num_steps), desc=f"Continual Step {step} Evaluation SNR {snr}", leave=False):
            (h_ls, h_true), (x_aux, mask, y_aux) = next(eval_data_iterator)
            
            h_pred, aux_pred = self.model((h_ls, (x_aux, mask)))
            
            step_channel_mse = mse(h_true, h_pred).numpy()
            step_aux_mse = mse(y_aux, aux_pred).numpy()
            
            eval_channel_mse.append(step_channel_mse)
            eval_aux_mse.append(step_aux_mse)

        # Calculate and log results
        avg_eval_channel_mse = np.mean(eval_channel_mse)
        avg_eval_aux_mse = np.mean(eval_aux_mse)
        
        wandb.log({
            f'continual/step_{step}_{scenario}_channel_mse': avg_eval_channel_mse,
        })
        
        print(f"Continual Step {step} - {scenario} SNR {snr} Results:")
        print(f"Channel Estimation MSE: {avg_eval_channel_mse:.6f}")
        print(f"Auxiliary Task MSE: {avg_eval_aux_mse:.6f}")

    def _evaluate_pre_ttt(self, scenario: str, speed: str, scenario_name: str):
        """Evaluate model performance before TTT training"""
        print(f"\nEvaluating on {scenario_name} {scenario} environment before TTT...")
        
        dataset_class = cds.get_dataset_class(self.config.eval_dataset_name)
        
        # Evaluate baselines
        if scenario in ["rt0", "rt1", "rt2", "rt3", "rt4", "uma", "umi"]:
            baselines = []
        else:
            baselines = ["ALMMSE", "LMMSE", "LS", "DDCE"]

        # Evaluate baselines
        print(f"{scenario_name} {scenario} Baselines: {baselines}...")

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

            self.eval_loader = self.dataset.get_eval_loader(self.config.eval_batch_size, "test", "both")
            eval_data_iterator = iter(self.eval_loader)

            eval_channel_mse = []
            eval_aux_mse = []
            
            for step in tqdm(range(num_steps), desc=f"Model Evaluation SNR {snr}", leave=False):
                (h_ls, h_true), (x_aux, mask, y_aux) = next(eval_data_iterator)
                
                h_pred, aux_pred = self.model((h_ls, (x_aux, mask)))
                
                step_channel_mse = mse(h_true, h_pred).numpy()
                step_aux_mse = mse(y_aux, aux_pred).numpy()
                
                eval_channel_mse.append(step_channel_mse)
                eval_aux_mse.append(step_aux_mse)

            # Calculate and log average MSE
            avg_channel_mse = np.mean(eval_channel_mse)
            avg_aux_mse = np.mean(eval_aux_mse)
            
            wandb.log({
                f'pre_ttt_{scenario_name}_channel_mse': avg_channel_mse,
                f'pre_ttt_{scenario_name}_aux_mse': avg_aux_mse
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

    def _perform_detection_and_recovery(self, h_pred, tx_symbols, rx_symbols, noise_lin, actual_batch_size, calculate_flops=False):
        """Perform symbol detection and recovery with optional FLOPs calculation"""
        h_hat = real_to_complex_batch(h_pred) 
        h_hat = tf.reshape(h_hat, [h_pred.shape[0], 1, 1, 1, 1, h_pred.shape[1], h_pred.shape[2]])
        
        # Make sure h_hat matches rx_symbols batch size
        if h_hat.shape[0] != actual_batch_size:
            h_hat = h_hat[:actual_batch_size]

        # Define bits_shape for symbol output
        bits_shape = [h_pred.shape[0], 1, 1, self.n] 
        err_var = 0.0

        # Detection with FLOPs calculation
        if calculate_flops:
            if self.config.det_method == "mmse-pic":
                if self.output == "bit":
                    prior_shape = bits_shape
                elif self.output == "symbol":
                    prior_shape = tf.concat([tf.shape(tx_symbols), [self.num_bits_per_symbol]], axis=0)
                prior = tf.zeros(prior_shape)
                detection_flops = self.calculate_operation_flops(
                    self.detector, rx_symbols, h_hat, prior, err_var, noise_lin
                )
                det_out = self.detector(rx_symbols, h_hat, prior, err_var, noise_lin)
            else:
                detection_flops = self.calculate_operation_flops(
                    self.detector, rx_symbols, h_hat, err_var, noise_lin
                )
                det_out = self.detector(rx_symbols, h_hat, err_var, noise_lin)
            
            self.recovery_flops['detection'] += detection_flops
        else:
            # Normal detection without FLOPs calculation
            if self.config.det_method == "mmse-pic":
                if self.output == "bit":
                    prior_shape = bits_shape
                elif self.output == "symbol":
                    prior_shape = tf.concat([tf.shape(tx_symbols), [self.num_bits_per_symbol]], axis=0)
                prior = tf.zeros(prior_shape)
                det_out = self.detector(rx_symbols, h_hat, prior, err_var, noise_lin)
            else:
                det_out = self.detector(rx_symbols, h_hat, err_var, noise_lin)

        # Recovery based on output format with FLOPs calculation
        if self.output == "bit":
            llr = tf.reshape(det_out, bits_shape)
            
            if calculate_flops:
                # Calculate FLOPs for each step
                decoder_flops = self.calculate_operation_flops(self.channel_decoder, llr)
                self.recovery_flops['channel_decoding'] += decoder_flops
                
                b_hat = self.channel_decoder(llr)
                
                encoder_flops = self.calculate_operation_flops(self.channel_encoder, b_hat)
                self.recovery_flops['channel_encoding'] += encoder_flops
                
                c_hat = self.channel_encoder(b_hat)
                
                mapper_flops = self.calculate_operation_flops(self.mapper, c_hat)
                self.recovery_flops['mapping'] += mapper_flops
                
                recov_tx_data_sym = self.mapper(c_hat)
                
                rg_mapper_flops = self.calculate_operation_flops(self.dataset.env.rg_mapper, recov_tx_data_sym)
                self.recovery_flops['rg_mapping'] += rg_mapper_flops
                
                recov_tx_symbols = self.dataset.env.rg_mapper(recov_tx_data_sym)
            else:
                # Normal recovery without FLOPs calculation
                b_hat = self.channel_decoder(llr)
                c_hat = self.channel_encoder(b_hat)
                recov_tx_data_sym = self.mapper(c_hat) 
                recov_tx_symbols = self.dataset.env.rg_mapper(recov_tx_data_sym)
                
            recovered_symbols = recov_tx_symbols
            
        elif self.output == "symbol":
            if calculate_flops:
                # Calculate FLOPs for symbol path
                symbol_mapping_flops = self.calculate_operation_flops(map_indices_to_symbols, det_out)
                self.recovery_flops['symbol_mapping'] += symbol_mapping_flops
                
                est_tx_data_sym = map_indices_to_symbols(det_out)
                
                rg_mapper_flops = self.calculate_operation_flops(self.dataset.env.rg_mapper, est_tx_data_sym)
                self.recovery_flops['rg_mapping'] += rg_mapper_flops
                
                est_tx_symbols = self.dataset.env.rg_mapper(est_tx_data_sym)
            else:
                # Normal symbol recovery without FLOPs calculation
                est_tx_data_sym = map_indices_to_symbols(det_out)
                est_tx_symbols = self.dataset.env.rg_mapper(est_tx_data_sym)
                
            recovered_symbols = est_tx_symbols

        # Compute Symbol Error Rate
        ser = get_ser(tx_symbols, recovered_symbols)
        
        return recovered_symbols, ser

    def _perform_ttt_training(self, total_samples, is_continual=False, step_scenario=None, continual_step=None):
        """Perform the actual TTT training process"""
        # Noise computation can be on CPU
        noise_lin = tf.pow(10.0, -self.config.train_snr / 10.0)
        
        domain_idx = 0
        num_steps = (total_samples + self.config.train_batch_size - 1) // self.config.train_batch_size
        
        # Initialize refined time tracking
        time_tracker = {
            'detection_recovery': {'total': 0.0, 'count': 0},
            'aux_data_generation': {'total': 0.0, 'count': 0},
            'ttt_training': {'total': 0.0, 'count': 0}
        }
        training_step_counter = 0

        # # Pre-adaptation evaluation
        # before_adapt_test_data = iter(self.train_loader)
        # before_adapt_channel_mse = []
        # before_adapt_aux_mse = []

        # print(f"Starting pre-adaptation evaluation with {num_steps} steps...")
        # for step in tqdm(range(num_steps), desc="Pre-adaptation Evaluation", leave=False):
        #     (h_ls, h_true), (x_aux, mask, y_aux) = next(before_adapt_test_data)
        #     h_pred, aux_pred = self.model((h_ls, (x_aux, mask)))
        #     step_channel_mse = mse(h_true, h_pred).numpy()
        #     step_aux_mse = mse(y_aux, aux_pred).numpy()
        #     before_adapt_channel_mse.append(step_channel_mse)
        #     before_adapt_aux_mse.append(step_aux_mse)

        # pre_adapt_avg_channel = np.mean(before_adapt_channel_mse)
        # pre_adapt_avg_aux = np.mean(before_adapt_aux_mse)

        # print(f"Pre-adaptation - Channel MSE: {pre_adapt_avg_channel:.6f}, Aux MSE: {pre_adapt_avg_aux:.6f}")

        # wandb.log({
        #     'ttt/epoch_channel_mse': pre_adapt_avg_channel,
        #     'ttt/epoch_aux_loss': pre_adapt_avg_aux,
        # })

        # if is_continual:
        #     wandb.log({
        #         f'continual/pre_adapt_step_{continual_step}_{step_scenario}_channel_mse': pre_adapt_avg_channel,
        #     })
        
        # Training epochs
        for epoch in tqdm(range(self.config.epochs), desc="TTT Training Epochs"):
            print(f"\n=== Starting TTT Epoch {epoch+1}/{self.config.epochs} ===")
            channel_mse = []
            aux_mse = []
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
                
                # Get symbols for detection - data retrieval can be on CPU  
                tx_symbols = self.dataset.x_samples[domain_idx][self.dataset.test_indices[domain_idx]][start:end]
                rx_symbols = self.dataset.y_samples[domain_idx][self.dataset.test_indices[domain_idx]][start:end] 

                # Perform detection and recovery with timing
                detection_start = time.time()
                recovered_symbols, ser = self._perform_detection_and_recovery(
                    h_pred, tx_symbols, rx_symbols, noise_lin, actual_batch_size, 
                    calculate_flops=getattr(self.config, 'calculate_recovery_flops', False)
                )
                detection_time = time.time() - detection_start
                time_tracker['detection_recovery']['total'] += detection_time
                time_tracker['detection_recovery']['count'] += 1

                epoch_sers.append(ser)

                # Generate auxiliary data with timing
                aux_gen_start = time.time()
                (aug_x_aux, aug_mask, aug_y_aux) = generate_aux_data_online(
                    rx_symbols, recovered_symbols, self.aug_times, self.config.masking_type
                )
                aux_gen_time = time.time() - aux_gen_start
                time_tracker['aux_data_generation']['total'] += aux_gen_time
                time_tracker['aux_data_generation']['count'] += 1

                # TTT training step with timing
                ttt_start = time.time()
                with tf.GradientTape() as tape:
                    online_aux_pred = self.model.aux_branch(aug_x_aux, aug_mask)
                    h_pred = self.model.main_branch(h_ls)
                    main_loss = self.model.compiled_loss(h_true, h_pred)
                    aux_loss = self.model.compiled_loss(aug_y_aux, online_aux_pred)

                step_main_mse = mse(h_true, h_pred).numpy()
                step_aux_mse = mse(aug_y_aux, online_aux_pred).numpy()
                channel_mse.append(step_main_mse)
                aux_mse.append(step_aux_mse)

                # Apply gradients based on training mode
                if self.config.ssl:
                    self.model.encoder.trainable = True
                    self.model.main_decoder.trainable = False 
                    self.model.aux_decoder.trainable = False
                    grads = tape.gradient(aux_loss, self.model.trainable_weights)
                    self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                elif self.config.supervised:
                    self.model.encoder.trainable = True
                    self.model.main_decoder.trainable = False
                    self.model.aux_decoder.trainable = False
                    grads = tape.gradient(main_loss, self.model.trainable_weights)
                    self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                ttt_time = time.time() - ttt_start
                time_tracker['ttt_training']['total'] += ttt_time
                time_tracker['ttt_training']['count'] += 1

                training_step_counter += 1

                # Log step-wise metrics to wandb
                wandb.log({
                    'ttt/step_channel_mse': step_main_mse,
                    'ttt/step_aux_loss': step_aux_mse,
                    'ttt/step_ser': ser,
                })

                # Debug output every few steps
                if training_step_counter % 10 == 0:
                    debug_msg = f"Step {training_step_counter} - SER: {ser:.4f} | Channel MSE: {step_main_mse:.6f} | Detection: {detection_time:.3f}s | AuxGen: {aux_gen_time:.3f}s | TTT: {ttt_time:.3f}s"
                    
                    # Add FLOPs info if enabled
                    if getattr(self.config, 'calculate_recovery_flops', False):
                        total_flops = sum(self.recovery_flops.values())
                        debug_msg += f" | Total FLOPs: {total_flops:,}"
                    
                    print(debug_msg)

            # Log epoch metrics
            avg_ser = np.mean(epoch_sers)
            avg_channel_mse = np.mean(channel_mse)
            avg_aux_mse = np.mean(aux_mse)
            
            # Calculate timing averages
            step_count = time_tracker['detection_recovery']['count']
            avg_detection_time = time_tracker['detection_recovery']['total'] / step_count if step_count > 0 else 0
            avg_aux_gen_time = time_tracker['aux_data_generation']['total'] / step_count if step_count > 0 else 0
            avg_ttt_time = time_tracker['ttt_training']['total'] / step_count if step_count > 0 else 0
            
            # Log to wandb with clear output format distinction
            wandb.log({
                'ttt/epoch_channel_mse': avg_channel_mse,
                'ttt/epoch_aux_loss': avg_aux_mse,
                'ttt/epoch_avg_ser': avg_ser,
            })
            
            print(f"Epoch {epoch+1} Summary (Output Format: {self.config.output}):")
            print(f"Average Channel MSE: {avg_channel_mse:.6f}")
            print(f"Average Auxiliary MSE: {avg_aux_mse:.6f}")
            print(f"Average Symbol Error Rate: {avg_ser:.6f}")
            print(f"Average Detection/Recovery Time: {avg_detection_time:.4f}s")
            print(f"Average Aux Data Gen Time: {avg_aux_gen_time:.4f}s") 
            print(f"Average TTT Time: {avg_ttt_time:.4f}s")
                
            print("-" * 50)

        # Log final time summary
        self._log_final_time_summary(time_tracker)
        
        # Log final FLOPs summary if enabled
        if getattr(self.config, 'calculate_recovery_flops', False):
            print("\n=== Final FLOPs Summary ===")
            total_flops = sum(self.recovery_flops.values())
            print(f"Total Recovery FLOPs: {total_flops:,}")
            print(f"Detection FLOPs: {self.recovery_flops['detection']:,}")
            if self.output == "bit":
                print(f"Channel Decoding FLOPs: {self.recovery_flops['channel_decoding']:,}")
                print(f"Channel Encoding FLOPs: {self.recovery_flops['channel_encoding']:,}")
                print(f"Mapping FLOPs: {self.recovery_flops['mapping']:,}")
            else:
                print(f"Symbol Mapping FLOPs: {self.recovery_flops['symbol_mapping']:,}")
            print(f"RG Mapping FLOPs: {self.recovery_flops['rg_mapping']:,}")
            
            # Log to wandb
            for operation, flops in self.recovery_flops.items():
                wandb.log({f'recovery_flops/{operation}': flops})
            wandb.log({f'recovery_flops/total_recovery': total_flops})

    def _log_final_time_summary(self, time_tracker):
        """Log final time consumption summary with clear output format distinction"""
        step_count = time_tracker['detection_recovery']['count']
        
        # Calculate averages
        avg_detection_time = time_tracker['detection_recovery']['total'] / step_count if step_count > 0 else 0
        avg_aux_gen_time = time_tracker['aux_data_generation']['total'] / step_count if step_count > 0 else 0
        avg_ttt_time = time_tracker['ttt_training']['total'] / step_count if step_count > 0 else 0
        
        # Log to wandb with output format distinction
        wandb.log({
            f'time/final_avg_detection_time_{self.config.output}': avg_detection_time,
            f'time/final_avg_aux_gen_time_{self.config.output}': avg_aux_gen_time,
            f'time/final_avg_ttt_time_{self.config.output}': avg_ttt_time,
            f'time/final_total_detection_time_{self.config.output}': time_tracker['detection_recovery']['total'],
            f'time/final_total_aux_gen_time_{self.config.output}': time_tracker['aux_data_generation']['total'],
            f'time/final_total_ttt_time_{self.config.output}': time_tracker['ttt_training']['total'],
        })
        
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
            
            # Evaluate the model - NO device forcing
            self.eval_loader = self.dataset.get_eval_loader(self.config.eval_batch_size, "test", "both")
            eval_data_iterator = iter(self.eval_loader)

            eval_channel_mse = []
            eval_aux_mse = []
            
            for step in tqdm(range(num_steps), desc=f"Post-TTT Model Evaluation SNR {snr}", leave=False):
                (h_ls, h_true), (x_aux, mask, y_aux) = next(eval_data_iterator)
                
                h_pred, aux_pred = self.model((h_ls, (x_aux, mask)))
                
                step_channel_mse = mse(h_true, h_pred).numpy()
                step_aux_mse = mse(y_aux, aux_pred).numpy()
                
                eval_channel_mse.append(step_channel_mse)
                eval_aux_mse.append(step_aux_mse)

            # Calculate and log results
            avg_eval_channel_mse = np.mean(eval_channel_mse)
            avg_eval_aux_mse = np.mean(eval_aux_mse)
            
            wandb.log({
                f'post_ttt_{scenario_name}_channel_mse': avg_eval_channel_mse,
                f'post_ttt_{scenario_name}_aux_mse': avg_eval_aux_mse,
            })
            
            # Save results
            mses.loc[len(mses)] = [snr, avg_eval_channel_mse, "Model", self.config.seed]
            results_file = os.path.join(self.log_dir, f"post_ttt_{scenario_name}_{scenario}_results_snr_{snr}.csv")
            mses.to_csv(results_file, index=False)
            
            print(f"SNR {snr} Evaluation Results:")
            print(f"Channel Estimation MSE: {avg_eval_channel_mse:.6f}")
            print(f"Auxiliary Task MSE: {avg_eval_aux_mse:.6f}")
            print(f"Results saved to {results_file}")
    
    def get_recovery_flops_report(self):
        """Get a formatted report of FLOPs for recovery operations"""
        total_flops = sum(self.recovery_flops.values())
        self.recovery_flops['total_recovery'] = total_flops
        
        report = "\n=== Symbol Recovery FLOPs Report ===\n"
        report += f"{'Operation':<20} {'FLOPs':<15} {'Percentage':<10}\n"
        report += "-" * 50 + "\n"
        
        for operation, flops in self.recovery_flops.items():
            if operation != 'total_recovery':
                percentage = (flops / total_flops * 100) if total_flops > 0 else 0
                report += f"{operation:<20} {flops:<15,} {percentage:<10.2f}%\n"
        
        report += "-" * 50 + "\n"
        report += f"{'Total':<20} {total_flops:<15,} {'100.00%':<10}\n"
        report += "=" * 50 + "\n"
        
        return report
    
    def reset_recovery_flops(self):
        """Reset the FLOPs counters for recovery operations"""
        self.recovery_flops = {
            'detection': 0,
            'channel_decoding': 0,
            'channel_encoding': 0,
            'mapping': 0,
            'rg_mapping': 0,
            'symbol_mapping': 0,
            'total_recovery': 0
        }
    
    def log_recovery_flops(self, wandb_log=True, verbose=True):
        """Log FLOPs to wandb and optionally print report"""
        if verbose:
            report = self.get_recovery_flops_report()
            print(report)
        
        if wandb_log:
            # Log individual FLOPs to wandb
            for operation, flops in self.recovery_flops.items():
                wandb.log({f'recovery_flops/{operation}': flops})
        

def main(args):
    wandb_config = {**vars(args)}
    run = wandb.init(project='INFOCOM2026', # INFOCOM2026 or city_pairs
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
    parser.add_argument("--supervised", type=int, default=0)
    parser.add_argument("--ssl", type=int, default=1)
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

    # Add continual TTT arguments
    parser.add_argument("--continual", type=int, default=0, 
                       help="Enable continual TTT across multiple scenarios")
    parser.add_argument("--scenarios", type=str, default="", 
                       help="Comma-separated list of scenarios for continual TTT")
    
    # Add FLOPs calculation argument
    parser.add_argument("--calculate_recovery_flops", type=int, default=1, 
                       help="Calculate and report FLOPs for symbol recovery operations (0=disabled, 1=enabled)")

    with tf.device('/CPU'):
        print("Forcing CPU for evaluation, unless APPLE fixes MPS support")
        main(parser.parse_args(sys.argv[1:]))
