"""
This file evaluates the online TTT performance of a two-branch model with shared encoder
        -- main_decoder
- Encoder 
        -- aux_decoder

This is a single-branch version of online_ttt_v3.py, where we only train the main branch with pseudo-labels and without baseline evaluations.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import sys
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)

import swanlab as wandb
wandb.login()
from dataclasses import dataclass, asdict
import pandas as pd
import time
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tqdm import tqdm
import numpy as np
import cebed.datasets_with_ssl as cds
import cebed.models_with_ssl as cm
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
    
    # evaluation configs
    eval_snr_min: int = -2 
    eval_snr_max: int = 2
    eval_snr_step: int = 1
    eval_base_dir: str = "./data/ps2_p72"
    scenario: str = "umi"     
    speed: str = "30"         

    # dataset-related
    eval_dataset_name: str = "RandomMask"
    eval_batch_size: int = 64
    train_batch_size: int = 64
    ttt_split: float = 0.5  
    eval_split: float = 0.5 
    main_input_type: str = "low"
    aux_input_type: str = "low"
    aug_noise_std: float = 0.0
    sym_error_rate: float = 0.0

    # training configs
    epochs: int = 5
    supervised: int = 1
    ssl: int = 0
    learning_rate: float = 5e-4

    # method configs
    det_method: str = "lmmse"
    output: str = "symbol"
    aug_times: int = 2
    masking_type: str = "discrete"

    # other configs
    seed: int = 43
    verbose: int = 1
    output_dir: str = "experiment_results/v4"
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

        self.log_dir = os.path.join(self.config.output_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        # set the model and dataset objects
        self.model = None
        self.dataset = None
        self.test_loader = None 

        # Add channel application objects
        self.apply_noiseless_channel = ApplyOFDMChannel(add_awgn=False, dtype=tf.as_dtype(tf.complex64))
        self.apply_noisy_channel = ApplyOFDMChannel(add_awgn=True, dtype=tf.as_dtype(tf.complex64))

    def setup(self):
        """Setup the evaluator"""
        # -------------------- Bind with data and create datasets -------------------- #
        # create dataset with aug_factor=1 to match v3's behavior
        dataset_class = cds.get_dataset_class(self.config.eval_dataset_name)
        assert self.config.eval_dataset_name == "RandomMask" # MUST be RandomMask for ttt-task
        # Construct data directory in online_adapt(), here is just for initialization
        print("Initializing dataset with train_data_dir:", self.config.train_data_dir)
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
        '''Single-branch with pseudo-label version of online adaptation'''
        
        # Load model and compile
        self.model.load_weights(self.model_dir)
        self.model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(self.config.learning_rate),
            loss="mse"
        )
        
        # Prepare training data
        print("Preparing training data for TTT...")
        total_samples = self._prepare_training_data()

        # Perform TTT training
        print("Performing TTT training...")
        self._perform_ttt_training(total_samples)
        
        # Post-TTT evaluation
        print("Performing post-ttt evaluation...")
        self._evaluate_cross_snr()

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
                prior_shape = [h_pred.shape[0], 1, 1, self.k]
            elif self.output == "symbol":
                prior_shape = [h_pred.shape[0], 1, 1, self.num_tx]
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

    def _prepare_training_data(self):
        """Prepare and merge training data from multiple domains"""
        dataset_class = cds.get_dataset_class(self.config.eval_dataset_name)
        
        # Initialize dataset with training data directory (consistent with v3)
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

    def _perform_ttt_training(self, total_samples):
        """Perform the actual TTT training process"""
        # Noise computation
        noise_lin = tf.pow(10.0, -self.config.train_snr / 10.0)
        
        domain_idx = 0
        num_steps = (total_samples + self.config.train_batch_size - 1) // self.config.train_batch_size
        

        # Pre-adaptation evaluation (improved consistency with v3)
        before_adapt_test_data = iter(self.train_loader)
        before_adapt_mse = []

        print(f"Starting pre-adaptation evaluation with {num_steps} steps...")
        for step in tqdm(range(num_steps), desc="Pre-adaptation Evaluation", leave=False):
            (h_ls, h_true), _ = next(before_adapt_test_data)
            h_pred = self.model.main_branch(h_ls)
            step_mse = mse(h_true, h_pred).numpy()
            before_adapt_mse.append(step_mse)

        pre_adapt_avg = np.mean(before_adapt_mse)
        print(f"Pre-adaptation Channel MSE: {pre_adapt_avg:.6f}")

        # Log pre-adaptation metrics (consistent with v3)
        wandb.log({
            'ttt/epoch_channel_mse': pre_adapt_avg,
        })
        
        # Training epochs
        for epoch in tqdm(range(self.config.epochs), desc="TTT Training Epochs"):
            print(f"\n=== Starting TTT Epoch {epoch+1}/{self.config.epochs} ===")
            channel_mse = []
            epoch_sers = []
            batch_data = iter(self.train_loader)
            
            for step in tqdm(range(num_steps), desc=f"Epoch {epoch+1} Training Steps", leave=False):
                start = step * self.config.train_batch_size
                end = min((step + 1) * self.config.train_batch_size,
                         len(self.dataset.test_indices[domain_idx]))
                actual_batch_size = end - start
                
                # Get batch data
                (h_ls, h_true), _ = next(batch_data)
                
                # Channel estimation
                h_pred = self.model.main_branch(h_ls)

                # Use domain_idx consistently for tx/rx symbols
                tx_symbols = self.dataset.x_samples[domain_idx][self.dataset.test_indices[domain_idx]][start:end]
                rx_symbols = self.dataset.y_samples[domain_idx][self.dataset.test_indices[domain_idx]][start:end]

                # Symbol detection and recovery with timing
                recovered_symbols, ser = self._perform_detection_and_recovery(
                    h_pred, tx_symbols, rx_symbols, noise_lin, actual_batch_size
                )
                
                epoch_sers.append(ser)

                (aug_x_aux, aug_mask, aug_y_aux) = generate_aux_data_online(
                    rx_symbols, 
                    recovered_symbols, 
                    self.aug_times,
                    masking_type=self.config.masking_type
                )

                with tf.GradientTape() as tape:
                    h_pred = self.model.main_branch(h_ls)
                    loss = self.model.compiled_loss(aug_y_aux, h_pred)  # Use pseudo-labels for main branch

                step_mse = mse(aug_y_aux, h_pred).numpy()
                channel_mse.append(step_mse)

                # Apply gradients (pseudo-label training only)
                if self.config.ssl:
                    self.model.encoder.trainable = True
                    self.model.main_decoder.trainable = False
                    self.model.aux_decoder.trainable = False
                    grads = tape.gradient(loss, self.model.trainable_weights)
                    self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                

                # Log step-wise metrics to wandb (consistent with v3)
                wandb.log({
                    'ttt/step_channel_mse': step_mse,
                    'ttt/step_ser': ser,
                    'step_ser': ser
                })

            # Log epoch metrics (improved consistency)
            avg_ser = np.mean(epoch_sers)
            avg_channel_mse = np.mean(channel_mse)
            
            # Log to wandb with consistent structure
            wandb.log({
                'ttt/epoch_channel_mse': avg_channel_mse,
                'ttt/epoch_avg_ser': avg_ser,
            })
            


    def _evaluate_cross_snr(self):
        """Evaluate model performance across different SNRs after training"""
        dataset_class = cds.get_dataset_class(self.config.eval_dataset_name)
        
        snr_range = range(self.config.eval_snr_min, self.config.eval_snr_max + 1, self.config.eval_snr_step)
        for snr in tqdm(snr_range, desc="Post-TTT Cross-SNR Evaluation"):
            print(f"Evaluating SNR {snr} dB...")
            # Construct eval data directory for current SNR
            next_snr = snr + 1
            eval_data_dir = os.path.join(
                self.config.eval_base_dir,
                self.config.scenario,
                f"snr{snr}to{next_snr}_speed{self.config.speed}"
            )
            
            # Update dataset with current eval directory
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

            # Initialize tracking (consistent with v3)
            mses = pd.DataFrame(columns=["snr", "mse", "method", "seed"])
            
            # Evaluate the model on various SNRs
            self.eval_loader = self.dataset.get_eval_loader(self.config.eval_batch_size, "test", "both")
            eval_data_iterator = iter(self.eval_loader)
            num_steps = (
                len(self.dataset.test_indices[0]) + self.config.eval_batch_size - 1
            ) // self.config.eval_batch_size

            eval_channel_mse = []
            eval_aux_mse = []
            
            for step in tqdm(range(num_steps), desc=f"Model Evaluation SNR {snr}", leave=False):
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
            
            # Log evaluation results (consistent with v3 structure)
            wandb.log({
                'post_ttt_online_channel_mse': avg_eval_channel_mse,
                'post_ttt_online_aux_mse': avg_eval_aux_mse,
            })
            
            # Add model results to the mses DataFrame
            mses.loc[len(mses)] = [snr, avg_eval_channel_mse, "Model", self.config.seed]
            
            print(f"SNR {snr} Evaluation Results:")
            print(f"Channel Estimation MSE: {avg_eval_channel_mse:.6f}")
            print(f"Auxiliary Task MSE: {avg_eval_aux_mse:.6f}")
            
            # Save results to CSV for this SNR
            results_file = os.path.join(self.log_dir, f"post_pseudo_online_{self.config.scenario}_results_snr_{snr}.csv")
            mses.to_csv(results_file, index=False)
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
    print("Running online TTT channel estimation script...")
    parser = argparse.ArgumentParser(description="Online TTT Channel Estimation")

    # Model related
    parser.add_argument("--trained_model_dir", type=str, required=True)
    
    # Dataset related
    parser.add_argument("--eval_dataset_name", type=str, default="RandomMask")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--ttt_split", type=float, default=0.5)
    parser.add_argument("--eval_split", type=float, default=0.5)
    parser.add_argument("--main_input_type", type=str, default="low", choices=["low", "raw"])
    parser.add_argument("--aux_input_type", type=str, default="low", choices=["low", "raw"])
    
    # Training configs
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--supervised", type=int, default=1)
    parser.add_argument("--ssl", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    
    # Method configs
    parser.add_argument("--det_method", type=str, default="lmmse", choices=["lmmse", "k-best", "ep", "mmse-pic"])
    parser.add_argument("--output", type=str, default="symbol", choices=["symbol", "bit"])
    parser.add_argument("--aug_times", type=int, default=2)
    parser.add_argument("--masking_type", type=str, default="discrete", 
                       choices=["discrete", "random_symbols", "fixed", "fix_length"])
    
    # Other configs
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output_dir", type=str, default="experiment_results/v4")
    parser.add_argument("--wandb_name", type=str, required=True)
    
    # Training specific
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--train_snr", type=int, default=20)
    
    # Evaluation specific
    parser.add_argument("--eval_base_dir", type=str, default="./data/ps2_p72")
    parser.add_argument("--scenario", type=str, default="umi")
    parser.add_argument("--speed", type=str, default="30")
    parser.add_argument("--eval_snr_min", type=int, default=-2)
    parser.add_argument("--eval_snr_max", type=int, default=2)
    parser.add_argument("--eval_snr_step", type=int, default=1)
    
    with tf.device('/CPU'):
        print("Forcing CPU for pretraining, unless APPLE fixes MPS support")
        main(parser.parse_args(sys.argv[1:]))
