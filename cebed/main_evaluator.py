"""
Evaluation a model and baselines
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
from pathlib import Path

root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)
from typing import List, Dict
import time
from dataclasses import dataclass, asdict
from collections import defaultdict
from copy import deepcopy
from functools import partial

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from types import MethodType
from tqdm import tqdm
import numpy as np
import pandas as pd
from sionna.channel import ApplyOFDMChannel
from sionna.ofdm import LSChannelEstimator, LMMSEInterpolator

# from cebed.datasets.sionna import MultiDomainDataset
import cebed.datasets_with_ssl as cds
import cebed.models as cm # one-branch model

from cebed.baselines import evaluate_baselines,evaluate_lmmse,evaluate_ls,evaluate_almmse
from cebed.utils import unflatten_last_dim, write_metadata, read_metadata
from cebed.datasets_with_ssl.utils import postprocess
from cebed.utils_eval import mse, plot_2Dimage


@dataclass
class MainEvalConfig:
    # model-related
    trained_model_dir: str = ""
    model_name: str = "MaeFixMask"

    # dataset-related
    eval_data_dir: str = "./data/ps2_p72/speed5" 
    eval_dataset_name: str = "FixMask"
    eval_batch_size: int = 32
    train_split: float = 0.9
    main_input_type: str = "low" # help us to build a dataset class
    aux_input_type: str = "low"
    aug_noise_std: float = 0.0

    # other configs
    seed: int = 43
    verbose: int = 1
    output_dir: str = "main_eval_output"


class MainEvaluator:
    """
    Evaluate the trained model and the related baselines
    """

    def __init__(self, config: MainEvalConfig):
        self.config = config
        self.model_dir = self.config.trained_model_dir

        # read the config.yaml file from self.model_dir
        saved_train_config = read_metadata(os.path.join(self.model_dir, "config.yaml"))
        self.train_exp_name = saved_train_config["experiment_name"]
        self.model_name = saved_train_config["model_name"]
        self.task = saved_train_config["task"]
        
        # set log dir
        os.makedirs(self.config.output_dir, exist_ok=True)
        # self.log_dir = os.path.join(self.config.output_dir,
        #                             self.mode,
        #                             self.config.eval_data_dir.split('/')[-1]
        #                             )

        self.log_dir = os.path.join(self.config.output_dir, "AttnDecoder")
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
      
    def load_model_from_path(self, path):
        '''explicitly load the model from a specific path'''
        self.model.load_weights(path).expect_partial()

    def setup(self):
        """Setup the evaluator"""
        # -------------------- Bind with data and create datasets -------------------- #
        # create dataset
        dataset_class = cds.get_dataset_class(self.config.eval_dataset_name)
        self.dataset = dataset_class(
            self.config.eval_data_dir, # we can freely set this eval_data_dir; it can be different with train_data_dir
            train_split=self.config.train_split,
            main_input_type=self.config.main_input_type,
            aux_input_type = self.config.aux_input_type,
            # aug_noise_std=self.config.aug_noise_std,
            seed=self.config.seed
        )

        # ------------------------------- Create model ------------------------------- #
        # get model hyper-parameters from .yaml file
        model_hparams = cm.get_model_hparams(
            self.model_name, self.train_exp_name
        )

        # get the class of model
        model_class = cm.get_model_class(self.model_name)
        if "output_dim" not in model_hparams:
            model_hparams["output_dim"] = self.dataset.output_shape
            # output_shape: [num_symbol, num_subcarrier, 2] for one sample in siso case

        # initializ the model
        self.model = model_class(model_hparams)
        input_shape_main, input_shape_aux = self.dataset.get_input_shape()
        
        if self.config.model_name == "MaeFixMask":
            pilot_mask = self.dataset.env.get_mask()
            self.model.set_mask(pilot_mask)
            print("The mask for MAE is set up, used to pad the latent embedding")
        
            # Only pass a single tensor for MaeFixMask
            input1 = tf.zeros([1, input_shape_main[0], input_shape_main[1], input_shape_main[2]])
            self.model(input1)
        else:
            self.model.build(
                tf.TensorShape([None, input_shape_main[0], input_shape_main[1], input_shape_main[2]]) 
            )

    def save(self):
        '''Save EvalConfig and other metadata'''
        config = deepcopy(asdict(self.config)) 
        # check whether in-distribution or out-of-distribution
        saved_train_config = read_metadata(os.path.join(self.model_dir, "config.yaml"))
        if self.config.eval_data_dir == saved_train_config["data_dir"]:
            config.update({"eval_type": "in-distribution"})
        else:
            config.update({"eval_type": "out-of-distribution"})
        print(config)
        write_metadata(os.path.join(self.log_dir, "eval_config.yaml"), config)
    
    def visualize_main(self, snr_range: List[int]) -> None:
        apply_noiseless_channel = ApplyOFDMChannel(add_awgn=False, dtype=tf.as_dtype(tf.complex64))
        apply_noisy_channel = ApplyOFDMChannel(add_awgn=True, dtype=tf.as_dtype(tf.complex64))

        if self.dataset.env is None:
            raise ValueError("Env cannot be None")
        for i, snr in enumerate(tqdm(snr_range)):
            noise_lin = tf.pow(10.0, -snr / 10.0)

            # take the first batch of test samples
            start, end = 0, self.config.eval_batch_size
            batch_size = self.config.eval_batch_size

            if self.dataset.x_samples is None:
                symbols = self.dataset.env.generate_symbols(batch_size)
            else: # select the test symbols 
                symbols = self.dataset.x_samples[i][self.dataset.test_indices[i]][start:end]
            # must use the channel samples in the test datasets
            channels = self.dataset.test_y_main[i][start:end]
            if self.dataset.y_samples is None:
                noisy_signals = apply_noisy_channel([symbols, channels, noise_lin])
            else:
                noisy_signals = self.dataset.y_samples[i][self.dataset.test_indices[i]][start:end] 

            # main task: channel estimation
            inputs_main = self.dataset.test_x_main[i][start:end] # LS estimates as the input of neural estimator
            pre_inputs_main = tf.map_fn(
                                lambda x: self.dataset.preprocess_main(x, None, False)[0],
                                inputs_main,
                                fn_output_signature=tf.float32
                            ) # "low"-dimension inputs
            
            pred_main = self.model(pre_inputs_main) # full channels
            h_hat = tf.map_fn(postprocess, pred_main, fn_output_signature=tf.complex64) # reshaping to (batch_size,num_symbol, num_subcarrier)

            # one-batch loss at a specific SNR domain
            h_true = tf.squeeze(channels)
            batch_loss = mse(h_true,h_hat).numpy()
            print(f"Average test loss over this batch is {batch_loss}.")

            # Initialize default h_tensors
            h_tensors = {
                "h_true": h_true[0],
                "h_hat": h_hat[0],
                "h_ls": inputs_main[0]
            }

            # Add model-specific tensors
            if self.model_name == "ChannelNet":
                resize_inputs = tf.keras.layers.Resizing(self.model.output_dim[0], self.model.output_dim[1], interpolation=self.model.int_type)(pre_inputs_main)
                mid_output = self.model.sr_model(resize_inputs)
                h_hat_before_denoise = tf.map_fn(postprocess, mid_output, fn_output_signature=tf.complex64)
                h_tensors["h_before_denoise"] = h_hat_before_denoise[0]

            # Plot all tensors
            for k,v in h_tensors.items():
                filename = f"snr{snr}_" + k + ".png"
                plot_2Dimage(v,filename)
            
            # move on to the next SNR point
            continue
    
    def evaluate_main(
        self, snr_range: List[int], baselines: List[str] = [], save: bool = True
    ) -> None:
        """Evaluate the performance of denoising task
        baselines: [LMMSE, ALMMSE, LS]"""

        apply_noiseless_channel = ApplyOFDMChannel(
            add_awgn=False, dtype=tf.as_dtype(tf.complex64)
        )
        # record channel estimation mses
        mses = pd.DataFrame(columns=["snr", "mse", "method", "seed"])

        # record test losses for aux task
        losses = pd.DataFrame(columns=["snr", "loss", "seed"])

        if self.dataset.env is None:
            raise ValueError("Env cannot be None")

        for i, snr in enumerate(tqdm(snr_range)): # over a range of SNRs
            noise_lin = tf.pow(10.0, -snr / 10.0)

            test_mse = defaultdict(list)
            test_loss = list()
            num_steps = (
                len(self.dataset.test_indices[0]) + self.config.eval_batch_size - 1
            ) // self.config.eval_batch_size
            # test_indices: a list of length num_domains,
            # each element is a list of indices of test samples in the corresponding domain

            for step in range(num_steps): # number of batches in the i-th domain of test dataset
                start, end = step * self.config.eval_batch_size, min(
                    (step + 1) * self.config.eval_batch_size,
                    len(self.dataset.test_indices[0]),
                )

                batch_size = min(self.config.eval_batch_size, end - start + 1) # satifying the last batch (whose size may be smaller than the batch size)


                if self.dataset.x_samples is None:
                    raise ValueError("x_samples cannot be None")
                    # symbols = self.dataset.env.generate_symbols(batch_size)
                else: # select the test symbols 
                    # here, we do not need preprocessing inputs and labels for testing
                    # domain i: the i-th SNR points
                    symbols = self.dataset.x_samples[i][self.dataset.test_indices[i]][
                        start:end
                    ]

                # use the channel in the test datasets
                channels = self.dataset.test_y_main[i][start:end]

                if self.dataset.y_samples is None:
                    # noisy_signals = apply_noisy_channel([symbols, channels, noise_lin])
                    raise ValueError("y_samples cannot be None")
                else:
                    noisy_signals = self.dataset.y_samples[i][
                        self.dataset.test_indices[i]
                    ][start:end] # received signal

                noiseless_signals = apply_noiseless_channel([symbols, channels]) 

                # Add main task evaluation
                inputs_main = self.dataset.test_x_main[i][start:end]
                pre_inputs_main = tf.map_fn(
                    lambda x: self.dataset.preprocess_main(x, None, False)[0],
                    inputs_main,
                    fn_output_signature=tf.float32
                )
                
                # Get model predictions
                pred_main = self.model(pre_inputs_main)
                h_hat = tf.map_fn(postprocess, pred_main, fn_output_signature=tf.complex64)
                
                # Calculate loss for this batch
                h_true = tf.squeeze(channels)
                batch_loss = mse(h_true, h_hat).numpy()
                test_loss.append(batch_loss)

                # ----------------------------- evaluate baselines ---------------------------- #
                if len(baselines) > 0:
                    baseline_mses = self.evaluate_baselines(
                        noisy_signals, noiseless_signals, channels, snr, baselines
                    )
                    for baseline in baselines:
                        test_mse[baseline].append(baseline_mses[baseline])

            # Calculate average MSEs and losses
            test_mse = {k: np.mean(v) for k, v in test_mse.items()}
            test_loss = np.mean(test_loss) if test_loss else float('nan')

            # Record results
            for mid, (k, v) in enumerate(test_mse.items()):
                mses.loc[len(mses)] = [snr, v, k, self.config.seed]
            
            losses.loc[i] = [snr, test_loss, self.config.seed]

        if save:
            mses.to_csv(os.path.join(self.log_dir, "test_main_mses.csv"), index=False)
            losses.to_csv(os.path.join(self.log_dir, "test_main_losses.csv"), index=False)

        if self.config.verbose > 0:
            print(mses)
            print(losses)

