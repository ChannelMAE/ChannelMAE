"""
Evaluation a model and baselines
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from typing import List, Dict
import time
from dataclasses import dataclass, asdict
from collections import defaultdict
from copy import deepcopy
from functools import partial

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tqdm import tqdm
import numpy as np
import pandas as pd
from sionna.channel import ApplyOFDMChannel
from sionna.ofdm import LSChannelEstimator, LMMSEInterpolator

# from cebed.datasets.sionna import MultiDomainDataset
import cebed.datasets_with_ssl as cds
import cebed.models as cm # one-branch model

from types import MethodType
from cebed.baselines import evaluate_baselines,evaluate_lmmse,evaluate_ls,evaluate_almmse
from cebed.utils import unflatten_last_dim, write_metadata, read_metadata
from cebed.datasets_with_ssl.utils import postprocess
from cebed.utils_eval import mse, plot_2Dimage, expand_masked_input
from cebed.envs import OfdmEnv

@dataclass
class AuxEvalConfig:
    # model-related
    trained_model_dir: str = ""

    # dataset-related
    eval_data_dir: str = "./datasets/ps2_p72/speed5" 
    eval_dataset_name: str = "Denoise"
    eval_batch_size: int = 64
    train_split: float = 0.9
    main_input_type: str = "low" # help us to build a dataset class
    aux_input_type: str = "low"
    aug_noise_std: float = 0.0
    sym_error_rate: float = 0.001

    # other configs
    seed: int = 43
    verbose: int = 1
    output_dir: str = "eval_output"


class AuxEvaluator:
    """
    Evaluate the trained model and the related baselines
    """

    def __init__(self, config: AuxEvalConfig):
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

        self.log_dir = os.path.join(self.config.output_dir, "DiscreteRandomAttnDecoder")
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
        if self.config.eval_dataset_name == "Denoise":
            self.dataset = dataset_class(
                self.config.eval_data_dir,
                train_split=self.config.train_split,
                main_input_type=self.config.main_input_type,
                aux_input_type = self.config.aux_input_type,
                aug_noise_std=self.config.aug_noise_std,
                seed=self.config.seed
            )
        elif self.config.eval_dataset_name == "FixMask" or self.config.eval_dataset_name == "RandomMask":
            self.dataset = dataset_class(
                self.config.eval_data_dir,
                train_split=self.config.train_split,
                main_input_type=self.config.main_input_type,
                aux_input_type = self.config.aux_input_type,
                sym_error_rate = self.config.sym_error_rate,
                seed=self.config.seed
            )
        else:
            self.dataset = dataset_class(
                self.config.eval_data_dir,
                train_split=self.config.train_split,
                main_input_type=self.config.main_input_type,
                aux_input_type = self.config.aux_input_type,
                seed=self.config.seed
            )
        self.test_loader = self.dataset.get_eval_loader(self.config.eval_batch_size, "test", "aux")

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

        # initialize the model
        self.model = model_class(model_hparams)
        
        # initial inputs to the encoder
        _, input_shape_aux = self.dataset.get_input_shape()

        # build the model
        if self.model_name == "MaeRandomMask":
            low_dim_input = tf.zeros([1,input_shape_aux[0], input_shape_aux[1], input_shape_aux[2]])
            example_mask = self.dataset.env.get_mask()
            example_mask = tf.squeeze(example_mask)
            example_mask = tf.expand_dims(example_mask, axis=0) # [batch, 14, 72]
            self.model((low_dim_input, example_mask))
        elif self.model_name == "MaeFixMask":
            # custom model building
            pilot_mask = self.dataset.env.get_mask()
            self.model.set_mask(pilot_mask)
            print("The mask for MAE is set up, used to pad the latent embedding")
            # Only pass the input tensor for FixMask model
            low_dim_input = tf.zeros([1,input_shape_aux[0], input_shape_aux[1], input_shape_aux[2]])
            self.model(low_dim_input)
        else:
            # custom model-building
            self.model.build(
                tf.TensorShape([None, input_shape_aux[0], input_shape_aux[1], input_shape_aux[2]]) 
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
    

    def visualize_aux(self, snr_range: List[int]) -> None:

        num_batch_per_domain = (
                len(self.dataset.test_indices[0]) + self.config.eval_batch_size - 1
            ) // self.config.eval_batch_size
        
        domain_id = 0
        
        for batch_index, (test_x, mask, test_y) in enumerate(self.test_loader):

            # select one specific batch from this domain
            if batch_index == 0 or (batch_index + 1) % num_batch_per_domain == 1:
                
                # get the SNR of the i-th domain
                snr = snr_range[domain_id]

                # Handle different model types
                if self.model_name == "MaeRandomMask":
                    pred_y = self.model((test_x, mask))
                elif self.model_name == "MaeFixMask":
                    # Only pass the input tensor for FixMask model
                    pred_y = self.model(test_x)
                else:
                    pred_y = self.model(test_x)

                batch_loss = mse(test_y, pred_y).numpy()
                print(f"Average sample loss over the {batch_index}-th batch is {batch_loss}.")

                # -------------------- visualize one sample in this batch -------------------- #
                if self.config.eval_dataset_name in ["FixMask", "RandomMask"]:
                    expand_test_x = expand_masked_input(test_x, mask)
                    signals = {
                        "masked_image": expand_test_x[0],
                        "recon_image": pred_y[0],
                        "full_image": test_y[0],
                        "mask": mask[0]
                    }
                
                    for k,v in signals.items():
                        filename = os.path.join(self.log_dir, f"snr{snr}_{k}.png")
                        plot_2Dimage(v, filename)
            
                # move on to the next SNR point
                domain_id += 1
                continue

    def evaluate_aux(
        self, snr_range: List[int], baselines: List[str] = [], save: bool = True
    ) -> None:
        """Evaluate the performance of denoising task
        baselines: [ALMMSE, LS]"""

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

                # ----------------------------- Auxiliary task ---------------------------- #
                # Get inputs and labels with explicit dtype
                if hasattr(self.dataset, 'test_x_aux'):
                    inputs_aux = tf.cast(self.dataset.test_x_aux[i][start:end], tf.float32)
                elif hasattr(self.dataset, 'test_x1_aux'):
                    inputs_aux = tf.cast(self.dataset.test_x1_aux[i][start:end], tf.float32)
                else:
                    raise AttributeError("Dataset must have either test_x_aux or test_x1_aux attribute")

                # Get labels with explicit dtype
                if hasattr(self.dataset, 'test_y_aux'):
                    labels_aux = tf.cast(self.dataset.test_y_aux[i][start:end], tf.float32)
                else:
                    labels_aux = tf.cast(noisy_signals, tf.float32)

                # Get mask based on model type with explicit dtype
                if self.model_name == "MaeRandomMask":
                    if hasattr(self.dataset, 'inputs2_aux'):
                        mask = tf.cast(
                            self.dataset.inputs2_aux[i][self.dataset.test_indices[i]][start:end],
                            tf.float32
                        )
                    else:
                        mask = tf.cast(self.dataset.env.get_mask(), tf.float32)
                elif self.model_name == "MaeFixMask":
                    mask = tf.cast(self.dataset.env.get_mask(), tf.float32)
                else:
                    mask = None

                # Preprocess each sample in the batch
                if self.model_name == "MaeRandomMask":
                    def preprocess_fn(args):
                        input_sample, mask_sample, label_sample = args
                        processed_input, processed_mask, _ = self.dataset.preprocess_aux(
                            input_sample, mask_sample, label_sample, train=False
                        )
                        return processed_input, processed_mask
                    pre_inputs_aux, pre_mask = tf.map_fn(
                        preprocess_fn,
                        (inputs_aux, mask, labels_aux),
                        fn_output_signature=(tf.float32, tf.float32)
                    )
                    pred_aux = self.model((pre_inputs_aux, pre_mask))
                else:
                    def preprocess_fn(args):
                        input_sample, mask_sample, label_sample = args
                        processed_input, _, _ = self.dataset.preprocess_aux(
                            input_sample, mask_sample, label_sample, train=False
                        )
                        return processed_input
                    pre_inputs_aux = tf.map_fn(
                        preprocess_fn,
                        (inputs_aux, mask, labels_aux),
                        fn_output_signature=tf.float32
                    )
                    pred_aux = self.model(pre_inputs_aux)

                y_hat = tf.map_fn(postprocess, pred_aux, fn_output_signature=tf.complex64) # reshaping
                y_hat = tf.expand_dims(tf.expand_dims(y_hat, axis=1), axis=2) # (batch_size, 1, 1, num_symbol, num_subcarrier)

                # one-batch loss at a specific SNR domain
                denoise_batch_loss = mse(noisy_signals,y_hat).numpy()
                test_loss.append(denoise_batch_loss)

                # ----------------------------- evalute baselines ---------------------------- #
                if len(baselines) > 0:

                    # Use the denoised version of (y + epsilon) to conduct ALMMSE and LS estimation
                    baseline_aux_mses = self.evaluate_baselines(
                        y_hat, noiseless_signals, channels, snr, baselines
                    )

                    # Use the true noisy received signals to conduct ALMMSE and LS estimation
                    baseline_mses = self.evaluate_baselines(
                        noisy_signals, noiseless_signals, channels, snr, baselines
                    )
                    
                    for baseline in baselines:
                        test_mse[baseline+"_aux"].append(baseline_aux_mses[baseline])
                        test_mse[baseline].append(baseline_mses[baseline])

            test_mse = {k: np.mean(v) for k, v in test_mse.items()}
            test_loss = np.mean(test_loss)

            step = 2*len(baselines) # 
            for mid, (k, v) in enumerate(test_mse.items()):
                mses.loc[step * i + mid] = [snr, v, k, self.config.seed]
            
            losses.loc[i] = [snr, test_loss, self.config.seed] # record various denoising effect at different SNRs

        if save:
            mses.to_csv(os.path.join(self.log_dir, "test_denoise_mses.csv"), index=False)
            losses.to_csv(os.path.join(self.log_dir, "test_denoise_losses.csv"), index=False)

        if self.config.verbose > 0:
            print(mses)
            print(losses)
