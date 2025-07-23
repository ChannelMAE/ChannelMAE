"""
This file evaluates the online TTT performance of a two-branch model with shared decoder
- Main encoder -
               --- Shared decoder ---
- Aux encoder  -

Compared with online_ttt_v1.py, this file changes the pretrained model architecture.
Accordingly, during TTT, only Aux-encoder and shared-decoder can be trained.

"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import sys
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)

import wandb
wandb.login()
from typing import List, Dict
from dataclasses import dataclass, asdict
from copy import deepcopy
from functools import partial
import time

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
from cebed.utils_eval import mse, plot_2Dimage, expand_masked_input,get_next_batch
from cebed.utils import set_random_seed
import argparse


@dataclass
class EvalConfig:
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
    supervised: int = 0
    ssl: int = 0
    epochs: int = 1


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
        self.model_name = "ReconMAEV2" # fix mask for main, random mask for aux
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
      
    
    def load_model_from_path(self, path):
        '''explicitly load the model from a specific path'''
        self.model.load_weights(path).expect_partial()

    
    def setup(self):
        """Setup the evaluator"""
        # -------------------- Bind with data and create datasets -------------------- #
        # create dataset
        dataset_class = cds.get_dataset_class(self.config.eval_dataset_name)
        assert self.config.eval_dataset_name == "RandomMask" # MUST be RandomMask for ttt-task
        self.dataset = dataset_class(
            self.config.eval_data_dir,
            train_split=self.config.train_split,
            main_input_type=self.config.main_input_type,
            aux_input_type = self.config.aux_input_type,
            sym_error_rate = self.config.sym_error_rate,
            seed=self.config.seed
        )
        
        # ------------------------------- Create model ------------------------------- #
        # get model hyper-parameters from .yaml file
        assert self.model_name == "ReconMAEV2"
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
    

    def online_adapt(self,snr_range: List[int]) -> None:
        '''
        This function's logic is similar to online_ttt()'s logic in ttt.py
        '''
        ## initialize a random-mask model that can handle low_dim_input and mask
        # write it in the setup() function

        # load pretrained two-branch model parameters
        # self.load_model_from_path(os.path.join(self.model_dir, "cp.ckpt"))
        self.model.load_weights(self.model_dir)

        # start online adaptation
        self.loader = self.dataset.get_eval_loader(self.config.eval_batch_size, "test", "both")
        # self.model.set_train_mode("ttt")
        # self.encoder.trainable = True
        # self.main_decoder.trainable = False
        # self.aux_decoder.trainable = True        

        self.model.compile(optimizer=tf.keras.optimizers.legacy.Adam(0.001),loss="mse")
        
        for i,snr in enumerate(snr_range):
            
            # only test the channel estimator on a certain SNR level
            if snr == 20:
                noise_lin = tf.pow(10.0, -snr / 10.0)
                num_steps = (
                    len(self.dataset.test_indices[0]) + self.config.eval_batch_size - 1
                ) // self.config.eval_batch_size

                
                # aux_loss=[]
                for epoch in range(self.config.epochs):

                    channel_mse=[]
                    aux_mse=[]
                    batch_data = iter(self.loader)
                    
                    # for batch_id, (h_ls, h_true) in enumerate(main_batch_data):
                
                    for step in range(num_steps):

                        ### test the channel estimation task on the same test batch
                        (h_ls, h_true),(x_aux, mask, y_aux) = next(batch_data)
                        # pilot_mask = tf.tile(tf.expand_dims(one_pilot_mask, axis=0), [h_ls.shape[0], 1, 1]) # [batch, 14, 72]
                        
                        # if epoch==0 and step==0:
                        #     pass # remain the pretrained model
                        # else:
                        #     self.model.main_encoder.set_weights(self.model.aux_encoder.get_weights())

                        # time_start = time.time()
                        with tf.GradientTape() as tape:
                            h_pred, aux_pred = self.model((h_ls, (x_aux, mask)))
                            main_loss = self.model.compiled_loss(h_true, h_pred)
                            aux_loss = self.model.compiled_loss(y_aux, aux_pred)

                        step_main_mse = mse(h_true, h_pred).numpy()
                        step_aux_mse = mse(y_aux, aux_pred).numpy()
                        channel_mse.append(step_main_mse)
                        aux_mse.append(step_aux_mse)
                        # print("Per batch channel estimation MSE: ", channel_mse)

                        ### supervised: directly train the channel estimator
                        if self.config.ssl:
                            self.model.main_encoder.trainable = False
                            self.model.aux_encoder.trainable = True 
                            self.model.shared_decoder.trainable = True
                            grads = tape.gradient(aux_loss, self.model.trainable_weights)
                            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                            
                            wandb.log({'batch_step': epoch*(num_steps) + step,
                                        'channel_est_mse': main_loss,
                                        'ssl_train_loss': aux_loss})
                            
                        elif self.config.supervised:
                            self.model.main_encoder.trainable = True
                            self.model.aux_encoder.trainable = False
                            self.model.shared_decoder.trainable = True
                            grads = tape.gradient(main_loss, self.model.trainable_weights)
                            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                        
                            wandb.log({'batch_step': epoch*(num_steps) + step,
                                        'channel_est_mse': main_loss,
                                        'ssl_train_loss': aux_loss})
                            
                        else: # no training is conducted
                            wandb.log({'batch_step': epoch*(num_steps) + step,
                                        'channel_est_mse': main_loss,
                                        'ssl_train_loss': aux_loss})
                        # compute the moving average loss until the current step



                        # print("Per-step training time: ", time.time()-time_start)
                        
                    # average the aux-task training loss & channel estimation MSE
                    avg_channel_mse = np.mean(channel_mse)
                    avg_aux_mse = np.mean(aux_mse)  
                    wandb.log({'epoch': epoch,
                            'epoch_channel_mse': avg_channel_mse,
                            'epoch_aux_loss': avg_aux_mse})

                print("Last-epoch average channel estimation MSE at SNR {}: {}".format(snr,avg_channel_mse))

    
def main(args):

    wandb_config = {**vars(args)} # one-line code 
    run = wandb.init(project='Globecom2025', config=wandb_config)

    # saved_config = read_metadata(os.path.join(args.trained_model_dir, "config.yaml"))  # saved_config must be a dict
    # saved_config = read_metadata(os.path.join(args.trained_model_dir, "config.yaml"))  # saved_config must be a dict    
    # start_snr = saved_config["start_ds"]
    # end_snr = saved_config["end_ds"]
    # step = (end_snr - start_snr) / saved_config["num_domains"]
    # snr_range = np.arange(saved_config["start_ds"], saved_config["end_ds"], step)
    snr_range = np.arange(0, 25, 5)
    eval_config = EvalConfig(**vars(args))

    # initialize evaluator
    set_random_seed(eval_config.seed)
    evaluator = Evaluator(eval_config)
    evaluator.setup()

    # load trained model into the evaluator
    evaluator.online_adapt(snr_range)


if __name__ == "__main__":
    # The following arguments are exactly the same as the dataclass EvalConfig in cebed/TTT/trainer.py
    parser = argparse.ArgumentParser(description="Channel Estimation")

    ########################## General args ###################################
    parser.add_argument("--trained_model_dir", type=str, default="./pretrained_weights_v2.h5") # uma_speed5
    parser.add_argument("--eval_data_dir", type=str, default="./data/ps2_p72/rma/snr0to25_speed30") # related to OOD test dataset
    parser.add_argument("--eval_dataset_name", type=str, default="RandomMask")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_split", type=float, default=0.1)
    parser.add_argument("--main_input_type", type=str, default="low", choices=["low", "raw"])
    parser.add_argument("--aux_input_type", type=str, default="low", choices=["low", "raw"])
    parser.add_argument("--aug_noise_std", type=float, default=0)
    parser.add_argument("--sym_error_rate", type=float, default=0)

    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="output_ttt")

    parser.add_argument("--supervised", type=int, default=0)
    parser.add_argument("--ssl", type=int, default=1)

    main(parser.parse_args(sys.argv[1:]))