"""
Evaluation a model and baselines
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

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tqdm import tqdm
import numpy as np

# from cebed.datasets.sionna import MultiDomainDataset
import cebed.datasets_with_ssl as cds
import cebed.models as cm # one-branch model

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
        self.model_name = "MaeFixMask"
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
        if self.config.eval_dataset_name == "Denoise":
            self.dataset = dataset_class(
                self.config.eval_data_dir,
                train_split=self.config.train_split,
                main_input_type=self.config.main_input_type,
                aux_input_type = self.config.aux_input_type,
                aug_noise_std=self.config.aug_noise_std,
                seed=self.config.seed
            )
        elif self.config.eval_dataset_name == "FixMask" or "RandomMask":
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

        # ------------------------------- Create model ------------------------------- #
        self.model_name = "MaeRandomMask"
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
        if self.model_name =="MaeRandomMask":
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
            self.model.build(
                tf.TensorShape([None, input_shape_aux[0], input_shape_aux[1], input_shape_aux[2]]) 
            )
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
    

    def online_adapt(self,snr_range: List[int]) -> None:
        '''
        This function's logic is similar to online_ttt()'s logic in ttt.py
        '''
        ## initialize a random-mask model that can handle low_dim_input and mask
        # write it in the setup() function

        # load the parameters of trained model into the placeholder
        self.load_model_from_path(os.path.join(self.model_dir, "cp.ckpt"))
        # self.model.decoder.save_weights('./original_decoder_weights')

        # start online adaptation
        self.aux_loader = self.dataset.get_eval_loader(self.config.eval_batch_size, "test", "aux")
        self.main_loader = self.dataset.get_eval_loader(self.config.eval_batch_size, "test", "main")

        # set trainable parts
        self.model.encoder.trainable = True
        self.model.decoder.trainable = True
        self.model.compile(optimizer=tf.keras.optimizers.legacy.Adam(0.001),loss="mse")

        # set the fixed mask for the model call() to work
        one_pilot_mask = self.dataset.env.get_mask()
        one_pilot_mask = tf.squeeze(one_pilot_mask)
        pilot_mask = tf.tile(tf.expand_dims(one_pilot_mask, axis=0), [self.config.eval_batch_size, 1, 1]) # [batch, 14, 72]
        

        for i,snr in enumerate(snr_range):
            noise_lin = tf.pow(10.0, -snr / 10.0)
            num_steps = (
                len(self.dataset.test_indices[0]) + self.config.eval_batch_size - 1
            ) // self.config.eval_batch_size

            
            # aux_loss=[]
            for epoch in range(self.config.epochs):

                test_mse=[]
                aux_batch_data = iter(self.aux_loader)
                main_batch_data = iter(self.main_loader)
                
                # for batch_id, (h_ls, h_true) in enumerate(main_batch_data):
                for step in range(num_steps-1):

                    ### test the channel estimation task on the same test batch
                    h_ls, h_true = next(main_batch_data)
                    # pilot_mask = tf.tile(tf.expand_dims(one_pilot_mask, axis=0), [h_ls.shape[0], 1, 1]) # [batch, 14, 72]
                    
                    h_pred = self.model((h_ls, pilot_mask))
                    channel_mse = mse(h_true, h_pred).numpy()
                    test_mse.append(channel_mse)
                    # print("Per batch channel estimation MSE: ", channel_mse)

                    ### supervised: directly train the channel estimator
                    if self.config.supervised:
                        step_metric = self.model.train_step((h_ls, pilot_mask, h_true))
                        supervised_train_loss = step_metric['loss'].numpy()
                        # print("supervised train loss: ", supervised_train_loss)

                        wandb.log({'batch_step': epoch*(num_steps-1) + step,
                                'batch_test_loss': channel_mse, # moving-average-loss: per-sample loss until the current step
                                'supervised_train_loss':supervised_train_loss})
                        
                    ### SSL: get one batch of training pair of aux task
                    elif self.config.ssl:
                        one_batch_aux = next(aux_batch_data) # x, mask, y in the new channel environment
                        step_metric = self.model.train_step(one_batch_aux)  
                        ssl_train_loss = step_metric['loss'].numpy()
                        # print("Step metric: ", step_metric['loss'].numpy())

                        wandb.log({'batch_step': epoch*(num_steps-1) + step,
                                    'batch_test_loss': channel_mse,
                                    'ssl_train_loss': ssl_train_loss}) # moving-average-loss: per-sample loss until the current step
                    
                    else: # no training is conducted
                        wandb.log({'batch_step': epoch*(num_steps-1) + step,
                                    'batch_test_loss': channel_mse})
                    
                # average the aux-task training loss & channel estimation MSE
                avg_test_mse = np.mean(test_mse)
                wandb.log({'epoch': epoch,
                           'epoch_test_loss': avg_test_mse})
                
                print("Average channel estimation MSE at SNR {}: {}".format(snr,avg_test_mse))

    
def main(args):

    wandb_config = {**vars(args)} # one-line code 
    run = wandb.init(project='Globecom2025', config=wandb_config)

    # saved_config = read_metadata(os.path.join(args.trained_model_dir, "config.yaml"))  # saved_config must be a dict
    # saved_config = read_metadata(os.path.join(args.trained_model_dir, "config.yaml"))  # saved_config must be a dict    
    # start_snr = saved_config["start_ds"]
    # end_snr = saved_config["end_ds"]
    # step = (end_snr - start_snr) / saved_config["num_domains"]
    # snr_range = np.arange(saved_config["start_ds"], saved_config["end_ds"], step)
    snr_range = np.arange(20, 21, 1)
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
    parser.add_argument("--trained_model_dir", type=str, default="../pretrained_weights.h5") # uma_speed5
    parser.add_argument("--eval_data_dir", type=str, default="./data/ps2_p72/tdl/snr0to25_speed10") # related to OOD test dataset
    parser.add_argument("--eval_dataset_name", type=str, default="FixMask")
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_split", type=float, default=0.01)
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