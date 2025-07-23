import os
import sys
from pathlib import Path
root_path = str(Path(__file__).parent.parent.parent)
sys.path.append(root_path)

from typing import Tuple
from functools import partial
import numpy as np
import tensorflow as tf
from copy import deepcopy
import glob
from cebed.datasets_with_ssl.utils import (
    complex_to_real,
    make_split,
    prepare_dataset,
    read_dataset_from_file,
    combine_datasets,
    flip_sign
)
from cebed.datasets_with_ssl.base_multi_inputs import BaseMultiInputDataset
from cebed.envs import OfdmEnv, EnvConfig
from cebed.utils import read_metadata


class MAEDatasetFixMask(BaseMultiInputDataset):

    def __init__(
        self,
        path: str,
        train_split: float = 0.9,
        main_input_type: str = "low",
        aux_input_type: str = "raw",
        sym_error_rate: float = 0.001,
        seed: int = 0,
    ):
        self.sym_error_rate = sym_error_rate
        super().__init__(path, train_split, main_input_type, aux_input_type, seed)
        
    def load_data(self):
        """Load data from disk and create inputs and labels"""
        data_path = glob.glob(f"{self.main_path}/data*")[0]
        data = read_dataset_from_file(data_path)

        self.y_samples = np.array(data["y"][:])
        self.x_samples = np.array(data["x"][:])
        self.h_samples = np.array(data["h"][:])

        # Create random errors in x_samples
        est_x_samples = deepcopy(self.x_samples)
        num_symbols = est_x_samples.shape[-2] * est_x_samples.shape[-1]
        num_errors = int(num_symbols * self.sym_error_rate)

        for d in range(self.x_samples.shape[0]):
            for s in range(self.x_samples.shape[1]):
                all_indices = np.arange(num_symbols)
                indices = np.random.choice(all_indices, num_errors, replace=False)
                for idx in indices:
                    sym_idx = idx // est_x_samples.shape[-1]
                    sc_idx = idx % est_x_samples.shape[-1]
                    est_x_samples[d,s,0,0,sym_idx,sc_idx] = flip_sign(est_x_samples[d,s,0,0,sym_idx,sc_idx])

        # Main task setup (channel estimation)
        self.labels_main = self.h_samples
        self.num_domains = self.labels_main.shape[0]
        self.size = np.prod(self.labels_main.shape[:2])
        self.inputs_main = []
        for d in range(self.num_domains):
            ds_inputs = self.env.estimate_at_pilot_locations(self.y_samples[d]).numpy()
            self.inputs_main.append(ds_inputs)

        # Auxiliary task setup (reconstruction task)
        self.labels_aux = self.y_samples / est_x_samples

        # NOTE: Create fixed mask (using pilot pattern)
        fixed_mask = np.zeros(self.labels_aux.shape)

        # Set 1s at pilot positions (assuming pilot symbols are at positions 2 and 9)
        fixed_mask[:,:,:,:,2,:] = 1
        fixed_mask[:,:,:,:,9,:] = 1

        self.inputs1_aux = self.labels_aux * fixed_mask
        self.inputs2_aux = fixed_mask

    def preprocess_inputs_aux(self, input1, input2, input_type, mask):
        """Preprocess auxiliary inputs consistently with random mask version"""
        if input_type == "low" and input2 is None:
            raise ValueError(f"Mask is needed for the requested input type {input_type}")

        if len(input1.shape) > 4:
            raise ValueError("Input shape cannot have more than 4 dimensions")

        x = tf.squeeze(input1)
        if len(x.shape) > 2:
            x = tf.transpose(x, [1, 2, 0])

        if input_type == "raw":
            pre_x = x
        elif input_type == "low":
            mask_indices = tf.where(input2)
            symbol_indices, _ = tf.unique(mask_indices[:, -2])
            low_x = tf.gather(indices=symbol_indices, params=x, axis=0)
            subc_indices, _ = tf.unique(mask_indices[:, -1])
            pre_x = tf.gather(indices=subc_indices, params=low_x, axis=1)
        else:
            raise ValueError(f"Unknown input mode {input_type}")

        pre_x = complex_to_real(pre_x)
        pre_mask = tf.squeeze(input2)
        return pre_x, pre_mask

    def preprocess_labels_aux(self, label):
        if len(label.shape) > 4:
            raise ValueError("Input shape cannot have more than 4 dimensions")

        label = tf.squeeze(label)
        if len(label.shape) > 2:
            label = tf.transpose(label, [1, 2, 0])

        label = complex_to_real(label)
        return label

    def preprocess_aux(self, input1, input2, label, train=True):
        if train == True:
            label = self.preprocess_labels_aux(label)
        input1, input2 = self.preprocess_inputs_aux(input1, input2, self.aux_input_type, self.mask)
        return input1, input2, label
    
if __name__ == "__main__":
    MyDataset = MAEDatasetFixMask(path="./data/ps2_p72/Rayleigh/snr0to25_speed5", 
                           train_split=0.9, 
                           main_input_type="low",
                           aux_input_type = "low",
                           sym_error_rate=0,
                           seed=0)
    # already set up the dataset in the above line

    train_loader, eval_loader = MyDataset.get_loaders(
        train_batch_size=64,
        eval_batch_size=64,
        task = "both"
    )

    # Print dataset statistics
    # print(f"Dataset size: {MyDataset.size}")
    # print(f"Number of domains: {MyDataset.num_domains}")

    for batch_data in train_loader:
        # x,y = batch_data
        # print(x.shape, y.shape)

        x, mask, y = batch_data
        print(x.shape, mask.shape, y.shape)
        break
