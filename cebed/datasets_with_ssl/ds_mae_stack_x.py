"""
Defines dataset classes
"""
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



class MAEDataset(BaseMultiInputDataset):
    """
    Multi-domain dataset with MAE as the auxilary task with stacked inputs
    
    The dataset contains N domains and each domain consists of X number of samples
    All the domains are used for training

    For main task (channel estimation) :
    ----------------------------------------------------------------------------
    Before preprocessing: 
    input_main = incomplete LS estimates: [n_domain, num_samples, num_r, num_r_ants, n_t, n_t_ants, num_symbols, num_subcarriers]
    label_main = full channel: [n_domain, num_samples, num_r, num_r_ants, n_t, n_t_ants, num_symbols, num_subcarriers]
    
    After preprocessing ('raw' or 'low'):
    train_x_main = [n_train_sample, ns, nf, n_channel] or [n_train_sample, n_pilot_s, n_pilot_sc, n_channel]
    train_y_main = [n_train_sample, ns, nf, n_channel]
    # ---------------------------------------------------------------------------- #


    For auxilary task (received signal reconstruction with stacked inputs):
    ----------------------------------------------------------------------------
    Before preprocessing:
    inputs1_aux = stacked tensor of masked received signals and estimated transmitted symbols: [n_domain, n_sample, 1, n_r_ants+n_t_ants, ns, nf]
    inputs2_aux = random mask: [n_domain, n_sample, n_r, n_r_ants, ns, nf]
    labels_aux = received signals (y_samples): [n_domain, n_sample, n_r, n_r_ants, ns, nf]

    After preprocessing ('raw'):
    train_x1_aux = [n_train_sample, ns, nf, (n_r_ants+n_t_ants)*2]
    train_x2_aux = [n_train_sample, ns, nf] 
    train_y_aux = [n_train_sample, ns, nf, n_r*n_r_ants*2]
    # ---------------------------------------------------------------------------- #

    """
    
    def __init__(
        self,
        path: str,
        train_split: float = 0.9,
        main_input_type: str = "low", # this only refers to input_type of the main task
        aux_input_type: str = "raw", # this only refers to input_type of the auxilary task
        sym_error_rate: float = 0.001,
        seed: int = 0,
    ):
        self.sym_error_rate = sym_error_rate
        super().__init__(path, train_split, main_input_type, aux_input_type, seed)

    def generate_random_mask(self, shape):
        """Generate random mask for a batch of samples using discrete masking only"""
        random_mask = np.zeros(shape)
        n_sample = shape[1]

        for i in range(n_sample):
            # Discrete two symbols unmasked
            sym_indices = np.random.choice(range(14), size=2, replace=False)
            random_mask[:,i,:,:,sym_indices,:] = 1
                
        return random_mask


    def load_data(self):
        """
        Load data from disk
        create inputs and labels
        """
        
        # load a data dict from the disk
        data_path = glob.glob(f"{self.main_path}/data*")[0]
        data = read_dataset_from_file(data_path)

        self.y_samples = np.array(data["y"][:]) # "y" shape: [num_domains, num_samples, num_r, num_r_ants, num_symbols, num_subcarriers]
        self.x_samples = np.array(data["x"][:])
        self.h_samples = np.array(data["h"][:])

        # create input-label pairs for main task (channel estimation task)
        self.labels_main = np.array(data["h"][:])
        self.num_domains = self.labels_main.shape[0]
        self.size = np.prod(self.labels_main.shape[:2])
        self.inputs_main = [] # a list of inputs for each domain
        for d in range(self.num_domains):
            ds_inputs = self.env.estimate_at_pilot_locations(self.y_samples[d]).numpy()
            self.inputs_main.append(ds_inputs) 
        # ---------------------------------------------------------------------------- #
        # labels_main shape: [num_domains, num_samples, num_r, num_r_ants, n_t, n_t_ants, num_symbols, num_subcarriers]
        # inputs_main[0] shape: [num_samples, num_r, num_r_ants, n_t, n_t_ants, num_symbols, num_subcarriers]
        # ---------------------------------------------------------------------------- #

        ## create input-label pairs for auxilary task (reconstruction task)
        self.labels_aux = self.y_samples
        
        # Generate random mask
        random_mask = self.generate_random_mask(self.y_samples.shape)

        # Manually create random errors in the ground-truth x_samples ==> est_x_samples
        est_x_samples = deepcopy(self.x_samples)
        num_symbols = est_x_samples.shape[-2] * est_x_samples.shape[-1]
        num_errors = int(num_symbols * self.sym_error_rate)

        for d in range(self.x_samples.shape[0]):
            for s in range(self.x_samples.shape[1]):
                # Randomly select indices to flip
                all_indices = np.arange(num_symbols)
                indices = np.random.choice(all_indices, num_errors, replace=False)
                for idx in indices:
                    sym_idx = idx // est_x_samples.shape[-1]
                    sc_idx = idx % est_x_samples.shape[-1]
                    est_x_samples[d,s,0,0,sym_idx,sc_idx] = flip_sign(est_x_samples[d,s,0,0,sym_idx,sc_idx])

        # Create masked version of y_samples
        masked_y = self.y_samples * random_mask 

        # Stack masked received signals and estimated transmitted symbols
        # masked_y shape: [num_domains, num_samples, num_r, num_r_ants, num_symbols, num_subcarriers]
        # est_x_samples shape: [num_domains, num_samples, n_t, n_t_ants, num_symbols, num_subcarriers]
        # Since num_r = num_t = 1, target shape: [num_domains, num_samples, 1, num_r_ants + num_t_ants, num_symbols, num_subcarriers]


        stacked_input  = np.concatenate([masked_y, est_x_samples], axis=3)
        self.inputs1_aux = stacked_input  # Stacked tensor
        self.inputs2_aux = random_mask     # Random mask
        # ---------------------------------------------------------------------------- #
        # labels_aux shape: [num_domains, num_samples, num_r, num_r_ants, num_symbols, num_subcarriers]
        # inputs1_aux shape: [num_domains, num_samples, 1, num_r_ants + num_t_ants, num_symbols, num_subcarriers]
        # inputs2_aux shape: [num_domains, num_samples, num_r, num_r_ants, num_symbols, num_subcarriers]
        # ---------------------------------------------------------------------------- #
    
    def preprocess_inputs_aux(self, input1, input2, input_type, mask):
        '''
        # input1 shape (stacked): [num_domains, num_samples, 1, num_r_ants + num_t_ants, num_symbols, num_subcarriers]
        # input2 shape (mask): [num_domains, num_samples, num_r, num_r_ants, num_symbols, num_subcarriers]
        '''
        # ---------------------------- preprocess input1: ---------------------------- #
        if input_type == "low" and input2 is None:
            raise ValueError(f"Mask is needed for the requested input type {input_type}")
       
        if len(input1.shape) > 4:
            raise ValueError("Input 1 shape cannot have more than 6 dimensions")
        
        # Remove dimensions that are equal to 1
        x = tf.squeeze(input1) # [num_r_ants + num_t_ants, n_symbols, n_sc]
        if len(x.shape) > 2:
            x = tf.transpose(x, [1, 2, 0])  # [num_symbols, num_subcarriers, n_ants]
        
        # construct inputs based on the input_type
        if input_type == "raw":
            pre_x1 = x
        # elif input_type == "low":
        #     # Use the mask from input2 to determine pilot locations
        #     mask_indices = tf.where(input2) # pilot-location 1; non-pilot-location 0
        #     # gather the pilot symbols
        #     symbol_indices, _ = tf.unique(mask_indices[:, -2])
        #     low_x = tf.gather(indices=symbol_indices, params=x, axis=0)
        #     # gather the pilot subcarriers
        #     subc_indices, _ = tf.unique(mask_indices[:, -1])
        #     pre_x1 = tf.gather(indices=subc_indices, params=low_x, axis=1)
        #     # pre_x shape:  [num_unique_symbols, num_unique_subcarriers, n_ants]
        else:
            raise ValueError(f"Unknown input mode {input_type}")

        # Convert to real
        pre_x1 = complex_to_real(pre_x1) # [n_symbols, n_scs, n_ants*2]

        # ---------------------------- preprocess input2: ---------------------------- #
        if len(input2.shape) > 4:
            raise ValueError("Input 2 shape cannot have more than 4 dimensions")
        # Remove dimensions that are equal to 1
        pre_x2 = tf.squeeze(input2) 
        
        return pre_x1, pre_x2


    def preprocess_labels_aux(self, label):
        # labels_aux shape: [num_domains, num_samples, num_r, num_r_ants, num_symbols, num_subcarriers]
        if len(label.shape) > 4:
            raise ValueError("Input shape cannot have more than 4 dimensions")

        # Remove dimensions that are equal to 1
        label = tf.squeeze(label) # [n_symbols, n_sc]

        if len(label.shape) > 2 : # [num_r_ants, num_symbols, num_subcarriers]
            label = tf.transpose(label, [1, 2, 0]) # [num_symbols, num_subcarriers, num_r_ants]

        label = complex_to_real(label)
        return label

    def preprocess_aux(self, input1, input2, label, train=True):
        if train == True:
            label = self.preprocess_labels_aux(label)
        input1, input2 = self.preprocess_inputs_aux(input1, input2, self.aux_input_type, self.mask)
        return input1, input2, label
    

# ---------------------------------------------------------------------------- #
# Test Dataset Class
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test the MAEDataset
    print(f"\n{'='*50}")
    print(f"Testing MAEDataset")
    print(f"{'='*50}")
    
    MyDataset = MAEDataset(
        path="./data_TTTtrain/ps2_p72/rt1/snr10to20_speed5", 
        train_split=0.9, 
        main_input_type="low",
        aux_input_type="raw",
        sym_error_rate=0.001,
        seed=42
    )

    train_loader, eval_loader = MyDataset.get_loaders(
        train_batch_size=64,
        eval_batch_size=64,
        task="both"
    )

    
    # Get a batch of data
    for (x_main, y_main), (x1_aux, x2_aux, y_aux) in train_loader.take(1):
        print("\nMain task:")
        print(f"x1_main shape: {x_main.shape}")
        print(f"y_main shape: {y_main.shape}")

        print("\nAuxiliary task:")
        print(f"x1_aux shape: {x1_aux.shape}")
        print(f"x2_aux shape: {x2_aux.shape}")
        print(f"y_aux shape: {y_aux.shape}")
        
        # Visualize a sample
        plt.figure(figsize=(15, 10))
        
        # Get number of channels in x1_aux
        num_channels = x1_aux.shape[-1]
        
        # Calculate grid size based on number of channels
        total_plots = num_channels + 3  # channels + main input/output + mask
        rows = 2 + (num_channels // 2 + num_channels % 2)  # Main task row + rows needed for channels
        cols = 2
        
        # Main task input and output (top row)
        plt.subplot(rows, cols, 1)
        plt.title("Main Task Input (x1)")
        plt.imshow(np.abs(x_main[0, :, :, 0]))
        plt.colorbar()
        
        plt.subplot(rows, cols, 2)
        plt.title("Main Task Label")
        plt.imshow(np.abs(y_main[0, :, :, 0]))
        plt.colorbar()
        
        # Aux task input - show all channels
        for ch in range(num_channels):
            plt.subplot(rows, cols, 3 + ch)
            plt.title(f"Aux Task Input (x1) - Channel {ch}")
            plt.imshow(np.abs(x1_aux[0, :, :, ch]))
            plt.colorbar()
        
        # Aux task mask and output (last row)
        plt.subplot(rows, cols, 3 + num_channels)
        plt.title("Aux Task Mask (x2)")
        plt.imshow(x2_aux[0])
        plt.colorbar()
        
        plt.subplot(rows, cols, 4 + num_channels)
        plt.title("Aux Task Label")
        plt.imshow(np.abs(y_aux[0, :, :, 0]))
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig("mae_dataset_visualization.png")
        plt.close()
        print("\nVisualization saved as 'mae_dataset_visualization.png'")

        print(x1_aux[0, :, :, 1])
        print(x1_aux[0, :, :, 3])
        break
