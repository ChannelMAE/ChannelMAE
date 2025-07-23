"""
Only care about the main task of channel estimation:
input: pilot-based LS estimates
output: full channel
"""
import sys
from pathlib import Path
root_path = str(Path(__file__).parent.parent.parent)
sys.path.append(root_path)
import numpy as np
import glob
from cebed.datasets_with_ssl.utils import (
    read_dataset_from_file,
    complex_to_real
)
from cebed.datasets_with_ssl.base import BaseDataset
import tensorflow as tf

class ClassicDataset(BaseDataset):
    """
    
    The dataset contains N domains and each domain consists of X number of samples
    H: [n_domain, n_sample, num_r, num_r_ants, n_t, n_t_streams, ns, nf]

    Only focus on the main task of channel estimation
    ----------------------------------------------------------------------------
    Before preprocessing (same as MAEDataset): 
    inputs_main = incomplete LS: [n_domain, num_samples, num_r, num_r_ants, n_t, n_t_streams, num_symbols, num_subcarriers]
    labels_main = full channel: [n_domain, num_samples, num_r, num_r_ants, n_t, n_t_streams, num_symbols, num_subcarriers]
    
    After preprocessing ('raw' or 'low'):
    train_x_main = [n_train_sample, ns, nf, n_channel] or [n_train_sample, n_pilot_symbol, n_pilot_sc, n_channel]
    train_y_main = [n_train_sample, ns, nf, n_channel]
    
    """

    # NOTE: use the same __init__ method as in BaseDataset in base.py

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
        # labels_main shape: [num_domains, num_samples, num_r, num_r_ants, n_t, n_t_streams, num_symbols, num_subcarriers]
        # inputs_main[0] shape: [num_samples, num_r, num_r_ants, n_t, n_t_streams, num_symbols, num_subcarriers]
        # ---------------------------------------------------------------------------- #


        # create input-label pairs for denoising
        self.labels_aux = np.array(data["y"][:])
        self.inputs_aux = np.array(data["y"][:])
        # ---------------------------------------------------------------------------- #

    def preprocess_inputs_aux(self, input, input_type, mask):
        '''
        # inputs_aux shape: [num_domains, num_samples, num_r, num_r_ants, num_symbols, num_subcarriers]
        '''

        if input_type == "low" and mask is None:
            raise ValueError(f"Mask is needed for the requested input type {input_type}")

        if len(input.shape) > 4:
            raise ValueError("Input shape cannot have more than 4 dimensions")

        # Remove dimensions that are equal to 1
        x = tf.squeeze(input) # [n_symbols, n_sc]
        if len(x.shape) > 2 : # [num_r_ants, num_symbols, num_subcarriers]
            x = tf.transpose(x, [1, 2, 0]) # [num_symbols, num_subcarriers, num_r_ants]
        
        if input_type == "raw":
            pre_x = x
        elif input_type == "low":
            pilot_indices = tf.where(mask)
            # gather the pilot symbols
            symbol_indices, _ = tf.unique(pilot_indices[:, -2])
            low_x = tf.gather(indices=symbol_indices, params=x, axis=0)
            # gather the pilot subcarriers
            subc_indices, _ = tf.unique(pilot_indices[:, -1])
            pre_x = tf.gather(indices=subc_indices, params=low_x, axis=1)
        else:
            raise ValueError(f"Unknown input mode {input_type}")
        
        pre_x = complex_to_real(pre_x)
        return pre_x


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
    

if __name__ == "__main__":
    MyDataset = ClassicDataset(path="./data/ps2_p72/rayleigh/snr0to25_speed5", 
                                train_split=0.9, 
                                main_input_type="raw",
                                aux_input_type = "low",
                                seed=0)
    
    # set up the dataset in the above line

    train_loader, eval_loader = MyDataset.get_loaders(
        train_batch_size=64,
        eval_batch_size=64,
        task = "main" # MUST BE "main" for ClassicDataset
    )

    # Print dataset statistics
    print(f"Dataset size: {MyDataset.size}")
    print(f"Number of domains: {MyDataset.num_domains}")

    for batch_data in train_loader:
        x,y = batch_data
        print(x.shape, y.shape)
        break
    