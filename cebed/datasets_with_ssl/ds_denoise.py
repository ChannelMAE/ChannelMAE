"""
Defines dataset classes
"""
import sys
from pathlib import Path
root_path = str(Path(__file__).parent.parent.parent)
sys.path.append(root_path)
import numpy as np
from copy import deepcopy
import glob
from cebed.datasets_with_ssl.utils import (
    read_dataset_from_file,
    complex_normal,
    complex_to_real
)
from cebed.datasets_with_ssl.base import BaseDataset
from cebed.utils import read_metadata
import tensorflow as tf

class DenoiseDataset(BaseDataset):
    """
    Dataset for denoising task
    
    The dataset contains N domains and each domain consists of X number of samples
    H: [n_domain, n_sample, num_r, num_r_ants, n_t, n_t_streams, ns, nf]

    For main task (channel estimation) :
    ----------------------------------------------------------------------------
    Before preprocessing (same as MAEDataset): 
    inputs_main = incomplete LS: [n_domain, num_samples, num_r, num_r_ants, n_t, n_t_streams, num_symbols, num_subcarriers]
    labels_main = full channel: [n_domain, num_samples, num_r, num_r_ants, n_t, n_t_streams, num_symbols, num_subcarriers]
    
    After preprocessing ('raw' or 'low'):
    train_x_main = [n_train_sample, ns, nf, n_channel] or [n_train_sample, n_pilot_symbol, n_pilot_sc, n_channel]
    train_y_main = [n_train_sample, ns, nf, n_channel]
    # ---------------------------------------------------------------------------- #


    For auxilary task (received signal denoising):
    ----------------------------------------------------------------------------
    Before preprocessing:
    inputs_aux = received_signal_add_noise "y": [n_domain, n_sample, num_r, num_r_ants, ns, nf]
    labels_aux = received_signal "y": [n_domain, n_sample, num_r, num_r_ants, ns, nf]

    After preprocessing (only 'raw' is allowed):
    train_x_aux = [n_train_sample, ns, nf, n_channel] 
    train_y_aux = [n_train_sample, ns, nf, n_channel]
    # ---------------------------------------------------------------------------- #

    """
    def __init__(
        self,
        path: str,
        train_split: float = 0.9,
        main_input_type: str = "low", # this only refers to input_type of the main task
        aux_input_type: str = "raw", # this only refers to input_type of the auxilary task
        aug_noise_std: float = 0.5,
        seed: int = 0,
    ):
        self.aug_noise_std = aug_noise_std
        super().__init__(path, train_split, main_input_type, aux_input_type, seed)
    

    def load_data(self):
        """
        Load data from disk
        create inputs and labels
        """
        
        # load a data dict from the disk
        data_path = glob.glob(f"{self.main_path}/data*")[0]
        data = read_dataset_from_file(data_path)
        
        self.y_samples = np.array(data["y"][:]) # "y" shape: [num_domains, num_samples, num_r, num_r_ants, num_symbols, num_subcarriers]
        self.x_samples = np.array(data["x"][:]) # "x" shape: [num_domains, num_samples, num_t, num_t_ants, num_symbols, num_subcarriers]
        
        # # check pilot and data symbols: x_samples are all the symbols over the OFDM resource grid.
        # print(f"x_samples shape: {self.x_samples.shape}")
        # print(f"all the symbols: {self.x_samples[0, 0, 0, 0, :, 0]}")

        self.h_samples = np.array(data["h"][:])

        # input: h_ls at two pilot symbols
        self.labels_main = np.array(data["h"][:])
        self.num_domains = self.labels_main.shape[0]
        self.size = np.prod(self.labels_main.shape[:2])

        self.inputs_main = [] 
        for d in range(self.num_domains):
            ds_inputs = self.env.estimate_at_pilot_locations(self.y_samples[d]).numpy()
            self.inputs_main.append(ds_inputs)
        # ---------------------------------------------------------------------------- #
        # labels_main shape: [num_domains, num_samples, num_r, num_r_ants, n_t, n_t_streams, num_symbols, num_subcarriers]
        # inputs_main[0] shape: [num_samples, num_r, num_r_ants, n_t, n_t_streams, num_symbols, num_subcarriers]
        # ---------------------------------------------------------------------------- #

        # create input-label pairs for denoising
        self.labels_aux = np.array(data["y"][:]) 
        env_config = read_metadata(f"{self.main_path}/metadata.yaml") # env_config is still an EnvConfig object
        num_domains = env_config.num_domains
        start_snr = env_config.start_ds
        end_snr = env_config.end_ds

        # add 1 time noise to the received signal --> inputs_aux 
        self.inputs_aux = deepcopy(self.labels_aux) # avoid changing label_aux
        step = int((end_snr - start_snr) / num_domains)
        for i, snr_db in enumerate(range(start_snr, end_snr, step)):
            # var_no = np.power(10, -snr_db / 10) # noise power = noise variance for zero-mean Gaussian
            added_noise = complex_normal(mean = 0.0, std = self.aug_noise_std, shape = self.inputs_aux[i].shape)
            self.inputs_aux[i] = self.inputs_aux[i] + added_noise
        

        # ---------------------------------------------------------------------------- #
        # labels_aux shape: [num_domains, num_samples, num_r, num_r_ants, num_symbols, num_subcarriers]
        # inputs_aux shape: [num_domains, num_samples, num_r, num_r_ants, num_symbols, num_subcarriers]
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
        # elif input_type == "low":
        #     pilot_indices = tf.where(mask)
        #     # gather the pilot symbols
        #     symbol_indices, _ = tf.unique(pilot_indices[:, -2])
        #     low_x = tf.gather(indices=symbol_indices, params=x, axis=0)
        #     # gather the pilot subcarriers
        #     subc_indices, _ = tf.unique(pilot_indices[:, -1])
        #     pre_x = tf.gather(indices=subc_indices, params=low_x, axis=1)
        else:
            raise ValueError(f"Set input_type to 'raw' for denoising task")
        
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
    MyDataset = DenoiseDataset(path="./data/ps2_p72/rt1/snr10to20_speed5", 
                                train_split=0.9, 
                                main_input_type="low",
                                aux_input_type="raw",
                                seed=0)
    
    # already set up the dataset in the above line

    train_loader, eval_loader = MyDataset.get_loaders(
        train_batch_size=64,
        eval_batch_size=64,
        task = "both"  # "aux" for pretrainig; "main" for online inference; Training task is always "aux"
    )

    for (x_main, y_main), (x_aux, y_aux) in train_loader.take(1):
        print(f"x_main shape: {x_main.shape}")
        print(f"y_main shape: {y_main.shape}")
        print(f"x_aux shape: {x_aux.shape}")
        print(f"y_aux shape: {y_aux.shape}")

        import matplotlib.pyplot as plt

        # Create a 2x2 subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 4))

        # Visualize x_main[0,:,:,0]
        im1 = axes[0,0].imshow(x_main[0,:,:,0].numpy(), cmap='viridis', aspect='auto')
        axes[0,0].set_title('x_main[0,:,:,0]')
        axes[0,0].set_xlabel('Subcarrier')
        axes[0,0].set_ylabel('Symbol')
        plt.colorbar(im1, ax=axes[0,0])

        # Visualize y_main[0,:,:,0]
        im2 = axes[0,1].imshow(y_main[0,:,:,0].numpy(), cmap='viridis', aspect='auto')
        axes[0,1].set_title('y_main[0,:,:,0]')
        axes[0,1].set_xlabel('Subcarrier')
        axes[0,1].set_ylabel('Symbol')
        plt.colorbar(im2, ax=axes[0,1])

        # Visualize x_aux[0,:,:,0]
        im3 = axes[1,0].imshow(x_aux[0,:,:,0].numpy(), cmap='viridis', aspect='auto')
        axes[1,0].set_title('x_aux[0,:,:,0]')
        axes[1,0].set_xlabel('Subcarrier')
        axes[1,0].set_ylabel('Symbol')
        plt.colorbar(im3, ax=axes[1,0])

        # Visualize y_aux[0,:,:,0]
        im4 = axes[1,1].imshow(y_aux[0,:,:,0].numpy(), cmap='viridis', aspect='auto')
        axes[1,1].set_title('y_aux[0,:,:,0]')
        axes[1,1].set_xlabel('Subcarrier')
        axes[1,1].set_ylabel('Symbol')
        plt.colorbar(im4, ax=axes[1,1])

        plt.tight_layout()
        plt.savefig('denoise_example_data.png', dpi=300)