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
from copy import deepcopy

class LabelPilotDataset(BaseDataset):
    """
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
        self.label_pilot_mask = np.zeros((1, 1, 14, 72))  # [tx, stream, symbols, subcarriers]
        self.label_pilot_mask[0, 0, [4, 11], :] = 1  # NOTE: set label pilots at symbols 4,11 (original pilots at symbol 3, 10)
        self.labels_aux = self.h_samples * self.label_pilot_mask[np.newaxis, np.newaxis, :, :, :, :]  # NOTE: use ground-truth channels of label pilots

        # # NOTE: another option: use LS estimates of label pilots
        # self.labels_main =  self.y_samples/self.x_samples * self.mask 

        self.num_domains = self.labels_aux.shape[0]
        self.size = np.prod(self.labels_aux.shape[:2])
        self.inputs_aux = [] # a list of inputs for each domain
        for d in range(self.num_domains):
            ds_inputs = self.env.estimate_at_pilot_locations(self.y_samples[d]).numpy()
            self.inputs_aux.append(ds_inputs) 
        # ---------------------------------------------------------------------------- #
        # labels_main shape: [num_domains, num_samples, num_r, num_r_ants, n_t, n_t_streams, num_symbols, num_subcarriers]
        # inputs_main[0] shape: [num_samples, num_r, num_r_ants, n_t, n_t_streams, num_symbols, num_subcarriers]
        # ---------------------------------------------------------------------------- #
        
        # create input-label pairs for main task (channel estimation task)
        self.labels_main = np.array(data["h"][:])
        self.inputs_main = deepcopy(self.inputs_aux)
        
    def preprocess_inputs_aux(self, input, input_type, mask):
        return self.preprocess_inputs_main(input, input_type, mask)
        

    def preprocess_labels_aux(self, label):
        """
        Per-sample preprocessing for function mapping in prepare_dataset
        """
        if len(label.shape) > 6:
            raise ValueError("Input shape can to have more than 6 dimensions")

        label = tf.squeeze(label)

        if len(label.shape) == 4:
            # Stack the nr and nt dimensions
            nr, nt, ns, nf = label.shape
            label = tf.reshape(label, (-1, ns, nf))

        if len(label.shape) > 2:
            label = tf.transpose(label, [1, 2, 0]) # [num_symbols, num_subcarriers, num_channels]

        # keep the label's shape as [batch_size, 2, 72, 2]
        label_pilot_indices = tf.where(self.label_pilot_mask) # pilot-location 1; non-pilot-location 0

        # gather the pilot symbols
        symbol_indices, _ = tf.unique(label_pilot_indices[:, -2])
        low_label = tf.gather(indices=symbol_indices, params=label, axis=0)
        # gather the pilot subcarriers
        subc_indices, _ = tf.unique(label_pilot_indices[:, -1])
        label = tf.gather(indices=subc_indices, params=low_label, axis=1)
        
        label = complex_to_real(label)
        
        return label
    

if __name__ == "__main__":
    MyDataset = LabelPilotDataset(path="./data/ps2_p72/rt1/snr10to20_speed5", 
                                train_split=0.9, 
                                main_input_type="low",
                                aux_input_type="low",
                                seed=0)
    
    # set up the dataset in the above line

    train_loader, eval_loader = MyDataset.get_loaders(
        train_batch_size=64,
        eval_batch_size=64,
        task = "both" 
    )

    for (x_main, y_main), (x_aux, y_aux) in train_loader:
        print(x_main.shape, y_main.shape)
        print(x_aux.shape, y_aux.shape)
        break
