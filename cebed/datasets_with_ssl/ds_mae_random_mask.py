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


class MAEDatasetRandomMask(BaseMultiInputDataset):
    def __init__(
        self,
        path: str,
        train_split: float = 0.9,
        main_input_type: str = "low", # this only refers to input_type of the main task
        aux_input_type: str = "raw", # this only refers to input_type of the auxilary task
        sym_error_rate: float = 0.001,
        seed: int = 0,
        aug_factor: int = 1,  # Number of random masks per sample
        masking_type: str = "discrete"  # "discrete" or "contiguous" or "fix_length" or "random_symbols"
    ):  
        self.sym_error_rate = sym_error_rate
        self.aug_factor = aug_factor
        self.masking_type = masking_type
        super().__init__(path, train_split, main_input_type, aux_input_type, seed)

    def generate_random_mask(self, shape):
        """Generate random mask for a batch of samples"""
        random_mask = np.zeros(shape)
        n_sample = shape[1]

        for i in range(n_sample):
            if self.masking_type == "discrete":  # Discrete two symbols unmasked --> encoder X
                sym_indices = np.random.choice(range(shape[-2]), size=2, replace=False)  # Use shape[-2] instead of hardcoded 14
                random_mask[:,i,:,:,sym_indices,:] = 1
                
            elif self.masking_type == "contiguous":  # Contiguous two symbols unmasked --> encoder X
                start_idx = np.random.randint(0, shape[-2] - 1)  # Use shape[-2] instead of hardcoded 13
                sym_indices = [start_idx, start_idx + 1]
                random_mask[:,i,:,:,[sym_indices[0],sym_indices[1]],:] = 1
            
            elif self.masking_type == "random_symbols":
                num_unmasked = 3
                sym_indices = np.random.choice(range(shape[-2]), size=num_unmasked, replace=False)  # Use shape[-2] instead of hardcoded 14
                random_mask[:,i,:,:,sym_indices,:] = 1
                
            elif self.masking_type == "fix_length":
                # Get dimensions
                _, _, _, _, num_symbols, num_subcarriers = shape
                
                # Calculate total grid points and number of points to unmask (20%)
                total_points = num_symbols * num_subcarriers
                num_unmasked_symbols = 2
                num_unmasked = num_unmasked_symbols * num_subcarriers
                
                # Create flattened mask and randomly set points to 1
                flat_mask = np.zeros(total_points)
                unmasked_indices = np.random.choice(total_points, size=num_unmasked, replace=False)
                flat_mask[unmasked_indices] = 1
                
                # Reshape the mask to correct dimensions
                reshaped_mask = flat_mask.reshape(num_symbols, num_subcarriers)
                
                # Apply the same mask to all channels and antennas
                random_mask[:,i,:,:,:,:] = reshaped_mask

            elif self.masking_type == "fixed":
                random_mask[:,i,:,:,[2,9],:] = 1

            else:
                raise ValueError(f"Unknown masking type {self.masking_type}")
                
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
        self.x_samples = np.array(data["x"][:]) # "x" shape: [num_domains, num_samples, n_t, num_t_ants, num_symbols, num_subcarriers]
        self.h_samples = np.array(data["h"][:])

        ## manually create random errors in the ground-truth x_samples ==> est_x_samples
        est_x_samples = deepcopy(self.x_samples)
        num_symbols = est_x_samples.shape[-2] * est_x_samples.shape[-1]
        num_errors = int(num_symbols * self.sym_error_rate)

        # Exclude the pilot symbols from being flipped
        # exclude_indices = [2 * est_x_samples.shape[-1], 9 * est_x_samples.shape[-1]]
        for d in range(self.x_samples.shape[0]):
            for s in range(self.x_samples.shape[1]):
                # Randomly select indices to flip
                # Create a pool of indices excluding the 3rd and 10th symbols
                all_indices = np.arange(num_symbols)
                # valid_indices = np.setdiff1d(all_indices, exclude_indices)
                indices = np.random.choice(all_indices, num_errors, replace=False)
                for idx in indices:
                    sym_idx = idx // est_x_samples.shape[-1]
                    sc_idx = idx % est_x_samples.shape[-1]
                    est_x_samples[d,s,0,0,sym_idx,sc_idx] = flip_sign(est_x_samples[d,s,0,0,sym_idx,sc_idx])

        # create input-label pairs for main task (channel estimation task)
        self.labels_main = self.h_samples
        self.num_domains = self.labels_main.shape[0]
        self.base_size = np.prod(self.labels_main.shape[:2])  # Store original size
        self.inputs_main = [] 
        for d in range(self.num_domains):
            ds_inputs = self.env.estimate_at_pilot_locations(self.y_samples[d]).numpy()
            self.inputs_main.append(ds_inputs)

        ## create input-label pairs for auxilary task (reconstruction task)
        self.labels_aux = self.y_samples / est_x_samples

        # Augment both main and auxiliary data if aug_factor > 1
        if self.aug_factor > 1:
            # Augment main task data
            self.labels_main = np.tile(self.labels_main, (1, self.aug_factor, 1, 1, 1, 1, 1, 1))
            for d in range(self.num_domains):
                self.inputs_main[d] = np.tile(self.inputs_main[d], (self.aug_factor, 1, 1, 1, 1, 1, 1))
            
            # Augment auxiliary task data
            self.labels_aux = np.tile(self.labels_aux, (1, self.aug_factor, 1, 1, 1, 1))
            
            # Update size to reflect augmented data
            self.size = self.base_size * self.aug_factor
        else:
            self.size = self.base_size

        # Generate masks after augmentation
        random_mask = self.generate_random_mask(self.labels_aux.shape)
        
        self.inputs1_aux = self.labels_aux * random_mask
        self.inputs2_aux = random_mask

    def preprocess_inputs_aux(self, input1, input2, input_type, mask):
        """
        Per-sample preprocessing for function mapping in prepare_dataset
        """
        if input_type == "low" and input2 is None:
            raise ValueError(f"Mask is needed for the requested input type {input_type}")

        if len(input1.shape) > 4:
            raise ValueError("Input shape cannot have more than 4 dimensions")

        # Remove dimensions that are equal to 1
        x = tf.squeeze(input1)  # [n_symbols, n_sc]
        if len(x.shape) > 2:
            x = tf.transpose(x, [1, 2, 0])  # [num_symbols, num_subcarriers, num_channels]
        
        # construct inputs based on the input_type
        if input_type == "raw":
            pre_x = x
        elif input_type == "low":
            if self.masking_type == "fix_length":
                # Find indices where mask is 1 (unmasked points)
                mask_indices = tf.where(tf.squeeze(input2))
                # Gather the unmasked points
                unmasked_values = tf.gather_nd(x, mask_indices)
                # Reshape to (2, 72, channels)
                pre_x = tf.reshape(unmasked_values, [2, 72, -1])
            else:
                mask_indices = tf.where(input2)  # unmasked 1; masked 0
                # gather the pilot symbols
                symbol_indices, _ = tf.unique(mask_indices[:, -2])
                low_x = tf.gather(indices=symbol_indices, params=x, axis=0)
                # gather the pilot subcarriers
                subc_indices, _ = tf.unique(mask_indices[:, -1])
                pre_x = tf.gather(indices=subc_indices, params=low_x, axis=1)
        else:
            raise ValueError(f"Unknown input mode {input_type}")

        # Convert to real
        pre_x = complex_to_real(pre_x)  # For fix_length: [2, 72, n_channels*2]
        pre_mask = tf.squeeze(input2)  # [n_symbols, n_scs]
        return pre_x, pre_mask

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
    
    # Test different masking types
    masking_types = ["discrete"]
    aug_factors = [1]  # Added more aug_factors to test

    for mask_type in masking_types:
        for aug_factor in aug_factors:
            print(f"\n{'='*50}")
            print(f"Testing masking_type='{mask_type}' with aug_factor={aug_factor}")
            print(f"{'='*50}")
            
            MyDataset = MAEDatasetRandomMask(
                path="./data/ps2_p72/rt1/snr10to20_speed5", 
                train_split=0.9, 
                main_input_type="low",
                aux_input_type="low",
                aug_factor=aug_factor,
                seed=0,
                masking_type=mask_type
            )

            train_loader, eval_loader = MyDataset.get_loaders(
                train_batch_size=64,
                eval_batch_size=64,
                task="both"
            )

            # Count total number of batches
            train_batches = sum(1 for _ in train_loader)
            eval_batches = sum(1 for _ in eval_loader)

            # Print dataset statistics
            print("\nDataset Statistics:")
            print(f"Dataset size: {MyDataset.size}")
            print(f"Number of domains: {MyDataset.num_domains}")
            print(f"Number of training batches: {train_batches}")
            print(f"Number of evaluation batches: {eval_batches}")

            # Examine first batch only
            for batch_data in train_loader.take(1):
                (x_main, y_main), (x, mask, y) = batch_data
                print(f"\nFirst Batch Shapes:")
                print(f"Main input shape: {x_main.shape}")
                print(f"Main label shape: {y_main.shape}")
                print(f"Aux input shape: {x.shape}")
                print(f"Mask shape: {mask.shape}")
                print(f"Aux label shape: {y.shape}")
                
                # Count masked vs unmasked elements
                num_masked = tf.reduce_sum(tf.cast(mask == 0, tf.int32))
                num_total = tf.size(mask)
                mask_ratio = float(num_masked) / float(num_total)
                print(f"Masking ratio: {mask_ratio:.2%}")
                
                # Take first sample from batch for visualization
                sample_idx = 0
                
                # Create a figure with 3 subplots
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Plot the mask
                mask_plot = axes[0].imshow(mask[sample_idx], aspect='auto', cmap='binary')
                axes[0].set_title('Mask Pattern')
                axes[0].set_xlabel('Subcarriers')
                axes[0].set_ylabel('Symbols')
                plt.colorbar(mask_plot, ax=axes[0])
                
                # Plot the original data (magnitude)
                data_mag = tf.abs(y[sample_idx, ..., 0])  # Take first channel
                orig_plot = axes[1].imshow(data_mag, aspect='auto')
                axes[1].set_title('Original Data (Magnitude)')
                axes[1].set_xlabel('Subcarriers')
                axes[1].set_ylabel('Symbols')
                plt.colorbar(orig_plot, ax=axes[1])
                
                # Plot the masked data (magnitude)
                masked_mag = tf.abs(x[sample_idx, ..., 0])  # Take first channel
                masked_plot = axes[2].imshow(masked_mag, aspect='auto')
                axes[2].set_title('Masked Data (Magnitude)')
                axes[2].set_xlabel('Subcarriers')
                axes[2].set_ylabel('Symbols')
                plt.colorbar(masked_plot, ax=axes[2])
                
                plt.tight_layout()
                plt.savefig("mae_random_mask_example.png")
                

                # Create additional figure for main task data
                fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot main input (channel estimates at pilot locations)
                x_main_mag = tf.abs(x_main[sample_idx, ..., 0])  # Take first channel
                main_input_plot = axes2[0].imshow(x_main_mag, aspect='auto')
                axes2[0].set_title('Main Input (Channel Estimates)')
                axes2[0].set_xlabel('Subcarriers')
                axes2[0].set_ylabel('Symbols')
                plt.colorbar(main_input_plot, ax=axes2[0])
                
                # Plot main label (true channel)
                y_main_mag = tf.abs(y_main[sample_idx, ..., 0])  # Take first channel
                main_label_plot = axes2[1].imshow(y_main_mag, aspect='auto')
                axes2[1].set_title('Main Label (True Channel)')
                axes2[1].set_xlabel('Subcarriers')
                axes2[1].set_ylabel('Symbols')
                plt.colorbar(main_label_plot, ax=axes2[1])
                
                plt.tight_layout()
                plt.savefig("main_task_data.png")
