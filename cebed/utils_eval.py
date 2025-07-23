"""Eval helper functions"""
import os
from typing import Tuple
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from cebed.envs import OfdmEnv
from cebed.datasets_with_ssl.utils import complex_to_real
from sionna.phy.mapping import Constellation


def real_to_complex_batch(tensor: tf.Tensor) -> tf.Tensor:
    """
    Convert a real tensor to a complex tensor
    Assume that the real and imaginary parts are stacked along the last dimension

    :param tensor: a 4D real tensor [bs,h,w,c]
    :return: complex tensor [bs, h,w,c/2]
    """

    if tensor.shape[-1] == 2:
        c_mat = tf.complex(tensor[:, :, :, 0], tensor[:, :, :, 1])

        return c_mat
    elif tensor.shape[-1] > 2:
        nc = int(tensor.shape[-1] / 2)
        c_mat = tf.complex(tensor[:, :, :, :nc], tensor[:, :, :, nc:])

        return c_mat
    else:
        raise ValueError()

def map_indices_to_symbols(indices):
    # Define the constellation points for QPSK
    cons = Constellation('qam', 2)

    # Map indices to symbols using TensorFlow's advanced indexing
    symbols = tf.gather(cons.points, indices)

    return symbols



def get_ser(est_tx_symbols, tx_symbols):
    # Ensure both tensors have the same shape
    assert est_tx_symbols.shape == tx_symbols.shape, "Tensors must have the same shape"
    
    # Compare the real and imaginary parts separately
    real_errors = tf.not_equal(tf.math.real(est_tx_symbols), tf.math.real(tx_symbols))
    imag_errors = tf.not_equal(tf.math.imag(est_tx_symbols), tf.math.imag(tx_symbols))
    
    # Combine the errors
    total_errors = tf.logical_or(real_errors, imag_errors)
    
    # Calculate the number of symbol errors
    num_errors = tf.reduce_sum(tf.cast(total_errors, tf.float32))
    
    # Calculate the total number of symbols
    total_symbols = tf.size(tx_symbols, out_type=tf.float32)
    
    # Calculate the Symbol Error Rate (SER)
    ser = num_errors / total_symbols
    
    return ser


def generate_aux_data_online(rx_symbols, tx_symbols, aug_time, masking_type="discrete"):
    '''
    Given a batch of (y, x_hat), construct a batch of (masked_image, mask, full_image) for training the aux task
    @input
    rx_symbols [batch_size, 1, 1, num_symbols, num_subcarriers], tensor
    tx_symbols [batch_size, 1, 1, num_symbols, num_subcarriers], tensor
    aug_times: int, the number of times to augment the data
    masking_type: str, "discrete" or "random_symbols" or "fixed" or "fix_length"
    '''
    labels_aux = rx_symbols / tx_symbols  # this division is only correct for SISO!!!
    all_random_masks = []
    n_sample = labels_aux.shape[0]
    num_unmasked = 5

    for i in range(n_sample):
        random_mask = np.zeros([aug_time, 1, 1, 14, 72])
        for aug_idx in range(aug_time):
            if masking_type == "fix_length":
                # Get dimensions
                _, _, num_symbols, num_subcarriers = random_mask[aug_idx].shape
                
                # Calculate total grid points and number of points to unmask (fixed to 2 symbols worth)
                total_points = num_symbols * num_subcarriers
                num_unmasked_symbols = 2
                num_unmasked = num_unmasked_symbols * num_subcarriers  # 2 * 72 points
                
                # Create flattened mask and randomly set points to 1
                flat_mask = np.zeros(total_points)
                unmasked_indices = np.random.choice(total_points, size=num_unmasked, replace=False)
                flat_mask[unmasked_indices] = 1
                
                # Reshape the mask to correct dimensions and assign to random_mask
                reshaped_mask = flat_mask.reshape(num_symbols, num_subcarriers)
                random_mask[aug_idx, 0, 0] = reshaped_mask
                
            elif masking_type == "discrete":
                # Discrete two symbols unmasked
                sym_indices = np.random.choice(range(14), size=2, replace=False)
                random_mask[aug_idx, :, :, [sym_indices[0], sym_indices[1]], :] = 1
                
            elif masking_type == "random_symbols":
                # Random number of symbols (2-5)
                sym_indices = np.random.choice(range(14), size=num_unmasked, replace=False)
                random_mask[aug_idx, :, :, sym_indices, :] = 1
                
            elif masking_type == "fixed":
                # Fixed pilot positions (symbols 2 and 9)
                random_mask[aug_idx, :, :, [2, 9], :] = 1

        all_random_masks.append(random_mask)

    # Rest of the processing remains the same
    mask = np.concatenate(all_random_masks, axis=0)
    y_aux = np.repeat(labels_aux, aug_time, axis=0)
    x_aux = mask * y_aux

    # ------------------------------- Preprocessing ------------------------------ #
    pre_mask = np.squeeze(mask) # [Batch_size*aug_time, num_symbols, num_subcarriers], array
    y_aux = np.squeeze(y_aux) # [Batch_size*aug_time, num_symbols, num_subcarriers]
    x_aux = np.squeeze(x_aux)
    if(len(y_aux.shape) == 2):
        pre_mask = np.expand_dims(pre_mask, axis=0)
        y_aux = np.expand_dims(y_aux, axis=0)
        x_aux = np.expand_dims(x_aux, axis=0)
    pre_y_aux = np.zeros([y_aux.shape[0], y_aux.shape[1], y_aux.shape[2], 2])
    # Remove the fixed shape allocation and create it dynamically for each sample
    all_processed_x = []

    # Update preprocessing for fix_length case
    if masking_type == "fix_length":
        for sample_id in range(x_aux.shape[0]):
            x = x_aux[sample_id]
            # Find indices where mask is 1 (unmasked points)
            mask_indices = tf.where(pre_mask[sample_id])
            # Gather the unmasked points
            unmasked_values = tf.gather_nd(x, mask_indices)
            # Reshape to (2, 72, channels)
            processed_x = tf.reshape(unmasked_values, [2, 72, -1])
            # Convert to real values
            processed_x = tf.cast(complex_to_real(processed_x), tf.float32)
            all_processed_x.append(processed_x)
            y = y_aux[sample_id]
            pre_y_aux[sample_id] = tf.cast(complex_to_real(y), tf.float32)
    else:
        for sample_id in range(x_aux.shape[0]):
            x = x_aux[sample_id]
            mask_indices = tf.where(x)
            symbol_indices, _ = tf.unique(mask_indices[:, -2])
            low_x = tf.gather(indices=symbol_indices, params=x, axis=0)
            processed_x = tf.cast(complex_to_real(low_x), tf.float32)
            all_processed_x.append(processed_x)
            y = y_aux[sample_id]
            pre_y_aux[sample_id] = tf.cast(complex_to_real(y), tf.float32)

    # Stack and ensure float32 type
    pre_x_aux = tf.stack(all_processed_x)
    pre_mask = tf.cast(pre_mask, tf.float32)
    pre_y_aux = tf.cast(pre_y_aux, tf.float32)

    return (pre_x_aux, pre_mask, pre_y_aux)


def get_next_batch(iterator, dataset):
    try:
        return next(iterator)
    except StopIteration:
        # Create a new iterator and get the next batch
        iterator = iter(dataset)
        return next(iterator), iterator

class NBatchLogger(tf.keras.callbacks.Callback):
    """
    A Logger that log average performance per `display` steps.
    """
    def __init__(self, display):
        self.step = 0
        self.display = display
        self.metric_cache = {}

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for k in self.params['metrics']:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
        if self.step % self.display == 0:
            metrics_log = ''
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)
            print('step: {}/{} ... {}'.format(self.step,
                                        self.params['steps'],
                                        metrics_log))
            self.metric_cache.clear()

def is_complex_tensor(tensor):
    dtype = tf.as_dtype(tensor.dtype)
    return dtype == tf.dtypes.complex64 or dtype == tf.dtypes.complex128


def plot_2Dimage(tensor, filename:str):
    '''tensor/numpy arrays'''
    # Ensure tensor is complex
    # if not tf.is_tensor(tensor) or not tf.math.is_complex(tensor):
    #     raise ValueError("The input tensor must be a complex tensor.")

    # Squeeze the tensor to remove dimensions of size 1
    squeezed_tensor = tf.transpose(tf.squeeze(tensor))
    if len(squeezed_tensor.shape) > 3 or len(squeezed_tensor.shape) < 2:
        raise ValueError("The input tensor must be 2D or 3D.")
    
    if is_complex_tensor(squeezed_tensor) and len(squeezed_tensor.shape) == 2:
        # Separate the real and imaginary parts
        real_part = tf.math.real(squeezed_tensor)
        imag_part = tf.math.imag(squeezed_tensor)

        # Plot the real part as a 2D image
        plt.figure(figsize=(8,6))
        plt.subplot(1, 2, 1)
        plt.imshow(real_part, cmap='viridis')
        plt.title('Real Part')
        plt.colorbar()

        # Plot the imaginary part as a 2D image
        plt.subplot(1, 2, 2)
        plt.imshow(imag_part, cmap='viridis')
        plt.title('Imaginary Part')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(filename)
    
    if is_complex_tensor(squeezed_tensor)==False and len(squeezed_tensor.shape) == 2:
        plt.figure(figsize=(8,6))
        plt.imshow(squeezed_tensor, cmap='viridis')
        plt.colorbar()
        plt.title('Mask pattern')
        plt.savefig(filename)


    if len(squeezed_tensor.shape) == 3:
        real_part = squeezed_tensor[0,:,:]
        imag_part = squeezed_tensor[1,:,:]

        # Plot the real part as a 2D image
        plt.figure(figsize=(8,6))
        plt.subplot(1, 2, 1)
        plt.imshow(real_part, cmap='viridis')
        plt.title('Real Part')
        plt.colorbar()

        # Plot the imaginary part as a 2D image
        plt.subplot(1, 2, 2)
        plt.imshow(imag_part, cmap='viridis')
        plt.title('Imaginary Part')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(filename)


def expand_masked_input(low_embed: tf.Tensor, mask:tf.Tensor) -> tf.Tensor:
        
    # [num_pilots, 2]
    mask_indices = tf.where(mask == 1) # [batch, 14, 72]

    # [batch, nps, npf, c]
    batch_size = tf.shape(low_embed)[0] 
    n_channel = tf.shape(low_embed)[-1]

    # [batch*nps*npf, c]
    low_embed = tf.reshape(low_embed, [-1,n_channel])
    high_embed = tf.scatter_nd(
        mask_indices, # [batch*nps*npf, 3]
        low_embed,  # [ batch*nps*npf, c]
        tf.cast([
            batch_size, # batch
            mask.shape[1], # n_symbol
            mask.shape[2], # n_subcarrier
            n_channel
        ], dtype=tf.int64),
    )
    return high_embed



def mse(true: tf.Tensor, pred: tf.Tensor) -> float:
    """Computes mean squared error"""

    return tf.reduce_mean(tf.square(tf.abs(true - pred)))


def estimate_covariance_matrices(
    env: OfdmEnv, num_it: int = 10, batch_size: int = 1000, save_dir: str = "./eval"
) -> Tuple[tf.Tensor]:
    """
    Estimates the second order statistics of the channel

    Taken from
    https://nvlabs.github.io/sionna/examples/OFDM_MIMO_Detection.html#

    """

    if os.path.isfile(f"{save_dir}/freq_cov_mat"):
        freq_cov_mat = np.load(f"{save_dir}/freq_cov_mat.npy")
        time_cov_mat = np.load(f"{save_dir}/time_cov_mat.npy")
        space_cov_mat = np.load(f"{save_dir}/space_cov_mat.npy")
        freq_cov_mat = tf.constant(freq_cov_mat, tf.complex64)
        time_cov_mat = tf.constant(time_cov_mat, tf.complex64)
        space_cov_mat = tf.constant(space_cov_mat, tf.complex64)

        return freq_cov_mat, time_cov_mat, space_cov_mat

    freq_cov_mat = tf.zeros([env.config.fft_size, env.config.fft_size], tf.complex64)
    time_cov_mat = tf.zeros(
        [env.config.num_ofdm_symbols, env.config.num_ofdm_symbols], tf.complex64
    )
    space_cov_mat = tf.zeros(
        [env.config.num_rx_antennas, env.config.num_rx_antennas], tf.complex64
    )

    for _ in tf.range(num_it):
        # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
        h_samples = env.sample_channel(batch_size)

        #################################
        # Estimate frequency covariance
        #################################
        # [batch size, num_rx_ant, fft_size, num_ofdm_symbols]
        h_samples_ = tf.transpose(h_samples, [0, 1, 3, 2])
        # [batch size, num_rx_ant, fft_size, fft_size]
        freq_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [fft_size, fft_size]
        freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=(0, 1))
        # [fft_size, fft_size]
        freq_cov_mat += freq_cov_mat_

        ################################
        # Estimate time covariance
        ################################
        # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
        time_cov_mat_ = tf.matmul(h_samples, h_samples, adjoint_b=True)
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat_ = tf.reduce_mean(time_cov_mat_, axis=(0, 1))
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat += time_cov_mat_

        ###############################
        # Estimate spatial covariance
        ###############################
        # [batch size, num_ofdm_symbols, num_rx_ant, fft_size]
        h_samples_ = tf.transpose(h_samples, [0, 2, 1, 3])
        # [batch size, num_ofdm_symbols, num_rx_ant, num_rx_ant]
        space_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [num_rx_ant, num_rx_ant]
        space_cov_mat_ = tf.reduce_mean(space_cov_mat_, axis=(0, 1))
        # [num_rx_ant, num_rx_ant]
        space_cov_mat += space_cov_mat_

    freq_cov_mat /= tf.complex(
        tf.cast(env.config.num_ofdm_symbols * num_it, tf.float32), 0.0
    )
    time_cov_mat /= tf.complex(tf.cast(env.config.fft_size * num_it, tf.float32), 0.0)
    space_cov_mat /= tf.complex(tf.cast(env.config.fft_size * num_it, tf.float32), 0.0)

    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/freq_cov_mat", freq_cov_mat.numpy())
    np.save(f"{save_dir}/time_cov_mat", time_cov_mat.numpy())
    np.save(f"{save_dir}/space_cov_mat", space_cov_mat.numpy())

    return freq_cov_mat, time_cov_mat, space_cov_mat
