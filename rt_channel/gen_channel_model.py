import sys
import os

import numpy as np
os.environ['DRJIT_LIBLLVM_PATH'] = '/opt/homebrew/Cellar/llvm@16/16.0.6_1/lib/libLLVM.dylib'

import matplotlib.pyplot as plt
import sys
import pathlib
# Add the project root directory to the Python path
root_dir = pathlib.Path(__file__).parent
sys.path.append(str(root_dir))

import sionna
# Set random seed for reproducibility
# sionna.config.seed = 42  # This is no longer available in this version of Sionna
import tensorflow as tf
tf.random.set_seed(42)  # Use TensorFlow's seed instead

import matplotlib.pyplot as plt
import numpy as np
import time

# For link-level simulations
import tensorflow as tf
import pickle


class CIRGenerator:
    """Creates a generator from a given dataset of channel impulse responses.

    The generator samples ``num_tx`` different transmitters from the given path
    coefficients `a` and path delays `tau` and stacks the CIRs into a single tensor.

    Note that the generator internally samples ``num_tx`` random transmitters
    from the dataset. For this, the inputs ``a`` and ``tau`` must be given for
    a single transmitter (i.e., ``num_tx`` =1) which will then be stacked
    internally.

    Parameters
    ----------
    a : [batch size, num_rx, num_rx_ant, 1, num_tx_ant, num_paths, num_time_steps], complex
        Path coefficients per transmitter.

    tau : [batch size, num_rx, 1, num_paths], float
        Path delays [s] per transmitter.

    num_tx : int
        Number of transmitters

    Output
    -------
    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths], tf.float
        Path delays [s]
    """

    def __init__(self,
                 a,
                 tau,
                 num_tx):

        # Copy to tensorflow
        self._a = tf.constant(a, tf.complex64)
        self._tau = tf.constant(tau, tf.float32)
        self._dataset_size = self._a.shape[0]

        self._num_tx = num_tx

    def __call__(self):

        # Generator implements an infinite loop that yields new random samples
        while True:
            # Sample 4 random users and stack them together
            idx,_,_ = tf.random.uniform_candidate_sampler(
                            tf.expand_dims(tf.range(self._dataset_size, dtype=tf.int64), axis=0),
                            num_true=self._dataset_size,
                            num_sampled=self._num_tx,
                            unique=True,
                            range_max=self._dataset_size)

            a = tf.gather(self._a, idx)
            tau = tf.gather(self._tau, idx)

            # Transpose to remove batch dimension
            a = tf.transpose(a, (3,1,2,0,4,5,6))
            tau = tf.transpose(tau, (2,1,0,3))

            # And remove batch-dimension
            a = tf.squeeze(a, axis=0)
            tau = tf.squeeze(tau, axis=0)

            yield a, tau


# num_ue = 1
# batch_size = 20 # Must be the same for the BER simulations as CIRDataset returns fixed batch_size
# num_rx = 1
# num_rx_ant = 1
# num_tx = 1
# num_tx_ant = 1
# num_time_steps = 14 # NOTE = num_ofdm_symbols
# max_num_paths = 75

# # pickle a,tau from files
# with open('rt0/a.pkl', 'rb') as f:
#     a = pickle.load(f)
# with open('rt0/tau.pkl', 'rb') as f:
#     tau = pickle.load(f)

# print(a.shape)
# print(tau.shape)

# # Init CIR generator
# cir_generator = CIRGenerator(a,
#                              tau,
#                              num_ue)

# # Initialises a channel model that can be directly used by OFDMChannel layer
# channel_model = CIRDataset(cir_generator,
#                            batch_size,
#                            num_rx,
#                            num_rx_ant,
#                            num_tx,
#                            num_tx_ant,
#                            max_num_paths,
#                            num_time_steps)
# print(channel_model)

# self.channel = OFDMChannel(
#     channel_model=self.channel_model,
#     resource_grid=self.rg,
#     add_awgn=True,
#     normalize_channel=config.normalize_channel,
#     return_channel=True,
# )