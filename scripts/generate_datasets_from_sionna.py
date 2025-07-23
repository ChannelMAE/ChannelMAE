"""
Script to generate data from Sionna
"""
import sys
from pathlib import Path
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import argparse
from typing import Dict, Any
from collections import defaultdict

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# from tqdm import trange
import h5py

from cebed.envs import OfdmEnv, EnvConfig
from cebed.utils import set_random_seed, write_metadata


def generate_samples_from_env(
    env: OfdmEnv, snr_db: int, size: int, batch_size=100
) -> Dict[str, np.ndarray]:
    """
    Generate samples from env
    The total number of samples = size

    Output: dict with keys 'h', 'y', 'x'

    """

    i, n_samples = 0, 0
    n_iter = int(size / batch_size) # n_iter = number of batches

    ysamples_set = tf.TensorArray(tf.complex64, size=n_iter)
    xsamples_set = tf.TensorArray(tf.complex64, size=n_iter)
    hsamples_set = tf.TensorArray(tf.complex64, size=n_iter)

    print(f"Generation {size} samples with SNR {snr_db}\t")

    while n_samples < size:
        bt = min(batch_size, size - n_samples)

        # generate one batch, which could be accelerated by @tf.function
        x, rx_y, channel = env(bt, snr_db, return_x=True) # return_x = True
        # x:     [batch size, 1, num_rx_ant, num_ofdm_symbols, fft_size]
        # rx_y:  [batch size, 1, num_rx_ant, num_ofdm_symbols, fft_size]
        # channel: [batch size, 1, num_rx_ant, 1, 1, num_ofdm_symbols, fft_size]

        # create the inaccurate data symbols (corrected data symbols) 
        


        xsamples_set = xsamples_set.write(i, x)
        ysamples_set = ysamples_set.write(i, rx_y)
        hsamples_set = hsamples_set.write(i, channel)
        n_samples += bt
        i += 1

    ysamples_set = ysamples_set.concat() # concat along the 1-st axis, so that the shape of ysamples_set is [size, 1, num_rx_ant, num_ofdm_symbols, fft_size]
    hsamples_set = hsamples_set.concat() # shape of hsamples_set is [size, 1, num_rx_ant, 1, 1, num_ofdm_symbols, fft_size]
    xsamples_set = xsamples_set.concat() # shape of xsamples_set is [size, 1, num_rx_ant, num_ofdm_symbols, fft_size]

    data = dict(h=hsamples_set.numpy(), y=ysamples_set.numpy(), x=xsamples_set.numpy())
    # keys: 'h', 'y', 'x'
    # values: numpy arrays

    return data


def get_data(env: OfdmEnv, args: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Collect dataset

    Input:
    -------
    env: OfdmEnv
    args: Dict[str, Any], data generation arguments

    Output:
    -------
    data: Dict[str, np.ndarray], each value is a list of numpy arrays   
    """

    # if args.num_domains == 1:
    #     # only generate one type of dataset (i.e., one domain)
    #     return generate_samples_from_env(env, args.start_ds, args.size, args.batch_size)
    
    
    data = defaultdict(list) # any nonexistent key will be initialized with an empty list.
    step = int((args.end_ds - args.start_ds) / args.num_domains) 

    for snr_db in range(args.start_ds, args.end_ds, step):

        if args.num_domains == 1:
            print(f"Generating single-domain dataset at SNR {snr_db}") # NOTE: the single-domain data is still stored in a list

        task_data = generate_samples_from_env(env, snr_db, args.size, args.batch_size)

        assert task_data["h"].shape[0] == args.size

        for k, v in task_data.items():
            data[k].append(v)
        
    assert len(data["h"]) == args.num_domains 
    # 'data' is a list of dimension args.num_domains
    # the final total number of samples = args.size * args.num_domains

    return data


def main(args):
    print("Args:")

    for k, v in sorted(vars(args).items()):
        print("\t{}: {}".format(k, v))

    set_random_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    env_config = EnvConfig() 
    # args: command line arguments
    # env_config: configuration for intializing OFDM environment
    for k, v in vars(args).items():
        if hasattr(env_config, k):
            setattr(env_config, k, v)


    env = OfdmEnv(env_config)
    print("Starting data generation.....This may take a while....")
    start_time = time.time()

    out_dir = os.path.join(
        args.output_dir,
        f"ps{env.n_pilot_symbols}_p{env.n_pilot_subcarriers}", # pilot sparsity, LS estimates = n_pilot_symbols * n_pilot_subcarriers
        args.scenario,
        f"snr{env.config.start_ds}to{env.config.end_ds}_speed{env.config.ue_speed}" # end_snr is not included!!
    )
    os.makedirs(out_dir, exist_ok=True)

    data = get_data(env, args)

    with h5py.File(f"{out_dir}/data.hdf5", "w") as hf:

        for k in data:
            data[k] = np.array(data[k])
            hf.create_dataset(
                f"{k}",
                data=data[k],
                shape=data[k].shape,
                compression="gzip",
                chunks=True,
            )

    print(f"Finished data generation in {time.time()-start_time:.2f}s")
    write_metadata(f"{out_dir}/metadata.yaml", env_config)
    print(f"Data save in {out_dir}")


if __name__ == "__main__":
    """Main function to generate multi-domain datasets"""
    
    parser = argparse.ArgumentParser("Data Generation")
    parser.add_argument("--output_dir", type=str, default="./data_TTTtrain")
    parser.add_argument("--size", type=int, default=400) # number of samples per domain
    parser.add_argument("--batch_size", type=int, default=100) # must be dividable by size
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num_domains", type=int, default=1, help="Number of domains/tasks generated" # Four domains: [0,5,10,15]
    )
    parser.add_argument("--start_ds", default = 10, type=int, help="Start domain ID")
    parser.add_argument("--end_ds", default= 11, type=int, help="Last domain ID") # not included

    parser.add_argument("--scenario", type=str, default="rt4")
    parser.add_argument(
        "-nf", "--fft_size", type=int, default=72, help="Number of subcarriers"
    )
    parser.add_argument("-gc", "--guard_carriers", nargs="+", help="Guard subcarriers")
    parser.add_argument("-ps", "--p_spacing", type=int, help="Pilot spacing", default=1) # TODO: p_spacing =1 means continuous subcarriers are used as pilots
    parser.add_argument("--pilot_pattern", type=str, default="block")

    parser.add_argument(
        "-ns", "--num_ofdm_symbols", type=int, default=14, help="number of symbols"
    )
    parser.add_argument(
        "-cf", "--carrier_frequency", type=float, default=2.1e9, help="frequency GHz"
    )
    parser.add_argument(
        "--subcarrier_spacing", type=float, default=30e3, help="Sub-carrier spacing kHz"
    )
    parser.add_argument("--ue_speed", type=int, default=6, help="User speed")
    parser.add_argument(
        "-nr",
        "--num_rx_antennas",
        type=int,
        default=1,
        help="Number of receive antennas",
    )
    parser.add_argument("-nu", "--n_ues", type=int, default=1, help="Number of UEs")
    parser.add_argument(
        "--pilot_ofdm_symbol_indices", nargs="+", type=int, default=[3, 10]
    )

    parser.add_argument("--los", action="store_true", help="Line of sight scenario")
    parser.add_argument("--path_loss", action="store_true", help="Add path loss or not")
    parser.add_argument("--shadowing", action="store_true", help="Shadowing")
    parser.add_argument("--dc_null", action="store_true")
    parser.add_argument("--encode", action="store_true")

    args = parser.parse_args()
    with tf.device('/CPU'):
        main(args)


