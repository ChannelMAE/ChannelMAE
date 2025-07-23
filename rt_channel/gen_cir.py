import sys
import os
import argparse

import numpy as np
os.environ['DRJIT_LIBLLVM_PATH'] = '/opt/homebrew/Cellar/llvm@16/16.0.6_1/lib/libLLVM.dylib'

import sionna as sn
from sionna.rt import load_scene, Transmitter, Camera, CoverageMap
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

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera, AntennaArray
import tensorflow as tf
import pickle

import os
import yaml

# configs for ray tracing
# NOTE：rt_config must align with the arguments in 'generated_datasets_from_sionna.py'
rt_config = {
    "subcarrier_spacing": 30e3,
    "num_ofdm_symbols": 14,  # Total number of ofdm symbols per slot
    
    "num_tx": 1,  # Number of users
    "num_rx": 1,  # Only one receiver considered
    "num_tx_ant": 1,  # Each user has 1 antenna
    "num_rx_ant": 1,  # The receiver is equipped with 1 antenna
    
    # one_batch_size for CIR generation
    "batch_size_cir": 1000,
    "target_num_cirs": 8000,  # Defines how many different CIRS are generated.
    
    "max_depth": 5,
    "rt_num_samples": 2e6,  # Number of ray samples per tx
    
    # Sample points within a 10-400m radius around the transmitter
    "min_dist": 10,  # in m
    "max_dist": 400,  # in m
    "min_gain_db": -130,  # in dB / ignore any position with less than -130 dB path gain
    "max_gain_db": 0,  # in dB / ignore any position with more than 0 dB path gain
    
    "max_num_paths": 75,
    "rx_velocities": [3.0, 3.0, 0], # x_v, y_v, z_v

    "frequency": 2.1e9, # 2.1GHz
    "bandwidth": 2.16e6, # 2.16MHz
    "cm_size": [800.,800.], # Total size of the coverage map
    "cm_cell_size": (1., 1.), # Cell size of the coverage map
}

# ---------------------------------------------------------------------------- #

custom_scenes = {
    "rt0": sionna.rt.scene.simple_street_canyon_with_cars, # simple street scene
    "rt1": sionna.rt.scene.etoile, # paris sparse buildings
    "rt2": sionna.rt.scene.munich, # dense buildings
    "rt3": "ray_tracing_scenes/florence/florence.xml", # very dense buildings
    #"rt4": "ray_tracing_scenes/san_francisco/san_francisco.xml", # add uneven terrain data
    # 'rt5': 'ray_tracing_scenes/edinburgh/edinburgh.xml'
}

# Add a transmitter on top of a building
tx_poses = {
    "rt0": [22.7, 5.6, 0.75], # in front of car 2
    "rt1": [8.5,21,27],
    "rt2": [-36.59, -65.02, 25.],
    "rt3": [-81.5, 99, 25.],
    # "rt4": [468, 106, 70],
    # "rt5": [100, 100, 100], # NOTE: set tx_pos
}
tx_orientation=[np.pi,0,0]

def main(scene_name):
    a = None
    tau = None

    # Load integrated scene
    scene = load_scene(custom_scenes[scene_name]) # Try also sionna.rt.scene.etoile
    scene.frequency = rt_config["frequency"] # in Hz; implicitly updates RadioMaterials
    scene.bandwidth = rt_config["bandwidth"]

    # Remove old tx from scene
    scene.remove("tx")
    # scene.synthetic_array = True # Emulate multiple antennas to reduce ray tracing complexity
    # Transmitter (=basestation) has an antenna pattern from 3GPP 38.901
    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=1, # We want to transmitter to be equiped with the 16 rx antennas
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5, 
                                 pattern="tr38901", # NOTE: directional beam pattern
                                 polarization="V")

    # Create transmitter
    tx = Transmitter(name="tx",
                     position=tx_poses[scene_name],
                     orientation=tx_orientation,
                     power_dbm=44)
    scene.add(tx)

    scene.remove("rx")
    for i in range(rt_config["batch_size_cir"]):
        scene.remove(f"rx-{i}")
    # Create batch_size receivers
    for i in range(rt_config["batch_size_cir"]):
        rx = Receiver(name=f"rx-{i}",
                      position=[0.,0.,0.] # NOTE:Random position sampled from coverage map
                      )
        scene.add(rx)
        
    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="dipole", # a more practical pattern
                                 polarization="V")
    
    # set the cm_center as the tx_pos but with a height of 1.5m
    cm_center_pos = tx_poses[scene_name]
    cm_center_pos[2] = 1.5 # the receiver plane is 1.5m above the ground

    # compute coverage map for each cell on the map
    cm = scene.coverage_map(max_depth=rt_config["max_depth"],
                            diffraction=True,
                            cm_center=cm_center_pos,   # Center of the coverage map
                            cm_orientation=[0, 0, 0], # parallel to the xy plane
                            cm_size=rt_config["cm_size"],    # Total size of the coverage map
                            cm_cell_size=rt_config["cm_cell_size"],
                            combining_vec=None,
                            precoding_vec=None,
                            num_samples=int(10e6))

    # Each simulation returns batch_size_cir results
    num_runs = int(np.ceil(rt_config["target_num_cirs"]/rt_config["batch_size_cir"]))
    for idx in range(num_runs):
        print(f"Progress: {idx+1}/{num_runs}", end="\r")

        # Sample random user positions
        ue_pos, _ = cm.sample_positions(
                            num_pos=rt_config["batch_size_cir"],
                            metric="path_gain",
                            min_val_db=rt_config["min_gain_db"],
                            max_val_db=rt_config["max_gain_db"],
                            min_dist=rt_config["min_dist"],
                            max_dist=rt_config["max_dist"])
        ue_pos = tf.squeeze(ue_pos)

        # Update all receiver positions
        for idx in range(rt_config["batch_size_cir"]):
            scene.receivers[f"rx-{idx}"].position = ue_pos[idx]

        # Simulate CIR
        paths = scene.compute_paths(
                        max_depth=rt_config["max_depth"],
                        diffraction=True,
                        num_samples=rt_config["rt_num_samples"]
                        ) # shared between all tx in a scene; 
        # this number should be increased with the number of txs in the scene to guarantee the accuracy.

        # Transform paths into channel impulse responses
        paths.reverse_direction = True # Convert to uplink direction

        # apply_doppler transforms the path data into CIR data
        paths.apply_doppler(sampling_frequency=rt_config["subcarrier_spacing"],
                            num_time_steps=rt_config["num_ofdm_symbols"],
                            tx_velocities=[0.,0.,0], # fix tx
                            rx_velocities=rt_config["rx_velocities"]) 

        # We fix here the maximum number of paths to 75 which ensures
        # that we can simply concatenate different channel impulse reponses
        a_, tau_ = paths.cir(num_paths=rt_config["max_num_paths"])

        # print(a_.shape)
        # print(tau_.shape)

        del paths # Free memory

        if a is None:
            a = a_.numpy()
            tau = tau_.numpy()
        else:
            # Concatenate along the num_tx dimension
            a = np.concatenate([a, a_], axis=3)
            tau = np.concatenate([tau, tau_], axis=2)

    del cm # Free memory

    # Exchange the num_tx and batchsize dimensions
    a = np.transpose(a, [3, 1, 2, 0, 4, 5, 6])
    tau = np.transpose(tau, [2, 1, 0, 3])

    # Remove CIRs that have no active link (i.e., a is all-zero)
    p_link = np.sum(np.abs(a)**2, axis=(1,2,3,4,5,6))
    a = a[p_link>0.,...]
    tau = tau[p_link>0.,...]

    print("Shape of a:", a.shape)
    print("Shape of tau: ", tau.shape)

    os.makedirs(f'ray_tracing_data/{scene_name}', exist_ok=True)
    with open(f'ray_tracing_data/{scene_name}/a.pkl', 'wb') as f:
        pickle.dump(a, f)
    with open(f'ray_tracing_data/{scene_name}/tau.pkl', 'wb') as f:
        pickle.dump(tau, f)

    # Save the configuration to a YAML file
    with open(f"ray_tracing_data/{scene_name}/rt_config.yaml", "w") as f:
        yaml.dump(rt_config, f, default_flow_style=False)

    print(f"The CIR data of the current scene have been saved in the folder {scene_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Channel Impulse Response (CIR) data using ray tracing")
    parser.add_argument("--scene_name", type=str, default="rt4", 
                        choices=list(custom_scenes.keys()),
                        help="Name of the scene to use for ray tracing")
    
    args = parser.parse_args()
    main(args.scene_name)