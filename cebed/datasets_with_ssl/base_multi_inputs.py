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
)
from cebed.envs import OfdmEnv, EnvConfig
from cebed.utils import read_metadata


class BaseMultiInputDataset:
    '''
    Base dataset class for multi-domain with SSL
    '''    
    def __init__(
        self,
        path: str,
        train_split: float = 0.9,
        main_input_type: str = "low", 
        aux_input_type: str = "low", 
        aug_noise_std: float = 0.9, # 0 if there is no noise augmentation for aux task
        seed: int = 0
    ):
        self.aug_noise_std = aug_noise_std
        self.num_domains = None
        self.main_path = path
        self.seed = seed
        self.train_split = train_split
        self.main_input_type = main_input_type
        self.aux_input_type = aux_input_type
        self.env = None

        # original data generated from wireless env
        self.x_samples = None
        self.y_samples = None
        self.h_samples = None

        # NN labels and inputs for two tasks
        self.labels_main = None
        self.inputs_main = None
        self.labels_aux = None
        self.inputs1_aux = None
        self.inputs2_aux = None

        # sliced datasets for two tasks
        self.train_x_main = None
        self.train_y_main = None
        self.val_x_main = None
        self.val_y_main = None
        self.test_x_main = None
        self.test_y_main = None

        self.train_x1_aux = None
        self.train_x2_aux = None
        self.train_y_aux = None
        self.val_x1_aux = None
        self.val_x2_aux = None
        self.val_y_aux = None
        self.test_x1_aux = None
        self.text_x2_aux = None
        self.test_y_aux = None

        self.size = None # equal to __len__
        self.setup()

    def setup(self):
        """Initial Setup"""
        # Create ofdm environment
        self.create_env()
        self.mask = self.env.get_mask() # get pilot mask
        self.load_data() # read dataset from disk
        self.split()
    
    def __len__(self) -> int:
            """Returns the size of the whole dataset"""

            if self.size is not None:
                return self.size
            return 0

    # ------------------------ Funtions of calling setup() ----------------------- #
    def create_env(self):
        """
        Create OFDM env using saved metadata

        We explicitly override create_env() just for clarity
        This method can be deleted
        """
        assert os.path.isfile(os.path.join(self.main_path, "metadata.yaml"))
        env_config = read_metadata(os.path.join(self.main_path, "metadata.yaml"))

        if isinstance(env_config, dict):
            # deepcopy the config to avoid changing the original config
            saved_config = deepcopy(env_config)
            env_config = EnvConfig()
            env_config.from_dict(saved_config)

        # Create OFDM environment for managing this offline dataset
        self.env = OfdmEnv(env_config)

    def load_data(self, *args, **kwargs) -> None:
        """Read dataset from disk"""
        raise NotImplementedError
    
    def split(self):
        """Split the dataset for each domain into train, val and test sets"""

        train_x_main, train_y_main = [], []
        val_x_main, val_y_main = [], []
        test_x_main, test_y_main = [], []

        train_x1_aux, train_x2_aux, train_y_aux = [], [],[]
        val_x1_aux, val_x2_aux, val_y_aux = [], [],[]
        test_x1_aux, test_x2_aux, test_y_aux = [], [],[]

        self.test_indices = []

        for ds in range(self.num_domains):
            # deal with each domain
            train_indices, test_indices = make_split(
                len(self.inputs_main[ds]), train_fraction=self.train_split
            )

            assert len(train_indices) > 1, "train split cannot be empty"

            # split train data into train and validation sets
            train_indices, val_indices = make_split(
                len(train_indices), self.train_split
            )

            # record test indices
            self.test_indices.append(test_indices)

            # main task
            train_x_main.append(self.inputs_main[ds][train_indices])
            train_y_main.append(self.labels_main[ds][train_indices])

            val_x_main.append(self.inputs_main[ds][val_indices])
            val_y_main.append(self.labels_main[ds][val_indices])

            test_x_main.append(self.inputs_main[ds][test_indices])
            test_y_main.append(self.labels_main[ds][test_indices])

            # aux task
            train_x1_aux.append(self.inputs1_aux[ds][train_indices])
            train_x2_aux.append(self.inputs2_aux[ds][train_indices])
            train_y_aux.append(self.labels_aux[ds][train_indices])

            val_x1_aux.append(self.inputs1_aux[ds][val_indices])
            val_x2_aux.append(self.inputs2_aux[ds][val_indices])
            val_y_aux.append(self.labels_aux[ds][val_indices])

            test_x1_aux.append(self.inputs1_aux[ds][test_indices])
            test_x2_aux.append(self.inputs2_aux[ds][test_indices])
            test_y_aux.append(self.labels_aux[ds][test_indices])
            
        # record train, val, test into the class itself
        self.train_x_main = np.array(train_x_main) # shape: [num_domains, num_samples, num_r, num_r_ants, n_t, n_t_ants, num_symbols, num_subcarriers]
        self.train_y_main = np.array(train_y_main) # shape: [num_domains, num_samples, num_r, num_r_ants, n_t, n_t_ants, num_symbols, num_subcarriers]
        self.val_x_main = np.array(val_x_main)
        self.val_y_main = np.array(val_y_main)
        self.test_x_main = np.array(test_x_main)
        self.test_y_main = np.array(test_y_main)

        self.train_x1_aux = np.array(train_x1_aux)
        self.train_x2_aux = np.array(train_x2_aux)
        self.train_y_aux = np.array(train_y_aux)
        self.val_x1_aux = np.array(val_x1_aux)
        self.val_x2_aux = np.array(val_x2_aux)
        self.val_y_aux = np.array(val_y_aux)
        self.test_x1_aux = np.array(test_x1_aux)
        self.test_x2_aux = np.array(test_x2_aux)
        self.test_y_aux = np.array(test_y_aux)

    # -------------------------------- Preprocess -------------------------------- #
    def preprocess_inputs_main(self, input, input_type, mask):
        """
        Per-sample preprocessing for function mapping in prepare_dataset
        """

        if input_type == "low" and mask is None:
            raise ValueError(f"Mask is needed for the requested input type {input_type}")

        if len(input.shape) > 6:
            raise ValueError("Input shape cannot have more than 6 dimensions")

        # Remove dimensions that are equal to 1
        x = tf.squeeze(input) # [n_symbols, n_sc]

        # Stack the nr and nt dimensions if they are different than 1
        if len(x.shape) == 4:
            # Stack the nr and nt dimensions
            nr, nt, ns, nf = x.shape
            x = tf.reshape(x, (nr * nt, ns, nf))

        if len(x.shape) > 2:
            x = tf.transpose(x, [1, 2, 0])  # [num_symbols, num_subcarriers, num_channels]
        
        # construct inputs based on the input_type
        if input_type == "raw":
            pre_x = x
        elif input_type == "low":
            pilot_indices = tf.where(mask) # pilot-location 1; non-pilot-location 0
            # gather the pilot symbols
            symbol_indices, _ = tf.unique(pilot_indices[:, -2])
            low_x = tf.gather(indices=symbol_indices, params=x, axis=0)
            # gather the pilot subcarriers
            subc_indices, _ = tf.unique(pilot_indices[:, -1])
            pre_x = tf.gather(indices=subc_indices, params=low_x, axis=1)
            # pre_x shape:  [num_unique_symbols, num_unique_subcarriers, n_channel]
        else:
            raise ValueError(f"Unknown input mode {input_type}")

        # Convert to real
        pre_x = complex_to_real(pre_x) # [n_symbols, n_scs, n_channels*2]
        return pre_x
    
    def preprocess_labels_main(self, label):
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

        label = complex_to_real(label) # [num_symbols, num_subcarriers, num_channels*2]
        return label
    
    def preprocess_main(self, input, label, train=True):
        if train == True:
            label = self.preprocess_labels_main(label)
        input = self.preprocess_inputs_main(input, self.main_input_type, self.mask)
        return input, label
   
    # ----------------------------- Get data loaders ----------------------------- #
    def get_train_loader(self, batch_size:int, task: str = "both") -> tf.data.Dataset:
        """
        Get the train loader
        """
        # create main-task dataset and preprocess
        if task=="main":
            train_datasets = []
            for i in range(self.num_domains):
                ds = tf.data.Dataset.from_tensor_slices((self.train_x_main[i], self.train_y_main[i]))
                train_datasets.append(ds)
            main_ds = tf.data.Dataset.sample_from_datasets(train_datasets)
            
            # FIXME: the next preprocess_main function renders the None shape in main_ds, but why???
            main_ds = main_ds.map(self.preprocess_main, num_parallel_calls=tf.data.AUTOTUNE) 
            train_ds = prepare_dataset(main_ds, batch_size=batch_size, shuffle=True, preprocess_fn=None)
            # (train_x_main, train_y_main)

        elif task=="aux":
            # create aux-task dataset and preprocess
            train_datasets = []
            for i in range(self.num_domains):
                ds = tf.data.Dataset.from_tensor_slices((self.train_x1_aux[i], self.train_x2_aux[i], self.train_y_aux[i]))
                train_datasets.append(ds)
            # sample from all domains
            aux_ds = tf.data.Dataset.sample_from_datasets(train_datasets)
            aux_ds = aux_ds.map(self.preprocess_aux, num_parallel_calls=tf.data.AUTOTUNE) # work on each element/sample
            train_ds = prepare_dataset(aux_ds, batch_size=batch_size, shuffle=True, preprocess_fn=None)

        elif task=="both":
            train_datasets = []
            for i in range(self.num_domains):
                ds = tf.data.Dataset.from_tensor_slices((self.train_x_main[i], self.train_y_main[i]))
                train_datasets.append(ds)
            main_ds = tf.data.Dataset.sample_from_datasets(train_datasets)
            main_ds = main_ds.map(self.preprocess_main, num_parallel_calls=tf.data.AUTOTUNE) 
            # create aux-task dataset and preprocess
            train_datasets = []
            for i in range(self.num_domains):
                ds = tf.data.Dataset.from_tensor_slices((self.train_x1_aux[i], self.train_x2_aux[i], self.train_y_aux[i]))
                train_datasets.append(ds)
            # sample from all domains
            aux_ds = tf.data.Dataset.sample_from_datasets(train_datasets)
            aux_ds = aux_ds.map(self.preprocess_aux, num_parallel_calls=tf.data.AUTOTUNE) # work on each element/sample
        
            # zip two datasets together and shuffle together
            train_ds = tf.data.Dataset.zip((main_ds, aux_ds)) 
            # ((train_x_main, train_y_main),(train_x1_aux, train_x2_aux, train_y_aux))
            train_ds = prepare_dataset(train_ds, batch_size=batch_size, shuffle=True, preprocess_fn=None)
        else:
            raise ValueError("please set 'task' to either 'main' or 'aux' or 'both'")
        
        return train_ds

    def get_eval_loader(self, batch_size:int =32, setname: str = 'val', task: str = "both") -> tf.data.Dataset:
        
        # do not shuffle the dataset for either 'eval' or 'test'

        if setname == 'val':
            inputs_main = self.val_x_main
            labels_main = self.val_y_main
            inputs1_aux = self.val_x1_aux
            inputs2_aux = self.val_x2_aux
            labels_aux = self.val_y_aux
        elif setname == 'test':
            inputs_main = self.test_x_main
            labels_main = self.test_y_main
            inputs1_aux = self.test_x1_aux
            inputs2_aux = self.test_x2_aux
            labels_aux = self.test_y_aux
        else:
            raise ValueError("please set 'setname' to either 'val' or 'test'")
        
        # --------------------------------- main task -------------------------------- #
        inputs_main =combine_datasets(inputs_main)
        labels_main =combine_datasets(labels_main)
        
        # preprocess_fn = partial(self.preprocess_main, train=setname == "val")
        preprocess_fn = partial(self.preprocess_main, train=True)
        main_eval_ds = tf.data.Dataset.from_tensor_slices((inputs_main, labels_main))
        main_eval_ds = main_eval_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

        # --------------------------------- aux task --------------------------------- #
        inputs1_aux = combine_datasets(inputs1_aux)
        inputs2_aux = combine_datasets(inputs2_aux)
        labels_aux = combine_datasets(labels_aux)

        # preprocess_fn = partial(self.preprocess_aux, train=setname == "val")
        preprocess_fn = partial(self.preprocess_aux, train=True)
        aux_eval_ds = tf.data.Dataset.from_tensor_slices((inputs1_aux, inputs2_aux, labels_aux))
        aux_eval_ds = aux_eval_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

        if task == "both":
            # zip two datasets together
            eval_ds = tf.data.Dataset.zip((main_eval_ds, aux_eval_ds))
            # prepare datasets and DO NOT shuffle
            eval_ds = prepare_dataset(eval_ds, batch_size=batch_size, shuffle=False, preprocess_fn=None)
        elif task == "main":
            eval_ds = prepare_dataset(main_eval_ds, batch_size=batch_size, shuffle=False, preprocess_fn=None)
        elif task == "aux":
            eval_ds = prepare_dataset(aux_eval_ds, batch_size=batch_size, shuffle=False, preprocess_fn=None)
        else:
            raise ValueError("please set 'task' to either 'main' or 'aux' or 'both'")
        return eval_ds

    def get_loaders(
        self, train_batch_size: int, eval_batch_size: int, task: str = "both"
    ) -> Tuple[tf.data.Dataset]:
        """Get the dataloaders for train and validation"""
        train_loader = self.get_train_loader(train_batch_size, task)
        eval_loader = self.get_eval_loader(eval_batch_size, setname="val",task = task)
        return train_loader, eval_loader

    # ----------------------------- Other properties ----------------------------- #
    def get_input_shape(self):
        '''return main_input_shape, aux_input_shape'''

        if self.main_input_type == "raw": # full ofdm symbols with zeros at pilot locations
            main_input_shape = (
                self.env.config.num_ofdm_symbols,
                self.env.rg.num_effective_subcarriers,
                self.env.config.num_rx_antennas * self.env.config.n_ues * 2,
                # only one n_r (in a single-sector topology), n_t_ants (SIMO system)
            )
            
        elif self.main_input_type == "low": # only symbols at pilot locations
            main_input_shape = (
                self.env.n_pilot_symbols,
                self.env.n_pilot_subcarriers,
                self.env.config.num_rx_antennas * self.env.config.n_ues * 2,
            )

        else: 
            raise ValueError(f"Unknown input type {self.main_input_type}")
        
        if self.aux_input_type == "raw":
            aux_input_shape = (
                    self.env.config.num_ofdm_symbols,
                    self.env.rg.num_effective_subcarriers,
                    self.env.config.num_rx_antennas * 2,
                    # only one n_r (in a single-sector topology), n_t_ants (SIMO system)
                )
        elif self.aux_input_type == "low":
            aux_input_shape = (
                self.env.n_pilot_symbols,
                self.env.n_pilot_subcarriers,
                self.env.config.num_rx_antennas * 2,
            )
        else:
            raise ValueError(f"Unknown input type {self.aux_input_type}")
           
        return main_input_shape, aux_input_shape

    @property
    def output_shape(self): # main and aux tasks 
        main_output_shape = (
            self.env.config.num_ofdm_symbols,
            self.env.rg.num_effective_subcarriers,
            self.env.config.num_rx_antennas * self.env.config.n_ues * 2,
        )
        aux_output_shape = (
            self.env.config.num_ofdm_symbols,
            self.env.rg.num_effective_subcarriers,
            self.env.config.num_rx_antennas * 2,
        )
        # for simo-single-user: main_output_shape = aux_output_shape
        return main_output_shape, aux_output_shape 


    @property
    def pilots(self):
        return self.env.rg.pilot_pattern.pilots

    @property
    def num_symbols(self):
        return self.env.config.num_ofdm_symbols

    @property
    def num_subcarries(self):
        return self.env.config.fft_size

    @property
    def num_pilot_symbols(self):
        return self.env.n_pilot_symbols

    @property
    def num_pilot_subcarriers(self):
        return self.env.n_pilot_subcarriers
    # ---------------------------------------------------------------------------- #