"""
Code script to train a model on main task: channel estimation
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from typing import List, Dict
import time
from dataclasses import dataclass, asdict
from collections import defaultdict
from copy import deepcopy
import sys
from pathlib import Path
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import cebed.datasets_with_ssl as cds 
import cebed.models as cm 

# from cebed.baselines import linear_ls_baseline, lmmse_baseline
from cebed.utils import unflatten_last_dim, write_metadata,read_metadata



@dataclass
class MainTrainConfig:
    """Train config"""
    # throughout the process
    seed: int = 42
   
    # dataset-class-related: datasets_with_ssl
    data_dir: str = "../data/ps2_p72/speed5"
    dataset_name: str = "FixMask"
    train_split: float = 0.9
    main_input_type: str = "low"
    aux_input_type: str = "low"
    aug_noise_std: float = 0.0
    sym_error_rate: float = 0.001
    task: str = "main" # "main" or "aux"

    # model-class-related: models
    experiment_name: str = ""
    model_name: str = "MaeFixMask"
    verbose: int = 1
    output_dir: str = "main_train_output"

    # trainer-class-related
    epochs: int = 100
    train_batch_size: int = 32
    eval_batch_size: int = 32
    lr: float = 0.001
    loss_fn: str = "mse"
    early_stopping: bool = False


class MainTrainer:
    """
    Trainer class
    """

    def __init__(self, config: MainTrainConfig):
        self.config = config
        os.makedirs(self.config.output_dir, exist_ok=True)

        data_config = read_metadata(os.path.join(self.config.data_dir, "metadata.yaml"))  # saved_config must be a dict    
        self.start_snr = data_config.start_ds
        self.end_snr = data_config.end_ds

        self.run_name = f"bs{self.config.train_batch_size}_lr{self.config.lr}_std{self.config.aug_noise_std}_snr{self.start_snr}to{self.end_snr}_{self.config.model_name}_Attn*3"
        self.log_dir = os.path.join(
            self.config.output_dir,
            self.run_name
        )
        
        os.makedirs(self.log_dir, exist_ok=True)

        self.train_loader = None
        self.eval_loader = None
        self.model = None
        self.dataset = None
        self.optimizer = tf.keras.optimizers.legacy.Adam(self.config.lr)
        self.loss_fn = self.config.loss_fn

    def setup(self):
        """Setup the trainer"""
        # Create datasets
        dataset_class = cds.get_dataset_class(self.config.dataset_name)
        if self.config.dataset_name == "Denoise": # dataset created from ds_denoise.py
            self.dataset = dataset_class(
                self.config.data_dir,
                train_split=self.config.train_split,
                main_input_type=self.config.main_input_type,
                aux_input_type = self.config.aux_input_type,
                aug_noise_std=self.config.aug_noise_std,
                seed=self.config.seed
            )
        elif self.config.dataset_name == "FixMask" or self.config.dataset_name == "RandomMask" : # dataset created from ds_mae_random_mask.py
            self.dataset = dataset_class(
                self.config.data_dir,
                train_split=self.config.train_split,
                main_input_type=self.config.main_input_type,
                aux_input_type = self.config.aux_input_type,
                sym_error_rate = self.config.sym_error_rate,
                seed=self.config.seed
            )
        else: 
            self.dataset = dataset_class(
                self.config.data_dir,
                train_split=self.config.train_split,
                main_input_type=self.config.main_input_type,
                aux_input_type = self.config.aux_input_type,
                seed=self.config.seed
            )

        self.train_loader, self.eval_loader = self.dataset.get_loaders(
            train_batch_size=self.config.train_batch_size,
            eval_batch_size=self.config.eval_batch_size,
            task = "main"
        )

        # Create model
        model_hparams = cm.get_model_hparams(
            self.config.model_name, self.config.experiment_name  # self.config.data_dir
        )

        model_class = cm.get_model_class(self.config.model_name)
        if "output_dim" not in model_hparams:
            raise ValueError("output_dim is not in model_hparams")
            # model_hparams["output_dim"] = self.dataset.output_shape

        self.model = model_class(model_hparams)
        
        # initial inputs to the encoder
        input_shape_main, _ = self.dataset.get_input_shape()

        # conditions for adding a fixed mask
        if self.config.model_name == "MaeFixMask":
            pilot_mask = self.dataset.env.get_mask()
            self.model.set_mask(pilot_mask)
            print("The mask for MAE is set up, used to pad the latent embedding")
        
        # model building for the main task does not include random masks
        self.model.build(
            tf.TensorShape([None, input_shape_main[0], input_shape_main[1], input_shape_main[2]]) 
        )


    def save(self):
        config = deepcopy(asdict(self.config))
        if (self.dataset.env is not None) and (hasattr(self.dataset.env, "config")):
            config.update(asdict(self.dataset.env.config))
        print(config)
        write_metadata(os.path.join(self.log_dir, "config.yaml"), config)

    def train(self):
        """Train the model"""
        callbacks = self.get_training_callbacks()

        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)

        if self.config.verbose > 0:
            print(self.model.summary(expand_nested=True))
            print(
                f"Start training for \
                        seed {self.config.seed} \
                        with a learning rate {self.config.lr}"
            )

        # before training, get the val_loss on the untrained model
        train_results = self.model.evaluate(self.train_loader,
                                    verbose = 2, # one-line verbose
                                    return_dict = True)
        val_results = self.model.evaluate(self.eval_loader,
                                    verbose = 2, # one-line verbose
                                    return_dict = True)
        
        # with open(os.path.join(self.log_dir, "csv_logger.csv"), "w") as f:

        before_train_filename = os.path.join(self.log_dir, "before_train.txt")
        with open(before_train_filename, "w") as f:
            f.write("On the train dataset: \n")
            f.write(str(train_results))
            f.write("\n")
            f.write("On the val dataset: \n")
            f.write(str(val_results))
        

        # Training & Validation
        start_time = time.time()
        self.model.fit(
            self.train_loader,
            verbose=self.config.verbose,
            epochs=self.config.epochs,
            validation_data=self.eval_loader,
            callbacks=callbacks,
        )

        print(f"Finished training in {time.time()-start_time:.2f}")


    def get_training_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Returns the training callbacks"""

        ckpt_folder = os.path.join(self.log_dir, "cp.ckpt")
        # Checkpoint callback
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_folder,
            save_weights_only=True,
            verbose=self.config.verbose,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        )

        # Tensorboard callback
        tensorboard_folder = os.path.join(self.config.output_dir, "tensorboard", self.run_name)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_folder
        )

        # history logging callback
        csv_logger_filename = os.path.join(self.log_dir, "csv_logger.csv")
        history_logger = tf.keras.callbacks.CSVLogger(
            csv_logger_filename, separator=",", append=True
        )

        # training callbacks
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001
        )

        callbacks = [
            tensorboard_callback,
            checkpoint_callback,
            lr_callback,
            history_logger,
        ]

        if self.config.early_stopping:
            es_callback = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, min_delta=0.00001
            )
            callbacks.append(es_callback)

        return callbacks

    def load_model(self):
        self.model.load_weights(f"{self.log_dir}/cp.ckpt").expect_partial()
