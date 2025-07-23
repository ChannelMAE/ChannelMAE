import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple, List
import sys
from pathlib import Path
import os
import datetime
root_path = str(Path(__file__).parent.parent.parent)
sys.path.append(root_path)

from cebed.models.common import ResidualBlock
from cebed.models.transformers import Encoder, EncoderLayer
from cebed.models_with_ssl.base import BaseModel
from cebed.utils import write_metadata

class Decoder(tf.keras.layers.Layer):
    '''
    Decoder of MAE: cnn + residual blocks + cnn
    :param output_dim: A tuple of the output shape
    :param kernel_size: The convolution kernel size
    :param num_blocks: Number of residual blocks
    :param hidden_size: The hidden size for the input CNN
    '''

    def __init__(self, output_dim, num_blocks: int = 3, hidden_size: int = 12, kernel_size: int = 2):

        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(hidden_size, kernel_size, padding="same")
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

        self.blocks = [
            ResidualBlock(
                hidden_size=self.hidden_size, #12
                kernel_size=self.kernel_size, #2
                layernorm=True,
            )
            for _ in range(num_blocks)
        ]
        self.conv2 = tf.keras.layers.Conv2D(output_dim[2], kernel_size, padding="same")


    def call(self, inputs: tf.Tensor) -> tf.Tensor:

        # [batch, ns, nf, ???]--> [batch, ns, nf, hidden_size] --> [batch, ns, nf, c]
        x = self.conv1(inputs) # linear cnn

        # the dimensions do not change passing through the residual blocks
        for block in self.blocks: 
            x = block(x)

        x = self.conv2(x) # linear cnn
        return x
    
class ReconMAE_MainOnly(BaseModel):
    """
    A two-branch model with the shared encoder
            -- main_decoder
    - Encoder 
            -- aux_decoder
    """
    def __init__(self, hparams: Dict[str, Any], main_input_shape: List[int] = None):
        super().__init__(hparams)
        
        # Set default input shape if none provided
        if main_input_shape is None:
            main_input_shape = [2, 72, 2]  # [nps, nfs, c]
        else:
            main_input_shape = main_input_shape
        
        num_heads = main_input_shape[0]  # nps
        head_size = main_input_shape[1]  # nfs

        # We follow the original transformer paper
        # and keep the dimensions of all the sub-layers equal
        ff_dim = num_heads * head_size

        self.encoder = Encoder(
            num_layers=self.num_en_layers,
            key_dim=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=self.dropout_rate,
        )

        self.main_decoder = Decoder(
            self.output_dim, self.num_main_dc_layers, self.hidden_size, kernel_size=self.main_dc_kernel_size
        )


        self.mode = None
        self.pilot_mask = None
        self.masking_type = hparams["masking_type"]
        
    def set_mask(self, pilot_mask: tf.Tensor):
        self.pilot_mask = pilot_mask
    
    def set_train_mode(self, mode: str):
        """
        Set the mode of the model: "pretrain" or "test-time-train"
        """
        self.mode = mode
        if mode == "pretrain":
            self.encoder.trainable = True
            self.main_decoder.trainable = True
        elif mode == "ttt":
            self.encoder.trainable = True
            self.main_decoder.trainable = False
        else:
            raise ValueError(f"Unknown mode {mode}")

    def main_expand_batch(self, low_embed):
        # [num_pilots, 2]
        pilot_indices = tf.where(self.pilot_mask[0,0,:,:]==1) # pilot=1, non-pilot=0

        # [batch, nps*npf, c]
        batch_size = tf.shape(low_embed)[0] # use tf.shape to handle dynamic shape
        n_channel = tf.shape(low_embed)[2]

        # embed.shape: [nps* npf, batch, c] 
        low_embed = tf.transpose(low_embed, [1,0,2]) # [nps* npf, batch, c]
        high_embed = tf.scatter_nd(
            pilot_indices,
            low_embed,  # [nps* npf, batch, c]
            tf.cast([
                self.output_dim[0], # n_symbol
                self.output_dim[1], # n_subcarrier
                batch_size,
                n_channel
            ], dtype=tf.int64),
        ) # [n_symbol, n_subcarrier, batch, c]
        high_embed = tf.transpose(high_embed, [2,0,1,3]) # [batch, n_symbol, n_subcarrier, c]
        return high_embed
    
    def tokenize_input(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """Tokenize input based on masking type"""
        # [batch, nps, nfs, c]
        
        if self.masking_type == "discrete" or self.masking_type == "contiguous" or self.masking_type == "fixed":
            # [batch, c, nps*nfs]
            shape = input_tensor.shape
            # print(f"input_tensor shape: {shape}")
            input_tensor = tf.keras.layers.Reshape((-1, shape[-1]))(input_tensor)
            input_tensor = tf.keras.layers.Permute((2, 1))(input_tensor)
            
        return input_tensor

    def post_encoder_reshape(self, latent: tf.Tensor) -> tf.Tensor:
        """Reshape encoder output based on masking type"""
        if self.masking_type == "discrete" or self.masking_type == "contiguous" or self.masking_type == "fixed":
            # [batch, c, nps*npf] -> [batch, nps*npf, c]
            latent = tf.keras.layers.Permute((2, 1))(latent)
            
        
        return latent

    def main_branch(self, main_input:tf.Tensor, is_training: bool = True) -> tf.Tensor:
        latent = self.tokenize_input(main_input)
        latent = self.encoder(latent)
        # latent = self.bn_layer(latent, training=is_training)
        # latent = self.norm_layer(latent)
        latent = self.post_encoder_reshape(latent)
        expanded_latent = self.main_expand_batch(latent)
        main_outputs = self.main_decoder(expanded_latent)
        return main_outputs

    ## The simplified version of call function
    
    def call(self, inputs:tf.Tensor, is_training: bool = True) -> tf.Tensor:
        main_input, (low_dim_aux_input, mask) = inputs 
        
        # --------------------------------- Main Task -------------------------------- #
        # [batch, nps, nfs, c]
        main_outputs = self.main_branch(main_input)

        return main_outputs


    ### May inherit train_step and test_step from BaseModel
    # ---------------------------------------------------------------------------- #
    @tf.function
    def pretrain_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        """
        Pre-training step given a batch of data 
        Update both super-resolution network and denoising network based on combined loss
        """
        assert self.mode == "pretrain", "Mode should be 'pretrain'"
        (x_main, y_main), (x1_aux, x2_aux, y_aux) = data

        with tf.GradientTape() as tape:
            # Only get pred_main since we're only using main branch
            pred_main = self((x_main, (x1_aux, x2_aux)))
            loss = self.compiled_loss(y_main, pred_main)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.compiled_metrics.update_state(y_main, pred_main)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def pretrain_test_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        assert self.mode == "pretrain", "Mode should be 'pretrain'"
        
        (x_main, y_main), (x1_aux, x2_aux, y_aux) = data
        # Only get pred_main since we're only using main branch
        pred_main = self((x_main, (x1_aux, x2_aux)))
        loss = self.compiled_loss(y_main, pred_main)

        self.compiled_metrics.update_state(y_main, pred_main)
        return {m.name: m.result() for m in self.metrics}

    
    def get_training_callbacks(self, log_dir: str, verbose: int = 1) -> List[tf.keras.callbacks.Callback]:
        """Returns the training callbacks with cosine annealing"""
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Checkpoint callback - save best model
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(log_dir, "cp.ckpt"),
            save_weights_only=True,
            verbose=verbose,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        )

        # Tensorboard logging
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(log_dir, "tensorboard")
        )

        # CSV History logging
        history_logger = tf.keras.callbacks.CSVLogger(
            os.path.join(log_dir, "training_history.csv"),
            separator=",",
            append=True
        )

        # Learning rate annealing
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )

        return [tensorboard_callback, checkpoint_callback, lr_callback, history_logger]

    def train_model(self, 
                   train_loader,
                   eval_loader,
                   epochs: int = 50,
                   learning_rate: float = 0.001,
                   log_dir: str = "model_output",
                   weights_name = "RECON_MAE",
                   early_stopping: bool = True,
                   verbose: int = 1):
        
        """Training with cosine annealing and early stopping"""
        # Create timestamped output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(log_dir, f"ReconMAE_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)

        # Get callbacks with cosine annealing
        callbacks = self.get_training_callbacks(log_dir, verbose)
        
        # Add early stopping if requested
        if early_stopping:
            es_callback = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                min_delta=0.00001,
                verbose=1
            )
            callbacks.append(es_callback)

        self.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate),
            loss=tf.keras.losses.MeanAbsoluteError(name="loss")
        )
        
        # Set training mode
        self.set_train_mode("pretrain")

        # Get initial validation performance
        initial_eval = self.evaluate(
            eval_loader,
            verbose=0,
            return_dict=True
        )
        print(f"Initial validation loss: {initial_eval['loss']:.4f}")

        # Train model with adjusted epochs
        history = self.fit(
            train_loader,
            validation_data=eval_loader, 
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )

        # Save final model and config
        self.save_weights(os.path.join(log_dir, f"{weights_name}.h5"))
        
        # Save training config
        config = {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "early_stopping": early_stopping,
            "timestamp": timestamp
        }
        write_metadata(os.path.join(log_dir, "training_config.yaml"), config)

        return history, log_dir
