import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple, List
import sys
from pathlib import Path
import os
import datetime

root_path = str(Path(__file__).parent.parent.parent)
sys.path.append(root_path)

from cebed.models.transformers import Encoder
from cebed.models_with_ssl.base import BaseModel
from cebed.models_with_ssl.recon_net import Decoder
from cebed.utils import write_metadata


class ReconMAEX(BaseModel):
    """
    A two-branch model with one shared encoder
            -- main_decoder 
    - Shared Encoder 
            -- aux_decoder (uses pre_encoder_process from mae_stack_x.py)
    
    Only supports masking_type="discrete"
    """
    def __init__(self, hparams: Dict[str, Any], main_input_shape: List[int] = None):
        super().__init__(hparams)
        
        # Force discrete masking type
        self.masking_type = "discrete"
        
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

        # Single shared encoder for both branches
        self.encoder = Encoder(
            num_layers=self.num_en_layers,
            key_dim=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=self.dropout_rate,
        )

        # Main decoder 
        self.main_decoder = Decoder(
            self.output_dim, self.num_main_dc_layers, self.hidden_size, kernel_size=self.main_dc_kernel_size
        )

        # Aux decoder (same class as main decoder)
        self.aux_decoder = Decoder(
            self.output_dim, self.num_aux_dc_layers, self.hidden_size, kernel_size=self.aux_dc_kernel_size
        )

        self.mode = None
        self.pilot_mask = None
        
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
            self.aux_decoder.trainable = True
        elif mode == "ttt":
            self.encoder.trainable = True
            self.main_decoder.trainable = False
            self.aux_decoder.trainable = False
        else:
            raise ValueError(f"Unknown mode {mode}")

    def aux_expand_batch(self, low_embed: tf.Tensor, mask:tf.Tensor) -> tf.Tensor:
        '''
        Insert the encoded embeddings into the pilot locations and pad other locations with non-zero values for the entire batch
        '''
        # [num_pilots, 2]
        mask_indices = tf.where(mask == 1) # [batch*nps*npf, 3]
        # print(f"mask_indices shape: {mask_indices.shape}") 

        # [batch, nps*npf, c]
        batch_size = tf.shape(low_embed)[0] # must use dynamic shape!
        n_channel = tf.shape(low_embed)[2]
        low_embed = tf.reshape(low_embed, [-1,n_channel])
        # print(f"low_embed shape: {low_embed.shape}")
        high_embed = tf.scatter_nd(
            mask_indices, # [batch*nps*npf, 3]
            low_embed,  # [ batch*nps*npf, c]
            tf.cast([
                batch_size, # batch
                self.output_dim[0], # n_symbol
                self.output_dim[1], # n_subcarrier
                n_channel
            ], dtype=tf.int64),
        )
        return high_embed

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
        """Tokenize input for discrete masking type only - for main branch only"""
        # [batch, nps, nfs, 2] -> [batch, 2, nps*nfs]
        
        # Get dynamic dimensions
        batch_size = tf.shape(input_tensor)[0]
        channels = tf.shape(input_tensor)[-1]

        # Main branch: [batch, nps, nfs, 2] -> [batch, 2, nps*nfs] 
        input_tensor = tf.reshape(input_tensor, [batch_size, -1, channels])
        input_tensor = tf.transpose(input_tensor, [0, 2, 1])
        
        return input_tensor

    def post_encoder_reshape(self, latent: tf.Tensor) -> tf.Tensor:
        """Reshape encoder output for discrete masking type only - extract first 2 channels for main branch"""
        # [batch, 4, nps*nfs] -> [batch, nps*nfs, 2]
        # Only take the first 2 channels for main branch since we padded with zeros
        latent = latent[:, :2, :]  # [batch, 2, nps*nfs]
        latent = tf.transpose(latent, [0, 2, 1])  # [batch, nps*nfs, 2]
        return latent

    def pre_encoder_process(self, stacked_tensor: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """
        Pre-process stacked tensor before encoder by extracting masked positions.
        
        Args:
            stacked_tensor: [batch, ns, nf, 4] - stacked input tensor
            mask: [batch, ns, nf] - binary mask (1 for unmasked positions)
        
        Returns:
            low_stacked_tensor: [batch, 4, num_masked_positions] - ready for encoder
        """
        batch_size = tf.shape(mask)[0]
        num_seq = tf.shape(stacked_tensor)[-1]
        
        # Expand mask to match stacked_tensor dimensions
        mask_expanded = tf.expand_dims(mask, axis=-1)  # [batch, ns, nf, 1]
        mask_expanded = tf.tile(mask_expanded, [1, 1, 1, 4])  # [batch, ns, nf, 4]
        new_stacked_tensor = stacked_tensor * mask_expanded # [batch, ns, nf, 4]

        # Turn [batch, ns, nf, 4] into [batch, nps, npf, 4] - handle different masks per sample
        batch_size = tf.shape(mask)[0]

        # Get unmasked positions for each sample
        unmask_indices = tf.where(mask)  # [total_unmasked, 3] (batch, symbol, subcarrier)

        # Extract values at unmasked positions
        gathered_values = tf.gather_nd(new_stacked_tensor, unmask_indices)  # [total_unmasked, 4]

        # Count unmasked positions per batch
        num_unmasked_per_batch = tf.reduce_sum(tf.cast(mask, tf.int32), axis=[1, 2])  # [batch]
        max_unmasked = tf.reduce_max(num_unmasked_per_batch)

        # Create batch-local position indices for reshaping
        batch_indices = unmask_indices[:, 0]  # [total_unmasked] - int64
        cumsum = tf.concat([[0], tf.cumsum(num_unmasked_per_batch)[:-1]], axis=0)  # Cumulative sum offsets
        cumsum = tf.cast(cumsum, tf.int64)  # Cast to int64
        position_in_batch = tf.range(tf.shape(gathered_values)[0], dtype=tf.int64) - tf.gather(cumsum, batch_indices)

        # Scatter into final tensor shape [batch, max_unmasked, 4]
        scatter_indices = tf.stack([batch_indices, position_in_batch], axis=1)
        low_stacked_tensor = tf.scatter_nd(
            scatter_indices, 
            gathered_values, 
            [batch_size, max_unmasked, 4]
        )

        # Get actual pilot dimensions from first sample's mask
        first_mask_indices = tf.where(mask[0])
        unique_symbols = tf.unique(first_mask_indices[:, 0])[0]
        unique_subcarriers = tf.unique(first_mask_indices[:, 1])[0]
        nps = tf.shape(unique_symbols)[0]
        npf = tf.shape(unique_subcarriers)[0]

        # Reshape to [batch, nps, npf, 4]
        low_stacked_tensor = tf.reshape(low_stacked_tensor, [batch_size, nps, npf, 4])

        # Final reshape for encoder: [batch_size, nps*npf, 4] -> [batch_size, 4, nps*npf]
        low_stacked_tensor = tf.reshape(low_stacked_tensor, [batch_size, -1, num_seq]) # [batch_size, nps*npf, 4]
        low_stacked_tensor = tf.transpose(low_stacked_tensor, [0, 2, 1]) # [batch_size, 4, nps*npf]
        return low_stacked_tensor

    def main_branch(self, main_input:tf.Tensor, is_training: bool = True) -> tf.Tensor:
        """Main branch - uses shared encoder"""
        latent = self.tokenize_input(main_input)
        latent = self.encoder(latent)
        latent = self.post_encoder_reshape(latent)
        expanded_latent = self.main_expand_batch(latent)  # Use shared expand_batch
        main_outputs = self.main_decoder(expanded_latent)
        return main_outputs
    
    def aux_branch(self, stacked_tensor: tf.Tensor, mask: tf.Tensor, is_training: bool = True) -> tf.Tensor:
        """Aux branch - uses pre_encoder_process from mae_stack_x.py and shared encoder"""
        # Extract the second and fourth channels as est_x_symbol
        est_x = tf.stack([stacked_tensor[..., 1], stacked_tensor[..., 3]], axis=-1)  # [batch, ns, nf, 2]
        
        # Use pre_encoder_process to get properly formatted input for encoder
        low_stacked_tensor = self.pre_encoder_process(stacked_tensor, mask)  # [batch, 4, nps*npf]
        
        # Pass through shared encoder (no need for tokenize_input since pre_encoder_process formats correctly)
        latent = self.encoder(low_stacked_tensor)  # [batch, 4, nps*npf]
        
        # Reshape for expand_batch: [batch, 4, nps*npf] -> [batch, nps*npf, 4]
        latent = tf.transpose(latent, [0, 2, 1])
        
        # Transform the latent embedding into a 2D image with padding value using shared expand_batch
        # [batch, ns, nf, 4]
        expanded_latent = self.aux_expand_batch(latent, mask)  # Use shared expand_batch with mask
        expanded_latent = tf.concat([expanded_latent, est_x], axis=-1)  # [batch, ns, nf, 6]
        
        # [batch, ns, nf, c]
        aux_outputs = self.aux_decoder(expanded_latent)  # Use regular Decoder class
        
        return aux_outputs

    ## The simplified version of call function
    def call(self, inputs:tf.Tensor, is_training: bool = True) -> tf.Tensor:
        main_input, (stacked_aux_input, mask) = inputs 
        
        # --------------------------------- Main Task -------------------------------- #
        # [batch, nps, nfs, c]
        main_outputs = self.main_branch(main_input)
        
        # --------------------------------- Aux Task --------------------------------- #
        # [batch, nps, nfs, c] (using stacked input and MAE architecture)
        aux_outputs = self.aux_branch(stacked_aux_input, mask)

        return main_outputs, aux_outputs

    ### May inherit train_step() and test_step() from BaseModel
    # ---------------------------------------------------------------------------- #
    # @tf.function
    def pretrain_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        """
        Pre-training step given a batch of data 
        Update both super-resolution network and denoising network based on combined loss
        """
        assert self.mode == "pretrain", "Mode should be 'pretrain'"
        (x_main, y_main), (x1_aux, x2_aux, y_aux) = data

        with tf.GradientTape() as tape:
            pred_main, pred_aux = self((x_main, (x1_aux,x2_aux))) 

            # Compute combined loss for both tasks
            loss = self.compiled_loss(y_main, pred_main) + self.compiled_loss(y_aux, pred_aux)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # add channel estimation error as a metric
        self.compiled_metrics.update_state(y_main, pred_main)
        return {m.name: m.result() for m in self.metrics}

    # @tf.function
    def pretrain_test_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        assert self.mode == "pretrain", "Mode should be 'pretrain'"
        
        (x_main, y_main), (x1_aux, x2_aux, y_aux) = data
        pred_main, pred_aux = self((x_main, (x1_aux,x2_aux))) 
        loss = self.compiled_loss(y_main, pred_main) + self.compiled_loss(y_aux, pred_aux)

        # add channel estimation error as a metric
        self.compiled_metrics.update_state(y_main, pred_main)
        return {m.name: m.result() for m in self.metrics}

    # @tf.function
    def test_time_train_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        """
        Test-time training given a batch of data
        Only update the denoising network based on SSL loss
        """
        assert self.mode == "ttt", "Mode should be 'ttt'"
        (x_main, y_main), (x1_aux, x2_aux, y_aux) = data
        with tf.GradientTape() as tape:
            pred_main , pred_aux = self((x_main,(x1_aux,x2_aux))) # pass input tuple
            loss = self.compiled_loss(y_aux, pred_aux) 

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # add channel mse as a metric (track main-task performance)
        self.compiled_metrics.update_state(y_main, pred_main) # channel mse
        
        return {m.name: m.result() for m in self.metrics}

    # @tf.function
    def test_time_test_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        assert self.mode == "ttt", "Mode should be 'ttt'"
        (x_main, y_main), (x1_aux, x2_aux, y_aux) = data
        pred_main, pred_aux = self((x_main,(x1_aux,x2_aux)))
        loss = self.compiled_loss(y_aux, pred_aux) 

        self.compiled_metrics.update_state(y_main, pred_main)
        return {m.name: m.result() for m in self.metrics}
    
    # ---------------------------------------------------------------------------- #
    def get_training_callbacks(self, log_dir: str, verbose: int = 1) -> List[tf.keras.callbacks.Callback]:
        """Returns the training callbacks with annealing strategy"""
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
                   weights_name = "RECON_MAEX",
                   early_stopping: bool = True,
                   verbose: int = 1):
        
        """Improved training with annealing and early stopping"""
        # Create timestamped output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(log_dir, f"ReconMAEX_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)

        # Get callbacks
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

        # Compile model
        self.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate),
            loss=tf.keras.losses.MeanSquaredError(name="loss")
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
            "timestamp": timestamp,
            "masking_type": self.masking_type
        }
        write_metadata(os.path.join(log_dir, "training_config.yaml"), config)

        return history, log_dir 