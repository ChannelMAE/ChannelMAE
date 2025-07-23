import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple, List
import sys
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import datetime

# Add the project root to Python path to fix imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent  # Go up two levels to get to TTT_for_wireless
sys.path.insert(0, str(project_root))

# Configure GPU/MPS for Mac
def configure_gpu():
    """Configure GPU acceleration for Mac (MPS) or fallback to CPU"""
    return False
    # # Check for MPS (Metal Performance Shaders) availability on Apple Silicon
    # if tf.config.list_physical_devices('GPU'):
    #     print("GPU devices found:")
    #     for device in tf.config.list_physical_devices('GPU'):
    #         print(f"  {device}")
    #         # Enable memory growth to avoid allocating all GPU memory at once
    #         tf.config.experimental.set_memory_growth(device, True)
    
    # # For Apple Silicon Macs, check MPS availability
    # try:
    #     # Try to create a simple operation on MPS
    #     with tf.device('/GPU:0'):
    #         test_tensor = tf.constant([1.0, 2.0, 3.0])
    #         result = tf.reduce_mean(test_tensor)
    #     print("✅ GPU acceleration enabled (MPS)")
    #     return True
    # except:
    #     print("⚠️  GPU not available, using CPU")
    #     return False

# Configure GPU at module import
GPU_AVAILABLE = configure_gpu()

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
        # self.conv1 = tf.keras.layers.Conv2D(hidden_size, kernel_size, activation="relu", padding="same")
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
        inputs = features = self.conv1(inputs) # linear cnn

        # the dimensions do not change passing through the residual blocks
        for block in self.blocks: 
            features = block(features)

        # features = tf.keras.layers.Add()([inputs, features])
        
        return self.conv2(features)
    

class AttnDecoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers: int = 1,
        key_dim: int = 16,
        num_heads: int = 2,
        ff_dim: int = 16,
        dropout_rate: float = 0.1,
        kernel_size: int = 5,
    ):
        super().__init__()
        
        # Add an initial projection layer to match dimensions
        self.input_projection = tf.keras.layers.Conv1D(
            filters=ff_dim, kernel_size=kernel_size, padding="same"
        )
        
        self.dec_layers = [
            EncoderLayer(
                key_dim=key_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_layers)
        ]
        
        # Add final projection layer to match output dimensions
        self.output_projection = tf.keras.layers.Conv1D(
            filters=key_dim * num_heads, kernel_size=kernel_size, padding="same"
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Process input shape [batch, ns, nfs, c] through attention layers
        and restore original dimensions
        """
        batch_size = tf.shape(inputs)[0]
        ns = tf.shape(inputs)[1]
        nfs = tf.shape(inputs)[2]
        c = tf.shape(inputs)[3]
        
        # [batch, ns, nfs, c] -> [batch, nfs, ns*c]
        x = tf.transpose(inputs, [0, 2, 1, 3])
        x = tf.reshape(x, [-1, nfs, ns * c])
        
        # Project input to match attention dimensions
        x = self.input_projection(x)
        
        # Pass through attention layers
        for layer in self.dec_layers:
            x = layer(x)
            
        # Project back to original dimensions
        x = self.output_projection(x)
        
        # Restore dimensions: [batch, nfs, ns*c] -> [batch, ns, nfs, c]
        x = tf.reshape(x, [batch_size, nfs, ns, c])
        x = tf.transpose(x, [0, 2, 1, 3])
        
        return x


class ReconMAE(BaseModel):
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
        
        # Modify encoder parameters based on masking type
        if self.masking_type == "discrete" or self.masking_type == "contiguous" or self.masking_type == "fixed":
            # #==========[batch, c, nps*nfs]==========# DEFAULT one for previous experiments
            num_heads = main_input_shape[0]  # nps
            head_size = main_input_shape[1]  # nfs
            #==========[batch, nps, nfs*c]==========#
            # num_heads = main_input_shape[2]  # c
            # head_size = main_input_shape[1]  # nfs
            #==========[batch, nfs, nps*c]==========#
            # num_heads = main_input_shape[2]  # c
            # head_size = main_input_shape[0]  # nps
            #==========[batch, nps*c, nfs]==========#
            # num_heads = 2
            # head_size = 72

        elif self.masking_type == "random_symbols":
            # [batch, nps, nfs*c], which is the only choice for random_symbols
            num_heads = main_input_shape[2]  # c
            head_size = main_input_shape[1]  # nfs

        elif self.masking_type == "fix_length":
            # [batch, c, nps*nfs]
            num_heads = main_input_shape[0]  # nps
            head_size = main_input_shape[1]  # nfs
            
        else:
            raise ValueError(f"Unknown masking type: {self.masking_type}")

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

        self.aux_decoder = Decoder(
            self.output_dim, self.num_aux_dc_layers, self.hidden_size, kernel_size=self.aux_dc_kernel_size
        )

        self.main_decoder = Decoder(
            self.output_dim, self.num_main_dc_layers, self.hidden_size, kernel_size=self.main_dc_kernel_size
        )

        # Adc_num_heads = self.output_dim[2]  # number of channels
        # Adc_head_size = self.output_dim[0]  # number of symbols
        # Adc_ff_dim = Adc_num_heads * Adc_head_size

        # self.aux_decoder = AttnDecoder(
        #     num_layers=1,
        #     key_dim=Adc_head_size,
        #     num_heads=Adc_num_heads,
        #     ff_dim=Adc_ff_dim,
        #     dropout_rate=self.dropout_rate,
        #     kernel_size=7,
        # )

        # self.main_decoder = AttnDecoder(
        #     num_layers=1,
        #     key_dim=Adc_head_size,
        #     num_heads=Adc_num_heads,
        #     ff_dim=Adc_ff_dim,
        #     dropout_rate=self.dropout_rate,
        #     kernel_size=7,
        # )

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
        """Tokenize input based on masking type"""
        # [batch, nps, nfs, c]
        
        if self.masking_type == "discrete" or self.masking_type == "contiguous" or self.masking_type == "fixed":
            # Get dynamic dimensions
            batch_size = tf.shape(input_tensor)[0]
            nps = tf.shape(input_tensor)[1]
            nfs = tf.shape(input_tensor)[2]
            channels = tf.shape(input_tensor)[3]

            # #==========[batch, c, nps*nfs]==========# DEFAULT one for previous experiments
            input_tensor = tf.reshape(input_tensor, [batch_size, -1, channels])
            input_tensor = tf.transpose(input_tensor, [0, 2, 1])

            #==========[batch, nps, nfs*c]==========#
            # input_tensor = tf.reshape(input_tensor, [batch_size, nps, nfs * channels])

            # #==========[batch, nfs, nps*c]==========#
            # input_tensor = tf.transpose(input_tensor, [0, 1, 3, 2])
            # input_tensor = tf.reshape(input_tensor, [batch_size, nfs, nps * channels])
            # input_tensor = tf.transpose(input_tensor, [0, 2, 1])

            # #==========[batch, nps*c, nfs]==========#
            # input_tensor = tf.transpose(input_tensor, [0, 1, 3, 2])
            # input_tensor = tf.reshape(input_tensor, [batch_size, nps * channels, nfs])

        elif self.masking_type == "random_symbols":
            # Get dynamic dimensions
            batch_size = tf.shape(input_tensor)[0]
            nps = tf.shape(input_tensor)[1]
            nfs = tf.shape(input_tensor)[2]
            channels = tf.shape(input_tensor)[3]

            input_tensor = tf.reshape(input_tensor, [batch_size, nps, nfs * channels])

        elif self.masking_type == "fix_length":
            # [batch, c, nps*nfs]
            batch_size = tf.shape(input_tensor)[0]
            channels = tf.shape(input_tensor)[-1]

            # Reshape to [batch, 2*72, channels]
            input_tensor = tf.reshape(input_tensor, [batch_size, -1, channels])

            # Transpose to [batch, channels, 2*72]
            input_tensor = tf.transpose(input_tensor, [0, 2, 1])

        else:
            raise ValueError(f"Unknown masking type: {self.masking_type}")
            
        return input_tensor

    def post_encoder_reshape(self, latent: tf.Tensor) -> tf.Tensor:
        """Reshape encoder output based on masking type"""
        if self.masking_type == "discrete" or self.masking_type == "contiguous" or self.masking_type == "fixed":
            shape = tf.shape(latent)
            nps = 2  # number of symbols
            nfs = 72  # number of subcarriers
            c = 2    # number of channels

            # #==========[batch, c, nps*nfs] -> [batch, nps*nfs, c]==========# DEFAULT one
            latent = tf.transpose(latent, [0, 2, 1])

            # #==========[batch, nps, nfs*c] -> [batch, nps*nfs, c]==========#
            # latent = tf.reshape(latent, [shape[0], -1, c])

            #==========[batch, nfs, nps*c] -> [batch, nps*nfs, c]==========#
            # latent = tf.reshape(latent, [shape[0], nfs, nps, c])
            # latent = tf.transpose(latent, [0, 2, 1, 3])
            # latent = tf.reshape(latent, [shape[0], -1, c])

            # #==========[batch, nps*c, nfs] -> [batch, nps*nfs, c]==========#
            # latent = tf.reshape(latent, [shape[0], nps, c, nfs])
            # latent = tf.reshape(latent, [shape[0], -1, c])

        elif self.masking_type == "random_symbols":
            # [batch, variable_nps, nfs*c] -> [batch, variable_nps * nfs, c]
            shape = tf.shape(latent)
            nfs = self.output_dim[1]
            c = shape[-1] // nfs 
            latent = tf.reshape(latent, [shape[0], shape[1], nfs, c])
            latent = tf.reshape(latent, [shape[0], -1, c])
            
        elif self.masking_type == "fix_length":
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
    
    def aux_branch(self, low_dim_aux_input:tf.Tensor, mask:tf.Tensor, is_training: bool = True) -> tf.Tensor:
        latent = self.tokenize_input(low_dim_aux_input)
        latent = self.encoder(latent) 
        # latent = self.bn_layer(latent, training=is_training)
        # latent = self.norm_layer(latent)
        latent = self.post_encoder_reshape(latent)
        expanded_latent = self.aux_expand_batch(latent, mask)
        aux_outputs = self.aux_decoder(expanded_latent)
        return aux_outputs
    

    ## The simplified version of call function
    def call(self, inputs:tf.Tensor, is_training: bool = True) -> tf.Tensor:
        main_input, (low_dim_aux_input, mask) = inputs 
        
        # --------------------------------- Main Task -------------------------------- #
        # [batch, nps, nfs, c]
        main_outputs = self.main_branch(main_input)
        
        # --------------------------------- Aux Task --------------------------------- #
        # [batch, nps, nfs, c]
        aux_outputs = self.aux_branch(low_dim_aux_input, mask)

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

            # TODO: for aux-task, only compute the loss of masked parts!!)
            loss = self.compiled_loss(y_main, pred_main) + self.compiled_loss(y_aux, pred_aux) # sum-loss of two tasks

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
        loss = self.compiled_loss(y_main, pred_main) + self.compiled_loss(y_aux, pred_aux) # self.compute_loss

        # add channel estimation error as a metric
        self.compiled_metrics.update_state(y_main, pred_main)
        return {m.name: m.result() for m in self.metrics}


    # ---------------------------------------------------------------------------- #
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
        
        return {m.name: m.result() for m in self.metrics} # only recorded per epoch in model.fit()


    #### The following function is discarded; and the convergence of aux-task alone is checked by functions in 'models'
    # NOTE: This TTT validation step is only used for checking whether the denoising SSL task works
    # NOTE: when TTT during deployment, we only need 'test_time_train_step'
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
                   weights_name = "RECON_MAE",
                   early_stopping: bool = True,
                   verbose: int = 1):
        
        """Improved training with annealing and early stopping"""
        # Create timestamped output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(log_dir, f"ReconMAE_{timestamp}")
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
        best_checkpoint_path = os.path.join(log_dir, "cp.ckpt")
        if os.path.exists(best_checkpoint_path + ".index"):
            self.load_weights(best_checkpoint_path)
            print(f"Loaded best weights from checkpoint before saving .h5 file")

        self.save_weights(os.path.join(log_dir, f"{weights_name}.h5"))
        

        return history, log_dir
    