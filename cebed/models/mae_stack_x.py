import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple
import sys
from pathlib import Path
import time

root_path = str(Path(__file__).parent.parent.parent)
sys.path.append(root_path)

from cebed.models.transformers import Encoder, HA02
from cebed.models.common import ResidualBlock
from cebed.models.base import BaseModel

class MAEDecoder(tf.keras.layers.Layer):
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
    

class MAE(BaseModel):
    """
    Masked Autoencoder for both channel estimation and receive signal reconstruction
    """
    def __init__(self, hparams: Dict[str, Any]):
        super().__init__(hparams)

        # Number of heads in the number of pilot symbols
        num_heads = self.input_dim[0]
        # head size is set to be equal the number of pilot subcarriers
        head_size = self.input_dim[1]

        # ff_dim = num_pilot_symbols*num_pilot_subcarriers
        # We follow the original transformer paper
        # and keep the dimensions of all the sub-layers equal
        ff_dim = np.prod(self.input_dim[0:-1])

        self.encoder = Encoder(
            num_layers=self.num_en_layers,
            key_dim=head_size, # 72
            num_heads=num_heads, # 2
            ff_dim=ff_dim, # 144
            dropout_rate=self.dropout_rate,
        )

        self.decoder = MAEDecoder(
            self.output_dim, self.num_dc_layers, self.hidden_size, kernel_size=self.kernel_size
        )

    # def set_mask(self, pilot_mask: tf.Tensor):
    #     self.mask = pilot_mask


    # def expand_one_sample(self, low_dim_sample):
    #     '''
    #     insert the encoded embeddings into the pilot locations
    #     and pad other locations with zeros
    #     '''
    #     pilot_indices = tf.where(self.mask[0,0,:,:]==1) # pilot=1, non-pilot=0

    #     ## Padding the non-pilot locations with zeros
    #     # expanded_sample = tf.scatter_nd(
    #     #     indices=tf.cast(pilot_indices, tf.int64),  # Indices for 2d locations to put
    #     #     updates=tf.reshape(low_dim_sample, [-1, self.output_dim[-1]]),  # Flatten pre_x for scatter update
    #     #     shape=self.output_dim  # The full original shape
    #     # )

    #     ## Padding the non-pilot locations with non-zeros
    #     expanded_sample = tf.tensor_scatter_nd_add(
    #         tf.ones(self.output_dim, dtype=tf.float32),
    #         indices=tf.cast(pilot_indices, tf.int64),  # Indices for 2d locations to put
    #         updates=tf.reshape(low_dim_sample, [-1, self.output_dim[-1]]),  # Flatten pre_x for scatter update
    #     )
    #     return expanded_sample


    def expand_batch(self, low_embed: tf.Tensor, mask:tf.Tensor) -> tf.Tensor:
        """
        Inserts the encoded embeddings into the pilot locations and pads other locations with non-zero values for the entire batch.

        Args:
            low_embed (tf.Tensor): [batch_size, nps*npf, c]
            mask (tf.Tensor): [batch_size, ns, nf].

        Returns:
            shape [batch_size, ns, nf, c]
        """
        
        # [num_pilots, 3]
        mask_indices = tf.where(mask == 1) # mask: [batch, 14, 72]

        # [batch, nps*npf, c]
        batch_size = tf.shape(low_embed)[0] # must use dynamic shape!
        n_channel = tf.shape(low_embed)[2]

        # embed.shape: [nps* npf, batch, c] 
        # low_embed = tf.transpose(low_embed, [1,0,2]) # [ns, nf, batch, c]

        # [batch*nps*npf, c]
        low_embed = tf.reshape(low_embed, [-1,n_channel])
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
        # high_embed = tf.transpose(high_embed, [2,0,1,3])
        return high_embed
    

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
        # low_shape = [2,72,4]
        # inputs1_aux shape: [batch, ns, nf, 4]
        # inputs2_aux shape: [batch, ns, nf] - the mask
        
        # Expand mask to match est_x_symbol dimensions
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

        # ---------------------------------------------------------------------------- #
        # plot new_stacked_tensor and low_stacked_tensor
        # self.plot_tensors_debug(mask[0,:,:], new_stacked_tensor[0,:,:,0], low_stacked_tensor[0,:,:,0], est_x[0]) 

        # ---------------------------------------------------------------------------- #
        low_stacked_tensor = tf.reshape(low_stacked_tensor, [batch_size, -1, num_seq]) # [batch_size, nps*npf, 4]
        low_stacked_tensor = tf.transpose(low_stacked_tensor, [0, 2, 1]) # [batch_size, 4, nps*npf]
        return low_stacked_tensor

    # def plot_tensors_debug(self, mask_sample, new_tensor_sample, low_tensor_sample, est_x_sample):
    #     """Plot mask, new_stacked_tensor, low_stacked_tensor, and est_x channels for debugging"""
    #     import matplotlib.pyplot as plt
    #     import numpy as np
        
    #     fig, axes = plt.subplots(5, 1, figsize=(8, 20))
        
    #     # Convert tensors to numpy for plotting
    #     mask_np = mask_sample.numpy() if hasattr(mask_sample, 'numpy') else mask_sample
    #     new_np = new_tensor_sample.numpy() if hasattr(new_tensor_sample, 'numpy') else new_tensor_sample
    #     low_np = low_tensor_sample.numpy() if hasattr(low_tensor_sample, 'numpy') else low_tensor_sample
    #     est_x_np = est_x_sample.numpy() if hasattr(est_x_sample, 'numpy') else est_x_sample
        
    #     # Plot mask
    #     im1 = axes[0].imshow(mask_np, cmap='viridis')
    #     axes[0].set_title('Mask[0,:,:]')
    #     axes[0].set_xlabel('Subcarriers')
    #     axes[0].set_ylabel('OFDM Symbols')
    #     plt.colorbar(im1, ax=axes[0])
        
    #     # Plot new_stacked_tensor
    #     im2 = axes[1].imshow(new_np, cmap='viridis')
    #     axes[1].set_title('new_stacked_tensor[0,:,:,0]')
    #     axes[1].set_xlabel('Subcarriers')
    #     axes[1].set_ylabel('OFDM Symbols')
    #     plt.colorbar(im2, ax=axes[1])
        
    #     # Plot low_stacked_tensor
    #     im3 = axes[2].imshow(low_np, cmap='viridis')
    #     axes[2].set_title('low_stacked_tensor[0,:,:,0]')
    #     axes[2].set_xlabel('Subcarriers')
    #     axes[2].set_ylabel('OFDM Symbols')
    #     plt.colorbar(im3, ax=axes[2])
        
    #     # Plot est_x channel 0
    #     im4 = axes[3].imshow(est_x_np[:,:,0], cmap='viridis')
    #     axes[3].set_title('est_x[0,:,:,0]')
    #     axes[3].set_xlabel('Subcarriers')
    #     axes[3].set_ylabel('OFDM Symbols')
    #     plt.colorbar(im4, ax=axes[3])
        
    #     # Plot est_x channel 1
    #     im5 = axes[4].imshow(est_x_np[:,:,1], cmap='viridis')
    #     axes[4].set_title('est_x[0,:,:,1]')
    #     axes[4].set_xlabel('Subcarriers')
    #     axes[4].set_ylabel('OFDM Symbols')
    #     plt.colorbar(im5, ax=axes[4])
        
    #     plt.tight_layout()
    #     plt.savefig('tensor_debug_plot.png', dpi=150, bbox_inches='tight')
    #     plt.show()

    def call(self, inputs: tf.Tensor, is_training: bool = True) -> tf.Tensor:
        stacked_tensor, mask = inputs
        # Extract the second and fourth channels as est_x_symbol
        est_x = tf.stack([stacked_tensor[..., 1], stacked_tensor[..., 3]], axis=-1)  # [batch, ns, nf, 2]

        
        low_stacked_tensor = self.pre_encoder_process(stacked_tensor, mask)
        
        # Pass through encoder
        latent = self.encoder(low_stacked_tensor)  # [batch, 4, nps*npf]


        # [batch, nps*npf, c]
        latent = tf.transpose(latent, [0, 2, 1])

        # transform the latent embedding into a 2D image with padding value
        # [batch, ns, nf, c]
        expanded_latent = self.expand_batch(latent, mask)
        expanded_latent = tf.concat([expanded_latent, est_x], axis=-1)

        # [batch, ns, nf, c]
        outputs = self.decoder(expanded_latent)
        
        return outputs
    

    def train_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        """
        The logic of one training step given a minibatch of data
        """

        x1_batch, x2_batch, y_batch = data
        # x1_batch shape: [batch, ns, nf, (n_r_ants+n_t_ants)*2]
        # x2_batch shape: [batch, ns, nf] - mask
        # y_batch shape: [batch, ns, nf, n_r*n_r_ants*2]

        with tf.GradientTape() as tape:
            preds = self((x1_batch, x2_batch))
            loss = self.compiled_loss(y_batch, preds)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # self.compiled_metrics.update_state(y_batch, preds)

        return {m.name: m.result() for m in self.metrics}
    

    def test_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        # the data_loader here must be a single-task data loader (task = "aux")
        x1_batch, x2_batch, y_batch = data
        # x1_batch shape: [batch, ns, nf, (n_r_ants+n_t_ants)*2]
        # x2_batch shape: [batch, ns, nf] - mask
        # y_batch shape: [batch, ns, nf, n_r*n_r_ants*2]

        preds = self((x1_batch, x2_batch))
        loss = self.compiled_loss(y_batch, preds)
        self.compiled_metrics.update_state(y_batch, preds)
        return {m.name: m.result() for m in self.metrics}


if __name__ == "__main__":

    from cebed.datasets_with_ssl.ds_mae_stack_x import MAEDataset
    import cebed.models as cm
    MyDataset = MAEDataset(
        path="./data/ps2_p72/rt1/snr10to20_speed5", 
        train_split=0.9, 
        main_input_type="low",
        aux_input_type="raw", # aux_input [14,72,4]
        sym_error_rate=0.001,
        seed=42
    )
    # already set up the dataset in the above line

    train_loader, eval_loader = MyDataset.get_loaders(
        train_batch_size=64,
        eval_batch_size=64,
        task="aux"
    )


    ## prepare model
    experiment_name = "siso_1_umi_block_1_ps2_p72"
    model_name = "MaeRandomMask"
    model_hparams = cm.get_model_hparams(model_name, experiment_name)

    # initialize model
    MyModel = MAE(model_hparams)
    # build model
    # Get input shapes from dataset
    # main_input_shape, aux_input_shape = MyDataset.get_input_shape()
    
    # # inputs1_aux shape: [batch, ns, nf, (n_r_ants+n_t_ants)*2] 
    # inputs1_aux = tf.zeros([8, 14, 72, 4])  # 4 = (1+1)*2 for SISO case
    # # inputs2_aux shape: [batch, ns, nf] - mask
    # inputs2_aux = tf.zeros([8, 14, 72])
    

    for inputs1_aux, inputs2_aux, y_aux in train_loader.take(1):
        print("\nAuxiliary task:")
        print(f"x1_aux shape: {inputs1_aux.shape}")
        print(f"x2_aux shape: {inputs2_aux.shape}")
        print(f"y_aux shape: {y_aux.shape}")
        output = MyModel((inputs1_aux, inputs2_aux))
        print(f"output shape: {output.shape}")
       

    # print(MyModel.summary())

    # train model
    MyModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
                    loss=tf.keras.losses.MeanSquaredError(name="loss"))
    MyModel.fit(train_loader, validation_data=eval_loader, epochs = 10, callbacks=None)
