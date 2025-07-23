import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
from typing import Dict, Any, Tuple
import sys
from pathlib import Path

root_path = str(Path(__file__).parent.parent.parent)
sys.path.append(root_path)

from cebed.models.transformers import Encoder, EncoderLayer, HA02
from cebed.models.common import ResidualBlock
from cebed.models.base import BaseModel
import time

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
    
class AttnDecoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers: int = 1,
        key_dim: int = 16,
        num_heads: int = 2,
        ff_dim: int = 16,
        kernel_size: int = 5,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        # Add an initial projection layer to match dimensions
        self.input_projection = tf.keras.layers.Conv1D(
            filters=ff_dim, kernel_size=7, padding="same"
        )
        
        # Modified encoder layers to use consistent dimensions
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
            filters=key_dim * num_heads, kernel_size=7, padding="same"
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

class MaeFixMask(BaseModel):
    """
    Masked Autoencoder for both channel estimation and received signal reconstruction
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

        # self.decoder = MAEDecoder(
        #     self.output_dim, self.num_dc_layers, self.hidden_size, kernel_size=self.kernel_size
        # )

        Adc_num_heads = self.output_dim[2]  # number of channels
        Adc_head_size = self.output_dim[0]  # number of symbols
        Adc_ff_dim = Adc_num_heads * Adc_head_size

        self.decoder = AttnDecoder(
            num_layers=self.num_dc_layers,
            key_dim=Adc_head_size,
            num_heads=Adc_num_heads,
            ff_dim=Adc_ff_dim,
            dropout_rate=self.dropout_rate,
            kernel_size=self.kernel_size,
        )


    def set_mask(self, pilot_mask: tf.Tensor):
        '''
        Set a fixed mask for all the inputs (i.e., the pilot mask)
        '''
        self.mask = pilot_mask
    
    def expand_batch(self, low_embed):
        '''
        Insert the encoded embeddings into the pilot locations and pad other locations with non-zero values for the entire batch
        '''
        # [num_pilots, 2]
        pilot_indices = tf.where(self.mask[0,0,:,:]==1) # pilot=1, non-pilot=0

        # [batch, nps*npf, c]
        batch_size = tf.shape(low_embed)[0] # use tf.shape to handle dynamic shape
        n_channel = tf.shape(low_embed)[2]

        # embed.shape: [nps* npf, batch, c] 
        low_embed = tf.transpose(low_embed, [1,0,2]) # [ns, nf, batch, c]
        high_embed = tf.scatter_nd(
            pilot_indices,
            low_embed,  # [nps* npf, batch, c]
            tf.cast([
                self.output_dim[0], # n_symbol
                self.output_dim[1], # n_subcarrier
                batch_size,
                n_channel
            ], dtype=tf.int64),
        )
        high_embed = tf.transpose(high_embed, [2,0,1,3])
        return high_embed
    

    def call(self, inputs:tf.Tensor, is_training: bool = True) -> tf.Tensor:
        """
        Forward pass
        :param inputs: Either a single tensor or a tuple of (input_tensor, mask_tensor)
        """
        # Handle both single tensor and tuple inputs
        if isinstance(inputs, tuple):
            inputs = inputs[0]  # Take only the first tensor
        
        # [batch, nps, nfs, c]
        shape = inputs.shape

        # [batch, nps*npf, c], each real-value h_ls sequence over one frame is seen as a token sequence
        inputs = tf.keras.layers.Reshape((-1, shape[-1]))(inputs)

        # [batch, c, nps*npf]
        inputs = tf.keras.layers.Permute((2, 1))(inputs)

        # [batch, c, nps*npf] ==> seq_len = c, feature_dim = nps*npf
        latent = self.encoder(inputs) # latent length is the same as the input length

        # [batch, nps*npf, c]
        latent = tf.keras.layers.Permute((2, 1))(latent)

        # transform the latent embedding into a 2D image with padding value
        # [batch, ns, nf, c]
        expanded_latent = self.expand_batch(latent)
        
        # [batch, ns, nf, c]
        outputs = self.decoder(expanded_latent)
        
        return outputs
    
    ### inherit the train_step and test_step from BaseModel


# if __name__ == "__main__":

#     from cebed.datasets_with_ssl.ds_mae_random_mask import MAEDataset
#     import cebed.models as cm
#     MyDataset = MAEDataset(path="./data/ps2_p72/umi/snr0to25_speed5", 
#                            train_split=0.9, 
#                            main_input_type="low",
#                            aux_input_type = "low",
#                            seed=0)
#     # already set up the dataset in the above line

#     train_loader, eval_loader = MyDataset.get_loaders(
#         train_batch_size=64,
#         eval_batch_size=64,
#         task="main" # the main task is channel estimation
#     )

#     # Train for MAE 
#     # prepare model
#     experiment_name = "siso_1_umi_block_1_ps2_p72"
#     model_name = "MAE"
#     model_hparams = cm.get_model_hparams(model_name, experiment_name)

#     # initialize model and set up the mask
#     MyModel = MAE(model_hparams)
#     MyModel.set_mask(MyDataset.env.get_mask())
#     print("The mask for MAE is set up, used to pad the latent embedding")

#     # build model
#     h_ls = tf.zeros([1, 2, 72, 2])
#     MyModel(h_ls)
#     print(MyModel.summary())

#     # train model
#     MyModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
#                     loss=tf.keras.losses.MeanSquaredError(name="loss"))
#     MyModel.fit(train_loader, validation_data=eval_loader, epochs = 10, callbacks=None)

    # ---------------------------------------------------------------------------- #
    ## prepare model of HA02
    # experiment_name = "siso_1_umi_block_1_ps2_p72"
    # model_name = "HA02"
    # model_hparams = cm.get_model_hparams(model_name, experiment_name)

    # # initialize model and set up the mask
    # MyModel = HA02(model_hparams)

    # # build model
    # h_ls = tf.zeros([1, 2, 72, 2])
    # MyModel(h_ls)
    # print(MyModel.summary())

    # # train model
    # MyModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
    #                 loss=tf.keras.losses.MeanSquaredError(name="loss"))
    # # MyModel.fit(train_loader, validation_data=eval_loader, epochs = 10, callbacks=None)

    # i=0
    # for batch_data in train_loader:
    #     MyModel.train_step(batch_data)
    #     # print(MyModel.metrics)
    #     i+=1
    #     if i==10:
    #         break

    # # train model
    # MyModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
    #                 loss=tf.keras.losses.MeanSquaredError(name="loss"))
    # MyModel.fit(train_loader, validation_data=eval_loader, epochs = 10, callbacks=None)

