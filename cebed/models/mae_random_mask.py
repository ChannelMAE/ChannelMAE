import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple
import sys
from pathlib import Path
import time

root_path = str(Path(__file__).parent.parent.parent)
sys.path.append(root_path)

from cebed.models.transformers import Encoder, HA02, EncoderLayer
from cebed.models.common import ResidualBlock
from cebed.models.base import BaseModel

'''
This class is the model class for Aux Class
'''
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
            filters=ff_dim, kernel_size=5, padding="same"
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
            filters=key_dim * num_heads, kernel_size=5, padding="same"
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

class MaeRandomMask(BaseModel):
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

        # # for tokenization of the form [batch, nfs, nps*c], it should be
        # num_heads = 2
        # head_size = 2
        # ff_dim = 4

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

        # Adc_num_heads = self.output_dim[2]  # number of channels
        # Adc_head_size = self.output_dim[0]  # number of symbols
        # Adc_ff_dim = Adc_num_heads * Adc_head_size

        # self.decoder = AttnDecoder(
        #     num_layers=self.num_dc_layers,
        #     key_dim=Adc_head_size,
        #     num_heads=Adc_num_heads,
        #     ff_dim=Adc_ff_dim,
        #     dropout_rate=self.dropout_rate,
        #     kernel_size=self.kernel_size,
        # )


    def expand_batch(self, low_embed: tf.Tensor, mask:tf.Tensor) -> tf.Tensor:
        '''
        Insert the encoded embeddings into the pilot locations and pad other locations with non-zero values for the entire batch
        '''
        # [num_pilots, 2]
        mask_indices = tf.where(mask == 1) # [batch, 14, 72]

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
    


    def call(self, inputs:tf.Tensor, is_training: bool = True) -> tf.Tensor:
        """
        Forward pass
        :param inputs: A batch of inputs (low_dim_masked_image, random_mask)
        :return upscaled inputs
        """
        low_dim_input, mask = inputs 

        # [batch, nps, nfs, c]
        shape = low_dim_input.shape

        # ---------------------------------------------------------------------------- #
        #  # # try another way of "self-attention": [batch, nfs, nps*c]
        # low_dim_input = tf.transpose(low_dim_input, [0, 2, 1, 3])  # [batch, nfs, nps, c]
        # low_dim_input = tf.reshape(low_dim_input, [-1, shape[2], shape[1] * shape[3]])

        # # pass through encoder keeping shape [batch, nfs, nps*c]
        # latent = self.encoder(low_dim_input)

        # # reshape back to [batch, nps * nfs, c] for expand_batch
        # latent = tf.reshape(latent, [-1, shape[2], shape[1], shape[3]])  # [batch, nfs, nps, c]
        # latent = tf.transpose(latent, [0, 2, 1, 3])  # [batch, nps, nfs, c]
        # latent = tf.reshape(latent, [-1, shape[1] * shape[2], shape[3]])
        # ---------------------------------------------------------------------------- #
        # try another way of "self-attention": [batch, nps, nfs*c]
        low_dim_input = tf.keras.layers.Reshape((shape[1],-1))(low_dim_input)

        # [batch, nps, nfs*c]
        latent = self.encoder(low_dim_input) 

        # [batch, nps * nfs, c]
        latent = tf.keras.layers.Reshape((shape[1], shape[2], -1))(latent)
        latent = tf.keras.layers.Reshape((-1, shape[-1]))(latent)
        
        # ---------------------------------------------------------------------------- #
        # # try another way of "self-attention": [batch, c, nps*npf]
        # # [batch, nps*npf, c], each real-value low_dim_input sequence over one frame is seen as a token sequence
        # low_dim_input = tf.keras.layers.Reshape((-1, shape[-1]))(low_dim_input)

        # #  [batch, c, nps*npf]
        # low_dim_input = tf.keras.layers.Permute((2, 1))(low_dim_input)

        # # [batch, c, nps*npf] ==> seq_len = c, feature_dim = nps*npf
        # latent = self.encoder(low_dim_input) # the latent length is the same as the input length
        
        # # [batch, nps*npf, c]
        # latent = tf.keras.layers.Permute((2, 1))(latent)
        # ---------------------------------------------------------------------------- #
        # transform the latent embedding into a 2D image with padding value
        # [batch, ns, nf, c]
        expanded_latent = self.expand_batch(latent, mask)

        # [batch, ns, nf, c]
        outputs = self.decoder(expanded_latent)
        
        return outputs
    

    def train_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        """
        The logic of one training step given a minibatch of data
        """

        x1_batch, x2_batch, y_batch = data
        x1_batch = tf.reshape(x1_batch, [-1, self.input_dim[0], self.input_dim[1], self.input_dim[2]])

        with tf.GradientTape() as tape:
            preds = self((x1_batch, x2_batch)) # x_batch can be a tuple of batched input1 and input2.
            loss = self.compiled_loss(y_batch, preds)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # self.compiled_metrics.update_state(y_batch, preds)

        return {m.name: m.result() for m in self.metrics}
    

    def test_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        # the data_loader here must be a single-task data loader (task = "main" or "aux")
        x1_batch, x2_batch, y_batch = data
        x1_batch = tf.reshape(x1_batch, [-1, self.input_dim[0], self.input_dim[1], self.input_dim[2]])

        preds = self((x1_batch, x2_batch)) # x_batch can be a tuple of batched input1 and input2.
        loss = self.compiled_loss(y_batch, preds)
        self.compiled_metrics.update_state(y_batch, preds)
        return {m.name: m.result() for m in self.metrics}


if __name__ == "__main__":

    from cebed.datasets_with_ssl.ds_mae_random_mask import MAEDatasetRandomMask
    import cebed.models as cm
    MyDataset = MAEDatasetRandomMask(path="./data/ps2_p72/rt1/snr10to20_speed5", 
                                    train_split=0.9, 
                                    main_input_type="low",
                                    aux_input_type = "low",
                                    sym_error_rate=0,
                                    seed=0)
    # already set up the dataset in the above line

    train_loader, eval_loader = MyDataset.get_loaders(
        train_batch_size=64,
        eval_batch_size=64,
        task="aux"
    )


    # prepare model
    experiment_name = "siso_1_umi_block_1_ps2_p72"
    model_name = "MaeRandomMask"
    model_hparams = cm.get_model_hparams(model_name, experiment_name)

    # initialize model
    MyModel = MaeRandomMask(model_hparams)

    ##  build model for aux task
    low_dim_input = tf.zeros([1, 2, 72, 2])
    example_mask = MyDataset.env.get_mask()
    example_mask = tf.squeeze(example_mask)
    example_mask = tf.expand_dims(example_mask, axis=0) # [batch, 14, 72]
    MyModel((low_dim_input, example_mask))
    print(MyModel.summary())


#   # train model
    MyModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
                    loss=tf.keras.losses.MeanSquaredError(name="loss"))
    
#     # for batch_data in train_loader:
#     #     # train_x1_aux, train_x2_aux, train_y_aux = batch_data
#     #     MyModel.train_step(batch_data)
#     #     print(MyModel.metrics)
#     #     break

    MyModel.fit(train_loader, validation_data=eval_loader, epochs=10, callbacks=None)


    ##  prepare model
    # model_name = "HA02"
    # model_hparams = cm.get_model_hparams(model_name, experiment_name)

    # # initialize model
    # MyModel = HA02(model_hparams)

    # # build model
    # low_dim_input = tf.zeros([1, 2, 72, 2])
    # MyModel(low_dim_input)
    # print(MyModel.summary())


    # for batch_data in train_loader:
    #     # train_x1_aux, train_x2_aux, train_y_aux = batch_data
    #     MyModel.train_step(batch_data)
    #     print(MyModel.metrics)
    #     break

    # for batch_data in eval_loader:
    #     train_x1_aux, train_x2_aux, train_y_aux = batch_data
    #     MyModel.test_step(batch_data)
    #     print(MyModel.metrics)
    #     break

    # print(MyModel.metrics)
    # MyModel.fit(train_loader, validation_data=eval_loader, epochs = 10, callbacks=None)
    # print(MyModel.metrics)

