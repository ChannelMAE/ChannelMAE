import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple
import time

import sys
from pathlib import Path
root_path = str(Path(__file__).parent.parent.parent)
sys.path.append(root_path)

from cebed.models.common import ResidualBlock, FeedForward
from cebed.models.base import BaseModel

class GlobalSelfAttention(tf.keras.layers.Layer):
    """
    Self attention module as decribed in https://arxiv.org/abs/1706.03762
    """

    def __init__(self, **kwargs: Dict[str, Any]):
        '''
        kwargs:
            num_heads: The number of attention heads
            key_dim: The size of each attention head for query and key
            value_dim: The size of each attention head for value
            dropout: The dropout rate
            output_shape: The output shape
        '''
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Self attention logic: All linear projections with no activation function
        """
        # x.shape: [B, seq_len, feature_dim]
        # attn_output.shape: [B, seq_len, X], X = output_shape
        # if no output_shape is provided when initialization, then X = query seq. feature dim 
        attn_output = self.mha(query=x, value=x, key=x) # query seq. = value seq. = key seq. ==> self-attention!
        x = self.add([x, attn_output])
        x = self.layernorm(x) 

        return x  # [B, seq_len, X]
    

class EncoderLayer(tf.keras.layers.Layer):
    """
    Transformer Encoder layer as decribed in https://arxiv.org/abs/1706.03762

    :param key_dim: The size of each attention head for query and key
    :param num_heads: The number of attention heads
    :param ff_dim: The hidden dimension of the FF module
    :param dropout_rate: The dropout rate

    """

    def __init__(
        self,
        key_dim: int = 16, # key_dim: The size of each attention head for query and key
        num_heads: int = 2,
        ff_dim: int = 16,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate
        ) # value_dim = key_dim (= query_dim) if not provided

        self.ffn = FeedForward(key_dim=ff_dim, ff_dim=ff_dim) # FIXME： two-layer MLP

    
    def call(self, x: tf.Tensor) -> tf.Tensor:

        # x.shape (input): [B, seq_len, feature_dim]
        # x.shape (output): [B, seq_len, feature_dim]
        x = self.self_attention(x) 
        x = self.ffn(x)

        # return x.shape: [B, seq_len, ff_dim]
        return x


class Encoder(tf.keras.layers.Layer):
    """
    Transformer Encoder block
    :param num_layers: Number of encoder layers
    :param key_dim: The size of each attention head for query and key
    :param num_heads: The number of attention heads
    :param ff_dim: The hidden dimension of the FF module
    :param dropout_rate: The dropout rate
    """

    def __init__(
        self,
        num_layers: int = 2,
        key_dim: int = 16,
        num_heads: int = 2,
        ff_dim: int = 16,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.enc_layers = [
            EncoderLayer(
                key_dim=key_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_layers)
        ]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        The layer logic
        """

        for layer in self.enc_layers:
            x = layer(x)

        return x


class HA02Decoder(tf.keras.layers.Layer):
    """
    The HA02 Decoder: Residual blocks + Upsampling block
    :param num_layers: Number of decoder layers
    :param hidden_size: The hidden size for the residual block
    :param kernel_size: The convolution kernel size
    """

    def __init__(self, num_layers: int = 2, hidden_size: int = 2, kernel_size: int = 2):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(hidden_size, kernel_size, padding="same")
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

        self.blocks = [
            ResidualBlock(
                hidden_size=self.hidden_size,
                kernel_size=self.kernel_size,
                layernorm=True,
            )
            for _ in range(num_layers)
        ]

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.conv1(inputs)

        for block in self.blocks: # the decoder is just cnn + 1 residual block
            x = block(x)
        return x


class UpsamlingBlock(tf.keras.layers.Layer):
    """
    Upsampling block for HA02 architecture (Totally linear transformation!!!)

    :param output_dim: A tuple of the output shape
    :param kernel_size: The convolution kernel size
    """

    def __init__(self, output_dim: Tuple[int], kernel_size: int = 2):
        super().__init__()
        self.seq = tf.keras.Sequential(
            [   # TODO: before permute: [batch, nps*npf, 2, hidden_size]??? this is the output of the decoder
                # "2" means the real and imaginary parts

                # [batch, 2, hidden_size, nps*npf] 
                tf.keras.layers.Permute((2, 3, 1)),
                # [batch, 2, hidden_size, ns*nf]
                tf.keras.layers.Dense(output_dim[0] * output_dim[1]), # dense layer with no activation
                # [batch, ns*nf, 2, hidden_size]
                tf.keras.layers.Permute((3, 1, 2)),
                # [batch, ns*nf, 2, 1], still in a sequence shape
                tf.keras.layers.Conv2D(1, kernel_size, padding="same"), # a cnn layer with no activation
            ]
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Layer logic
        """

        return self.seq(inputs)    

# ---------------------------------------------------------------------------- #
class HA02(BaseModel):
    """
    Implementation of the paper Attention Based Neural Networks for Wireless
    Channel Estimation
    https://arxiv.org/pdf/2204.13465.pdf
    """
    def __init__(self, hparams: Dict[str, Any]):
        super().__init__(hparams)

        self.encoder = None
        self.decoder = None
        self.upsamling = None
       

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build model's layers
        input_shape = [batch, nps, nfs, c]
        """
        # Number of heads in the number of pilot symbols
        num_heads = input_shape[1]

        # head size is set to be equal the number of pilot subcarriers
        head_size = input_shape[2] 

        # ff_dim = num_pilot_symbols*num_pilot_subcarriers
        # We follow the original transformer paper
        # and keep the dimensions of all the sub-layers equal
        
        #ff_dim = np.prod(input_shape[1:-1]) # if the number of pilots changes, the model arch. also changes!
        ff_dim = num_heads * head_size

        # self.project_layer = tf.keras.layers.Dense(ff_dim) # project inputs to ff_dim before encoder
        self.encoder = Encoder(
            num_layers=self.num_en_layers,
            key_dim=head_size, # 72
            num_heads=num_heads, # 2
            ff_dim=ff_dim, # 144
            dropout_rate=self.dropout_rate,
        )

        self.decoder = HA02Decoder(
            self.num_dc_layers, self.hidden_size, kernel_size=self.kernel_size
        )

        self.upsamling = UpsamlingBlock(self.output_dim, self.kernel_size)
        super().build(input_shape)


    def call(self, inputs: tf.Tensor, is_training: bool = True) -> tf.Tensor:
        """
        Forward pass

        :param inputs: A batch of inputs
        :param training: True if model is training, False otherwise

        :return upscaled inputs
        """
        # [batch, nps, nfs, c]
        shape = inputs.shape # in "low" input_type, the shape is [batch, nps, nfs, c]
        
        # [batch, nps*npf, c], each real-value h_ls sequence over one frame is seen as a token sequence
        inputs = tf.keras.layers.Reshape((-1, shape[-1]))(inputs)

        # NOTE in the original paper,
        # the embedding size = n pilot symbols*n pilots subcarriers
        # We are using TF MultiHeadAttention where the emb dimension is
        # the channel dim, that is why we are permuting the inputs
        # [batch, c, nps*npf]
        inputs = tf.keras.layers.Permute((2, 1))(inputs)

        # Project inputs to ff_dim before encoder
        # inputs = self.project_layer(inputs)

        # [batch, c, nps*npf] ==> seq_len = c, feature_dim = nps*npf
        latent = self.encoder(inputs) # latent length is the same as the input length

        # [batch, nps*npf, c]
        latent = tf.keras.layers.Permute((2, 1))(latent)

        shape = latent.shape
        # Reshape before sending to the decoder
        # [batch, nps*npf, c, 1]
        latent = tf.keras.layers.Reshape([shape[1], shape[2], 1])(latent)

        # [batch, nps*npf, c, hidden_size]
        decoded = self.decoder(latent)

        # start_time = time.time()
        # [batch, ns*nf, 2, 1], where finally the interpolation happens
        upscaled = self.upsamling(decoded)
        # print("upsampling time: ", time.time()-start_time)

        outputs = tf.keras.layers.Reshape(self.output_dim)(upscaled)
        
        return outputs

class HA03(HA02):
    def __init__(self, hparams: Dict[str, Any]):
        super().__init__(hparams)
        
    
    def call(self, inputs: tf.Tensor, is_training: bool = True) -> tf.Tensor:
        # [batch, nps, nfs, c]
        shape = inputs.shape # in "low" input_type, the shape is [batch, nps, nfs, c]
        
        # [batch, nps*npf, c], each real-value h_ls sequence over one frame is seen as a token sequence
        inputs = tf.keras.layers.Reshape((-1, shape[-1]))(inputs)

        # NOTE in the original paper,
        # the embedding size = n pilot symbols*n pilots subcarriers
        # We are using TF MultiHeadAttention where the emb dimension is
        # the channel dim, that is why we are permuting the inputs
        # [batch, c, nps*npf]
        inputs = tf.keras.layers.Permute((2, 1))(inputs)

        # [batch, c, nps*npf] ==> seq_len = c, feature_dim = nps*npf
        latent = self.encoder(inputs) # latent length is the same as the input length

        # [batch, nps*npf, c]
        latent = tf.keras.layers.Permute((2, 1))(latent)

        shape = latent.shape
        # Reshape before sending to the decoder
        # [batch, nps*npf, c, 1]
        latent = tf.keras.layers.Reshape([shape[1], shape[2], 1])(latent)

        # [batch, nps*npf, c, hidden_size]
        decoded = self.decoder(latent)

        # start_time = time.time()
        # [batch, nps*npf, 2, 1], for HA03, the output is the channel estimates at pilot symbols
        upscaled = self.upsamling(decoded)
        outputs = tf.keras.layers.Reshape(self.output_dim)(upscaled)

        return outputs

class MTRE(BaseModel):
    """
    Implementation of the paper Channel Estimation Method Based on Transformer
    in High Dynamic Environment
    https://ieeexplore.ieee.org/abstract/document/9299821
    """

    def build(self, input_shape):
        self.embedding = tf.keras.layers.Conv2D(
            input_shape[-1], kernel_size=1, padding="same"
        )

        num_heads = input_shape[-1]
        head_size = input_shape[-2]
        ff_dim = num_heads * head_size
        self.encoder = tf.keras.Sequential(name="Encoder")

        for _ in range(self.num_layers):
            self.encoder.add(
                EncoderLayer(
                    head_size, num_heads, ff_dim, dropout_rate=self.dropout_rate
                )
            )
        self.out = tf.keras.layers.Reshape(self.output_dim) # only decoder

        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # NOTE we dont use activation in the embedding layer since it gives better results

        embs = self.embedding(inputs)

        # NOTE in the original paper the attention is done on frequency only
        # [batch, ns, nf*c]
        embs = tf.keras.layers.Reshape((embs.shape[1], -1))(embs)

        outputs = self.encoder(embs)

        outputs = self.out(outputs)

        return outputs

# test code:
if __name__ == "__main__":

    from cebed.datasets_with_ssl.ds_classic import ClassicDataset
    import cebed.models as cm
    
    # Alternative FLOP calculation using model summary and manual calculation
    def estimate_flops_from_params(model, input_shape):
        """Estimate FLOPs based on model parameters and operations"""
        total_params = model.count_params()
        print(f"Total parameters: {total_params:,}")
        
        # More detailed FLOP estimation for transformer models
        batch_size, seq_len, hidden_dim = input_shape[0], input_shape[1] * input_shape[2], input_shape[3]
        
        # Get model configuration if available
        if hasattr(model, 'num_layers'):
            num_layers = model.num_layers
        else:
            num_layers = 1  # Default fallback
            
        if hasattr(model, 'num_heads'):
            num_heads = model.num_heads
        else:
            num_heads = hidden_dim  # Default assumption
            
        head_dim = hidden_dim // num_heads if num_heads > 0 else hidden_dim
        ff_dim = 4 * hidden_dim  # Standard transformer FFN expansion
        
        print(f"Model configuration:")
        print(f"  - Sequence length: {seq_len}")
        print(f"  - Hidden dimension: {hidden_dim}")
        print(f"  - Number of layers: {num_layers}")
        print(f"  - Number of heads: {num_heads}")
        print(f"  - Head dimension: {head_dim}")
        print(f"  - FFN dimension: {ff_dim}")
        
        # Calculate FLOPs per layer
        # 1. Self-attention: Q, K, V projections + attention computation + output projection
        qkv_flops = 3 * seq_len * hidden_dim * hidden_dim  # Q, K, V projections
        attention_flops = seq_len * seq_len * hidden_dim  # Attention computation
        output_proj_flops = seq_len * hidden_dim * hidden_dim  # Output projection
        attention_total = qkv_flops + attention_flops + output_proj_flops
        
        # 2. Feed-forward network: two linear layers
        ff_flops = seq_len * (hidden_dim * ff_dim + ff_dim * hidden_dim)
        
        # Total per layer
        layer_flops = attention_total + ff_flops
        
        # Total for all layers
        transformer_flops = num_layers * layer_flops
        
        # Add embedding and output layer FLOPs (rough estimate)
        embedding_flops = seq_len * hidden_dim * hidden_dim  # Rough estimate
        output_flops = seq_len * hidden_dim * 2  # Assuming output dimension is 2
        
        total_flops = transformer_flops + embedding_flops + output_flops
        
        print(f"\nFLOP breakdown (per sample):")
        print(f"  - Attention (per layer): {attention_total:,}")
        print(f"  - Feed-forward (per layer): {ff_flops:,}")
        print(f"  - Per layer total: {layer_flops:,}")
        print(f"  - All {num_layers} layers: {transformer_flops:,}")
        print(f"  - Embedding & output: {embedding_flops + output_flops:,}")
        print(f"  - Total estimated FLOPs: {total_flops:,}")
        
        # Multiply by 2 for forward + backward pass during training
        training_flops = 2 * total_flops
        print(f"  - Training FLOPs (forward + backward): {training_flops:,}")
        
        return total_flops

    MyDataset = ClassicDataset(
        path="./data/ps2_p72/rt1/snr10to20_speed5", 
        train_split=0.9, 
        main_input_type="low"
    )
    # already set up the dataset in the above line

    train_loader, eval_loader = MyDataset.get_loaders(
        train_batch_size=64,
        eval_batch_size=64,
        task="main" # MUST BE "main" for ClassicDataset
    )

    ## prepare model
    experiment_name = "siso_1_umi_block_1_ps2_p72"
    model_name = "HA02"
    model_hparams = cm.get_model_hparams(model_name, experiment_name)

    # initialize model
    MyModel = HA02(model_hparams)

    for x, y in train_loader.take(1):
        print("\nMain task:")
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")
        
        
        # Build the model
        MyModel(x)
        print(MyModel.summary())
        break  # Only process first batch

    # tf.profiler.experimental.start('logdir/')
    # for x,y in train_loader.take(5):
    #     MyModel(x)        # ← must execute ops here
    # tf.profiler.experimental.stop()
        
        # # Method 2: Rough estimation based on parameters
        # print("\nMethod 2: Parameter-based estimation")
        # estimate_flops_from_params(MyModel, x.shape)
        
    # # train model
    # MyModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
    #                 loss=tf.keras.losses.MeanSquaredError(name="loss"))
    # MyModel.fit(train_loader, validation_data=eval_loader, epochs = 5, callbacks=None)


