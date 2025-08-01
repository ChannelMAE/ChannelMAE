"""
Common blocks for models

"""

from typing import Dict, Any
import tensorflow as tf


class ResidualBlock(tf.keras.layers.Layer):
    """
    Residual block as decribed in https://arxiv.org/abs/1707.02921

    :param hidden_size: Layers'  hidden size
    :param kernel_size: convolution kernel size
    """

    def __init__(self, hidden_size: int, kernel_size: int = 3, layernorm=False):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            hidden_size, kernel_size, activation="relu", padding="same" # with activation
        )
        self.conv2 = tf.keras.layers.Conv2D(hidden_size, kernel_size, padding="same")
        self.ln = None
        if layernorm:
            self.ln = tf.keras.layers.LayerNormalization()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        The layer logic
        """

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = tf.keras.layers.Add()([inputs, x])
        if self.ln is not None:
            # print("here")
            x = self.ln(x)

        return x


class BaseAttention(tf.keras.layers.Layer):
    """
    Base Attention Module

    :param **kwargs: Keyword arguments
    """

    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class FeedForward(tf.keras.layers.Layer):
    """
    Feed Forward module for transformers

    :param key_dim: The size of each attention head for query and key
    :param ff_dim: The hidden dimension of the FF module
    :param dropout_rate: The dropout rate
    :param activation: The activation fucntion
    """

    def __init__(
        self,
        key_dim: int = 16, # key and query dim
        ff_dim: int = 16,  # hidden layer dim = model dim
        dropout_rate: float = 0.1,
        activation: str = "gelu", # with activation
    ):
        super().__init__()
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(key_dim, activation=activation), 
                tf.keras.layers.Dense(ff_dim),
                tf.keras.layers.Dropout(dropout_rate),
            ]
        )
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        The logic of the feed forward module
        The first dense layer takes 'key_dim' as its output dimension, 
        and the second dense layer takes 'ff_dim' as its output dimension.
        the input size for the network is determined by the input tensor passed to the call() method.
        """
        x = self.add([x, self.seq(x)]) # residual connection
        x = self.layer_norm(x)

        # output_shape: [B, N, ff_dim]
        return x


class ConvBlock(tf.keras.layers.Layer):
    """
    A convolutional block

    :param hidden_size: The size of the hidden layers
    """

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(hidden_size, 3, padding="same")
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.ReLU()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        The logic of the convolutional block
        """
        x = self.conv(x)
        x = self.bn(x) # BN layer
        x = self.act(x)

        return x
