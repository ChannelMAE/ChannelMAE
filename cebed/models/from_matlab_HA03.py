'''
Directly translated from Matlab codes
NEED TO BE DEBUGGED.

Question:
Why the Matlab structure is different from the CeBed structure?
'''

import tensorflow as tf
import numpy as np

class HA03Model(tf.keras.Model):
    def __init__(self, num_heads=6, encoder_num_layers=1, decoder_num_layers=3, name='HA03Model'):
        """
        HA03 Channelformer model with correct hyperparameters:
        - num_heads: 6 (multi-head attention heads)
        - encoder_num_layers: 1 (single encoder layer)
        - decoder_num_layers: 3 (three decoder layers)
        """
        super(HA03Model, self).__init__(name=name)
        
        self.num_heads = num_heads
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        
        # Will be initialized in build()
        self.encoder_blocks = []
        self.decoder_blocks = []
        self.regression_conv = None
        self.final_fc = None
        self.final_conv = None
        
    def build(self, input_shape):
        # super(HA03Model, self).build(input_shape)
        
        # Build encoder blocks (1 layer for HA03)
        for i in range(self.encoder_num_layers):
            encoder_block = EncoderBlock(
                num_heads=self.num_heads,
                name=f'encoder_layer_{i+1}'
            )
            self.encoder_blocks.append(encoder_block)
        
        # Build regression convolution (initial decoder convolution)
        # Using 12 filters as specified in parameters_hybrid.m
        self.regression_conv = tf.keras.layers.Conv2D(
            filters=12,  # Number_of_filters = 12 from MATLAB code
            kernel_size=5,  # filterSize = [5 5] from MATLAB code
            padding='same',
            name='regression_conv'
        )
        
        # Build decoder blocks (3 layers for HA03)
        for j in range(self.decoder_num_layers):
            decoder_block = DecoderBlock(name=f'decoder_layer_{j+1}')
            self.decoder_blocks.append(decoder_block)
        
        # Build final dense layer
        # Feature_size from MATLAB corresponds to input feature dimension
        feature_size = input_shape[1] * input_shape[2]  # spatial dimensions flattened
        self.final_fc = tf.keras.layers.Dense(
            units=feature_size,
            name='final_fc'
        )
        
        # Build final convolution (5x5 kernel, 1 output filter)
        self.final_conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=5,  # filterSize = [5 5] from MATLAB code
            padding='same',
            name='final_conv'
        )
    
    def call(self, inputs, training=None):
        x = inputs
        
        # Apply encoder layers (1 layer for HA03)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, training=training)
        
        # Apply regression convolution
        x = self.regression_conv(x)
        
        # Apply decoder layers (3 layers for HA03)
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, training=training)
        
        # Apply final FC layer
        # Reshape for dense layer: (batch, height, width, channels) -> (batch, height*width*channels)
        batch_size = tf.shape(x)[0]
        height, width, channels = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        flattened = tf.reshape(x, [batch_size, -1])
        x = self.final_fc(flattened)
        
        # Reshape back to spatial format
        x = tf.reshape(x, [batch_size, height, width, -1])
        
        # Apply final convolution
        x = self.final_conv(x)
        
        return x


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, name='encoder_block'):
        super(EncoderBlock, self).__init__(name=name)
        self.num_heads = num_heads
        
        # Will be initialized in build()
        self.attention_layer = None
        self.layer_norm1 = None
        self.feedforward = None
        self.layer_norm2 = None
    
    def build(self, input_shape):
        super(EncoderBlock, self).build(input_shape)
        
        # Multi-head attention
        self.attention_layer = MultiHeadAttention(
            num_heads=self.num_heads,
            name='attention'
        )
        
        # Layer normalization
        self.layer_norm1 = tf.keras.layers.LayerNormalization(
            axis=1, name='layer_norm1'  # axis=1 to match MATLAB normalization dimension
        )
        
        # Feedforward network (uses Conv2D in HA03)
        self.feedforward = FeedForwardNetwork(name='feedforward')
        
        # Second layer normalization
        self.layer_norm2 = tf.keras.layers.LayerNormalization(
            axis=1, name='layer_norm2'  # axis=1 to match MATLAB normalization dimension
        )
    
    def call(self, inputs, training=None):
        # Multi-head attention with residual connection
        # Note: MATLAB uses permute operations for dimension matching
        attn_output = self.attention_layer(inputs, training=training)
        x = attn_output + inputs  # Residual connection
        x = self.layer_norm1(x)
        
        # Feedforward with residual connection
        ff_output = self.feedforward(x, training=training)
        x = ff_output + x  # Residual connection
        x = self.layer_norm2(x)
        
        return x


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, name='decoder_block'):
        super(DecoderBlock, self).__init__(name=name)
        
        # Will be initialized in build()
        self.conv1 = None
        self.conv2 = None
        self.layer_norm = None
    
    def build(self, input_shape):
        super(DecoderBlock, self).build(input_shape)
        
        # First convolution (5x5 kernel as per MATLAB code)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=12,  # Number_of_filters = 12 from MATLAB
            kernel_size=5,  # filterSize = [5 5] from MATLAB
            padding='same',
            activation='relu',
            name='conv1'
        )
        
        # Second convolution (5x5 kernel as per MATLAB code)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=12,  # Number_of_filters = 12 from MATLAB
            kernel_size=5,  # filterSize = [5 5] from MATLAB
            padding='same',
            name='conv2'
        )
        
        # Layer normalization
        self.layer_norm = tf.keras.layers.LayerNormalization(
            axis=1, name='layer_norm'  # axis=1 to match MATLAB normalization dimension
        )
    
    def call(self, inputs, training=None):
        # First convolution with ReLU
        x = self.conv1(inputs)
        
        # Second convolution
        x = self.conv2(x)
        
        # Residual connection
        x = x + inputs
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, name='multi_head_attention'):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        
        # Will be initialized in build()
        self.qkv_dense = None
        self.output_dense = None
    
    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        
        feature_dim = input_shape[-1]
        
        # Dense layer to generate Q, K, V (3x feature_dim as per MATLAB code)
        self.qkv_dense = tf.keras.layers.Dense(
            units=feature_dim * 3,  # 3x for Q, K, V
            name='qkv_projection'
        )
        
        # Output projection
        self.output_dense = tf.keras.layers.Dense(
            units=feature_dim,
            name='output_projection'
        )
    
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1] * tf.shape(inputs)[2]  # Flatten spatial dims
        feature_dim = inputs.shape[-1]
        head_dim = feature_dim // self.num_heads
        
        # Flatten spatial dimensions for attention computation
        x = tf.reshape(inputs, [batch_size, seq_len, feature_dim])
        
        # Generate Q, K, V
        qkv = self.qkv_dense(x)  # (batch, seq_len, 3*feature_dim)
        
        # Split into Q, K, V
        q, k, v = tf.split(qkv, 3, axis=-1)
        
        # Reshape for multi-head attention
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, head_dim])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, head_dim])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, head_dim])
        
        # Transpose for attention computation
        q = tf.transpose(q, [0, 2, 1, 3])  # (batch, heads, seq_len, head_dim)
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        # Scaled dot-product attention
        scores = tf.matmul(q, k, transpose_b=True)  # (batch, heads, seq_len, seq_len)
        scores = scores / tf.sqrt(tf.cast(head_dim, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Apply attention to values
        attended = tf.matmul(attention_weights, v)  # (batch, heads, seq_len, head_dim)
        
        # Concatenate heads
        attended = tf.transpose(attended, [0, 2, 1, 3])  # (batch, seq_len, heads, head_dim)
        attended = tf.reshape(attended, [batch_size, seq_len, feature_dim])
        
        # Output projection
        output = self.output_dense(attended)
        
        # Reshape back to spatial format
        height, width = tf.shape(inputs)[1], tf.shape(inputs)[2]
        output = tf.reshape(output, [batch_size, height, width, feature_dim])
        
        return output


class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, name='feedforward'):
        super(FeedForwardNetwork, self).__init__(name=name)
        
        # Will be initialized in build()
        self.conv1 = None
        self.conv2 = None
    
    def build(self, input_shape):
        super(FeedForwardNetwork, self).build(input_shape)
        
        # First convolution (3x3, 5 filters as per MATLAB parameters_hybrid.m)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=5,  # Number_of_filters_Encoder = 5 from MATLAB
            kernel_size=3,  # filterSize = [3 3] from MATLAB
            padding='same',
            name='conv1'
        )
        
        # Second convolution (3x3, back to 1 filter)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=1,  # Back to original channel dimension
            kernel_size=3,  # filterSize = [3 3] from MATLAB
            padding='same',
            name='conv2'
        )
    
    def call(self, inputs, training=None):
        # First convolution with GELU
        x = self.conv1(inputs)
        x = self.gelu(x)
        
        # Second convolution
        x = self.conv2(x)
        
        return x
    
    @staticmethod
    def gelu(x):
        """GELU activation function matching MATLAB implementation"""
        return 0.5 * x * (1.0 + tf.tanh(tf.sqrt(2.0 / np.pi) * (x + 0.044715 * tf.pow(x, 3.0))))


# Example usage with correct hyperparameters:
if __name__ == "__main__":
    # Create model instance with HA03 hyperparameters
    model = HA03Model(
        num_heads=6,           # Correct: 6 attention heads
        encoder_num_layers=1,  # Correct: 1 encoder layer  
        decoder_num_layers=3   # Correct: 3 decoder layers
    )
    
    # Build model with example input shape (batch, height, width, channels)
    # For channel estimation: complex channel coefficients (real/imag)
    # input_shape = (8, 72, 2, 2)  # Example: 64 subcarriers, 10 symbols, 2 channels
    # model.build(input_shape)
    
    # Create dummy input
    dummy_input = tf.random.normal([8, 72, 2, 2])
    
    # Forward pass
    output = model(dummy_input, training=False)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Print model summary
    model.summary()