import tensorflow as tf
from tensorflow import keras

import numpy as np
import random

# Set random seeds for reproducibility
tf.random.set_seed(1)
np.random.seed(0)
random.seed(0)

keras = tf.keras
layers = tf.keras.layers


class BasicBlock(layers.Layer):
    """Equivalent to PyTorch BasicBlock"""
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.expansion = 1
        
        # Conv2D: TensorFlow uses NHWC format by default
        self.conv1 = layers.Conv2D(
            filters=planes,
            kernel_size=3,
            strides=stride,
            padding='same',  # equivalent to padding=1 in PyTorch
            use_bias=False,
            data_format='channels_last'
        )
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
    
    def call(self, x, training=None):
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        return out


class DnCNN(keras.Model):
    """Equivalent to PyTorch DnCNN"""
    
    def __init__(self, depth=5, filters=64):
        super(DnCNN, self).__init__()
        
        self.depth = depth
        self.filters = filters
        
        # Build layer1: Sequential of BasicBlocks
        self.layer1_blocks = []
        for i in range(depth - 1):
            if i == 0:
                in_planes = 2  # Input channels
            else:
                in_planes = filters
            
            self.layer1_blocks.append(BasicBlock(in_planes, filters))
        
        # noise_layer1: Conv2D from filters to 2 channels
        self.noise_layer1 = layers.Conv2D(
            filters=2,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False,
            data_format='channels_last'
        )
        
        # Build layer2: Sequential of BasicBlocks  
        self.layer2_blocks = []
        for i in range(depth - 1):
            if i == 0:
                in_planes = 2  # Input channels (from out1)
            else:
                in_planes = filters
            
            self.layer2_blocks.append(BasicBlock(in_planes, filters))
        
        # noise_layer2: Conv2D from filters to 2 channels
        self.noise_layer2 = layers.Conv2D(
            filters=2,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False,
            data_format='channels_last'
        )
    
    def call(self, input_tensor, training=None):
        # input_tensor shape: (batch, height, width, channels) in TensorFlow
        
        # Layer1 forward pass
        out = input_tensor
        for block in self.layer1_blocks:
            out = block(out, training=training)
        
        # noise_layer1
        out1 = self.noise_layer1(out)
        
        # Channel mixing operations (equivalent to PyTorch channel operations)
        # PyTorch: out1[:,0,:,:] = 0.5*input[:,1,:,:] + 0.5*tanh(out1[:,0,:,:])
        # PyTorch: out1[:,1,:,:] = 0.5*input[:,0,:,:] + 0.5*tanh(out1[:,1,:,:])
        
        # In TensorFlow NHWC format, channels are the last dimension
        channel_0 = 0.5 * input_tensor[:, :, :, 1:2] + 0.5 * tf.tanh(out1[:, :, :, 0:1])
        channel_1 = 0.5 * input_tensor[:, :, :, 0:1] + 0.5 * tf.tanh(out1[:, :, :, 1:2])
        
        # Concatenate channels back
        out1 = tf.concat([channel_0, channel_1], axis=-1)
        
        # Layer2 forward pass
        out = out1
        for block in self.layer2_blocks:
            out = block(out, training=training)
        
        # noise_layer2
        out = self.noise_layer2(out)
        
        # Final residual connection: input - tanh(out)
        output = input_tensor - tf.tanh(out)
        
        return output


def create_model_summary(model, input_shape):
    """Create and print model summary"""
    print("="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    model.build(input_shape)
    model.summary()
    
    print(f"\nTotal trainable parameters: {model.count_params():,}")
    print("="*80)


# ---------------------------------------------------------------------------- #
if __name__ == '__main__':
    
    # Set random seeds
    tf.random.set_seed(1)
    np.random.seed(0)
    random.seed(0)
    
    # Model parameters
    batch_size = 4
    channels = 2  # Complex data (real + imaginary)
    height = 14
    width = 72
    
    print("Creating TensorFlow/Keras equivalent of PyTorch DnCNN...")
    
    # Create model
    model = DnCNN()
    
    # Create random test data (TensorFlow uses NHWC format)
    # rec_y = tf.random.normal((batch_size, height, width, channels))
    true_h = tf.random.normal((batch_size, height, width, channels))
    ls_h = tf.random.normal((batch_size, height, width, channels))
    
    print(f"\nInput shapes:")
    print(f"ls_h (input): {ls_h.shape}")
    
    # Forward pass
    est_h = model(ls_h, training=False)
    print(f"est_h (output): {est_h.shape}")
    
    # Print model summary
    # create_model_summary(model, (None, height, width, channels))
    
    from cebed.datasets_with_ssl.ds_denoise import DenoiseDataset
    MyDataset = DenoiseDataset(path="./data/ps2_p72/rt1/snr10to20_speed5", 
                                train_split=0.9, 
                                main_input_type="low",
                                aux_input_type="raw",
                                seed=0)
    
    # already set up the dataset in the above line

    train_loader, eval_loader = MyDataset.get_loaders(
        train_batch_size=64,
        eval_batch_size=64,
        task = "aux"  # "aux" for pretrainig; "main" for online inference; Training task is always "aux"
    )
