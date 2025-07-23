import sys
from pathlib import Path
root_path = str(Path(__file__).parent.parent.parent)
sys.path.append(root_path)

import tensorflow as tf
from cebed.models.cnn import create_srcnn_model, create_dncnn_model
from cebed.models_with_ssl.base import BaseModel


class DenoiseSingleBranch(BaseModel):
    """
    Denoising model with a single branch based on ChannelNet structure

    Main task: super-resolution CNN --> denoising CNN
    Aux task: denoising CNN
    """

    def __init__(self, hparams):
        super().__init__(hparams)

        if self.int_type not in ["bilinear", "bicubic", "nearest"]:
            raise ValueError(f"Interpolation type not supported {self.int_type}")

        # The super-resolution model
        self.sr_model = create_srcnn_model(
            num_channels=self.output_dim[-1],
            hidden_size=self.sr_hidden_size,
            kernel_size=self.sr_kernels,
        )
        # The denoising model
        self.denoiser = create_dncnn_model(
            num_channels=self.output_dim[-1],
            hidden_size=self.dc_hidden,
            num_layers=self.num_dc_layers,
        )
        self.mode = None


    def call(self, inputs): # def call(self, inputs, training=False):
        """
        Only forward pass:
        main_inputs: a batch of low-resolution LS estimates 
        ---> output: full-resolution channel estimates

        aux_inputs: a batch of full-resolution received (preprocess_y_sample + random noise) 
        ---> output: denoised received signals

        """
        main_inputs, aux_inputs = inputs
        # --------------------------------- aux task --------------------------------- #
        # for denoising task: x_aux must be in the full dimension (full noisy received signal 'y')
        # x_aux = tf.keras.layers.Resizing(self.output_dim[0], self.output_dim[1], interpolation=self.int_type)(aux_inputs)
        noise = self.denoiser(aux_inputs)
        pred_aux = aux_inputs - noise

        # --------------------------------- main task -------------------------------- #
        # Aligning with the paper 'ChannelNet': add a resizing layer before the super-resolution model
        x_main = tf.keras.layers.Resizing(self.output_dim[0], self.output_dim[1], interpolation=self.int_type)(main_inputs)
        x_main = self.sr_model(x_main)
        noise = self.denoiser(x_main)
        pred_main = x_main - noise # full channel estiamtes

        return pred_main, pred_aux