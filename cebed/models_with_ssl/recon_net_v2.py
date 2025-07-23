import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple
import sys
from pathlib import Path
root_path = str(Path(__file__).parent.parent.parent)
sys.path.append(root_path)

from cebed.models.common import ResidualBlock
from cebed.models.transformers import Encoder
from cebed.models_with_ssl.base import BaseModel
from cebed.models_with_ssl.recon_net import Decoder


class ReconMAEV2(BaseModel):
    """
    A two-branch model with shared decoder
    - Main encoder -
               --- Shared decoder ---
    - Aux encoder  -
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

        self.main_encoder = Encoder(
            num_layers=self.num_en_layers,
            key_dim=head_size, # 72
            num_heads=num_heads, # 2
            ff_dim=ff_dim, # 144
            dropout_rate=self.dropout_rate,
        )

        self.aux_encoder = Encoder(
            num_layers=self.num_en_layers,
            key_dim=head_size, # 72
            num_heads=num_heads, # 2
            ff_dim=ff_dim, # 144
            dropout_rate=self.dropout_rate,
        )

        self.shared_decoder = Decoder(
            self.output_dim, self.num_dc_layers, self.hidden_size, kernel_size=self.kernel_size
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
            self.main_encoder.trainable = True
            self.aux_encoder.trainable = True
            self.shared_decoder.trainable = True
        elif mode == "ttt":
            self.main_encoder.trainable = False
            self.aux_encoder.trainable = True
            self.shared_decoder.trainable = True
        else:
            raise ValueError(f"Unknown mode {mode}")

    def aux_expand_batch(self, low_embed: tf.Tensor, mask:tf.Tensor) -> tf.Tensor:
        '''
        Insert the encoded embeddings into the pilot locations and pad other locations with non-zero values for the entire batch
        '''
        # [num_pilots, 2]
        mask_indices = tf.where(mask == 1) # [batch, 14, 72]

        # [batch, nps*npf, c]
        batch_size = tf.shape(low_embed)[0] # must use dynamic shape!
        n_channel = tf.shape(low_embed)[2]
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
        return high_embed

    def main_expand_batch(self, low_embed):
        # [num_pilots, 2]
        pilot_indices = tf.where(self.pilot_mask[0,0,:,:]==1) # pilot=1, non-pilot=0

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
        main_input, (low_dim_aux_input, mask) = inputs 

        # --------------------------------- Main Task -------------------------------- #
        # [batch, nps, nfs, c]
        shape = main_input.shape
        main_input = tf.keras.layers.Reshape((-1, shape[-1]))(main_input)
        main_input = tf.keras.layers.Permute((2, 1))(main_input)
        latent = self.main_encoder(main_input) 
        latent = tf.keras.layers.Permute((2, 1))(latent)
        expanded_latent = self.main_expand_batch(latent)
        main_outputs = self.shared_decoder(expanded_latent)
        
        # --------------------------------- Aux Task --------------------------------- #
        shape = low_dim_aux_input.shape
        low_dim_aux_input = tf.keras.layers.Reshape((-1, shape[-1]))(low_dim_aux_input)
        low_dim_aux_input = tf.keras.layers.Permute((2, 1))(low_dim_aux_input)
        latent = self.aux_encoder(low_dim_aux_input) 
        latent = tf.keras.layers.Permute((2, 1))(latent)
        expanded_latent = self.aux_expand_batch(latent, mask) # add random mask
        aux_outputs = self.shared_decoder(expanded_latent)

        return main_outputs, aux_outputs
    
    ### May inherit train_step and test_step from BaseModel
    # ---------------------------------------------------------------------------- #
    @tf.function
    def pretrain_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        """
        Pre-training step given a batch of data 
        Update both super-resolution network and denoising network based on combined loss
        """
        assert self.mode == "pretrain", "Mode should be 'pretrain'"
        (x_main, y_main), (x1_aux, x2_aux, y_aux) = data

        with tf.GradientTape() as tape:
            pred_main, pred_aux = self((x_main, (x1_aux,x2_aux))) 
            loss = self.compiled_loss(y_main, pred_main) + self.compiled_loss(y_aux, pred_aux) # sum-loss of two tasks

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # add channel estimation error as a metric
        self.compiled_metrics.update_state(y_main, pred_main)
        return {m.name: m.result() for m in self.metrics}

    
    @tf.function
    def pretrain_test_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        assert self.mode == "pretrain", "Mode should be 'pretrain'"
        
        (x_main, y_main), (x1_aux, x2_aux, y_aux) = data
        pred_main, pred_aux = self((x_main, (x1_aux,x2_aux))) 
        loss = self.compiled_loss(y_main, pred_main) + self.compiled_loss(y_aux, pred_aux) # self.compute_loss

        # add channel estimation error as a metric
        self.compiled_metrics.update_state(y_main, pred_main)
        return {m.name: m.result() for m in self.metrics}


    # ---------------------------------------------------------------------------- #
    @tf.function
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
    @tf.function
    def test_time_test_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        assert self.mode == "ttt", "Mode should be 'ttt'"
        (x_main, y_main), (x1_aux, x2_aux, y_aux) = data
        pred_main, pred_aux = self((x_main,(x1_aux,x2_aux)))
        loss = self.compiled_loss(y_aux, pred_aux) 

        self.compiled_metrics.update_state(y_main, pred_main)
        return {m.name: m.result() for m in self.metrics}
    


# ------------------------- Test the two-branch model ------------------------ #
if __name__ == "__main__":

    from cebed.datasets_with_ssl.ds_mae_random_mask import MAEDatasetRandomMask
    import cebed.models_with_ssl as cm
    MyDataset = MAEDatasetRandomMask(path="./data/ps2_p72/uma/snr0to25_speed5", 
                                    train_split=0.9, 
                                    main_input_type="low",
                                    aux_input_type = "low",
                                    seed=0)

    train_loader, eval_loader = MyDataset.get_loaders(
        train_batch_size=64,
        eval_batch_size=64,
        task="both"
    )


    # prepare model
    experiment_name = "siso_1_umi_block_1_ps2_p72"
    model_name = "ReconMAEV2"
    model_hparams = cm.get_model_hparams(model_name, experiment_name)

    # initialize model
    MyModel = ReconMAEV2(model_hparams)
    MyModel.set_mask(MyDataset.env.get_mask())

    # build model for main task
    main_input = tf.ones([1, 2, 72, 2])
    aux_low_input = tf.ones([1, 2, 72, 2])
    example_mask = MyDataset.env.get_mask()
    example_mask = tf.squeeze(example_mask)
    example_mask = tf.expand_dims(example_mask, axis=0) # [batch, 14, 72]
    MyModel((main_input, (aux_low_input, example_mask)))
    print(MyModel.summary())

    # train model
    MyModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
                    loss=tf.keras.losses.MeanSquaredError(name="loss"))
    MyModel.set_train_mode("pretrain")

    MyModel.fit(train_loader, validation_data=None, epochs=10, callbacks=None)
    MyModel.save_weights("pretrained_model_v2.h5")
    