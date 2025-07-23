import tensorflow as tf
from typing import Tuple, Dict, Any


class BaseModel(tf.keras.Model):
    """
    The base model object
    :param input_shape: The input shape of the model
    :param output_shape: The model's output shape
    :param hparam: The model hyperparameters
    """

    def __init__(self, hparams: Dict[str, Any]) -> None:
        super().__init__()
        self.hparams = hparams

        for key, val in self.hparams.items():
            setattr(self, key, val)


    def set_train_mode(self, mode: str):
        """
        Set the mode of the model: "pretrain" or "test-time-train"
        """
        self.mode = mode
        if mode == "pretrain":
            self.sr_model.trainable = True
            self.denoiser.trainable = True
        elif mode == "ttt":
            self.sr_model.trainable = False
            self.denoiser.trainable = True
        else:
            raise ValueError(f"Unknown mode {mode}")
    

    def train_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        """
        The logic of one training step given a minibatch of data
        """
        if self.mode is None:
            raise ValueError("Please set the mode of the model before training")
        
        if self.mode == "pretrain":
            return self.pretrain_step(data)
        elif self.mode == "ttt":
            return self.test_time_train_step(data)
        else:
            raise ValueError(f"Unknown mode {self.mode}")
    

    def test_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        """
        The logic of one test step given a minibatch of data
        """
        if self.mode is None:
            raise ValueError("Please set the mode of the model before training")
        
        if self.mode == "pretrain":
            return self.pretrain_test_step(data)
        elif self.mode == "ttt":
            return self.test_time_test_step(data)
        else:
            raise ValueError(f"Unknown mode {self.mode}")
    
    
    # ---------------------------------------------------------------------------- #
    @tf.function
    def pretrain_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        """
        Pre-training step given a batch of data 
        Update both super-resolution network and denoising network based on combined loss
        """
        assert self.mode == "pretrain", "Mode should be 'pretrain'"
        (x_main, y_main), (x_aux, y_aux) = data

        with tf.GradientTape() as tape:
            pred_main, pred_aux = self((x_main, x_aux)) # pass an input tuple to the model
            loss = self.compiled_loss(y_main, pred_main) + self.compiled_loss(y_aux, pred_aux) # sum-loss of main task and SSL task

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # add channel estimation error as a metric
        self.compiled_metrics.update_state(y_main, pred_main)
        return {m.name: m.result() for m in self.metrics}

    
    @tf.function
    def pretrain_test_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        assert self.mode == "pretrain", "Mode should be 'pretrain'"
        
        (x_main, y_main), (x_aux, y_aux) = data
        pred_main, pred_aux = self((x_main, x_aux))
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
        (x_main , y_main), (x_aux, y_aux) = data
        with tf.GradientTape() as tape:
            pred_main , pred_aux = self((x_main, x_aux)) # pass input tuple
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
        (x_main , y_main), (x_aux, y_aux) = data
        pred_main, pred_aux = self((x_main, x_aux))
        loss = self.compiled_loss(y_aux, pred_aux) 

        self.compiled_metrics.update_state(y_main, pred_main)
        return {m.name: m.result() for m in self.metrics}
    
