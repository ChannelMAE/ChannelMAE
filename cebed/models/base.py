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


    def train_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        """
        The logic of one training step given a minibatch of data

        All the supervised-learning-based models follow this common logic,
        so that we can define it here in the base class
        """
        x_batch, y_batch = data

        # TODO: the following is a temporary fix but still Nonetype will appear in the data_loader
        x_batch = tf.reshape(x_batch, [-1, self.input_dim[0], self.input_dim[1], self.input_dim[2]])

        with tf.GradientTape() as tape:
            preds = self(x_batch) # x_batch can be a tuple of batched input1 and input2.
            loss = self.compiled_loss(y_batch, preds)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compiled_metrics.update_state(y_batch, preds)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        """
        """
        
        # the data_loader here must be a single-task data loader (task = "main" or "aux")
        # if self.task == "aux":
        # x_batch, mask, y_batch = data
        # elif self.task == "main":
        x_batch, y_batch = data

        # TODO: the following is a temporary fix but still Nonetype will appear in the data_loader
        x_batch = tf.reshape(x_batch, [-1, self.input_dim[0], self.input_dim[1], self.input_dim[2]])

        with tf.GradientTape() as tape:
            preds = self(x_batch) # x_batch can be a tuple of batched input1 and input2.
            loss = self.compiled_loss(y_batch, preds)

        # grads = tape.gradient(loss, self.trainable_weights)
        # self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compiled_metrics.update_state(y_batch, preds)

        return {m.name: m.result() for m in self.metrics}

    