import os
import tensorflow as tf


def get_training_callbacks(log_dir):
    """Returns the training callbacks for model training.
    
    Args:
        log_dir (str): Directory to save logs and checkpoints
        
    Returns:
        list: List of Keras callbacks
    """
    
    # Checkpoint callback - save best model weights
    ckpt_folder = os.path.join(log_dir, "cp.ckpt")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_folder,
        save_weights_only=True,
        verbose=1,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )
    
    # CSV logger for training history
    csv_logger_filename = os.path.join(log_dir, "training_history.csv")
    history_logger = tf.keras.callbacks.CSVLogger(
        csv_logger_filename, separator=",", append=False
    )
    
    # Learning rate reduction on plateau
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", 
        factor=0.5, 
        patience=5, 
        min_lr=0.00001,
        verbose=1
    )
    
    # Early stopping callback
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", 
        patience=10, 
        min_delta=0.00001,
        restore_best_weights=True,
        verbose=1
    )
    
    return [checkpoint_callback, history_logger, lr_callback, es_callback] 