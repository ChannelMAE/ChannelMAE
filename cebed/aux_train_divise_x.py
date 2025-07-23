'''
1. Pretrain aux-task using stacked x method
2. Evaluate
'''
import os
import sys
import time
import argparse
from pathlib import Path

# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)

# Imports
import tensorflow as tf
from cebed.datasets_with_ssl.ds_mae_random_mask import MAEDatasetRandomMask
from cebed.models.mae_random_mask import MaeRandomMask
import cebed.models as cm
from cebed.utils_callback import get_training_callbacks


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MAE with Divide X method')
    parser.add_argument('--epochs', type=int, default=1, 
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--run_name', type=str, default='MAE_diviseX_bs64_lr0.001_baseline',
                       help='Name for this training run (default: MAE_diviseX_bs64_lr0.001_baseline)')
    parser.add_argument('--log_dir', type=str, default='train_output',
                       help='Base directory for logging (default: train_output)')
    parser.add_argument('--model_name', type=str, default='MaeRandomMask',
                       help='Model name (default: MaeRandomMask)')
    parser.add_argument('--experiment_name', type=str, default='siso_1_umi_block_1_ps2_p72',
                       help='Experiment name (default: siso_1_umi_block_1_ps2_p72)')
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    MyDataset = MAEDatasetRandomMask(
        path="./data/ps2_p72/rt1/snr10to20_speed5", 
        train_split=0.9, 
        main_input_type="low",
        aux_input_type="low",  # aux_input [14,72,4]
        sym_error_rate=0,
        seed=42
    )
    # already set up the dataset in the above line

    train_loader, eval_loader = MyDataset.get_loaders(
        train_batch_size=64,
        eval_batch_size=64,
        task="aux"  # must be aux
    )

    ## prepare model
    model_hparams = cm.get_model_hparams(args.model_name, args.experiment_name)

    # initialize model
    MyModel = MaeRandomMask(model_hparams)
    # build model
    # Get input shapes from dataset
    # main_input_shape, aux_input_shape = MyDataset.get_input_shape()
    
    # # inputs1_aux shape: [batch, ns, nf, (n_r_ants+n_t_ants)*2] 
    # inputs1_aux = tf.zeros([8, 14, 72, 4])  # 4 = (1+1)*2 for SISO case
    # # inputs2_aux shape: [batch, ns, nf] - mask
    # inputs2_aux = tf.zeros([8, 14, 72])
    

    for inputs1_aux, inputs2_aux, y_aux in train_loader.take(1):
        print("\nAuxiliary task:")
        print(f"x1_aux shape: {inputs1_aux.shape}")
        print(f"x2_aux shape: {inputs2_aux.shape}")
        print(f"y_aux shape: {y_aux.shape}")
        output = MyModel((inputs1_aux, inputs2_aux))
        print(f"output shape: {output.shape}")
       

    # print(MyModel.summary())

    # Setup training configuration
    # Create output directory
    log_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Compile model
    MyModel.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(name="loss")
    )
    
    # Get callbacks
    callbacks = get_training_callbacks(log_dir)
    
    # Evaluate before training
    print("Evaluating model before training...")
    train_results = MyModel.evaluate(train_loader, verbose=1, return_dict=True)
    val_results = MyModel.evaluate(eval_loader, verbose=1, return_dict=True)
    
    # Save initial results
    before_train_filename = os.path.join(log_dir, "before_train.txt")
    with open(before_train_filename, "w") as f:
        f.write("Results before training:\n")
        f.write(f"Train dataset: {train_results}\n")
        f.write(f"Validation dataset: {val_results}\n")
    
    print(f"Initial - Train Loss: {train_results['loss']:.6f}, Val Loss: {val_results['loss']:.6f}")
    
    # Train the model
    print("\nStarting training...")
    start_time = time.time()
    
    history = MyModel.fit(
        train_loader,
        validation_data=eval_loader,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Print final results
    final_train_loss = min(history.history['loss'])
    final_val_loss = min(history.history['val_loss'])
    final_lr = history.history['lr'][-1]
    
    print(f"\nTraining Summary:")
    print(f"Best Train Loss: {final_train_loss:.6f}")
    print(f"Best Val Loss: {final_val_loss:.6f}")
    print(f"Final Learning Rate: {final_lr:.8f}")
    print(f"Training History saved to: {os.path.join(log_dir, 'training_history.csv')}")


if __name__ == "__main__":
    main()
