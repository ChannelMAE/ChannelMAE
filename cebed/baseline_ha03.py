'''
1. Pretrain HA03
2. Evaluate HA03
3. Online Adaptation
4. Online Evaluation
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
import numpy as np
from cebed.datasets_with_ssl.ds_label_pilot import LabelPilotDataset
from cebed.models.transformers import HA03
import cebed.models as cm
from cebed.utils_callback import get_training_callbacks


def visualize_model_performance(x, y, model_output, sample_idx=20, title_suffix="", filename="visualization.png"):
    """
    Visualize input, label, and model output for comparison.
    
    Args:
        x: Input tensor
        y: Label tensor  
        model_output: Model output tensor
        sample_idx: Index of sample to visualize
        title_suffix: Suffix to add to titles
        filename: Output filename for the plot
    """
    import matplotlib.pyplot as plt
    
    # Get the data for visualization
    input_data = x[sample_idx, :, :, 0].numpy()
    label_data = y[sample_idx, :, :, 0].numpy()
    output_data = model_output[sample_idx, :, :, 0].numpy()
    
    # Calculate unified range for all three plots
    vmin = min(input_data.min(), label_data.min(), output_data.min())
    vmax = max(input_data.max(), label_data.max(), output_data.max())

    fig, axs = plt.subplots(1, 3, figsize=(15, 2))
    
    # Plot input with colorbar
    im1 = axs[0].imshow(input_data, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_title(f'Input{title_suffix}', fontsize=12)
    axs[0].set_xlabel('Subcarriers')
    axs[0].set_ylabel('OFDM Symbols')
    plt.colorbar(im1, ax=axs[0], shrink=0.8)
    
    # Plot label with colorbar
    im2 = axs[1].imshow(label_data, aspect='auto', cmap='viridis')
    axs[1].set_title(f'Label{title_suffix}', fontsize=12)
    axs[1].set_xlabel('Subcarriers')
    axs[1].set_ylabel('OFDM Symbols')
    plt.colorbar(im2, ax=axs[1], shrink=0.8)
    
    # # Plot output with colorbar
    im3 = axs[2].imshow(output_data, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[2].set_title(f'Model Output{title_suffix}', fontsize=12)
    axs[2].set_xlabel('Subcarriers')
    axs[2].set_ylabel('OFDM Symbols')
    plt.colorbar(im3, ax=axs[2], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {filename}")
    plt.close()  # Close the figure to free memory


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train HA03 baseline')
    parser.add_argument('--epochs', type=int, default=80, 
                       help='Number of training epochs (default: 1)')
    parser.add_argument('--run_name', type=str, default='HA03_bs64_lr0.001_baseline',
                       help='Name for this training run (default: HA03_bs64_lr0.001_baseline)')
    parser.add_argument('--log_dir', type=str, default='train_output',
                       help='Base directory for logging (default: train_output)')
    parser.add_argument('--model_name', type=str, default='HA03',
                       help='Model name (default: HA03)')
    parser.add_argument('--experiment_name', type=str, default='siso_1_umi_block_1_ps2_p72',
                       help='Experiment name (default: siso_1_umi_block_1_ps2_p72)')
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    MyDataset = LabelPilotDataset(path="./data/ps2_p72/rt4/snr10to25_speed5", 
                                train_split=0.9, 
                                main_input_type="low",
                                aux_input_type="low",
                                seed=42)
    
    # set up the dataset in the above line

    train_loader, eval_loader = MyDataset.get_loaders(
        train_batch_size=64,
        eval_batch_size=64,
        task = "aux" 
    )

    ## prepare model
    model_hparams = cm.get_model_hparams(args.model_name, args.experiment_name)

    # initialize model
    MyModel = HA03(model_hparams)

    
    for x, y in train_loader.take(1):
        print("\nMain task:")
        print(f"Model input shape: {x.shape}")
        print(f"Label shape: {y.shape}")
        output = MyModel(x)

        # output shape: [batch, 2, 72, 2]
        print(f"Model output shape: {output.shape}")
    
        # Visualize before training
        visualize_model_performance(
            x, y, output, 
            sample_idx=0, 
            title_suffix=" (Before Training)",
            filename="ha03_data_example_before_training.png"
        )
        break

    print(MyModel.summary())
    
    # Create output directory
    log_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    
    # Compile model
    MyModel.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(name="loss"),
        # metrics=['mae']
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

    MyModel.save_weights(os.path.join(log_dir, "ha03.h5"))
    
    # Visualize after training with the same data
    print("\nGenerating post-training visualization...")
    for x, y in train_loader.take(1):
        # Get output from trained model
        output_after_training = MyModel(x)
        
        # Visualize after training
        visualize_model_performance(
            x, y, output_after_training,
            sample_idx=0,
            title_suffix=" (After Training)",
            filename="ha03_data_example_after_training.png"
        )
        break

    # We can use classic_evalutor.py
    # NOTE: Only when evaluation channels to compute to compute channel MSEs, we do bilinear resizing to the model output
    # interp_outputs = tf.image.resize(outputs, [14, 72], method=tf.image.ResizeMethod.BILINEAR)
    # tf.keras.layers.Resizing(target_height, target_width, interpolation="bilinear")


if __name__ == "__main__":
    main()