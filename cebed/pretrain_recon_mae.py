#!/usr/bin/env python3
"""
Pretraining script for ReconMAE model.
This script handles the pretraining phase of the ReconMAE model using the training dataset.
"""

import argparse
import os
import sys
import datetime
from pathlib import Path

import tensorflow as tf

root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)

from cebed.datasets_with_ssl.ds_mae_random_mask import MAEDatasetRandomMask
from cebed.datasets_with_ssl.ds_mae_fix_mask import MAEDatasetFixMask
from cebed.models_with_ssl.recon_net import ReconMAE
from cebed.models_with_ssl.recon_net_main_only import ReconMAE_MainOnly
import cebed.models_with_ssl as cm


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain ReconMAE Model")
    
    # Data configuration
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data directory")
    parser.add_argument("--train_split", type=float, default=0.9,
                        help="Training split ratio")
    parser.add_argument("--main_input_type", type=str, default="low",
                        choices=["low", "high"], help="Main input type")
    parser.add_argument("--aux_input_type", type=str, default="low", 
                        choices=["low", "high"], help="Auxiliary input type")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--aug_factor", type=int, default=1,
                        help="Data augmentation factor")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="ReconMAE",
                        choices=["ReconMAE", "ReconMAE_MainOnly"],
                        help="Model name - ReconMAE for dual-branch, ReconMAE_MainOnly for single-branch")
    parser.add_argument("--experiment_name", type=str, default="siso_1_umi_block_1_ps2_p72",
                        help="Experiment name for model hyperparameters")
    parser.add_argument("--masking_type", type=str, default="discrete",
                        choices=["discrete", "contiguous", "fixed", "random_symbols", "fix_length"],
                        help="Type of masking strategy")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=64,
                        help="Evaluation batch size")
    parser.add_argument("--early_stopping", action="store_true", default=True,
                        help="Enable early stopping")
    parser.add_argument("--no_early_stopping", dest="early_stopping", action="store_false",
                        help="Disable early stopping")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./model_output",
                        help="Output directory for model weights and logs")
    parser.add_argument("--weights_name", type=str, default=None,
                        help="Name for the saved weights file (auto-generated if not provided)")
    parser.add_argument("--verbose", type=int, default=1,
                        choices=[0, 1, 2], help="Verbosity level")
    
    return parser.parse_args()


def create_dataset(args):
    """Create dataset based on masking type"""
    print(f"Creating dataset with masking type: {args.masking_type}")
    
    if args.masking_type == "fixed":
        dataset = MAEDatasetFixMask(
            path=args.data_path,
            train_split=args.train_split,
            main_input_type=args.main_input_type,
            aux_input_type=args.aux_input_type,
            seed=args.seed
        )
    else:
        dataset = MAEDatasetRandomMask(
            path=args.data_path,
            train_split=args.train_split,
            main_input_type=args.main_input_type,
            aux_input_type=args.aux_input_type,
            seed=args.seed,
            aug_factor=args.aug_factor,
            masking_type=args.masking_type
        )
    
    return dataset


def main():
    args = parse_args()
    
    print("="*80)
    print("ReconMAE Pretraining Script")
    print("="*80)
    print(f"Data path: {args.data_path}")
    print(f"Model: {args.model_name}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Masking type: {args.masking_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch sizes: train={args.train_batch_size}, eval={args.eval_batch_size}")
    print("="*80)
    
    # Create dataset
    try:
        dataset = create_dataset(args)
        print(f"✅ Dataset created successfully")
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return 1
    
    # Get data loaders
    try:
        train_loader, eval_loader = dataset.get_loaders(
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            task="both"
        )
        
        train_size = sum(1 for _ in train_loader)
        eval_size = sum(1 for _ in eval_loader)
        print(f"✅ Data loaders created: {train_size} train batches, {eval_size} eval batches")
        
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        return 1
    
    # Get model hyperparameters
    try:
        model_hparams = cm.get_model_hparams(args.model_name, args.experiment_name)
        # Override masking type if specified
        model_hparams["masking_type"] = args.masking_type
        print(f"✅ Model hyperparameters loaded")
        
    except Exception as e:
        print(f"❌ Error loading model hyperparameters: {e}")
        return 1
    
    # Get input shape from dataset
    try:
        for batch_data in train_loader.take(1):
            if isinstance(batch_data, tuple):
                (x_main, y_main), (x1_aux, x2_aux, y_aux) = batch_data
                main_input_shape = x_main.shape[1:]
                # Take first examples from batch for model building
                main_input = x_main[0:1]
                aux_low_input = x1_aux[0:1]
                example_mask = x2_aux[0:1]
                break
        
        print(f"✅ Input shape determined: {main_input_shape}")
        
    except Exception as e:
        print(f"❌ Error getting input shape: {e}")
        return 1
    
    # Initialize model based on model_name
    try:
        if args.model_name == "ReconMAE":
            model = ReconMAE(model_hparams, main_input_shape=main_input_shape)
        elif args.model_name == "ReconMAE_MainOnly":
            model = ReconMAE_MainOnly(model_hparams, main_input_shape=main_input_shape)
        else:
            raise ValueError(f"Unknown model name: {args.model_name}")
            
        model.set_mask(dataset.env.get_mask())
        
        # Build model using real examples from dataset
        model((main_input, (aux_low_input, example_mask)))
        print(f"✅ Model initialized successfully: {args.model_name}")
        print(f"Model summary:")
        model.summary()
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return 1
    
    # Generate weights name if not provided
    if args.weights_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario = Path(args.data_path).parent.name  # e.g., "rt1"
        model_suffix = "MainOnly" if args.model_name == "ReconMAE_MainOnly" else "Full"
        args.weights_name = f"{scenario}_{model_suffix}_{timestamp}"
    
    # Train model
    try:
        print(f"🚀 Starting training for {args.model_name}...")
        history, log_dir = model.train_model(
            train_loader=train_loader,
            eval_loader=eval_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            log_dir=args.output_dir,
            weights_name=args.weights_name,
            early_stopping=args.early_stopping,
            verbose=args.verbose,
        )
        
        print(f"✅ Training completed successfully!")
        print(f"📁 Model saved in: {log_dir}")
        print(f"💾 Weights file: {args.weights_name}.h5")
        
        # Print final metrics
        if history.history:
            final_loss = history.history.get('loss', [])[-1] if history.history.get('loss') else 'N/A'
            final_val_loss = history.history.get('val_loss', [])[-1] if history.history.get('val_loss') else 'N/A'
            print(f"📊 Final training loss: {final_loss}")
            print(f"📊 Final validation loss: {final_val_loss}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    with tf.device('/CPU'):
        print("Forcing CPU for pretraining, unless APPLE fixes MPS support")
        main()
