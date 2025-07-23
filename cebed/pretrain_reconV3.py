import sys
from pathlib import Path
import os
import datetime
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)

# ------------------------- Test the two-branch model ------------------------ #
if __name__ == "__main__":
    from cebed.datasets_with_ssl.ds_mae_stack_x import MAEDataset
    import cebed.models_with_ssl as cm
    from cebed.models_with_ssl.recon_net_v3 import ReconMAEX
    
    # Get model config first to access masking type
    experiment_name = "siso_1_umi_block_1_ps2_p72"
    model_name = "ReconMAE"
    model_hparams = cm.get_model_hparams(model_name, experiment_name)
    
    # Create dataset with same masking type as model
    MyDataset = MAEDataset(
        path="./data/ps2_p72/rt1/snr10to25_speed5", 
        train_split=0.9,  # NOTE: this is the split ratio for pretraining
        main_input_type="low",
        aux_input_type="raw",
        seed=0
        # aug_factor=1,
        # masking_type=model_hparams["masking_type"]
    )

    train_loader, eval_loader = MyDataset.get_loaders(
        train_batch_size=64,
        eval_batch_size=64,
        task="both"
    )

    MyModel = ReconMAEX(model_hparams, main_input_shape=[2,72,2])

    # Get example data from dataset
    for (x_main, y_main), (x1_aux, x2_aux, y_aux) in train_loader.take(1):
        
        print(f"x_main shape: {x_main.shape}")
        print(f"y_main shape: {y_main.shape}")
        print(f"x1_aux shape: {x1_aux.shape}")
        print(f"x2_aux shape: {x2_aux.shape}")
        print(f"y_aux shape: {y_aux.shape}")
        
        MyModel.set_mask(MyDataset.env.get_mask())
        MyModel((x_main, (x1_aux, x2_aux)))
        print(MyModel.summary())
        break
    

    # train model
    history, log_dir = MyModel.train_model(
        train_loader=train_loader,
        eval_loader=eval_loader,
        epochs=2,
        learning_rate=0.001,
        log_dir="./new_output",
        weights_name="rt1",
        early_stopping=True,
        verbose=1
    )
    
    # The model weights will be automatically saved in log_dir
    print(f"Training completed. Model saved in {log_dir}")
