# ChannelMAE: Self-Supervised Learning Assisted Online Adaptation of Neural Channel Estimators

This repository contains the implementation for our paper "ChannelMAE: Self-Supervised Learning Assisted Online Adaptation of Neural Channel Estimators" submitted to **INFOCOM 2026**.


<!-- ## Overview

ChannelMAE introduces a self-supervised learning framework that enables neural channel estimators to adapt online to new environments without requiring labeled data. The key innovation lies in combining Masked Autoencoder (MAE) architecture with test-time training for robust channel estimation across diverse scenarios.

### Key Features

- **Self-supervised pretraining** using masked autoencoder architecture
- **Online adaptation** capabilities for new environments
- **Two-branch architecture** with shared encoder for main and auxiliary tasks
- **Support for multiple scenarios**: UMi, UMa, RMa, and ray-tracing environments
- **Comprehensive baseline comparisons** with classical methods (LS, LMMSE, ALMMSE) -->

<!-- ## Core Architecture

### ChannelMAE Model Implementation

The core ChannelMAE architecture is implemented in:
- **`cebed/models_with_ssl/recon_net.py`** - Main two-branch ReconMAE model with shared encoder
- **`cebed/models_with_ssl/recon_net_main_only.py`** - Single-branch variant for main task only
- **`cebed/models_with_ssl/recon_net_v3.py`** - Enhanced version (ReconMAEX) with improved auxiliary branch

Key model features:
- Transformer-based encoder for learning representations from pilot data
- Dual decoder architecture for main (channel estimation) and auxiliary (reconstruction) tasks
- Flexible masking strategies (random, fixed, contiguous)
- Support for different input dimensions and antenna configurations

### Online Adaptation Implementation

The online adaptation mechanisms are implemented in:
- **`cebed/online_ttt_v1.py`** - Basic test-time training with two-branch model
- **`cebed/online_ttt_v2.py`** - Enhanced TTT with improved training strategies
- **`cebed/online_ttt_v3.py`** - Advanced adaptation with pseudo-labeling
- **`cebed/online_ttt_v4.py`** - Single-branch pseudo-label adaptation
- **`cebed/online_adapt_dncnn.py`** - Adaptation using DnCNN baseline
- **`cebed/online_adapt_ha03.py`** - Adaptation using HA03 baseline -->

## Project Structure

```
ChannelMAE/
├── cebed/                         # Main package
│   ├── models/                    # Base model implementations
│   │   └── mae_random_mask.py     # Random mask MAE
│   ├── models_with_ssl/           # Self-supervised models
│   │   └── recon_net.py           # Main ChannelMAE implementation
│   ├── datasets_with_ssl/         # Dataset implementations
│   ├── online_ttt_v3.py           # Online adaptation scripts
│   ├── online_adapt_*.py          # Baseline adaptation scripts
│   └── ...
├── hyperparams/                   # Model hyperparameter configs
├── scripts/                       # Utility scripts
│   └── online_ttt.py              # Online TTT execution script
├── runs/                          # Training and evaluation scripts
└── ray_tracing_data/              # Ray tracing simulation data
```

## Results and Figures

The experimental results and figures referenced in the paper are in: **`plot_figs/`**


## Getting Started
<!-- 
### Prerequisites

```bash
# Create conda environment
conda create -n channelmae python=3.11
conda activate channelmae

# Install dependencies
pip install -r requirements.txt  
``` -->


1. **Pretraining ChannelMAE:**
```bash
cd runs
bash pretrain.sh
```

2. **Online adaptation and evaluation:**
```bash
bash ttt.sh
```

3. **Baseline evaluation:**
```bash
bash eval_baselines.sh
```

## License

This project is licensed under the same terms as the original CeBed project. Please refer to the [CeBed repository](https://github.com/SAIC-MONTREAL/CeBed) for license details.

## Acknowledgments

This codebase is built upon the excellent foundation provided by the [CeBed](https://github.com/SAIC-MONTREAL/CeBed) project. We thank the original authors for their valuable contribution to the wireless communication community.
