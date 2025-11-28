#!/bin/bash
#SBATCH --job-name=trn
#SBATCH --partition=fast
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Load required modules
ml matplotlib/3.9.2-gfbf-2024a
ml PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
ml scikit-learn/1.5.2-gfbf-2024a
ml tqdm/4.67.1-GCCcore-14.2.0
ml tensorboard/2.18.0-gfbf-2024a

# Run the training script
python /home.nfs/{your_fel_username}/{path_to_your_dir}/train.py
