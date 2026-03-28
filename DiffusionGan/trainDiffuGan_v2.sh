#!/bin/bash
#SBATCH --job-name=DiffuGan_v2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-24:00:00
#SBATCH --mem-per-cpu=100G
## GPU requirements
#SBATCH --gres gpu:1
#SBATCH -p gpu
#SBATCH --nice

export PYTHONUNBUFFERED=1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate bg3.9

# Change to the project directory
cd $HOME/CTG-Test

# Set PYTHONPATH to include CTG, Pplan, and the current directory
export PYTHONPATH=$HOME/CTG-Test/CTG/:$HOME/CTG-Test/Pplan:$HOME/CTG-Test:$PYTHONPATH

export WANDB_APIKEY=fca5ee0f0b5f26a561e73544b04d48a96daa94b4

# Run adversarial training with situation-aware v2 reward model
# --model_path: path to pre-trained v2 model (from Stage 3)
# --num_iterations: G-D alternation rounds
# --d_epochs: reward model retraining epochs per D step
python -m DiffusionGan.adversarialDiffusion_v2 \
    --model_path ./MaxEntIRL/irl_output/situation_aware_irl_v2.pt \
    --num_iterations 50 \
    --d_epochs 30 \
    --d_batch_size 16
