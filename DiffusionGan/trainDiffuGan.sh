#!/bin/bash
#SBATCH --job-name=DiffuGan
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
cd $HOME/CTGTest

# Set PYTHONPATH to include both CTG and the current directory
export PYTHONPATH=$HOME/CTGTest/CTG/:$HOME/CTGTest:$PYTHONPATH

# Define parameter combinations

export WANDB_APIKEY=fca5ee0f0b5f26a561e73544b04d48a96daa94b4
# Run the appropriate command based on SLURM_ARRAY_TASK_ID
python -m DiffusionGan.adversarialDiffusion

