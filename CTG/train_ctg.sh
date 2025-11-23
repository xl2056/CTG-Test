#!/bin/bash
#SBATCH --job-name=CTG
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-24:00:00
#SBATCH --mem-per-cpu=100G
## GPU requirements
#SBATCH --gres gpu:1
#SBATCH -p gpu
#SBATCH --nice

export PYTHONUNBUFFERED=1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate bg3.9

export PYTHONPATH=$HOME/CTGTest/CTG/

# Define parameter combinations

export WANDB_APIKEY=fca5ee0f0b5f26a561e73544b04d48a96daa94b4
# Run the appropriate command based on SLURM_ARRAY_TASK_ID
python ~/CTGTest/CTG/scripts/train.py --dataset_path ../behavior-generation-dataset/nuscenes --config_name trajdata_nusc_strive --remove_exp_dir

