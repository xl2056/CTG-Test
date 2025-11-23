#!/bin/bash
#SBATCH --job-name=DiffGanInfer
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

# Add ffmpeg directly to PATH (no bin subdirectory)
export PATH=$HOME/CTGTest/ffmpeg-7.0.2-amd64-static:$PATH

# Set PYTHONPATH to include both CTG and the current directory
export PYTHONPATH=$HOME/CTGTest/CTG/:$HOME/CTGTest:$PYTHONPATH


# Run the appropriate command based on SLURM_ARRAY_TASK_ID
python -m DiffusionGan.inference

