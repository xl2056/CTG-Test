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


# Run the appropriate command based on SLURM_ARRAY_TASK_ID
python ~/CTGTest/CTG/scripts/scene_editor.py  --results_root_dir nusc_results/  --num_scenes_per_batch 1  --dataset_path ../behavior-generation-dataset/nuscenes  --env trajdata  --policy_ckpt_dir diffuser_trained_models/test/run0  --policy_ckpt_key checkpoints/iter100000.ckpt  --eval_class Diffuser   --editing_source 'config' 'heuristic'  --registered_name 'trajdata_nusc_diff'  --render 

