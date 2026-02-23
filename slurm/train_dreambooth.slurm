#!/bin/bash
#
#SBATCH --job-name=diff_finetune # Job name for tracking
#SBATCH --partition=mlsys-ada  # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1     # Number of GPUs to use 
# 80GB of system RAM will be allocated with 1 GPU specified

#SBATCH --time=1-00:00:00      # Job time limit set to 2 days (48 hours)
#SBATCH --output=../joboutput/train_%j.out # Standard out from your job
#SBATCH --error=../joboutput/train_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
source ~/.bashrc
conda activate /dcs/large/u5649209/envs/zimage

## Execute your program(s) ##
python train_dreambooth.py