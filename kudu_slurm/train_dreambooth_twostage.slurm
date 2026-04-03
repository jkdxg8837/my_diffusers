#!/bin/bash
#
#SBATCH --job-name=rank95_analysis # Job name for tracking
#SBATCH --partition=mlsys-ada  # Partition you wish to use
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32      # Number of CPU threads
#SBATCH --gres=gpu:1             # Number of GPUs to use

#SBATCH --time=2-00:00:00        # Job time limit (2 days)
#SBATCH --output=../joboutput/twostage_%j.out # Standard out
#SBATCH --error=../joboutput/twostage_%j.err  # Standard error

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
source ~/.bashrc
conda activate /dcs/large/u5649209/envs/zimage

export WORKSPACE="/dcs/pg24/u5649209/data/workspace/diffusers"



echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"

## Execute your program(s) ##
python /dcs/pg24/u5649209/data/workspace/diffusers/kudu_slurm/train_dreambooth_twostage.py
