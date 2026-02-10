#!/bin/bash
#
#SBATCH --job-name=diff_finetune # Job name for tracking
#SBATCH --partition=gpu  # Partition you wish to use (see above for list)
#SBATCH --gres=gpu:lovelace_l40:1 # Number of GPUs to use
#SBATCH --account=su007-hw-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32      # Number of CPU threads used by your job
# 80GB of system RAM will be allocated with 1 GPU specified

#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#SBATCH --output=../joboutput/joboutput_%j.out # Standard out from your job
#SBATCH --error=../joboutput/joboutput_%j.err  # Standard error from your job

## Initialisation ##

source /home/j/jiayang_gu/.bashrc
source activate zimage
export WANDB_KEY="05e08cadfa8f29b4fb78407724e732c453648de6"
export PROJECT="fft_sd3"

## Execute your program(s) ##
python eval_fft.py