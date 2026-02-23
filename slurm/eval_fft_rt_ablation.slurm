#!/bin/bash
#
#SBATCH --job-name=eval_fft_rt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --account=su007-hw-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

#SBATCH --time=2-00:00:00
#SBATCH --output=../joboutput/eval_fft_rank_%j.out
#SBATCH --error=../joboutput/eval_fft_rank_%j.err

## Initialisation ##

source /home/j/jiayang_gu/.bashrc
source activate zimage

## Execute ##
for RT in 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
    echo "===== Running rank=64, scale_factor=1.0, rank_threshold=${RT} ====="
    python -u eval_fft_lora_init.py --rank 64 --scale_factor 1.0 --rank_threshold ${RT}
done
