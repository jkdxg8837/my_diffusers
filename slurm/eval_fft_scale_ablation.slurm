#!/bin/bash
#
#SBATCH --job-name=eval_fft_scale
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --account=su007-hw-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

#SBATCH --time=2-00:00:00
#SBATCH --output=../joboutput/joboutput_%j.out
#SBATCH --error=../joboutput/joboutput_%j.err

## Initialisation ##

source /home/j/jiayang_gu/.bashrc
source activate zimage

## Execute ##
RANK=64
for SCALE in 1 5 10 30 50 100 500 1000; do
    echo "===== Running rank=${RANK} scale_factor=${SCALE} ====="
    python -u eval_fft_lora_init.py --rank ${RANK} --scale_factor ${SCALE}
done
