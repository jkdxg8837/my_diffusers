#!/bin/bash
#
#SBATCH --job-name=analyze_weight
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --account=su007-hw-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --output=../joboutput/analyze_weight1_%j.out
#SBATCH --error=../joboutput/analyze_weight1_%j.err

## Initialisation ##
source /home/j/jiayang_gu/.bashrc
source activate zimage

## Run the analysis ##
python -u analyze_weight1.py
