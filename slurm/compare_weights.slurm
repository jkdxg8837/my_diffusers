#!/bin/bash
#
#SBATCH --job-name=compare_weights
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --account=su007-hw-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --output=../joboutput/compare_weights_%j.out
#SBATCH --error=../joboutput/compare_weights_%j.err

## Initialisation ##
source /home/j/jiayang_gu/.bashrc
source activate omini
export WANDB_KEY="05e08cadfa8f29b4fb78407724e732c453648de6"
export PROJECT="compare_weights"

## Run the 3 training experiments ##
python compare_weights.py

## Run the analysis ##
python analyze_weights.py
