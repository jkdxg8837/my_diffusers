#!/bin/bash
#
#SBATCH --job-name=eval_flux_lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:lovelace_l40:1
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
python eval_flux.py
