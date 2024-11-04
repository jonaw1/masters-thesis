#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --output=example.out
#SBATCH --error=error.out
#SBATCH --mem=80GB
#SBATCH --gpus=1

source ~/miniforge3/etc/profile.d/conda.sh
conda activate myenv
python3 src/main.py
conda deactivate
