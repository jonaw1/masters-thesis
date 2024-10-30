#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --output=example.out
#SBATCH --error=error.out
#SBATCH --mem=20000M
#SBATCH --gpus=1

module purge
module load cuda/11.7
source .venv/bin/activate
python3 main.py
