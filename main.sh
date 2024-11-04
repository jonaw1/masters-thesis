#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --output=example.out
#SBATCH --error=error.out
#SBATCH --mem=20GB
#SBATCH --gpus=1

module purge
module load cuda/11.7
source ~/miniforge3/etc/profile.d/conda.sh
conda activate myenv
python3 src/main.py
conda deactivate
