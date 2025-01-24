#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --output=example.out
#SBATCH --error=error.out
#SBATCH --mem=80GB
#SBATCH --gpus=1

source ~/miniforge3/etc/profile.d/conda.sh
conda activate myenv
pip install -r "./src/EasyEdit/requirements.txt"

python3 src/edit.py

conda deactivate
