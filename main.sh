#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --output=example.out
#SBATCH --error=error.out
#SBATCH --mem=80GB
#SBATCH --gpus=1

source ~/miniforge3/etc/profile.d/conda.sh
conda activate myenv

MODEL_ARG=$1
case "$MODEL_ARG" in
  gpt2)
    MODEL_NAME="GPT2_XL_MODEL_NAME"
    ;;
  gptj)
    MODEL_NAME="GPT_J_MODEL_NAME"
    ;;
  *)
    echo "Invalid model name. Use 'gpt2' or 'gptj'."
    exit 1
    ;;
esac

python3 src/main.py --model "$MODEL_NAME"

conda deactivate
