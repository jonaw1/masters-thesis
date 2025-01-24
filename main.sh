#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --output=example.out
#SBATCH --error=error.out
#SBATCH --mem=80GB
#SBATCH --gpus=1

source ~/miniforge3/etc/profile.d/conda.sh
conda activate myenv
conda env update --file env.yml --prune

MODEL_ARG=$1
case "$MODEL_ARG" in
  gpt2)
    MODEL_NAME="gpt2-xl"
    ;;
  gptj)
    MODEL_NAME="EleutherAI/gpt-j-6B"
    ;;
  *)
    echo "Invalid model name. Use 'gpt2' or 'gptj'."
    exit 1
    ;;
esac

python3 src/main.py --model "$MODEL_NAME"

conda deactivate
