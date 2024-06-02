#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time 1-10:00:00
#SBATCH --job-name=crystallm_eval
#SBATCH --array 0
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=6G
#SBATCH --output=logs/eval_%A_%a.out

python bin/evaluate_cifs.py gen_v1_small_CHILI-3K.tar.gz -o eval_v1_small_CHILI-3K.csv --workers 12
