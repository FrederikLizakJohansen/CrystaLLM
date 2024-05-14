#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time 1-10:00:00
#SBATCH --job-name=crystallm_eval
#SBATCH --array 0
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=6G
#SBATCH --output=logs/eval_crystallm_base_%A_%a.out

python bin/evaluate_cifs.py generated_crystallm_base_test.tar.gz -o crystallm_base_test_eval.csv --workers 12
