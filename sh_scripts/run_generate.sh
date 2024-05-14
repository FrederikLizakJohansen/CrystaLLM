#!/bin/bash
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time 0-05:00:00
#SBATCH --job-name=crystallm_gen
#SBATCH --array 0
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --output=logs/generate_CHILI-100K_small_finetune_%A_%a.out

python generate_samples.py
