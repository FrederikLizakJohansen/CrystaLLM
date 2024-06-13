#!/bin/bash
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time 1-12:00:00
#SBATCH --job-name=crystallm_make
#SBATCH --array 0
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=6G
#SBATCH --output=logs/make_%A_%a.out

# Function to display help message
usage() {
  echo "Usage: $0 [options]"
  echo "Pass any number of arguments and their values to the script, e.g. --a value1 --b value2"
  exit 1
}

# Check if any arguments are provided
if [ "$#" -eq 0 ]; then
  usage
fi

# Collect all aguments
ARGS=("$@")

# Display the arguments
echo "Arguments passed: ${ARGS[*]}"

python bin/prepare_dataset_prefix_prompt.py "${ARGS[@]}"
