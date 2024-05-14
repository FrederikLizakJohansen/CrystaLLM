#!/bin/bash
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time 0-05:00:00
#SBATCH --job-name=crystallm_train
#SBATCH --array 0
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=6G
#SBATCH --output=logs/train_%A_%a.out

# Function to display help message
usage() {
  echo "Usage: $0 [-c <arg1>]"
  exit 1
}

# Parsing command-line arguments
while getopts ":c:" opt; do
  case $opt in
    c) arg1="$OPTARG"
      ;;
    \?) echo "Invalid option -$OPTARG" >&2
        usage
      ;;
    :) echo "Option -$OPTARG requires an argument." >&2
        usage
      ;;
  esac
done

# Check if all arguments are provided
if [ -z "$arg1" ]; then
  echo "Error: -c (config) must be provided."
  usage
fi

# Display the arguments
echo "config: $arg1"

echo "Running training with the provided arguments..."

python bin/train.py --config="$arg1"
