#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time 0-03:00:00
#SBATCH --job-name=crystallm_preprocess
#SBATCH --array 0
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=6G
#SBATCH --output=logs/preprocess_%A_%a.out

# Function to display help message
usage() {
  echo "Usage: $0 [-i <arg1>] [-o <arg2>] [-w <arg3>]"
  exit 1
}

# Parsing command-line arguments
while getopts ":i:o:w:" opt; do
  case $opt in
    i) arg1="$OPTARG"
      ;;
    o) arg2="$OPTARG"
      ;;
    w) arg3="$OPTARG"
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
if [ -z "$arg1" ] || [ -z "$arg2" ] || [ -z "$arg3" ]; then
  echo "Error: All arguments -i (in), -o (out), and -w (workers) must be provided."
  usage
fi

# Display the arguments
echo "in: $arg1"
echo "out: $arg2"
echo "workers: $arg3"

echo "Running preprocessing with the provided arguments..."

python bin/preprocess.py "$arg1" --out "$arg2" --workers "$arg3"
