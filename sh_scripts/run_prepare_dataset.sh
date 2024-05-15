#!/bin/bash
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time 0-05:00:00
#SBATCH --job-name=crystallm_make
#SBATCH --array 0
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=6G
#SBATCH --output=logs/make_%A_%a.out

# Function to display help message
#usage() {
#  echo "Usage: $0 [-c <arg1>] [-d <arg2>] [-w <arg3>]"
#  exit 1
#}
#
## Parsing command-line arguments
#while getopts ":c:d:w:" opt; do
#  case $opt in
#    c) arg1="$OPTARG"
#      ;;
#    d) arg2="$OPTARG"
#      ;;
#    w) arg3="$OPTARG"
#      ;;
#    \?) echo "Invalid option -$OPTARG" >&2
#        usage
#      ;;
#    :) echo "Option -$OPTARG requires an argument." >&2
#        usage
#      ;;
#  esac
#done
#
## Check if all arguments are provided
#if [ -z "$arg1" ] || [ -z "$arg2" ] || [ -z "$arg3" ]; then
#  echo "Error: All arguments -c (cif_pkl), -d (dataset_name), and -w (workers) must be provided."
#  usage
#fi
#
## Display the arguments
#echo "cif_pkl: $arg1"
#echo "dataset_name: $arg2"
#echo "workers: $arg3"
#
#echo "Preparing dataset with the provided arguments..."

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

python bin/prepare_dataset.py "${ARGS[@]}"
