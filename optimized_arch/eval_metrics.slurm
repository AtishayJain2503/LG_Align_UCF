#!/bin/bash
#SBATCH --job-name=lgalign_eval
#SBATCH --output=logs/slurm_eval-%j.out
#SBATCH --error=logs/slurm_eval-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=highgpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --mem=96G

set -e

# Load modern python explicitly
module load python/python-3.11.4-gcc-12.2.0

cd $SLURM_SUBMIT_DIR

# Activate the venv they used for training
source venv/bin/activate

# Execute the python recovery with whatever argument was passed!
python recover_eval.py $1
