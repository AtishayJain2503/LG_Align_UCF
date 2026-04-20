#!/bin/bash
#SBATCH --job-name=eval_fahim
#SBATCH --output=logs/eval_slurm-%j.out
#SBATCH --error=logs/eval_slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=highgpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G

echo "Starting Fast Evaluation job"
date

module load python/python-3.11.4-gcc-12.2.0

export HF_HOME="$HOME/LG_Align_UCF/optimized_arch/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

# Enforce strict thread limits to prevent OpenBLAS from crashing SLURM
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd $SLURM_SUBMIT_DIR

# Use the full optimized environment so no missing modules
source "$HOME/LG_Align_UCF/optimized_arch/venv/bin/activate"

# Run the fast evaluator
python eval_checkpoint_fast.py

echo "Job completed"
date
