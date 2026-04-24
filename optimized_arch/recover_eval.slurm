#!/bin/bash
#SBATCH --job-name=recover_eval_v6
#SBATCH --output=logs/recover_eval-%j.out
#SBATCH --error=logs/recover_eval-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=highgpu
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G

echo "Starting V6 Recover Eval"
date

module load python/python-3.11.4-gcc-12.2.0

# Thread limits — prevents OpenBLAS from spawning 32 threads on a login-constrained node
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# HuggingFace cache — avoid re-downloading CLIP weights
export HF_HOME="$HOME/LG_Align_UCF/optimized_arch/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

cd $SLURM_SUBMIT_DIR

source venv/bin/activate

# Pass the ExpID as an argument: sbatch recover_eval.slurm 76837247
EXP_ID=${1:-76837247}
echo "Evaluating Experiment ID: $EXP_ID"

python recover_eval.py $EXP_ID

echo "Eval completed"
date
