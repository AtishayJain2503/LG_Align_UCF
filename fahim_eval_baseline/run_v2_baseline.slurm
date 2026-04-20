#!/bin/bash
#SBATCH --job-name=fahim_baseline
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=highgpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=32G

echo "Starting Fahim Baseline job"
date

module load python/python-3.11.4-gcc-12.2.0

cd $SLURM_SUBMIT_DIR

# ── Reuse the already-working venv from optimized_arch ──────────────────────
# That venv was verified working (20-epoch training ran successfully).
# If it doesn't exist for some reason, fall back to building our own.
SHARED_VENV="$HOME/LG_Align_UCF/optimized_arch/venv"
LOCAL_VENV="$SLURM_SUBMIT_DIR/venv"

if [ -d "$SHARED_VENV" ] && "$SHARED_VENV/bin/python" -c "import torch" 2>/dev/null; then
    echo "Reusing shared venv from optimized_arch..."
    source "$SHARED_VENV/bin/activate"
else
    echo "Shared venv not usable, building local venv..."
    if [ ! -d "$LOCAL_VENV" ]; then
        python3 -m venv "$LOCAL_VENV"
    fi
    source "$LOCAL_VENV/bin/activate"
    pip install --upgrade pip
    pip install "setuptools<70" wheel
    pip install git+https://github.com/openai/CLIP.git
    pip install -r requirements.txt
fi

# Create output folders
mkdir -p logs model_weights rank info losses fig embeddings

# GPU check
echo "--- GPU DIAGNOSTICS ---"
nvidia-smi || echo "nvidia-smi failed!"
python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available())"
echo "-----------------------"

echo "Running main.py..."
python main.py

echo "Job completed"
date
