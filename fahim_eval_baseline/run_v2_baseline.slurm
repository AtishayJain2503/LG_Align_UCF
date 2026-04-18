#!/bin/bash
#SBATCH --job-name=lgalign_train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=highgpu    # Use your cluster's high GPU partition
#SBATCH --gpus=1               # Explicitly request 1 GPU
#SBATCH --time=24:00:00        # Time limit
#SBATCH --mem=32G              # Memory limit

# STOP ON ERROR: Exit script immediately if any command fails (e.g. out of memory, missing packages)
set -e

echo "Starting LGAlign training job"
date

# Try to load modern python
module load python/python-3.11.4-gcc-12.2.0
# module load cuda/12.1

# Navigate to the project directory
cd $SLURM_SUBMIT_DIR

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and base tools FIRST (Ensure setuptools < 70 because newer versions deleted pkg_resources!)
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip "setuptools<70" wheel

# Install CLIP from GitHub using our isolated backwards-compatible tools
echo "Installing CLIP..."
pip install git+https://github.com/openai/CLIP.git --no-build-isolation
echo "Installing requirements..."
pip install -r requirements.txt

# Create output folders just in case
mkdir -p logs model_weights rank info losses fig embeddings

# Check GPU availability
echo "--- GPU DIAGNOSTICS ---"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi || echo "nvidia-smi failed!"
python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available()); print('Device Count:', torch.cuda.device_count())"
echo "-----------------------"

# Run the training script
echo "Running main.py..."

python main.py

echo "Job completed"
date
