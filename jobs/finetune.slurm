#!/bin/bash
#SBATCH --job-name=f2f-finetune
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/finetune_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:-.venv}/bin/activate"
nvidia-smi --query-gpu=name,memory.total --format=csv
python src/fire2flood_pipeline/finetune_prithvi.py "$@"
