#!/bin/bash
#SBATCH --job-name=f2f-ft-pair
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/finetune_pair_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:-.venv}/bin/activate"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

pids=()
gpu=0
for spec in "$@"; do
  log="logs/finetune_${SLURM_JOB_ID}_gpu${gpu}.out"
  echo "GPU ${gpu}: finetune_prithvi.py ${spec} -> ${log}"
  CUDA_VISIBLE_DEVICES=${gpu} python src/fire2flood_pipeline/finetune_prithvi.py ${spec} > "${log}" 2>&1 &
  pids+=($!)
  gpu=$((gpu + 1))
done

status=0
for p in "${pids[@]}"; do
  wait "${p}" || status=1
done
exit ${status}
