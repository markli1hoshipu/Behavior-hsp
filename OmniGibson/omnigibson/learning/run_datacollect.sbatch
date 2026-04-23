#!/bin/bash
#SBATCH --job-name=b1k_datacollect
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared_work/logs/%j_%x.out
#SBATCH --error=/shared_work/logs/%j_%x.err

set -euo pipefail

# --- Configurable parameters ---
# Pass via sbatch --export=... or positional args.
export TASK_NAME="${1:-picking_up_trash}"
export INSTANCE_ID="${INSTANCE_ID:-0}"
export LOG_DIR="${LOG_DIR:-/shared_work/logs/behavior_datacollect/${TASK_NAME}}"
export RECORDING_PATH="${RECORDING_PATH:-${LOG_DIR}/trajectories/${TASK_NAME}_${INSTANCE_ID}.hdf5}"

mkdir -p "${LOG_DIR}/trajectories"

# --- Per-job unique appdata (avoid Isaac Sim cache collisions across jobs on the same node) ---
export OMNIGIBSON_APPDATA_PATH="/tmp/og_appdata_${SLURM_JOB_ID}"
mkdir -p "${OMNIGIBSON_APPDATA_PATH}"

# --- Conda env ---
export CONDA_BASE="/shared_work/miniconda3"
export BEHAVIOR_ENV="${CONDA_BASE}/envs/behavior"
source "${CONDA_BASE}/bin/activate" "${BEHAVIOR_ENV}"

# --- Isaac Sim / OmniGibson ---
export OMNIGIBSON_HEADLESS=1
export OMNI_KIT_ACCEPT_EULA=YES
# Blackwell-safe GPU selection: use OMNIGIBSON_GPU_ID, not CUDA_VISIBLE_DEVICES
# (CUDA_VISIBLE_DEVICES causes close_stage() segfault on Isaac Sim 5.1 + sm_120).
# Slurm allocates one GPU per job via --gres=gpu:1; inside the job it appears as GPU 0.
export OMNIGIBSON_GPU_ID=0

# --- Dataset path (shared on trt cluster) ---
export OG_DATA_PATH="/shared_work/BEHAVIOR-1K/datasets"

echo '==========================================' 
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Task:         ${TASK_NAME}"
echo "Instance ID:  ${INSTANCE_ID}"
echo "Log dir:      ${LOG_DIR}"
echo "Recording:    ${RECORDING_PATH}"
echo "Appdata:      ${OMNIGIBSON_APPDATA_PATH}"
echo "GPU allocated: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start:        $(date)"
echo '=========================================='

cd /shared_work/BEHAVIOR-1K

python OmniGibson/omnigibson/learning/eval_data_gen.py \
    task.name="${TASK_NAME}" \
    instance_id="${INSTANCE_ID}" \
    log_path="${LOG_DIR}" \
    recording_path="${RECORDING_PATH}"

echo "Done: $(date)"

# Clean up per-job appdata on success
rm -rf "${OMNIGIBSON_APPDATA_PATH}"
