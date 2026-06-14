#!/bin/bash
#SBATCH --job-name=adp-pithtrain-qwen35-smoke
#SBATCH --time=1:00:00
#SBATCH --partition=flame
#SBATCH --qos=flame-16gpu_qos
#SBATCH --account=gneubig
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --output=/home/gneubig/workspace/project/7511d8c2946a4a53a9b2a4f643840479/pithtrain_runs/adp_qwen35_35b_a3b_smoke/logs/slurm_%x_%j.out

set -euo pipefail

REPO=/home/gneubig/workspace/project/7511d8c2946a4a53a9b2a4f643840479/pith-train
RUN_ROOT=/home/gneubig/workspace/project/7511d8c2946a4a53a9b2a4f643840479/pithtrain_runs/adp_qwen35_35b_a3b_smoke

cd "$REPO"
source .venv/bin/activate

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/wandb" "$RUN_ROOT/hf_cache"

export HF_HOME="$RUN_ROOT/hf_cache"
export HF_HUB_CACHE="$RUN_ROOT/hf_cache/hub"
export WANDB_DIR="$RUN_ROOT/wandb"
export WANDB_PROJECT="${WANDB_PROJECT:-adp-experiments}"
export WANDB_NAME="${WANDB_NAME:-adp-pithtrain-qwen35-35b-a3b-adp-seq2048-gbs32-pp2-ep8-2node-h100-smoke-${SLURM_JOB_ID}}"
export WANDB_GROUP="${WANDB_GROUP:-adp-pithtrain-smoke}"

export PITHTRAIN_MODEL_CONFIG="${PITHTRAIN_MODEL_CONFIG:-/project/flame/gneubig/adp/models/Qwen3.5-35B-A3B}"
export PITHTRAIN_SAVE_LOCATION="${PITHTRAIN_SAVE_LOCATION:-/project/flame/gneubig/adp/pithtrain_checkpoints/qwen3.5-35b-a3b}"
export PITHTRAIN_DATASET="${PITHTRAIN_DATASET:-$RUN_ROOT/toktxt/qwen35}"
export PITHTRAIN_SEQUENCE_LENGTH="${PITHTRAIN_SEQUENCE_LENGTH:-2048}"
export PITHTRAIN_GLOBAL_BATCH_SIZE="${PITHTRAIN_GLOBAL_BATCH_SIZE:-32}"
export PITHTRAIN_MICRO_BATCH_SIZE="${PITHTRAIN_MICRO_BATCH_SIZE:-1}"
export PITHTRAIN_MAX_STEPS="${PITHTRAIN_MAX_STEPS:-12}"
export PITHTRAIN_PIPELINE_PARALLEL_SIZE="${PITHTRAIN_PIPELINE_PARALLEL_SIZE:-2}"
export PITHTRAIN_EXPERT_PARALLEL_SIZE="${PITHTRAIN_EXPERT_PARALLEL_SIZE:-8}"
export PITHTRAIN_CONTEXT_PARALLEL_SIZE="${PITHTRAIN_CONTEXT_PARALLEL_SIZE:-1}"

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp6s0f0,enp141s0f0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SLURM_LAUNCH_NODE_IPADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

srun -W 0 examples/pretrain_lm/launch.sh adp-qwen35-35b-a3b-adp-smoke
