#!/bin/bash
# =============================================================================
# train_lora.sbatch -- LoRA fine-tune Qwen3.8-27B across 2 nodes x 8 B300.
#
# HOW TO USE:
#   sbatch /mnt/data/qwen38-demo/scripts/train_lora.sbatch
#
# MONITORING:
#   squeue --me
#   tail -f /mnt/data/qwen38-demo/logs/train_lora_<JOBID>.out
#   sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,ExitCode
# =============================================================================

#SBATCH --job-name=qwen38-lora
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # one task per node; torchrun forks the 8 GPU procs
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --partition=main
#SBATCH --output=/mnt/data/qwen38-demo/logs/train_lora_%j.out
#SBATCH --error=/mnt/data/qwen38-demo/logs/train_lora_%j.err

set -euo pipefail

export DEMO_DIR="${DEMO_DIR:-/mnt/data/qwen38-demo}"
source "$DEMO_DIR/venv/bin/activate"

# --- What to train -----------------------------------------------------------
export MODEL_PATH="$DEMO_DIR/models/Qwen3.8-27B"
# Overridable so A/B arms can write to separate directories without editing
# this file (see LORA_TARGET_GDN and #12).
export OUTPUT_DIR="${OUTPUT_DIR:-$DEMO_DIR/output/qwen3.8-27b-sql-lora}"
# 0 = attention + MLP only. Measured in #12: adding the 48 Gated DeltaNet
# projections costs 58.2M trainable params and ~15% throughput and changes no
# predictions (A-only=0 at both 25 and 500 steps). Set to 1 to re-enable.
export LORA_TARGET_GDN="${LORA_TARGET_GDN:-0}"
export DATASET_PATH="$DEMO_DIR/datasets/sql-create-context"

# 8 per GPU x 16 GPUs x 1 accum = effective batch 128, matching the Qwen3-era
# runs so the accuracy numbers stay comparable. Measured: 584 optimizer steps
# over the 74,648-example train split (78,577 raw, less the 5% eval split).
#
# There is a lot of headroom here: B300 has 268.6 GiB HBM against roughly 3GB/rank
# of sharded frozen base weights, so activations dominate. Raise
# PER_DEVICE_BATCH_SIZE before reaching for gradient accumulation.
export PER_DEVICE_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=1
export NUM_EPOCHS=1
export LEARNING_RATE=2e-4            # LoRA tolerates ~10x the full-FT rate
export MAX_SEQ_LEN=1024


# --- Short-run knobs ---------------------------------------------------------
# Defaults reproduce a full single-epoch run exactly. Override per-submission:
#   MAX_STEPS=25 SAVE_STEPS=10 MAX_EVAL_EXAMPLES=200 \
#       sbatch --export=ALL scripts/train_lora.sbatch
export MAX_STEPS="${MAX_STEPS:--1}"           # -1 = run NUM_EPOCHS epochs
export SAVE_STEPS="${SAVE_STEPS:-500}"
export EVAL_STEPS="${EVAL_STEPS:-$SAVE_STEPS}"
export MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:--1}"   # -1 = all 3,929

# Every rank imports sft_common from shared NFS, so without this all 16 race to
# write .pyc files into $DEMO_DIR/scripts/__pycache__. Same class of problem as
# TRITON_CACHE_DIR above and disable_caching() in prepare_datasets.
export PYTHONDONTWRITEBYTECODE=1

export LORA_R=32
export LORA_ALPHA=64
export LORA_DROPOUT=0.05

# --- Interconnect ------------------------------------------------------------
export NCCL_DEBUG=WARN               # INFO if you need to debug IB reachability
export NCCL_IB_DISABLE=0             # InfiniBand between nodes
export NCCL_NET_GDR_LEVEL=5          # GPUDirect RDMA
# Loading the 52GB base checkpoint across 16 ranks is hundreds of broadcasts,
# and the 1800s default is not generous over shared NFS. Pairs with
# ddp_timeout=7200 in train_lora.py. Unlike train_full.sbatch we leave the NCCL
# watchdog enabled: this path has no long state-dict gather to trip it, and the
# monitoring is worth keeping.
export NCCL_TIMEOUT=7200

# Triton compiles kernels per rank; point its cache at node-local scratch so 16
# ranks are not writing into the same directory on shared NFS. serve.sbatch
# already does this.
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-$SLURM_JOB_ID"
mkdir -p "$TRITON_CACHE_DIR"

# Sequence packing is unavailable for this architecture (the Gated DeltaNet
# recurrent state cannot be reset mid-sequence), so expect lower tokens/s per
# GPU than an equivalent dense Qwen3 run.

# --- Rendezvous --------------------------------------------------------------
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500

echo "=========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Nodes      : $SLURM_JOB_NODELIST"
echo "Master     : $MASTER_ADDR:$MASTER_PORT"
echo "GPUs/node  : $SLURM_GPUS_ON_NODE"
echo "Total GPUs : $(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))"
echo "Model      : $MODEL_PATH"
echo "Output     : $OUTPUT_DIR"
echo "=========================================="

srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$DEMO_DIR/scripts/train_lora.py"
