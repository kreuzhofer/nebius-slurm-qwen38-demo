#!/bin/bash
# =============================================================================
# train_full.sbatch -- full-parameter fine-tune of Qwen3.8-27B on 2 nodes x 8 B300.
#
# All ~26.9B text parameters are trained, sharded with FSDP2. This is the
# variant the H100-era version of this demo could not run for a *smaller*
# model -- 268.6 GiB of HBM per GPU turns it into a routine job (~20-28GB/GPU for
# weights, gradients and optimizer state).
#
# HOW TO USE:
#   sbatch /mnt/data/qwen38-demo/scripts/train_full.sbatch
#
#   # with mid-run checkpoints (costs a ~54GB gather+write each time):
#   SAVE_STRATEGY=steps SAVE_STEPS=200 sbatch --export=ALL \
#       /mnt/data/qwen38-demo/scripts/train_full.sbatch
#
# MONITORING:
#   squeue --me
#   tail -f /mnt/data/qwen38-demo/logs/train_full_<JOBID>.out
#   sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,ExitCode
#
# The output checkpoint is directly usable -- no LoRA merge step:
#   sbatch scripts/evaluate.sbatch <base> <output>
#   sbatch scripts/serve.sbatch <output>
# =============================================================================

#SBATCH --job-name=qwen38-full
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # one task per node; torchrun forks the 8 GPU procs
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --partition=main
#SBATCH --output=/mnt/data/qwen38-demo/logs/train_full_%j.out
#SBATCH --error=/mnt/data/qwen38-demo/logs/train_full_%j.err

set -euo pipefail

export DEMO_DIR="${DEMO_DIR:-/mnt/data/qwen38-demo}"
source "$DEMO_DIR/venv/bin/activate"

# --- What to train -----------------------------------------------------------
export MODEL_PATH="$DEMO_DIR/models/Qwen3.8-27B"
export OUTPUT_DIR="$DEMO_DIR/output/qwen3.8-27b-sql-full"
export DATASET_PATH="$DEMO_DIR/datasets/sql-create-context"

# 8 per GPU x 16 GPUs x 1 accum = effective batch 128, the same as
# train_lora.sbatch and as the Qwen3-era runs, so accuracy stays comparable.
# ~583 steps for 74.6k examples.
export PER_DEVICE_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=1
export NUM_EPOCHS=1
export LEARNING_RATE=1e-5            # full FT wants ~1/20th of the LoRA rate
export MAX_SEQ_LEN=1024

# Checkpointing: "no" writes the model once at the end. See the long comment in
# train_full.py -- a FULL_STATE_DICT gather of a 27B model is ~54GB and doing it
# every few hundred steps is what blew the distributed timeout in the Qwen3-era
# pipeline. Override to trade I/O for resumability.

# --- Short-run knobs ---------------------------------------------------------
# Defaults reproduce a full single-epoch run exactly. Override per-submission:
#   MAX_STEPS=25 SAVE_STEPS=10 MAX_EVAL_EXAMPLES=200 \
#       sbatch --export=ALL scripts/train_full.sbatch
export MAX_STEPS="${MAX_STEPS:--1}"           # -1 = run NUM_EPOCHS epochs
export SAVE_STEPS="${SAVE_STEPS:-500}"
export EVAL_STEPS="${EVAL_STEPS:-$SAVE_STEPS}"
export MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:--1}"   # -1 = all 3,929

# Every rank imports sft_common from shared NFS, so without this all 16 race to
# write .pyc files into $DEMO_DIR/scripts/__pycache__. Same class of problem as
# TRITON_CACHE_DIR above and disable_caching() in prepare_datasets.
export PYTHONDONTWRITEBYTECODE=1

export SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
export SAVE_STEPS="${SAVE_STEPS:-500}"

# --- Interconnect ------------------------------------------------------------
export NCCL_DEBUG=WARN               # INFO if you need to debug IB reachability
export NCCL_IB_DISABLE=0             # InfiniBand between nodes
export NCCL_NET_GDR_LEVEL=5          # GPUDirect RDMA
# The final state-dict gather is a long collective. Raise the NCCL watchdog to
# match ddp_timeout=7200 in train_full.py so it is not killed mid-save.
export NCCL_TIMEOUT=7200
export TORCH_NCCL_ENABLE_MONITORING=0

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
echo "Model      : $MODEL_PATH (full fine-tune)"
echo "Output     : $OUTPUT_DIR"
echo "Save       : $SAVE_STRATEGY"
echo "=========================================="

srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$DEMO_DIR/scripts/train_full.py"
