#!/bin/bash
# =============================================================================
# serve.sbatch -- vLLM OpenAI-compatible server for Qwen3.8-27B.
#
# HOW TO USE:
#   # serve the merged fine-tuned model (1 GPU, port 8000):
#   sbatch /mnt/data/qwen38-demo/scripts/serve.sbatch
#
#   # serve the base model alongside it for A/B comparison:
#   sbatch /mnt/data/qwen38-demo/scripts/serve.sbatch \
#       /mnt/data/qwen38-demo/models/Qwen3.8-27B 8001
#
# TP=1 is the default and is sufficient: ~50GB of bf16 weights on a 275GB B300.
# The KV cache is unusually cheap for a 27B model because only 16 of the 64
# layers cache at all -- the other 48 are linear attention, which keeps a
# small fixed-size recurrent state per sequence instead.
#
# If you do want tensor parallelism, the head counts constrain the valid sizes:
# 24 query / 4 KV / 16 linear-key / 48 linear-value, all read from config.json
# and confirmed against the downloaded checkpoint.
#
# TP=16 fails on the **24 query heads**, not the linear-key count -- 16 divides
# 16 exactly, so the earlier note here had the wrong reason. TP=3 and TP=6 also
# fail, on key_dim=2048, after passing every head-count check. TP in {1,2,4,8}
# is the safe set.
#
# Still unverified on this cluster: only TP=1 has been run. There is also a
# reported silent post-load hang at TP=4 for Qwen3.5/3.6 models, though on
# different hardware and an older vLLM. Test before relying on multi-GPU.
#
# QUERYING:
#   squeue --me                # NODELIST tells you which worker it landed on
#   bash scripts/query.sh
# =============================================================================

#SBATCH --job-name=qwen38-serve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=main
#SBATCH --output=/mnt/data/qwen38-demo/logs/serve_%j.out
#SBATCH --error=/mnt/data/qwen38-demo/logs/serve_%j.err

set -euo pipefail

export DEMO_DIR="${DEMO_DIR:-/mnt/data/qwen38-demo}"
source "$DEMO_DIR/venv/bin/activate"

MODEL_PATH=${1:-$DEMO_DIR/output/qwen3.8-27b-sql}
PORT=${2:-8000}
TP_SIZE=${3:-1}
MAX_MODEL_LEN=${4:-8192}   # native is 262144; 8192 is ample for SQL and starts faster

# Triton/torch.compile caches on shared NFS race between concurrent jobs.
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-$SLURM_JOB_ID"
mkdir -p "$TRITON_CACHE_DIR"

echo "=========================================="
echo "Model     : $MODEL_PATH"
echo "Node      : $(hostname)"
echo "Port      : $PORT"
echo "TP size   : $TP_SIZE"
echo "Max len   : $MAX_MODEL_LEN"
echo ""
echo "Health    : curl http://$(hostname):$PORT/v1/models"
echo "=========================================="

# Flag notes:
#   --reasoning-parser qwen3
#       Effectively mandatory. The chat template opens every assistant turn
#       with <think>, so without a reasoning parser the whole reasoning block
#       lands in message.content and any exact-match scoring reads garbage.
#   --default-chat-template-kwargs '{"enable_thinking": false}'
#       Matches how the model was fine-tuned. Without it the template injects
#       a "reasoning effort is set to xhigh" system message and the model
#       reasons at length before answering -- slow, and it can overrun
#       max_tokens before emitting any SQL.
#   --language-model-only
#       VERIFIED against vllm 0.28.0 on this cluster: accepted, and it shows up
#       as 'language_model_only': True in the server's parsed non-default args.
#       It does NOT "skip building the vision tower to free KV cache" as this
#       comment used to claim -- it disables multimodal *inputs*. The flag that
#       skips vision profiling is --skip-mm-profiling.
#       On a MERGED checkpoint it is a no-op either way: merging writes
#       model_type qwen3_5_text with no vision_config, so there is no vision
#       tower to disable. It is kept because it is correct and harmless for
#       both the base (multimodal) and merged (text-only) paths.
vllm serve "$MODEL_PATH" \
    --served-model-name "$(basename "$MODEL_PATH")" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --language-model-only \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}'
