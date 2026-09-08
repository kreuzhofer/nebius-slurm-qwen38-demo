#!/bin/bash
# =============================================================================
# download.sh -- fetch Qwen3.8-27B and the SQL dataset to shared storage.
#
# Both the login node and the workers have internet egress on this cluster, but
# downloading once to shared storage beats re-downloading per job.
#
# Model is ~56GB across 18 safetensors shards (27.8B params in bf16, which
# includes a vision tower and an MTP head we do not train -- see README).
#
# Usage:
#   source /mnt/data/qwen38-demo/activate.sh
#   bash /mnt/data/qwen38-demo/repo/models/qwen3.8-27b/download.sh
# =============================================================================
set -euo pipefail

DEMO_DIR="${DEMO_DIR:-/mnt/data/qwen38-demo}"
MODELS_DIR="$DEMO_DIR/models"
DATASETS_DIR="$DEMO_DIR/datasets"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"
MODEL_NAME="$(basename "$MODEL_ID")"

mkdir -p "$MODELS_DIR" "$DATASETS_DIR" "$DEMO_DIR/logs"

# --- Disk space check ------------------------------------------------------
AVAIL_GB=$(df -BG --output=avail "$DEMO_DIR" | tail -1 | tr -dc '0-9')
echo "Available on $(df --output=target "$DEMO_DIR" | tail -1): ${AVAIL_GB}GB"
if [ "$AVAIL_GB" -lt 130 ]; then
    echo "WARNING: <130GB free. You need ~56GB for the base model and another"
    echo "         ~56GB if you merge the LoRA adapter into a full checkpoint."
fi

# --- Model -----------------------------------------------------------------
echo ""
echo "=== Downloading $MODEL_ID (~56GB) ==="
if compgen -G "$MODELS_DIR/$MODEL_NAME/*.safetensors" > /dev/null; then
    echo "Already present at $MODELS_DIR/$MODEL_NAME, skipping."
else
    # `hf` is the current CLI; huggingface-cli is deprecated and its
    # --local-dir-use-symlinks flag is a no-op warning on recent hub versions.
    if command -v hf >/dev/null 2>&1; then
        hf download "$MODEL_ID" --local-dir "$MODELS_DIR/$MODEL_NAME"
    else
        huggingface-cli download "$MODEL_ID" --local-dir "$MODELS_DIR/$MODEL_NAME"
    fi
    echo "Downloaded to $MODELS_DIR/$MODEL_NAME"
fi

# --- Dataset ---------------------------------------------------------------
echo ""
echo "=== Downloading b-mc2/sql-create-context (~50MB) ==="
if [ -d "$DATASETS_DIR/sql-create-context" ]; then
    echo "Already present, skipping."
else
    python -c "
from datasets import load_dataset
ds = load_dataset('b-mc2/sql-create-context')
ds.save_to_disk('$DATASETS_DIR/sql-create-context')
print('Saved. Train examples:', len(ds['train']))
"
fi

echo ""
echo "=== Done ==="
echo "Next:  sbatch $DEMO_DIR/repo/models/qwen3.8-27b/train_lora.sbatch"
