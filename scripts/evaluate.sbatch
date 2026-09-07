#!/bin/bash
# =============================================================================
# evaluate.sbatch -- score base vs fine-tuned on the held-out SQL test split.
#
# One GPU is plenty: 27B in bf16 is ~50GB against 275GB of HBM, and the two
# models are loaded one at a time.
#
# HOW TO USE:
#   # merged checkpoint (default):
#   sbatch /mnt/data/qwen38-demo/scripts/evaluate.sbatch
#
#   # explicit paths:
#   sbatch /mnt/data/qwen38-demo/scripts/evaluate.sbatch \
#       /mnt/data/qwen38-demo/models/Qwen3.8-27B \
#       /mnt/data/qwen38-demo/output/qwen3.8-27b-sql
#
# Results land in /mnt/data/qwen38-demo/results/
# =============================================================================

#SBATCH --job-name=qwen38-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=main
#SBATCH --output=/mnt/data/qwen38-demo/logs/evaluate_%j.out
#SBATCH --error=/mnt/data/qwen38-demo/logs/evaluate_%j.err

set -euo pipefail

export DEMO_DIR="${DEMO_DIR:-/mnt/data/qwen38-demo}"
source "$DEMO_DIR/venv/bin/activate"

BASE_MODEL=${1:-$DEMO_DIR/models/Qwen3.8-27B}
TUNED_MODEL=${2:-$DEMO_DIR/output/qwen3.8-27b-sql}
NUM_EXAMPLES=${3:-100}

echo "=========================================="
echo "Base  : $BASE_MODEL"
echo "Tuned : $TUNED_MODEL"
echo "N     : $NUM_EXAMPLES"
echo "=========================================="

python "$DEMO_DIR/scripts/evaluate.py" \
    --base-model "$BASE_MODEL" \
    --tuned-model "$TUNED_MODEL" \
    --num-examples "$NUM_EXAMPLES"
