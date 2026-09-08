#!/bin/bash
# =============================================================================
# evaluate.sbatch -- score a base model against one or more fine-tunes on the
# held-out SQL test split.
#
# One GPU is plenty: 27B in bf16 is ~52GB against 268.6 GiB of HBM, and the models
# are loaded one at a time.
#
# Every argument is passed straight through to evaluate.py, so any flag it
# accepts works here. This used to take three positional arguments and forward
# only three flags, which meant --prefix could not be set -- so the documented
# full-fine-tune comparison silently overwrote the LoRA one's results and
# labelled them as LoRA.
#
# HOW TO USE:
#   # base vs merged LoRA, 500 examples (the default):
#   sbatch /mnt/data/qwen38-demo/repo/models/qwen3.8-27b/evaluate.sbatch
#
#   # three-way: base vs merged LoRA vs full fine-tune, one chart, base
#   # generated once:
#   sbatch /mnt/data/qwen38-demo/repo/models/qwen3.8-27b/evaluate.sbatch \
#       --tuned-model /mnt/data/qwen38-demo/output/qwen3.8-27b-sql \
#       --tuned-model /mnt/data/qwen38-demo/output/qwen3.8-27b-sql-full
#
#   # cheap smoke:
#   sbatch /mnt/data/qwen38-demo/repo/models/qwen3.8-27b/evaluate.sbatch --num-examples 100
#
# Results land in /mnt/data/qwen38-demo/results/, under a filename prefix
# derived from the tuned model names unless --prefix says otherwise.
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

# With no arguments, score the base model against the merged LoRA checkpoint.
# evaluate.py owns every other default, including --num-examples.
if [ "$#" -eq 0 ]; then
    set -- --base-model "$DEMO_DIR/models/Qwen3.8-27B" \
           --tuned-model "$DEMO_DIR/output/qwen3.8-27b-sql"
fi

echo "=========================================="
echo "evaluate.py $*"
echo "=========================================="

python "$DEMO_DIR/repo/common/evaluate.py" "$@"
