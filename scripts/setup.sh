#!/bin/bash
# =============================================================================
# setup.sh -- run ONCE on the login node.
#
# Creates a Python venv on the shared filesystem so every worker node can see
# the same interpreter and packages, then installs the pinned stack.
#
# Why a venv and not Miniconda: every node here runs the same image with
# /usr/bin/python3 == 3.12, so a venv on shared NFS is sufficient and has far
# fewer moving parts than a conda install on NFS.
#
# Usage:
#   bash scripts/setup.sh
#   source /mnt/data/qwen38-demo/activate.sh
# =============================================================================
set -euo pipefail

DEMO_DIR="${DEMO_DIR:-/mnt/data/qwen38-demo}"
VENV_DIR="$DEMO_DIR/venv"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Qwen3.8-27B demo setup ==="
echo "DEMO_DIR: $DEMO_DIR"

mkdir -p "$DEMO_DIR"/{models,datasets,output,results,logs,repo}

# --- Sanity-check the hardware we are pinning wheels for -------------------
# B300 is sm_103. If you are on a different GPU, the cu130 pin in
# requirements.txt may be wrong for you.
if command -v sinfo >/dev/null 2>&1; then
    echo ""
    echo "Cluster GPUs:"
    sinfo -o "  %P %N %G" | sed 1d
fi

# --- Step 1: venv ----------------------------------------------------------
if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo ""
    echo "Creating venv at $VENV_DIR (Python $(python3 --version 2>&1 | awk '{print $2}')) ..."
    python3 -m venv "$VENV_DIR"
else
    echo "venv already exists at $VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --quiet --upgrade pip setuptools wheel

# --- Step 2: install the pinned stack --------------------------------------
# A single `pip install -r` resolves the whole file at once, so the order of
# lines in requirements.txt is not install order -- see the corrected notes at
# the top of that file. What matters is the --extra-index-url, which makes the
# +cu130 local version resolvable.
echo ""
echo "Installing pinned stack (this takes several minutes -- ~8GB installed)..."
"$VENV_DIR/bin/pip" install -r "$REPO_DIR/requirements.txt"

# --- Step 3: copy scripts to the shared filesystem -------------------------
# Slurm jobs run on the workers, which can read $DEMO_DIR but not necessarily
# your home checkout. Keep the shared copy in sync with the repo.
echo ""
echo "Syncing repo to $DEMO_DIR/repo ..."
# rsync, not `cp scripts/*`, for two reasons that both bit in practice:
#   * cp without -r exits 1 on any subdirectory, and `set -e` then aborts this
#     script before the activate.sh and GPU-verification steps below. A
#     __pycache__ appears the moment anyone imports sft_common from the repo,
#     so this is a normal state, not an exotic one.
#   * cp never removes anything, so a renamed or deleted script lingers in
#     $DEMO_DIR forever and the Slurm jobs keep executing the stale copy.
# Sync into $DEMO_DIR/repo, NOT $DEMO_DIR directly: the weights live in
# $DEMO_DIR/models/Qwen3.8-27B and the code in models/qwen3.8-27b/, which differ
# only by case. Keeping the repo under its own prefix keeps code and data apart
# and makes "what do the Slurm jobs actually execute" answerable with one path.
mkdir -p "$DEMO_DIR/repo"
for d in common models scripts; do
    rsync -a --delete --exclude='__pycache__' \
        "$REPO_DIR/$d/" "$DEMO_DIR/repo/$d/"
done

# --- Step 4: activation helper ---------------------------------------------
cat > "$DEMO_DIR/activate.sh" << ACTIVATE
#!/bin/bash
# source /mnt/data/qwen38-demo/activate.sh
export DEMO_DIR="$DEMO_DIR"
source "$VENV_DIR/bin/activate"
echo "Qwen3.8 demo env active. DEMO_DIR=\$DEMO_DIR"
ACTIVATE
chmod +x "$DEMO_DIR/activate.sh"

# --- Step 5: verify the GPU build actually works ---------------------------
echo ""
echo "=== Verifying torch sees the B300s (submitting a 1-GPU job) ==="
srun --partition="${PARTITION:-main}" --nodes=1 --ntasks=1 --gpus-per-node=1 \
     --time=00:05:00 \
     "$VENV_DIR/bin/python" -c "
import torch
print('torch          :', torch.__version__)
print('cuda available :', torch.cuda.is_available())
print('device         :', torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
print('compute cap    :', f'sm_{cap[0]}{cap[1]}')
x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
print('matmul check   :', bool(torch.isfinite(x @ x).all()))
import transformers, peft
print('transformers   :', transformers.__version__)
print('peft           :', peft.__version__)
" || {
    echo ""
    echo "!! GPU verification FAILED."
    echo "!! NOTE: '+cu130' is not the thing to check -- no published torch"
    echo "!! 2.13.0 wheel contains sm_103 SASS at all. B300 runs the sm_100"
    echo "!! code path under Blackwell minor-version compatibility, and that is"
    echo "!! expected and working. Look instead at: driver/CUDA version on the"
    echo "!! worker (needs CUDA 13), whether the GPU was actually allocated"
    echo "!! (nvidia-smi inside the job), and the full error from torch."
    exit 1
}

echo ""
echo "=== Setup complete ==="
echo "Next:  source $DEMO_DIR/activate.sh"
echo "       bash $DEMO_DIR/repo/models/qwen3.8-27b/download.sh"
