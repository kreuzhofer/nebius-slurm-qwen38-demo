# Qwen3.8-27B fine-tuning on a Nebius Soperator Slurm cluster

Fine-tunes [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) for
natural-language-to-SQL across 2 nodes × 8 NVIDIA B300, then evaluates it
against the base model and serves it with vLLM.

Two training paths, same data and same effective batch size so the results are
directly comparable:

| | script | trainable params | LR | output |
|---|---|---|---|---|
| **LoRA** | `train_lora.sbatch` | ~0.4% (adapter) | 2e-4 | adapter, needs a merge step |
| **Full** | `train_full.sbatch` | all ~26.9B | 1e-5 | standalone checkpoint |

Task: [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context)
(74,659 train / 3,930 held out). Metric: exact-match accuracy on 100 held-out
questions after light normalization.

## Cluster

Written for and verified against:

- 2 worker nodes × 8 **NVIDIA B300 SXM6** (275 GB HBM each, **sm_103**), 192 vCPU / 2.5 TB RAM per node
- Slurm 25.11.3 via [Soperator](https://github.com/nebius/nebius-solutions-library/tree/main/soperator), partition `main`
- Driver 580.159.04 / CUDA 13.0
- Shared filesystem at `/mnt/data`; internet egress from both login and worker nodes

Check yours with `sinfo -o "%P %N %G"` before trusting the pins in
`requirements.txt`.

## Quick start

```bash
# 0. one-time environment setup on the login node (~3GB of wheels)
git clone https://github.com/kreuzhofer/nebius-slurm-qwen38-lora-demo.git
cd nebius-slurm-qwen38-lora-demo
bash scripts/setup.sh                 # ends with a GPU smoke test on a worker
source /mnt/data/qwen38-demo/activate.sh

# 1. fetch model (~56GB) + dataset (~50MB)
bash /mnt/data/qwen38-demo/scripts/download.sh

# 2. train on 16 GPUs -- pick one
sbatch /mnt/data/qwen38-demo/scripts/train_lora.sbatch    # LoRA adapter
sbatch /mnt/data/qwen38-demo/scripts/train_full.sbatch    # all 26.9B params
squeue --me
tail -f /mnt/data/qwen38-demo/logs/train_lora_<JOBID>.out

# 3. LoRA ONLY -- merge the adapter into a standalone checkpoint.
#    Skip this entirely if you ran train_full.sbatch.
srun --partition=main --nodes=1 --gpus-per-node=1 --time=01:00:00 \
    python /mnt/data/qwen38-demo/scripts/merge_lora.py \
    /mnt/data/qwen38-demo/output/qwen3.8-27b-sql-lora \
    /mnt/data/qwen38-demo/models/Qwen3.8-27B \
    /mnt/data/qwen38-demo/output/qwen3.8-27b-sql

# 4. score base vs fine-tuned
sbatch /mnt/data/qwen38-demo/scripts/evaluate.sbatch      # defaults to the merged LoRA path
sbatch /mnt/data/qwen38-demo/scripts/evaluate.sbatch \
    /mnt/data/qwen38-demo/models/Qwen3.8-27B \
    /mnt/data/qwen38-demo/output/qwen3.8-27b-sql-full      # full fine-tune
#    -> /mnt/data/qwen38-demo/results/qwen3.8-27b-lora_results.{json,md} + .png

# 5. serve and query
sbatch /mnt/data/qwen38-demo/scripts/serve.sbatch
bash scripts/query.sh
```

Re-run `bash scripts/setup.sh` after editing anything in `scripts/` — it
re-syncs the shared copy under `/mnt/data/qwen38-demo/scripts/` that the Slurm
jobs actually execute.

## Layout

```
requirements.txt          pinned stack; torch must come from the cu130 index first
scripts/
  setup.sh                venv on shared NFS + install + GPU smoke test
  download.sh             model + dataset to shared storage
  sft_common.py           prompt building, label masking, FSDP2 config, loading
  train_lora.py           LoRA SFT: hybrid-attention target modules
  train_lora.sbatch       2 nodes x 8 GPUs via srun + torchrun
  train_full.py           full-parameter SFT: fused AdamW, one-shot save
  train_full.sbatch       same topology, longer collective timeouts
  merge_lora.py           adapter -> standalone checkpoint (LoRA path only)
  evaluate.py             base vs fine-tuned exact-match + chart + JSON
  evaluate.sbatch         1 GPU
  serve.sbatch            vLLM OpenAI server
  query.sh                one-shot SQL request
```

`sft_common.py` holds the prompt construction and label masking. Both training
scripts and `evaluate.py` import it, so training and scoring cannot drift apart
— which is exactly how the Qwen3-era `train.py` and `train_lora.py` diverged.

## What is different about Qwen3.8-27B

Worth reading before adapting this to another model, because several of these
bit during the port.

**It is not its own architecture.** The checkpoint reports
`model_type: "qwen3_5"` / `architectures: ["Qwen3_5ForConditionalGeneration"]`.
Everything in `transformers` and `vllm` lives under `qwen3_5`, not `qwen3_8`.

**It is multimodal, with no text-only checkpoint.** The weights include a
27-block vision tower (`model.visual.*`) and a multi-token-prediction head
(`mtp.*`). For text-only SFT, load through `AutoModelForCausalLM`: it resolves
to `Qwen3_5ForCausalLM`, whose
`_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]` drops
both cleanly and leaves the ~26.9B-parameter text stack.

**Attention is hybrid, and this changes the LoRA config.** Of 64 layers only
**16** use classic attention (`self_attn`, `full_attention_interval: 4`); the
other **48** are Gated DeltaNet linear attention with projections named
`linear_attn.in_proj_{qkv,z,a,b}` and `linear_attn.out_proj`. A Qwen3-era
target list of `q/k/v/o_proj` + MLP silently adapts only a quarter of the
token-mixing blocks. `train_lora.py` targets both kinds, and skips
`in_proj_a`/`in_proj_b` (both `[48, 5120]`, where rank > 48 is degenerate).

**Thinking is on by default** at `reasoning_effort='xhigh'`. `enable_thinking=False`
does two distinct jobs: it keeps an injected "Reasoning effort is set to
xhigh…" sentence out of the system message, and it makes the generation prompt
pre-fill an *empty* think block `<think>\n\n</think>\n\n`. The training script
masks that prefix along with the rest of the prompt — leave it unmasked and the
model learns to emit a second empty think block after the one the server
already gave it.

**`transformers` v5 moved the FSDP goalposts.** `fsdp_config["version"]`
defaults to `2`, so the FSDP1-only knobs (`backward_prefetch`,
`forward_prefetch`, `use_orig_params`) are silently ignored, `fsdp` as a string
is deprecated, and `auto_wrap` is a documented no-op. Under FSDP you also want
`activation_checkpointing` inside `fsdp_config` rather than
`TrainingArguments(gradient_checkpointing=True)`, which adds a redundant
AllGather in the backward pass. `transformer_layer_cls_to_wrap` is set
explicitly to `Qwen3_5DecoderLayer` because auto-detection reads
`_no_split_modules` off the class, which still lists `Qwen3_5VisionBlock` —
a module that does not exist in a text-only instantiation.

**Sequence packing is unavailable.** The Gated DeltaNet recurrent state cannot
be reset mid-sequence, so expect lower tokens/s per GPU than a dense model of
similar size.

**Full fine-tuning is no longer a stretch, which is the headline result.** In
the H100-era version of this demo, full fine-tuning a *smaller* 32B model OOM'd
on 16×80 GB and had to fall back to LoRA. Here, sharded full SFT of 27B costs
roughly 20 GiB per GPU (54 GB bf16 params + 54 GB grads + 216 GB fp32 AdamW
states, sharded 16 ways), or ~28 GiB with fp32 master weights, against 275 GB
of HBM. LoRA is now a choice about iteration speed and adapter portability
rather than a workaround.

The one place full fine-tuning still costs you is checkpointing: a
`FULL_STATE_DICT` save gathers ~54 GB across ranks, and doing that every few
hundred steps is what blew the distributed timeout in the Qwen3-era pipeline.
`train_full.py` therefore defaults to `SAVE_STRATEGY=no` — write the model once
at the end — and raises `ddp_timeout` to 7200s with a matching `NCCL_TIMEOUT`.
Set `SAVE_STRATEGY=steps` if you want resumability and can afford the I/O.

## Known risks

Things I flagged during the port and could not fully verify. None of them block
step 0, but check them before quoting results.

- **`--language-model-only`** in `serve.sbatch` is reported to exist for this
  model family but is unverified against `vllm==0.28.0`. If the server rejects
  it, drop the flag — nothing else depends on it.
- **`--tensor-parallel-size 16` is expected to fail.** The head counts
  (24 query / 4 KV / 16 linear-key / 48 linear-value) constrain valid TP to
  `{1,2,4,8}`. The head counts are read from `config.json`; the divisibility
  constraint is inferred, not tested. TP=1 is the default here and sidesteps it.
- **A determinism bug in the chunked gated-delta-rule kernel** is reported to
  affect sm_103 specifically, fixed in `flash-linear-attention==0.5.2`. Only
  relevant if you install that classic fast path instead of Hub `kernels`.
  There is a commented-out pin in `requirements.txt`.
- **Fine-tuning purely on non-thinking data should be expected to degrade
  thinking mode.** That is fine for this SQL demo, which explicitly wants
  terse non-reasoning output, but do not reuse the adapter for general chat.
- **The save paths** are the most likely thing to need a tweak on the first
  run. `train_lora.py` falls back to recovering from the last step checkpoint
  if `adapter_model.safetensors` is missing. For `train_full.py`, the final
  `FULL_STATE_DICT` gather is the risky step — it takes several minutes and
  looks like a hang; if it times out anyway, raise `ddp_timeout`/`NCCL_TIMEOUT`
  further or switch to `SAVE_STRATEGY=steps` with
  `state_dict_type="SHARDED_STATE_DICT"` and consolidate afterwards.

## Provenance

`evaluate.py`, `merge_lora.py` and `query.sh` are adapted from
[kreuzhofer/nebius-slurm-ml-training-and-inference-demo](https://github.com/kreuzhofer/nebius-slurm-ml-training-and-inference-demo)
(MIT), which did the same exercise with Qwen3-8B and Qwen3-32B on H100s and
reached 88% and 84% exact match respectively from base rates of 2–3%. This repo
keeps only what the Qwen3.8 experiment needs; the Terraform, the 235B
multi-node Ray serving path, and the full-fine-tune scripts stayed behind.

MIT licensed — see [LICENSE](LICENSE).
