# Qwen3.8-27B fine-tuning on a Nebius Soperator Slurm cluster

Fine-tunes [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) for
natural-language-to-SQL across 2 nodes × 8 NVIDIA B300, then evaluates it
against the base model and serves it with vLLM.

📊 **[Full results and methodology → `docs/RESULTS.md`](docs/RESULTS.md)** —
every measurement behind this README, the charts, all 500 predictions, the eight
defects that running it exposed, and what was *not* measured.

**Measured on 500 held-out examples** (one epoch each, greedy decoding):

| Model | Accuracy | vs base | Usable as-is |
|---|---|---|---|
| Qwen3.8-27B base | 55.4% | — | 1.6% |
| **LoRA (1 epoch)** | **87.4%** | **+32.0pp** | 86.8% |
| Full fine-tune (1 epoch) | 86.6% | +31.2pp | 86.0% |

Full fine-tuning is **statistically indistinguishable** from LoRA here
(exact McNemar p = 0.42 over the same 500 examples), so a **4.3 GB adapter
matched a 51 GB checkpoint**.

Two training paths, same data and same effective batch size so the results are
directly comparable:

| | script | trainable params | LR | output |
|---|---|---|---|---|
| **LoRA** | `train_lora.sbatch` | 0.59% (adapter) | 2e-4 | adapter, needs a merge step |
| **Full** | `train_full.sbatch` | all ~26.9B | 1e-5 | standalone checkpoint |

Task: [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context).
The raw `train` split holds **78,577** examples; the seed-42 95/5 split used by
both training and evaluation gives **74,648 train / 3,929 held out**, so one
epoch is **584 optimizer steps** at effective batch 128. Metric: exact-match
accuracy on 500 held-out questions after light normalization (100 for smoke
runs). All figures measured, not estimated.

**The metric extracts the SQL before comparing it, and that step is load-bearing.**
`normalize_sql` pulls the query out of the response — a closed ```` ```sql ```` block
if there is one, otherwise from the last `SELECT` — strips chat-template
artefacts, normalises `'single'` quotes to the `"double"` quotes this dataset
stores, then lowercases and collapses whitespace. It is ported verbatim from the
reference implementation in
[kreuzhofer/dgx-manager-fine-tune-recipes](https://github.com/kreuzhofer/dgx-manager-fine-tune-recipes)
(`scripts/evaluate.py`), so numbers here are directly comparable with the
Qwen3.6/3.8 results measured on DGX Spark.

Measured on 100 held-out examples, base model vs a **25-step** smoke adapter:

| | base | 25-step LoRA |
|---|---|---|
| **Accuracy** (extract + normalise) | **58%** | **74%** |
| _Diagnostic:_ usable as-is, no post-processing | 1% | 72% |

Both columns are real, and the gap between them is the point. The base model
writes largely correct SQL and is unusable without post-processing: 72 of its
100 answers arrive wrapped in a markdown fence, and it uses ANSI-standard
`'single'` quotes where this dataset stores non-standard `"double"` ones.
Neither is a SQL error. Fine-tuning fixes the packaging almost immediately —
that is the 1% → 72% column — while adding a more modest **+16pp** of actual
correctness.

An earlier version of this file reported the base model at 1% as its headline.
That was a harness defect, not a property of the model: the shipped
`normalize_sql` had lost the extraction and quote-handling steps. Verified not
to be a prompt artifact — across three prompt variants the base produced double
quotes 0-1 times in 100 — and not an engine artifact, since vLLM and
`transformers.generate()` score identically on the same prompt.

## Cluster

Written for and verified against:

- 2 worker nodes × 8 **NVIDIA B300 SXM6** (**268.6 GiB / 288 GB HBM** each, **sm_103**), 192 vCPU, 2.66 TiB RAM per node
  — **4.20 TiB of VRAM across the 16 GPUs**, and 5.32 TiB of host RAM. Measured, not
  from a spec sheet: `nvidia-smi` reports 275,040 MiB per GPU. An earlier version of
  this file called that "275 GB", which is the MiB figure divided by 1000 and is
  neither unit.
- Slurm 25.11.3 via [Soperator](https://github.com/nebius/nebius-solutions-library/tree/main/soperator), partition `main`
- Driver 580.159.04 / CUDA 13.0
- Shared filesystem at `/mnt/data`; internet egress from both login and worker nodes

Check yours with `sinfo -o "%P %N %G"` before trusting the pins in
`requirements.txt`.

## Quick start

```bash
# 0. one-time environment setup on the login node (~8GB installed)
git clone https://github.com/kreuzhofer/nebius-slurm-qwen38-lora-demo.git
cd nebius-slurm-qwen38-lora-demo
bash scripts/setup.sh                 # ends with a GPU smoke test on a worker
source /mnt/data/qwen38-demo/activate.sh

# 1. fetch model (52GB, 18 shards) + dataset (17MB on disk)
bash /mnt/data/qwen38-demo/scripts/download.sh

# 2. smoke-test the whole path first: 25 steps, ~6 min, saves twice
MAX_STEPS=25 SAVE_STEPS=10 MAX_EVAL_EXAMPLES=200 \
    sbatch --export=ALL /mnt/data/qwen38-demo/scripts/train_lora.sbatch

# 3. train for real on 16 GPUs -- pick one. One epoch = 584 steps, ~15-20 min.
sbatch /mnt/data/qwen38-demo/scripts/train_lora.sbatch    # LoRA adapter
sbatch /mnt/data/qwen38-demo/scripts/train_full.sbatch    # all 26.9B params
squeue --me
tail -f /mnt/data/qwen38-demo/logs/train_lora_<JOBID>.out

# 4. LoRA ONLY -- merge the adapter into a standalone checkpoint (~5 min).
#    Skip this entirely if you ran train_full.sbatch.
#    NOTE: this is the one step that does not source the venv for you. Use the
#    venv interpreter explicitly, because bare `python` on the worker is
#    /usr/bin/python, which has no torch and no peft.
srun --partition=main --nodes=1 --gpus-per-node=1 --time=01:00:00 \
    /mnt/data/qwen38-demo/venv/bin/python \
    /mnt/data/qwen38-demo/scripts/merge_lora.py \
    /mnt/data/qwen38-demo/output/qwen3.8-27b-sql-lora \
    /mnt/data/qwen38-demo/models/Qwen3.8-27B \
    /mnt/data/qwen38-demo/output/qwen3.8-27b-sql

# 5. score base vs fine-tuned. Every argument is passed through to
#    evaluate.py, so pass --tuned-model once per model you want in one chart;
#    the base model's predictions are generated only once.
sbatch /mnt/data/qwen38-demo/scripts/evaluate.sbatch          # base vs merged LoRA, N=500
sbatch /mnt/data/qwen38-demo/scripts/evaluate.sbatch \
    --tuned-model /mnt/data/qwen38-demo/output/qwen3.8-27b-sql \
    --tuned-model /mnt/data/qwen38-demo/output/qwen3.8-27b-sql-full
#    Output filenames derive from the tuned model names, so separate
#    comparisons cannot overwrite each other:
#    -> results/qwen3.8-27b-sql_qwen3.8-27b-sql-full_results.{json,md} + .png

# 6. serve and query. vLLM takes several minutes to load 51GB, and query.sh
#    reads the hostname from squeue -- which is empty while the job is still
#    PENDING -- so wait for the server to answer before querying.
sbatch /mnt/data/qwen38-demo/scripts/serve.sbatch
HOST=$(squeue --noheader -n qwen38-serve -o "%N" | head -1)
until curl -sf -m 3 "http://$HOST:8000/v1/models" >/dev/null; do sleep 10; done
bash scripts/query.sh
```

Re-run `bash scripts/setup.sh` after editing anything in `scripts/` — it
re-syncs the shared copy under `/mnt/data/qwen38-demo/scripts/` that the Slurm
jobs actually execute. The sync is `rsync -a --delete`, so renamed and deleted
scripts are pruned rather than left behind as stale copies.

Downloads run unauthenticated unless you export `HF_TOKEN`, which the Hub warns
about and which costs you rate limit and speed on both the model fetch and the
Hub kernel fetch at model load. Neither model nor dataset is gated, so a token
is optional.

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

**Attention is hybrid — and, measured, this does *not* change the LoRA config.**
Of 64 layers only **16** use classic attention (`self_attn`,
`full_attention_interval: 4`); the other **48** are Gated DeltaNet linear
attention with projections named `linear_attn.in_proj_{qkv,z,a,b}` and
`linear_attn.out_proj`. The obvious inference — that a Qwen3-era target list of
`q/k/v/o_proj` + MLP adapts only a quarter of the token-mixing blocks and must
therefore be leaving capability on the table — is wrong, and this repo shipped
it as fact until it was tested.

Both configurations were trained identically and scored on the same 100
held-out examples:

| steps | attn + **GDN** + MLP | attn + MLP | won only by A | won only by B |
|---|---|---|---|---|
| 25 | 74% | 75% | **0** | 1 |
| 500 | 88% | **89%** | **0** | 1 |

`eval_loss` at 500 steps: 0.01749 vs 0.01754. Across two runs at two scales
there is not **one** held-out example the GDN adapters get right that the
narrow list misses, while the broad list costs 58.2M extra trainable
parameters (0.80% vs 0.59%) and about **15% throughput per step**.

So the default targets attention + MLP only, and skips the GDN projections.
`LORA_TARGET_GDN=1` puts them back if you want to re-test at another rank or
on another task — the arms here are not capacity-matched, so this measures
"which shipped config is better", not "does GDN placement matter in principle".

`in_proj_a`/`in_proj_b` are excluded regardless: both are `[48, 5120]`, so any
rank above 48 is degenerate there.

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
similar size. Measured on this cluster: steady-state LoRA training is
**~1.2-1.75 s/step** at effective batch 128 (so a 584-step epoch is roughly
15-20 minutes), which works out to ~0.55 samples/s/GPU or ~62 unpadded
tokens/s/GPU. Sequences are short here -- mean 112 tokens, max 239 -- so
`MAX_SEQ_LEN=1024` never actually truncates or drops anything.

**`use_kernels=True` does not get B300 the fused Gated DeltaNet *layer*
kernel.** That mapping is gated to compute capability exactly 121; B300 is 103,
so it falls through to ungated function-level kernels. This is neither the fused
layer path nor the pure-PyTorch fallback, and it means the flag buys less here
than on a cc-121 device. Note also that `use_kernels=True` makes a **live
HuggingFace Hub request at model load**, so training has a network dependency at
startup -- and it is unauthenticated unless you set `HF_TOKEN`.

**Full fine-tuning is no longer a stretch, which is the headline result.** In
the H100-era version of this demo, full fine-tuning a *smaller* 32B model OOM'd
on 16×80 GB and had to fall back to LoRA. Here, sharded full SFT of 27B costs
roughly 20 GiB per GPU (54 GB bf16 params + 54 GB grads + 216 GB fp32 AdamW
states, sharded 16 ways), or ~28 GiB with fp32 master weights, against **268.6 GiB**
of HBM per GPU.

**Measured, and the arithmetic holds up.** `torch.cuda.max_memory_allocated`
on rank 0:

| run | allocated | reserved | % of a 268.6 GiB card |
|---|---|---|---|
| full fine-tune, 584 steps (full epoch) | **35.45 GiB** | 49.78 GiB | 13.2% / 18.5% |
| full fine-tune, 25 steps | 25.06 GiB | 37.81 GiB | 9.3% / 14.1% |
| LoRA, 584 steps (full epoch) | 29.08 GiB | 63.83 GiB | 10.8% / 23.8% |

Note the first two rows: **the same configuration peaked 41% higher over 584
steps than over 25** (35.45 vs 25.06 GiB). Peak memory is a high-water mark, and
a longer run samples more of the batch-length distribution. Short runs
systematically understate it — budget from a full run, not a smoke test.

Sharded full fine-tuning of 27B really does fit in about 25 GiB per GPU, close
to the states-only estimate above, with activations adding only a few GiB at
`PER_DEVICE_BATCH_SIZE=8` / `MAX_SEQ_LEN=1024` and activation checkpointing on.

**Do not read those rows as "full FT is cheaper than LoRA".** They are not
comparable: the LoRA figure is a high-water mark over 584 steps and an
evaluation across all 3,929 held-out examples, while the full-FT figure covers
25 steps and 200 eval examples — far fewer batches sampled, and fewer chances
to hit a long one. Peak memory is a maximum, and a longer run samples more of
the distribution.

Note also the gap between *allocated* and *reserved*: during the LoRA run the
caching allocator held 63.83 GiB while only 29.08 GiB was live, and `nvidia-smi`
showed ~50 GiB average with 68 GiB peaks. Three numbers, one run — when
comparing against a prediction, say which you mean.

**And LoRA is not the faster option here, which was a surprise.** Both paths
ran one full epoch, 584 steps, same data and batch:

| | wall clock | s/step | trainable | artifact on disk | eval loss |
|---|---|---|---|---|---|
| LoRA (attn+MLP, r=32) | 15.1 min | 1.55 | 159 M | 4.3 GB adapter | **0.01691** |
| **Full fine-tune** | **11.9 min** | **1.22** | 27.1 B | 51 GB checkpoint | 0.01873 |

Full fine-tuning ran the epoch **21% faster** while training 170x more
parameters. Part of that is the LoRA run writing a mid-run checkpoint at step
500 that the full run skipped (`SAVE_STRATEGY=no`); correcting for it still
leaves full FT ahead by roughly 15%. The mechanism is that under FSDP
`full_shard` the dominant per-step cost is the parameter all-gather for forward
and backward, which **both** paths pay in full — LoRA saves on the optimizer
update and on gradient traffic, neither of which dominates here, while adding
PEFT wrapper overhead on 256 modules and still running the entire base model
forward and backward.

So on this hardware LoRA's remaining advantages are **adapter portability and
disk** (4.3 GB versus 51 GB, and you keep the base weights shared), not
iteration speed. It also reached a slightly *better* eval loss, though at a 20x
higher learning rate that was never separately tuned for the full path.

The one place full fine-tuning still costs you is checkpointing: a
`FULL_STATE_DICT` save gathers ~54 GB across ranks, and doing that every few
hundred steps is what blew the distributed timeout in the Qwen3-era pipeline.
`train_full.py` therefore defaults to `SAVE_STRATEGY=no` — write the model once
at the end — and raises `ddp_timeout` to 7200s with a matching `NCCL_TIMEOUT`.
Set `SAVE_STRATEGY=steps` if you want resumability and can afford the I/O.

## Known risks

Every item below has now been checked against this cluster, the downloaded
checkpoint, or the installed packages. Where the original guess was wrong, the
correction is stated rather than the guess softened.

- **`--language-model-only` is verified and stays.** vLLM 0.28.0 accepts it and
  reports `'language_model_only': True` in its parsed non-default args, with no
  argument rejection. Two corrections to what this file used to imply: the flag
  disables multimodal *inputs* rather than skipping construction of the vision
  tower (that is `--skip-mm-profiling`), and on a **merged** checkpoint it is a
  no-op regardless, because merging writes `model_type: qwen3_5_text` with no
  `vision_config`. Harmless and correct on both the base and merged paths.
- **`--tensor-parallel-size 16` fails on the 24 query heads**, not on the
  linear-key count as this file previously claimed -- 16 linear-key heads divide
  16 exactly. TP=3 and TP=6 also fail, on `key_dim=2048`, after passing every
  head-count check. Valid set is `{1,2,4,8}`. All four head counts are confirmed
  against the downloaded `config.json` (24 query / 4 KV / 16 linear-key /
  48 linear-value). Only TP=1 has actually been run here.
- **The sm_103 determinism bug does not apply as pinned, and the commented-out
  pin in `requirements.txt` is a trap rather than a remedy.** The bug is real
  (fla#945, fixed by #953 in 0.5.2) and worse than usually described -- the
  backward recomputes the forward state, so gradients disagree with the loss.
  But the pip `fla` module is imported whenever it is *importable*, regardless
  of `use_kernels`, so the rule is "never let `fla-core < 0.5.2` be importable".
  Verified after a real training run: `fla`, `fla_core`, `causal_conv1d` and
  `mamba_ssm` are all absent from the venv and nothing pulls them in, so the bug
  cannot reach this pipeline. Uncommenting that line is the only way to
  introduce it.
- **Fine-tuning purely on non-thinking data should be expected to degrade
  thinking mode.** That is fine for this SQL demo, which explicitly wants
  terse non-reasoning output, but do not reuse the adapter for general chat.
- **The save path did break on the first run, and the cause was not a
  timeout.** `save_model()`'s state-dict gather is collective but only rank 0
  writes. Without a barrier afterwards, ranks 1-15 returned from the gather,
  fell off the end of `main()` and exited; their CUDA contexts tore down, the
  driver shut down under rank 0 mid-write, and the process died between
  safetensors' write and its atomic rename. Symptom: a complete 870MB adapter
  left as an unrenamed `.tmp*` file, `adapter_model.safetensors` absent, and
  `CUDA driver error: unknown error` from `_hasPrimaryContext` at teardown.
  Fixed with `trainer.accelerator.wait_for_everyone()` at the end of `main()` in
  both training scripts (teardown error lines went 62 to 0). The same race sat
  in front of `train_full.py`'s 54GB write and is fixed there too.

- **Write throughput depends enormously on file shape — don't extrapolate from
  one to the other.** A LoRA step checkpoint is ~2.6GB spread over many files
  (870MB adapter, 1.7GB optimizer, 16 per-rank RNG states, scheduler, trainer
  state) and takes ~65s: about **40 MB/s**. The full fine-tune's terminal save
  is 50.1 GiB in **two** large safetensors shards and completed in **3m51s —
  about 214 MiB/s**, five times faster per byte. An earlier version of this file
  projected ~22 minutes for that save by scaling the small-file rate; the real
  figure is under four minutes. Both are comfortably inside
  `ddp_timeout=7200`, which is why `SAVE_STRATEGY=no` is a default worth
  revisiting rather than a necessity.

## Provenance

`evaluate.py`, `merge_lora.py` and `query.sh` are adapted from
[kreuzhofer/nebius-slurm-ml-training-and-inference-demo](https://github.com/kreuzhofer/nebius-slurm-ml-training-and-inference-demo)
(MIT), which did the same exercise with Qwen3-8B and Qwen3-32B on H100s and
reached 88% and 84% exact match respectively from base rates of 2–3%. This repo
keeps only what the Qwen3.8 experiment needs; the Terraform and the 235B
multi-node Ray serving path stayed behind.

**Do not carry the 2–3% base rate over as an expectation.** It is from the
Qwen3-era 8B/32B models on different hardware. On the architecturally identical
Qwen3.6-27B, this same dataset measures a base rate of roughly **39%**, so
expect something like 39% to the mid-70s rather than 2% to 88%. A measured base
rate near 2% here is evidence of a broken prompt or normalisation, not of a weak
model.

MIT licensed — see [LICENSE](LICENSE).
