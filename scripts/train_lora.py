"""
train_lora.py -- LoRA supervised fine-tuning of Qwen3.8-27B for SQL generation.

Run via train_lora.sbatch (torchrun launches one process per GPU).

Shared prompt/masking/FSDP logic lives in sft_common.py -- read the module
docstring there first, it explains what is Qwen3.8-specific about this pipeline.

The LoRA-specific concern is target module naming: Qwen3.8's attention is
hybrid, so the Qwen3-era target list covers only a quarter of the token-mixing
blocks. See LORA_TARGET_MODULES.
"""

import glob
import os
import shutil

from peft import LoraConfig, TaskType, get_peft_model
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from sft_common import (
    env_config,
    fsdp_config,
    load_model,
    load_tokenizer,
    prepare_datasets,
)

# Attention projections for the 16 full-attention layers, the Gated DeltaNet
# projections for the other 48, and the MLP for all 64.
#
# Deliberately excluded:
#   in_proj_a / in_proj_b -- these are [48, 5120]; a LoRA rank above 48 is
#                            degenerate there for no real capacity gain.
#   conv1d                -- nn.Conv1d (depthwise), not a linear layer.
ATTENTION_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]   # 16 full-attn layers
GDN_TARGETS = ["in_proj_qkv", "in_proj_z", "out_proj"]         # 48 Gated DeltaNet
MLP_TARGETS = ["gate_proj", "up_proj", "down_proj"]            # all 64 MLPs


def lora_target_modules(target_gdn=True):
    """
    Which modules LoRA adapts.

    LORA_TARGET_GDN=0 drops the Gated DeltaNet projections, leaving attention +
    MLP. That is the knob behind the A/B in #12: this repo targets the GDN
    projections on the reasoning that a Qwen3-era attention-only list adapts
    only 16 of 64 token-mixing blocks, while every sibling recipe for this
    architecture targets attention only -- and two of them state outright that
    LoRA should not be applied to the linear-attention projections. Neither
    position had ever been measured.

    Module counts, verified against the checkpoint (see #7):
        with GDN    400 modules, 217.58M trainable at r=32 (0.809% of 26.896B)
        without GDN 256 modules, 159.38M trainable at r=32 (0.593%)

    Note the arms are NOT capacity-matched at equal rank -- the broad list
    carries 1.37x the trainable parameters -- so a difference cannot be
    attributed to placement alone without a rank-matched third arm.

    RESULT (#12), 100 held-out examples, both arms trained identically:

        steps   arm A (with GDN)   arm B (attn+MLP)   A-only   B-only
           25         74%                75%             0        1
          500         88%                89%             0        1

    eval_loss at 500 steps: 0.01749 vs 0.01754. Across two runs at two scales
    there is not ONE example the GDN adapters get right that the narrow list
    misses, and arm B is ~15% faster per step. So GDN targeting is off by
    default: it costs 58.2M trainable parameters and 15% throughput to change
    nothing measurable. Set LORA_TARGET_GDN=1 to put it back.

    This is a measured negative result, not an assumption inherited from the
    sibling recipes -- which reached the same configuration by reasoning that
    turned out to be partly wrong (they claimed GDN layers lack MLP
    projections; all 64 layers have them, see #7).
    """
    mods = list(ATTENTION_TARGETS)
    if target_gdn:
        mods += GDN_TARGETS
    return mods + MLP_TARGETS


def main():
    cfg = env_config()
    is_main = cfg["is_main"]
    output_dir = os.environ.get(
        "OUTPUT_DIR", f"{cfg['demo_dir']}/output/qwen3.8-27b-sql-lora"
    )
    learning_rate = float(os.environ.get("LEARNING_RATE", "2e-4"))

    lora_r = int(os.environ.get("LORA_R", "32"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", "64"))
    lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.05"))
    # Default 0: MEASURED to make no difference. See lora_target_modules().
    target_gdn = os.environ.get("LORA_TARGET_GDN", "0") not in ("0", "false", "False")
    target_modules = lora_target_modules(target_gdn)

    if is_main:
        print(f"Model      : {cfg['model_path']}")
        print(f"Output     : {output_dir}")
        print(f"Dataset    : {cfg['dataset_path']}")
        print(f"Batch      : {cfg['per_device_bs']}/GPU x {cfg['grad_accum']} accum")
        print(f"LoRA       : r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        print(f"Targets    : {'attn+GDN+MLP' if target_gdn else 'attn+MLP (GDN dropped)'}"
              f" -> {target_modules}")
        print(f"Max seq len: {cfg['max_seq_len']}")
        if cfg["max_steps"] > 0:
            print(f"Steps      : capped at {cfg['max_steps']} (MAX_STEPS set)")
        else:
            print(f"Steps      : {cfg['num_epochs']} epoch(s), no cap")
        print(f"Save/eval  : every {cfg['save_steps']}/{cfg['eval_steps']} steps")

    tokenizer = load_tokenizer(cfg["model_path"])
    model = load_model(cfg["model_path"])

    # peft ships no default target-module mapping for qwen3_5, so an explicit
    # list is mandatory (target_modules=None would raise).
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        ),
    )
    if is_main:
        model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["num_epochs"],
        # -1 by default, which is how transformers spells "use num_train_epochs".
        # Set MAX_STEPS to cap the run; it then takes precedence over epochs.
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["per_device_bs"],
        per_device_eval_batch_size=cfg["per_device_bs"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=learning_rate,
        weight_decay=0.01,
        # transformers v5 removed warmup_ratio and folded it into warmup_steps,
        # which is now a float: >=1 means exact steps, and a value in [0, 1) is
        # a ratio of total steps (get_warmup_steps does
        # math.ceil(num_training_steps * warmup_steps)). So 0.03 here is exactly
        # the old warmup_ratio=0.03, not an approximation of it.
        warmup_steps=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=True,
        # Loading the 52GB base checkpoint fans out into hundreds of broadcasts
        # across 16 ranks, and the default 1800s process-group timeout is not
        # generous over shared NFS. The full-parameter path already raises this
        # for its state-dict gather; the LoRA path needs it for the *load*, even
        # though its own saves are small.
        ddp_timeout=7200,
        fsdp=True,
        # Only the adapter is trainable, so the state-dict gather is small and
        # a full checkpoint every save_steps is cheap -- unlike train_full.py.
        fsdp_config=fsdp_config(state_dict_type="FULL_STATE_DICT"),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=2,
        dataloader_num_workers=4,
        report_to="none",
    )

    train_ds, eval_ds = prepare_datasets(
        cfg["dataset_path"],
        tokenizer,
        cfg["max_seq_len"],
        is_main,
        cfg["max_eval_examples"],
    )
    if is_main:
        print(f"Train examples: {len(train_ds)}, eval examples: {len(eval_ds)}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,  # v5 renamed Trainer(tokenizer=...)
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, pad_to_multiple_of=8
        ),
    )

    if is_main:
        print("Starting LoRA training...")
    trainer.train()

    # Only the adapter weights are trainable, so this gather is small and safe
    # (unlike a full-model FULL_STATE_DICT gather, which is what blew the
    # rendezvous timeout in the Qwen3 pipeline).
    trainer.save_model(output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)
        print(f"\nLoRA training complete. Adapter saved to {output_dir}")
        # Fallback if a future transformers/peft combination writes nothing
        # useful here: recover from the last step checkpoint.
        if not os.path.exists(os.path.join(output_dir, "adapter_model.safetensors")):
            # Sort by step number, not lexicographically: sorted() puts
            # "checkpoint-1000" before "checkpoint-500", so [-1] would restore
            # the *oldest* checkpoint and look like it had worked.
            ckpts = sorted(
                glob.glob(os.path.join(output_dir, "checkpoint-*")),
                key=lambda p: int(p.rsplit("-", 1)[1]),
            )
            if ckpts:
                print(f"adapter_model.safetensors missing; copying from {ckpts[-1]}")
                for fname in os.listdir(ckpts[-1]):
                    src = os.path.join(ckpts[-1], fname)
                    if os.path.isfile(src) and not fname.startswith(
                        ("rng_state", "optimizer", "scheduler", "training_args")
                    ):
                        shutil.copy2(src, output_dir)

    # Hold every rank here until rank 0 has finished writing.
    #
    # save_model()'s state-dict gather is collective, but only rank 0 writes to
    # disk. Without this barrier ranks 1..15 return from the gather, skip the
    # is_world_process_zero() block, fall off the end of main() and exit -- their
    # CUDA contexts tear down, the driver shuts down under rank 0 mid-write, and
    # the process dies between safetensors' write and its atomic rename.
    #
    # Measured in job 811: the 870MB adapter was written completely and left as
    # an unrenamed .tmp* file, with "CUDA driver error: unknown error" from
    # _hasPrimaryContext at teardown. The recovery fallback above could not run
    # either, because the process was already gone.

    # Peak GPU memory, measured rather than assumed. Trainer's own memory
    # metrics are off by default (skip_memory_metrics=True) and nothing else
    # here sampled the device, so the README's "~20 GiB/GPU" prediction for the
    # full-parameter path went unverified through every run on this map.
    #
    # These are PyTorch's allocator counters for THIS rank. All ranks are
    # symmetric under FSDP, so rank 0 is representative. Note `reserved` is what
    # the caching allocator holds, which is the number to compare against HBM;
    # actual device usage is a little higher again (NCCL buffers, kernels, CUDA
    # context) and only nvidia-smi sees that.
    if is_main:
        import torch as _torch
        if _torch.cuda.is_available():
            print(
                f"Peak GPU mem (rank 0): "
                f"{_torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated, "
                f"{_torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved"
            )
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
