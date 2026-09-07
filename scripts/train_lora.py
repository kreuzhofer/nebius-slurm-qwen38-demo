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
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",          # 16 full-attention layers
    "in_proj_qkv", "in_proj_z", "out_proj",          # 48 Gated DeltaNet layers
    "gate_proj", "up_proj", "down_proj",             # all 64 MLPs
]


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

    if is_main:
        print(f"Model      : {cfg['model_path']}")
        print(f"Output     : {output_dir}")
        print(f"Dataset    : {cfg['dataset_path']}")
        print(f"Batch      : {cfg['per_device_bs']}/GPU x {cfg['grad_accum']} accum")
        print(f"LoRA       : r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
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
            target_modules=LORA_TARGET_MODULES,
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
        warmup_ratio=0.03,
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


if __name__ == "__main__":
    main()
