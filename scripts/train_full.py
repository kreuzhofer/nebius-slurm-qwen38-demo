"""
train_full.py -- full-parameter supervised fine-tuning of Qwen3.8-27B for SQL.

Run via train_full.sbatch (torchrun launches one process per GPU).

All ~26.9B text parameters are trained. The vision tower and MTP head are not
loaded at all (see sft_common.load_model), so nothing needs freezing.

WHY THIS FITS, when the H100-era version of this demo had to fall back to LoRA
for a smaller model: with FSDP2 full_shard across 16 ranks the steady-state cost
is roughly

    bf16 params  54GB
  + bf16 grads   54GB
  + fp32 AdamW   216GB   (m and v)
  ---------------------
    ~324GB total  ->  ~20GB per GPU

and about 28GB per GPU if fp32 master weights are also kept. Against 275GB of
HBM on a B300 that is uncontended, which is why activation checkpointing and
batch size are tuning choices here rather than survival tactics.

CHECKPOINTING IS THE INTERESTING PART -- see the comment on SAVE_STRATEGY below.
"""

import os

from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from sft_common import (
    env_config,
    fsdp_config,
    load_model,
    load_tokenizer,
    prepare_datasets,
)


def main():
    cfg = env_config()
    is_main = cfg["is_main"]
    output_dir = os.environ.get(
        "OUTPUT_DIR", f"{cfg['demo_dir']}/output/qwen3.8-27b-sql-full"
    )
    learning_rate = float(os.environ.get("LEARNING_RATE", "1e-5"))

    # Full-model checkpoints are ~54GB each and a FULL_STATE_DICT save requires
    # gathering unsharded weights across all 16 ranks. Doing that every
    # save_steps is what blew the distributed timeout in the Qwen3-era pipeline.
    #
    # Default here is "no": for a single-epoch demo run -- 584 optimizer steps,
    # measured, at effective batch 128 over the 74,648-example train split --
    # write the
    # model once at the end and skip mid-run checkpoints entirely. The trade-off
    # is that a crash means restarting rather than resuming -- set
    # SAVE_STRATEGY=steps (and SAVE_STEPS) if you want resumability and can
    # afford the I/O.
    save_strategy = os.environ.get("SAVE_STRATEGY", "no")
    save_steps = cfg["save_steps"]

    if is_main:
        print(f"Model      : {cfg['model_path']}")
        print(f"Output     : {output_dir}")
        print(f"Dataset    : {cfg['dataset_path']}")
        print(f"Batch      : {cfg['per_device_bs']}/GPU x {cfg['grad_accum']} accum")
        print(f"LR         : {learning_rate}")
        print(f"Max seq len: {cfg['max_seq_len']}")
        if cfg["max_steps"] > 0:
            print(f"Steps      : capped at {cfg['max_steps']} (MAX_STEPS set)")
        else:
            print(f"Steps      : {cfg['num_epochs']} epoch(s), no cap")
        print(f"Save       : strategy={save_strategy} steps={save_steps}")

    tokenizer = load_tokenizer(cfg["model_path"])
    model = load_model(cfg["model_path"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["num_epochs"],
        # -1 by default = use num_train_epochs. MAX_STEPS caps the run and then
        # takes precedence, which is what makes a cheap smoke run possible.
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
        # Fused AdamW is a straight win here and the optimizer states are the
        # dominant memory term, so it is worth being explicit.
        optim="adamw_torch_fused",
        fsdp=True,
        # FULL_STATE_DICT so the saved model is directly loadable by
        # transformers and vLLM with no consolidation step. This is affordable
        # because we save once; see the SAVE_STRATEGY comment above.
        fsdp_config=fsdp_config(state_dict_type="FULL_STATE_DICT"),
        # The final gather-and-write of a 54GB state dict can take a while.
        # Default process-group timeout is 1800s, which is cutting it fine over
        # shared NFS; give it two hours. Pairs with NCCL_TIMEOUT in the sbatch.
        ddp_timeout=7200,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy=save_strategy,
        **({"save_steps": save_steps, "save_total_limit": 1}
           if save_strategy != "no" else {}),
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
        print("Starting full fine-tune...")
    trainer.train()

    # Gathers the full state dict across ranks and writes ~54GB of safetensors.
    # Expect this to take several minutes; it is not hung.
    if is_main:
        print("Training done. Gathering full state dict and saving...")
    trainer.save_model(output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)
        print(f"\nFull fine-tune complete. Model saved to {output_dir}")
        print("This checkpoint is directly usable -- no LoRA merge step needed:")
        print(f"  sbatch scripts/evaluate.sbatch <base> {output_dir}")
        print(f"  sbatch scripts/serve.sbatch {output_dir}")


if __name__ == "__main__":
    main()
