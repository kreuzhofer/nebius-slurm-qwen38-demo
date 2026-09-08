"""
model.py -- everything about this pipeline that is specific to Qwen3.8-27B.

Split out from the task-level code in `common/` so that a second model can sit
beside this one without either inheriting the other's assumptions. What lives
here is architecture-shaped: how the model is loaded, how FSDP wraps it, and
where its weights live. What lives in `common/` is task-shaped -- the dataset
split, the prompt, the metric -- and MUST stay shared, because that is what
makes numbers from two models comparable at all.

Three things here are specific to Qwen3.8-27B:

1. ARCHITECTURE. It reports `model_type: "qwen3_5"` /
   `Qwen3_5ForConditionalGeneration`, is natively multimodal, and ships a vision
   tower plus a multi-token-prediction head. `AutoModelForCausalLM` resolves to
   `Qwen3_5ForCausalLM`, whose `_keys_to_ignore_on_load_unexpected` drops both,
   leaving the 26.896B-parameter text stack (measured).

2. HYBRID ATTENTION. 16 of 64 layers use classic attention; the other 48 are
   Gated DeltaNet. This drives the FSDP wrap class below, and it drove the LoRA
   target-list question that train_lora.py answers by measurement.

3. FSDP2. transformers v5 defaults `fsdp_config["version"]` to 2, so the
   FSDP1-only knobs are silently ignored and are absent here rather than
   carried over as dead config.
"""

import os

def load_model(model_path):
    """Load the text-only stack, dropping the vision tower and MTP head."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,        # `torch_dtype` is deprecated in transformers v5
        attn_implementation="sdpa",  # only affects the 16 full-attention layers
        use_kernels=True,            # fused Gated DeltaNet kernels from the Hub;
                                     # without this, 48/64 layers run the slow
                                     # pure-PyTorch fallback
    )
    # Avoids allocating the ~152MB/sequence GDN recurrent state every step.
    model.config.use_cache = False
    return model


def fsdp_config(state_dict_type="FULL_STATE_DICT"):
    """
    FSDP2 settings.

    transformers v5 defaults fsdp_config["version"] to 2, and the FSDP1-only
    knobs from the Qwen3-era scripts (backward_prefetch, forward_prefetch,
    use_orig_params) are silently ignored under it -- so they are gone here
    rather than carried over as dead config.
    """
    return {
        "version": 2,
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        # Set explicitly. Auto-detection reads _no_split_modules off the model
        # class, which still lists Qwen3_5VisionBlock even though a text-only
        # instantiation has no such module.
        "transformer_layer_cls_to_wrap": ["Qwen3_5DecoderLayer"],
        "reshard_after_forward": True,   # equivalent to full_shard
        # Prefer this over TrainingArguments(gradient_checkpointing=True): the
        # latter adds a redundant AllGather in the backward pass under FSDP
        # (transformers#30404).
        "activation_checkpointing": True,
        "cpu_ram_efficient_loading": True,
        "state_dict_type": state_dict_type,
    }


def env_config():
    """Read the knobs the .sbatch files set, with defaults."""
    demo_dir = os.environ.get("DEMO_DIR", "/mnt/data/qwen38-demo")
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    return {
        "demo_dir": demo_dir,
        "model_path": os.environ.get("MODEL_PATH", f"{demo_dir}/models/Qwen3.8-27B"),
        "dataset_path": os.environ.get(
            "DATASET_PATH", f"{demo_dir}/datasets/sql-create-context"
        ),
        "per_device_bs": int(os.environ.get("PER_DEVICE_BATCH_SIZE", "8")),
        "grad_accum": int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "1")),
        "num_epochs": int(os.environ.get("NUM_EPOCHS", "1")),
        "max_seq_len": int(os.environ.get("MAX_SEQ_LEN", "1024")),
        # Short-run knobs. MAX_STEPS=-1 means "run NUM_EPOCHS epochs", which is
        # how transformers itself spells "no step cap" -- so the default here is
        # exactly the previous behaviour: one full epoch, 584 optimizer steps at
        # effective batch 128 over the 74,648-example train split.
        #
        # A real smoke run is MAX_STEPS=25 SAVE_STEPS=10, which exercises the
        # checkpoint-save path twice in a couple of minutes instead of once at
        # the very end of a full run.
        "max_steps": int(os.environ.get("MAX_STEPS", "-1")),
        "save_steps": int(os.environ.get("SAVE_STEPS", "500")),
        # Defaults to SAVE_STEPS so a short run evaluates as often as it saves;
        # override independently when that is too expensive.
        "eval_steps": int(
            os.environ.get("EVAL_STEPS", os.environ.get("SAVE_STEPS", "500"))
        ),
        # -1 = the whole 3,929-example eval split. A 25-step smoke run that
        # evaluates over all of it spends far longer evaluating than training,
        # so cap it there.
        "max_eval_examples": int(os.environ.get("MAX_EVAL_EXAMPLES", "-1")),
        "rank": rank,
        "is_main": rank == 0,
    }
