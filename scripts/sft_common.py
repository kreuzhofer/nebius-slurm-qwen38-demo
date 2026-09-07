"""
sft_common.py -- shared pieces for train_lora.py and train_full.py.

The prompt construction and label masking below are the subtle part of this
demo, and they must be identical for both training modes and for evaluate.py --
so they live here once rather than being copy-pasted.

Three things here are specific to Qwen3.8 and are the reason this is not just
the Qwen3 pipeline with a new path in it:

1. ARCHITECTURE. Qwen3.8-27B reports `model_type: "qwen3_5"` /
   `Qwen3_5ForConditionalGeneration`. It is natively multimodal and ships a
   vision tower plus a multi-token-prediction head in the checkpoint. Loading
   through AutoModelForCausalLM resolves to `Qwen3_5ForCausalLM`, whose
   `_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]`
   drops both cleanly, leaving the ~26.9B-param text stack. That is what we
   want for a text-only SQL task.

2. HYBRID ATTENTION. Only 16 of the 64 layers use classic attention
   (`self_attn`, with q/k/v/o_proj). The other 48 are Gated DeltaNet linear
   attention, whose projections are named `linear_attn.in_proj_{qkv,z,a,b}`
   and `linear_attn.out_proj`. This matters for LoRA targeting (see
   train_lora.py) and for FSDP wrapping (see fsdp_config below).

3. THINKING MODE. Thinking is ON by default with reasoning_effort='xhigh'.
   Passing enable_thinking=False is load-bearing -- see build_example().
"""

import os

from transformers import AutoTokenizer

# torch and the model classes are imported inside load_model() rather than at
# module scope, so the prompt/masking helpers can be imported and unit-tested
# with nothing but transformers + jinja2 installed.

SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema and a question, write the "
    "correct SQL query. Output only the SQL query, nothing else."
)


def build_prompt(tokenizer, schema, question):
    """
    Render the inference-time prompt for one schema+question.

    Shared with evaluate.py so that training and scoring see byte-identical
    prompts.

    Why enable_thinking=False matters twice over:

      (a) With thinking enabled (the default), the template injects
          "Reasoning effort is set to xhigh. Please think carefully through..."
          into the system message -- and fabricates a system block if the
          conversation has none. That sentence would end up in every prompt.

      (b) With add_generation_prompt=True and enable_thinking=False, the
          template emits a *pre-filled empty* think block:
              <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
          The server emits those tokens for us at inference time, which is
          why build_example() masks them out of the labels.
    """
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_example(example, tokenizer):
    """
    Tokenize one SQL example into a prompt-masked training sample.

    The chat template is applied to the prompt only, then the answer is
    tokenized separately. That gives an exact boundary to mask on: everything
    up to and including the pre-filled `<think>\\n\\n</think>\\n\\n` is context
    (label -100), and only the SQL plus its `<|im_end|>` is supervised.

    Masking the empty think block is the important bit. It is part of the
    prompt at inference, so if it were supervised the model would learn to
    emit a *second* empty think block after the one the server already gave it.
    """
    prompt_text = build_prompt(tokenizer, example["context"], example["question"])

    # The template already emits all special tokens, so don't add more.
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(
        example["answer"] + "<|im_end|>", add_special_tokens=False
    )["input_ids"]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "length": len(input_ids),
    }


def prepare_datasets(
    dataset_path, tokenizer, max_seq_len, is_main, max_eval_examples=-1
):
    """
    Load, split 95/5 on seed 42 (same as evaluate.py), tokenize, length-filter.

    Measured at MAX_SEQ_LEN=1024: the split yields 74,648 train / 3,929 eval and
    the length filter drops nothing, so the "dropped N of M" line below never
    prints at the shipped setting. It stays because a smaller MAX_SEQ_LEN would
    silently discard answers otherwise.

    max_eval_examples caps the eval split (-1 = all of it), so a short run can
    evaluate without paying for all 3,929 examples.
    """
    from datasets import disable_caching, load_from_disk

    # Nothing here may write to the dataset directory: it lives on shared NFS
    # and all 16 ranks run this same code at the same time. Measured without
    # this: train_test_split alone drops two cache-*.arrow files into
    # datasets/sql-create-context/train/ on every run.
    #
    # disable_caching() rather than keep_in_memory=True on the split, because
    # train_test_split in datasets 4.6.0 forwards both keep_in_memory and an
    # auto-derived indices_cache_file_name to select(), which rejects the pair
    # with "Please use either `keep_in_memory` or `indices_cache_file_name`".
    disable_caching()

    dataset = load_from_disk(dataset_path)["train"]
    split = dataset.train_test_split(test_size=0.05, seed=42)

    def prepare(ds, desc):
        # keep_in_memory=True is load-bearing here, not an optimisation. Without
        # it, every one of the 16 ranks writes an Arrow cache file into the
        # dataset directory on shared NFS at the same time, which is a
        # documented route to pyarrow SIGBUS. The tokenized set is small enough
        # (~78k short SQL examples) that holding it in RAM is free on a 2.5TB
        # node.
        ds = ds.map(
            lambda x: build_example(x, tokenizer),
            remove_columns=ds.column_names,
            desc=f"Tokenizing {desc}",
            keep_in_memory=True,
        )
        # Drop rather than truncate: a clipped answer with no <|im_end|> would
        # teach the model to run on past the query.
        before = len(ds)
        ds = ds.filter(lambda x: x["length"] <= max_seq_len, keep_in_memory=True)
        if is_main and len(ds) < before:
            print(f"  {desc}: dropped {before - len(ds)} of {before} over {max_seq_len} tokens")
        return ds.remove_columns(["length"])

    train_ds = prepare(split["train"], "train")
    eval_ds = prepare(split["test"], "eval")
    if 0 <= max_eval_examples < len(eval_ds):
        if is_main:
            print(f"  eval: capped to {max_eval_examples} of {len(eval_ds)} examples")
        eval_ds = eval_ds.select(range(max_eval_examples))
    return train_ds, eval_ds


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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
