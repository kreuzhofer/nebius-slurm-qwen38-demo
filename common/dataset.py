"""
dataset.py -- the TASK definition, shared by every model in this repo.

The prompt construction, the label masking and the 95/5 split live here once.
They must be identical for every model and for evaluation, because they are what
make two models' numbers comparable. A model-specific copy of any of this is a
drift waiting to happen -- see docs/RESULTS.md, where exactly that kind of drift
made two runs of the same task look irreconcilable.

Anything architecture-shaped (how to load the weights, how FSDP wraps them)
belongs in models/<model>/model.py instead.

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


# The 95/5 split is defined exactly once, here. Training and evaluation must
# agree on it or "held out" means nothing -- evaluate.py used to re-declare the
# same two constants, which is a drift waiting to happen rather than the
# guarantee this module is supposed to provide.
SPLIT_TEST_SIZE = 0.05
SPLIT_SEED = 42


def load_split(dataset_path):
    """
    Load the dataset and apply the canonical 95/5 split, untokenized.

    Returns a DatasetDict with "train" and "test". Callers needing text (like
    evaluate.py) use it directly; prepare_datasets() tokenizes on top of it.

    disable_caching() lives here rather than at each call site: the dataset is
    on shared NFS and train_test_split writes its index mapping as a
    cache-*.arrow file into that directory, which with 16 ranks means 16
    concurrent writers. It cannot take keep_in_memory itself -- datasets 4.6.0
    forwards both that and an auto-derived indices_cache_file_name to select(),
    which rejects the pair.
    """
    from datasets import disable_caching, load_from_disk

    disable_caching()
    dataset = load_from_disk(dataset_path)["train"]
    return dataset.train_test_split(test_size=SPLIT_TEST_SIZE, seed=SPLIT_SEED)


def prepare_datasets(
    dataset_path, tokenizer, max_seq_len, is_main, max_eval_examples=-1
):
    """
    Tokenize and length-filter the canonical split from load_split().

    Measured at MAX_SEQ_LEN=1024: the split yields 74,648 train / 3,929 eval and
    the length filter drops nothing, so the "dropped N of M" line below never
    prints at the shipped setting. It stays because a smaller MAX_SEQ_LEN would
    silently discard answers otherwise.

    max_eval_examples caps the eval split (-1 = all of it), so a short run can
    evaluate without paying for all 3,929 examples.
    """
    split = load_split(dataset_path)

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
