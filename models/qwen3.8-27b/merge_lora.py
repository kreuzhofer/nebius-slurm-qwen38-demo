"""
merge_lora.py -- merge a LoRA adapter into the base weights.

Produces a standalone checkpoint that evaluate.py and serve.sbatch can consume
with no adapter plumbing.

Merging is the recommended path rather than serving the adapter dynamically:
vLLM fuses these projections internally (in_proj_qkv + in_proj_z -> in_proj_qkvz,
q/k/v -> qkv_proj), so a PEFT adapter targeting the unfused names has to be
re-sliced at load time, and that path has a history of bugs on GQA models --
which this is, with 24 query heads against 4 KV heads.

Usage:
  python merge_lora.py <adapter_dir> <base_model_dir> <output_dir>

Example (needs 1 GPU or plenty of host RAM; ~56GB is written out):
  srun --partition=main --nodes=1 --gpus-per-node=1 --time=01:00:00 \
      python /mnt/data/qwen38-demo/repo/models/qwen3.8-27b/merge_lora.py \
      /mnt/data/qwen38-demo/output/qwen3.8-27b-sql-lora \
      /mnt/data/qwen38-demo/models/Qwen3.8-27B \
      /mnt/data/qwen38-demo/output/qwen3.8-27b-sql
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into base weights")
    parser.add_argument("adapter_path")
    parser.add_argument("base_model_path")
    parser.add_argument("output_path")
    args = parser.parse_args()

    print(f"Loading base model from {args.base_model_path} ...")
    # Resolves to Qwen3_5ForCausalLM, which drops the vision tower and MTP head.
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        dtype=torch.bfloat16,   # `torch_dtype` is deprecated in transformers v5
        device_map="auto",
    )

    print(f"Loading adapter from {args.adapter_path} ...")
    model = PeftModel.from_pretrained(base, args.adapter_path)

    print("Merging ...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_path} ...")
    model.save_pretrained(args.output_path, safe_serialization=True)

    # Take the tokenizer from the adapter dir (train_lora.py saved it there) so
    # the chat template travels with the merged checkpoint.
    AutoTokenizer.from_pretrained(args.adapter_path).save_pretrained(args.output_path)

    print(f"Done. Merged model at {args.output_path}")


if __name__ == "__main__":
    main()
