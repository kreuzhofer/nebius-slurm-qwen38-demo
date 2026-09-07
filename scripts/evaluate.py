"""
evaluate.py -- base vs fine-tuned Qwen3.8-27B on exact-match SQL generation.

Loads each model in turn, generates SQL for N held-out questions, and writes
accuracy numbers, a side-by-side of the examples fine-tuning fixed, a bar
chart, and a JSON dump.

The test split uses the same seed (42) and ratio (0.05) as train_lora.py, so
these examples were never trained on.

Adapted from the Qwen3 demo (MIT). Qwen3.8-specific changes:
  * dtype= instead of the deprecated torch_dtype=
  * prompts come from sft_common.build_prompt, the same function the training
    scripts use, so scoring cannot drift from training
  * explicit <think> stripping, because a stray thinking block would otherwise
    be scored as the SQL
"""

import argparse
import json
import os

import matplotlib
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_common import build_prompt

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


def generate_sql(model, tokenizer, schema, question, max_new_tokens=256):
    # Identical prompt to training, including enable_thinking=False and the
    # pre-filled empty think block.
    text = build_prompt(tokenizer, schema, question)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy, for reproducibility
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    # Belt and braces: if the model emitted a think block anyway (it can, since
    # thinking is its default behaviour), keep only what follows it.
    if "</think>" in response:
        response = response.split("</think>")[-1]
    return response.strip()


def normalize_sql(sql):
    """Lowercase, collapse whitespace, drop a trailing semicolon."""
    return " ".join(sql.lower().strip().rstrip(";").split())


def evaluate_model(model_path, test_data, label):
    print(f"\n{'=' * 60}\nEvaluating: {label}\nPath: {model_path}\n{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ~50GB of bf16 weights; comfortable on one 275GB B300.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        use_kernels=True,
    )
    model.eval()

    preds = []
    for i, ex in enumerate(test_data):
        preds.append(generate_sql(model, tokenizer, ex["context"], ex["question"]))
        if i == 0 or (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(test_data)}] {ex['question'][:60]}...")
            print(f"       -> {preds[-1][:80]}")

    del model, tokenizer
    torch.cuda.empty_cache()
    return preds


def main():
    demo_dir = os.environ.get("DEMO_DIR", "/mnt/data/qwen38-demo")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=f"{demo_dir}/models/Qwen3.8-27B")
    parser.add_argument("--tuned-model", default=f"{demo_dir}/output/qwen3.8-27b-sql")
    parser.add_argument("--dataset", default=f"{demo_dir}/datasets/sql-create-context")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--results-dir", default=f"{demo_dir}/results")
    parser.add_argument("--prefix", default="qwen3.8-27b-lora_")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    dataset = load_from_disk(args.dataset)["train"]
    test_split = dataset.train_test_split(test_size=0.05, seed=42)["test"]
    num = min(args.num_examples, len(test_split))
    test_data = test_split.select(range(num))
    ground_truth = [ex["answer"] for ex in test_data]
    print(f"Evaluating on {num} held-out examples")

    base_preds = evaluate_model(args.base_model, test_data, "Base model")
    tuned_preds = evaluate_model(args.tuned_model, test_data, "Fine-tuned model")

    base_correct = sum(
        normalize_sql(p) == normalize_sql(g) for p, g in zip(base_preds, ground_truth)
    )
    tuned_correct = sum(
        normalize_sql(p) == normalize_sql(g) for p, g in zip(tuned_preds, ground_truth)
    )
    base_acc = base_correct / num * 100
    tuned_acc = tuned_correct / num * 100

    print(f"\n{'=' * 60}")
    print(f"  RESULTS ({num} examples)")
    print(f"{'=' * 60}")
    print(f"  Base       : {base_acc:5.1f}%  ({base_correct}/{num})")
    print(f"  Fine-tuned : {tuned_acc:5.1f}%  ({tuned_correct}/{num})")
    print(f"  Improvement: {tuned_acc - base_acc:+5.1f}%")
    print(f"{'=' * 60}")

    fixed = [
        i
        for i in range(num)
        if normalize_sql(base_preds[i]) != normalize_sql(ground_truth[i])
        and normalize_sql(tuned_preds[i]) == normalize_sql(ground_truth[i])
    ]

    print("\nExamples fine-tuning fixed:\n")
    for i in fixed[:5]:
        print(f"  Example {i + 1}:")
        print(f"    Question    : {test_data[i]['question']}")
        print(f"    Ground truth: {ground_truth[i]}")
        print(f"    Base        : {base_preds[i]}")
        print(f"    Fine-tuned  : {tuned_preds[i]}\n")
    if not fixed:
        print("  (none -- try more examples)")

    # --- Chart -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        ["Base Qwen3.8-27B", "LoRA fine-tuned"],
        [base_acc, tuned_acc],
        color=["#2196F3", "#4CAF50"],
        width=0.5,
    )
    ax.set_ylabel("Exact Match Accuracy (%)")
    ax.set_title("SQL Generation: Qwen3.8-27B base vs LoRA fine-tuned")
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, [base_acc, tuned_acc]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{acc:.1f}%",
            ha="center",
            fontweight="bold",
            fontsize=14,
        )
    plt.tight_layout()
    chart_path = os.path.join(args.results_dir, f"{args.prefix}accuracy_comparison.png")
    plt.savefig(chart_path, dpi=150)
    print(f"Chart  : {chart_path}")

    # --- JSON --------------------------------------------------------------
    results_path = os.path.join(args.results_dir, f"{args.prefix}results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "base_model": args.base_model,
                "tuned_model": args.tuned_model,
                "num_examples": num,
                "base_accuracy_pct": round(base_acc, 2),
                "tuned_accuracy_pct": round(tuned_acc, 2),
                "improvement_pct": round(tuned_acc - base_acc, 2),
                "examples": [
                    {
                        "question": test_data[i]["question"],
                        "schema": test_data[i]["context"],
                        "ground_truth": ground_truth[i],
                        "base_prediction": base_preds[i],
                        "tuned_prediction": tuned_preds[i],
                        "base_correct": normalize_sql(base_preds[i])
                        == normalize_sql(ground_truth[i]),
                        "tuned_correct": normalize_sql(tuned_preds[i])
                        == normalize_sql(ground_truth[i]),
                    }
                    for i in range(num)
                ],
            },
            f,
            indent=2,
        )
    print(f"JSON   : {results_path}")

    # --- Markdown ----------------------------------------------------------
    md_path = os.path.join(args.results_dir, f"{args.prefix}results.md")
    with open(md_path, "w") as f:
        f.write(f"# Qwen3.8-27B LoRA SQL results ({num} held-out examples)\n\n")
        f.write("| | Model | Accuracy | Correct |\n|---|---|---|---|\n")
        f.write(f"| Base | `{args.base_model}` | {base_acc:.1f}% | {base_correct}/{num} |\n")
        f.write(
            f"| Fine-tuned | `{args.tuned_model}` | {tuned_acc:.1f}% | {tuned_correct}/{num} |\n\n"
        )
        f.write(f"**Improvement: {tuned_acc - base_acc:+.1f}%**\n\n")
        f.write("## Examples fine-tuning fixed\n\n")
        for i in fixed[:5]:
            f.write(f"### Example {i + 1}\n\n")
            f.write(f"**Question:** {test_data[i]['question']}\n\n")
            f.write(f"**Ground truth:** `{ground_truth[i]}`\n\n")
            f.write(f"**Base:** `{base_preds[i]}`\n\n")
            f.write(f"**Fine-tuned:** `{tuned_preds[i]}`\n\n")
        if not fixed:
            f.write("(none -- try more examples)\n")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
