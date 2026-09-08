"""
evaluate.py -- compare a base model against one or more fine-tunes on
exact-match SQL generation.

Takes an arbitrary number of models: one base (the reference every "fixed by
fine-tuning" comparison is made against) and one or more tuned models. The base
model's predictions are generated once no matter how many tuned models are
scored against it, which is why a single three-way run is cheaper than two
two-way runs -- as well as producing one chart instead of two that cannot be
placed side by side.

Every label, chart title, filename prefix and Markdown heading is derived from
the models actually passed. Nothing here says "LoRA": an earlier version
hardcoded that, so pointing it at the full fine-tune produced a report that
misattributed its own result, and both runs wrote to the same filenames so the
second silently overwrote the first.

The test split comes from sft_common.load_split and the prompts from
sft_common.build_prompt, so scoring cannot drift from training. Qwen3.8
specifics: dtype= rather than the deprecated torch_dtype=, and explicit <think>
stripping, because thinking is this model's default and a stray reasoning block
would otherwise be scored as the SQL.
"""

import argparse
import json
import os
import sys

# Make the repo root importable (it is DEMO_DIR on the cluster), so `common`
# resolves whether this runs via evaluate.sbatch, srun, or a bare python call.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
import torch
from transformers import AutoModelForCausalLM

from common.dataset import build_prompt, load_split, load_tokenizer
from common.metric import normalize_sql, raw_exact

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

# Base first, then one colour per tuned model, cycled if there are more.
BASE_COLOR = "#2196F3"
TUNED_COLORS = ["#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#E91E63"]


def generate_sql(model, tokenizer, schema, question, max_new_tokens=256):
    # Identical prompt to training, including enable_thinking=False and the
    # pre-filled empty think block.
    text = build_prompt(tokenizer, schema, question)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )

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


def evaluate_model(model_path, test_data, label, max_new_tokens):
    print(f"\n{'=' * 60}\nEvaluating: {label}\nPath: {model_path}\n{'=' * 60}")

    tokenizer = load_tokenizer(model_path)  # shared pad-token handling
    # Load exactly as sft_common.load_model does, attn_implementation included:
    # scoring a model under a different attention implementation than it was
    # trained with is a silent way to make the comparison meaningless.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        use_kernels=True,
    )
    model.eval()

    preds = []
    for i, ex in enumerate(test_data):
        preds.append(
            generate_sql(
                model, tokenizer, ex["context"], ex["question"], max_new_tokens
            )
        )
        if i == 0 or (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(test_data)}] {ex['question'][:60]}...")
            print(f"       -> {preds[-1][:80]}")

    del model, tokenizer
    torch.cuda.empty_cache()
    return preds


def slug(path):
    """Filesystem- and title-safe stem for a model path."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", os.path.basename(path.rstrip("/")))


def resolve_labels(models, explicit):
    """
    Derive one display label per (path) in `models`, base first.

    Explicit --label values win, in order. Otherwise the basename is used, with
    " (base)" appended to the reference model so a chart is readable without
    knowing which bar is which. Duplicate basenames get an index suffix rather
    than two identically-labelled bars.
    """
    if explicit:
        if len(explicit) != len(models):
            raise SystemExit(
                f"--label given {len(explicit)} times but {len(models)} models "
                f"were passed (1 base + {len(models) - 1} tuned)"
            )
        return list(explicit)

    labels = [f"{slug(models[0])} (base)"] + [slug(p) for p in models[1:]]
    seen = {}
    out = []
    for lab in labels:
        seen[lab] = seen.get(lab, 0) + 1
        out.append(lab if seen[lab] == 1 else f"{lab} #{seen[lab]}")
    return out


def main():
    demo_dir = os.environ.get("DEMO_DIR", "/mnt/data/qwen38-demo")
    parser = argparse.ArgumentParser(
        description="Base vs one or more fine-tuned models, exact-match SQL."
    )
    parser.add_argument("--base-model", default=f"{demo_dir}/models/Qwen3.8-27B")
    parser.add_argument(
        "--tuned-model",
        action="append",
        default=None,
        help="Repeat for each fine-tune to score. Defaults to the merged LoRA "
        "checkpoint if not given.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional display label, once per model, base first. Derived from "
        "the paths if omitted.",
    )
    parser.add_argument("--dataset", default=f"{demo_dir}/datasets/sql-create-context")
    parser.add_argument("--num-examples", type=int, default=500)
    # 512 to match the reference DGX Spark runs, so numbers are comparable.
    # Measured here: answers never exceed ~230 chars, so this is slack, not need.
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--results-dir", default=f"{demo_dir}/results")
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Derived from the tuned model names if "
        "omitted, so separate comparisons cannot overwrite each other.",
    )
    args = parser.parse_args()

    tuned = args.tuned_model or [f"{demo_dir}/output/qwen3.8-27b-sql"]
    paths = [args.base_model] + tuned
    labels = resolve_labels(paths, args.label)
    prefix = args.prefix if args.prefix is not None else "_".join(
        slug(p) for p in tuned
    ) + "_"

    os.makedirs(args.results_dir, exist_ok=True)

    test_split = load_split(args.dataset)["test"]
    num = min(args.num_examples, len(test_split))
    test_data = test_split.select(range(num))
    ground_truth = [ex["answer"] for ex in test_data]
    norm_truth = [normalize_sql(g) for g in ground_truth]
    print(f"Evaluating {len(paths)} models on {num} held-out examples")
    for lab, p in zip(labels, paths):
        print(f"  {lab:40} {p}")

    # One pass per model; the base is generated once however many tuned models
    # are being compared against it.
    preds = {}
    for lab, path in zip(labels, paths):
        preds[lab] = evaluate_model(path, test_data, lab, args.max_new_tokens)

    norm = {lab: [normalize_sql(p) for p in preds[lab]] for lab in labels}
    correct = {lab: sum(n == g for n, g in zip(norm[lab], norm_truth)) for lab in labels}
    acc = {lab: correct[lab] / num * 100 for lab in labels}

    # Diagnostic only: exact match with no extraction and no quote handling.
    # Answers "was the output directly usable as-is", NOT "was the SQL right".
    raw_truth = [raw_exact(g) for g in ground_truth]
    raw_correct = {
        lab: sum(raw_exact(p) == t for p, t in zip(preds[lab], raw_truth))
        for lab in labels
    }
    raw_acc = {lab: raw_correct[lab] / num * 100 for lab in labels}

    base_label, tuned_labels = labels[0], labels[1:]
    width = max(len(lab) for lab in labels)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS ({num} examples)")
    print(f"{'=' * 60}")
    print(f"  {base_label:{width}} : {acc[base_label]:5.1f}%  ({correct[base_label]}/{num})")
    for lab in tuned_labels:
        delta = acc[lab] - acc[base_label]
        print(f"  {lab:{width}} : {acc[lab]:5.1f}%  ({correct[lab]}/{num})  {delta:+5.1f}pp")
    print(f"{'=' * 60}")
    print("  Diagnostic - directly usable with no post-processing:")
    for lab in labels:
        print(f"    {lab:{width}} : {raw_acc[lab]:5.1f}%  ({raw_correct[lab]}/{num})")
    print(f"{'=' * 60}")

    # Examples each fine-tune fixed, measured against the base model.
    fixed = {
        lab: [
            i
            for i in range(num)
            if norm[base_label][i] != norm_truth[i] and norm[lab][i] == norm_truth[i]
        ]
        for lab in tuned_labels
    }
    for lab in tuned_labels:
        print(f"\nExamples {lab} fixed ({len(fixed[lab])} total):\n")
        for i in fixed[lab][:5]:
            print(f"  Example {i + 1}:")
            print(f"    Question    : {test_data[i]['question']}")
            print(f"    Ground truth: {ground_truth[i]}")
            print(f"    {base_label}: {preds[base_label][i]}")
            print(f"    {lab}: {preds[lab][i]}\n")
        if not fixed[lab]:
            print("  (none)\n")

    # --- Chart -------------------------------------------------------------
    colors = [BASE_COLOR] + [
        TUNED_COLORS[i % len(TUNED_COLORS)] for i in range(len(tuned_labels))
    ]
    fig, ax = plt.subplots(figsize=(max(8, 2.6 * len(labels)), 5))
    bars = ax.bar(labels, [acc[lab] for lab in labels], color=colors, width=0.5)
    ax.set_ylabel("Exact Match Accuracy (%)")
    ax.set_title(f"SQL generation, {num} held-out examples")
    ax.set_ylim(0, 100)
    for bar, lab in zip(bars, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{acc[lab]:.1f}%",
            ha="center",
            fontweight="bold",
            fontsize=13,
        )
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    chart_path = os.path.join(args.results_dir, f"{prefix}accuracy_comparison.png")
    plt.savefig(chart_path, dpi=150)
    print(f"Chart  : {chart_path}")

    # --- JSON --------------------------------------------------------------
    results_path = os.path.join(args.results_dir, f"{prefix}results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "num_examples": num,
                "base_label": base_label,
                "models": [
                    {
                        "label": lab,
                        "path": path,
                        "is_base": lab == base_label,
                        "accuracy_pct": round(acc[lab], 2),
                        "correct": correct[lab],
                        "improvement_pct": round(acc[lab] - acc[base_label], 2),
                        "raw_exact_pct": round(raw_acc[lab], 2),
                        "raw_exact_correct": raw_correct[lab],
                        "fixed_vs_base": len(fixed.get(lab, [])),
                    }
                    for lab, path in zip(labels, paths)
                ],
                "examples": [
                    {
                        "question": test_data[i]["question"],
                        "schema": test_data[i]["context"],
                        "ground_truth": ground_truth[i],
                        "predictions": {
                            lab: {
                                "sql": preds[lab][i],
                                "correct": norm[lab][i] == norm_truth[i],
                            }
                            for lab in labels
                        },
                    }
                    for i in range(num)
                ],
            },
            f,
            indent=2,
        )
    print(f"JSON   : {results_path}")

    # --- Markdown ----------------------------------------------------------
    md_path = os.path.join(args.results_dir, f"{prefix}results.md")
    with open(md_path, "w") as f:
        f.write(f"# SQL exact-match results ({num} held-out examples)\n\n")
        f.write("| Model | Path | Accuracy | Correct | vs base | Usable as-is |\n")
        f.write("|---|---|---|---|---|---|\n")
        for lab, path in zip(labels, paths):
            delta = "-" if lab == base_label else f"{acc[lab] - acc[base_label]:+.1f}pp"
            f.write(
                f"| {lab} | `{path}` | {acc[lab]:.1f}% | "
                f"{correct[lab]}/{num} | {delta} | {raw_acc[lab]:.1f}% |\n"
            )
        f.write(
            "\n**Accuracy** extracts the SQL from the response (markdown fence or "
            "last `SELECT`) and normalises quote style, case and whitespace before "
            "comparing. **Usable as-is** is the same comparison with no extraction "
            "and no quote handling — a diagnostic for whether the output can be "
            "used without post-processing, not a measure of SQL correctness.\n\n"
        )
        for lab in tuned_labels:
            f.write(f"## Examples {lab} fixed ({len(fixed[lab])} total)\n\n")
            for i in fixed[lab][:5]:
                f.write(f"### Example {i + 1}\n\n")
                f.write(f"**Question:** {test_data[i]['question']}\n\n")
                f.write(f"**Ground truth:** `{ground_truth[i]}`\n\n")
                f.write(f"**{base_label}:** `{preds[base_label][i]}`\n\n")
                f.write(f"**{lab}:** `{preds[lab][i]}`\n\n")
            if not fixed[lab]:
                f.write("(none)\n\n")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
