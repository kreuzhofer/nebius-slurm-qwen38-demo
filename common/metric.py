"""
metric.py -- how a predicted SQL string is compared to ground truth.

THIS FILE DEFINES THE PUBLISHED NUMBER. Every model in this repo must score
through it unchanged, or results stop being comparable -- across models here,
and against the reference runs it was ported from.

Ported verbatim from `scripts/evaluate.py` in
kreuzhofer/dgx-manager-fine-tune-recipes, the implementation behind the
Qwen3.6/3.8 numbers measured on DGX Spark. Adopting it rather than inventing our
own is deliberate: see docs/RESULTS.md for what happened when a local variant
had drifted from it.
"""

import re

def raw_exact(sql):
    """
    The old normalisation: lowercase, collapse whitespace, drop a trailing `;`.

    Kept only as a **diagnostic**, never as the headline. It answers "did the
    model emit SQL that is directly usable with no post-processing at all",
    which is a real question -- but it is not a measure of SQL correctness, and
    reporting it as one is how this repo previously scored the base model at 1%.
    """
    return " ".join(sql.lower().strip().rstrip(";").split())


def normalize_sql(sql):
    """
    Extract the SQL from a model response, then normalise it for comparison.

    Ported from `scripts/evaluate.py` in kreuzhofer/dgx-manager-fine-tune-recipes,
    which is the reference implementation used for the Qwen3.6/3.8 numbers
    measured on DGX Spark. Adopting it verbatim is deliberate: it makes results
    from this repo directly comparable with those, instead of incomparable.

    WHY THIS MATTERS, measured on 100 held-out examples with the base model:
    without the extraction and quote steps below, the base scores **1%**; with
    them it scores **58%**. The difference is entirely parsing. 72 of 100 base
    answers arrive inside a markdown fence, and the base writes ANSI-standard
    `'single'` quotes while this dataset stores non-standard `"double"` ones.
    Neither is a SQL error. Verified not to be a prompt artifact: across three
    prompt variants the base produced double quotes 0-1 times in 100.

    Handles three output styles:
      1. Plain SQL (raw SELECT ...)
      2. Closed markdown code block: ```sql ... ``` (chat-model style)
      3. Verbose reasoning output where SQL follows a label, possibly inside an
         unclosed backtick span, possibly truncated mid-thought.
    """
    if sql is None:
        # Thinking-mode responses that burn max_tokens on reasoning and never
        # emit SQL come back as content=null with finish_reason="length".
        # Count as a miss rather than crashing the accuracy computation.
        return ""

    s = sql.strip()

    # 1. Closed markdown code block -- most reliable signal.
    code_block = re.search(r"```(?:sql)?\s*\n?(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if code_block:
        s = code_block.group(1).strip()
    else:
        # 2. Otherwise take from the LAST SELECT, which is almost always the
        #    final answer, and cut at whatever follows it.
        matches = list(re.finditer(r"\bSELECT\b", s, re.IGNORECASE))
        if matches:
            tail = s[matches[-1].start():]
            cuts = [tail.find(c) for c in
                    [";", "```", "\n`", "\n\nNote", "\n\nThis ", "\n\nExplanation"]]
            cuts = [c for c in cuts if c > 0]
            if cuts:
                tail = tail[:min(cuts)]
            s = tail.strip().lstrip("`").strip()

    # Chat-template artefacts.
    for tag in ["<end_of_turn>", "<start_of_turn>", "<|im_end|>", "<|im_start|>",
                "model", "user"]:
        s = s.split(tag)[0]
    s = s.strip().rstrip(";").rstrip("`").strip()

    # Quote convention: the model writes ANSI '...', the dataset stores "...".
    s = s.replace("'", '"')
    return " ".join(s.lower().split())
