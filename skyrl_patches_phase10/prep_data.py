"""Convert GSM8K + MATH datasets to SkyRL parquet format for Phase 10.

Produces:
   /root/autodl-tmp/data/math/gsm8k_train_52step.parquet  (5K rows ≈ 10 step × 512)
   /root/autodl-tmp/data/math/math_train_52step.parquet   (26K rows = 52 step × 512)
   /root/autodl-tmp/data/math/math_val.parquet            (MATH-500 test, 500 rows)

Usage:
    python prep_data.py --output-dir /root/autodl-tmp/data/math
"""
from __future__ import annotations

import argparse
import os
import re
import uuid
from typing import List

import pandas as pd

# System prompt template for math + Python tool use (Phase 10 baseline)
SYSTEM_PROMPT = """You are a math reasoning assistant. To solve a math problem, you can:
1. Reason step-by-step inside <think>...</think>
2. Execute Python code by writing <python>...</python> — you'll see the output between <output>...</output>
3. When ready, give the final answer inside <answer>...</answer>

You can call <python> as many times as needed (up to 8 turns). Available libraries: sympy, numpy, math, fractions, itertools, scipy.

Example:
<think>I need to find x such that x^2 + 5x - 14 < 0.</think>
<python>
import sympy as sp
x = sp.Symbol('x')
print(sp.solve(x**2 + 5*x - 14 < 0, x))
</python>
<output>(-7 < x) & (x < 2)</output>
<think>Largest integer x is 1.</think>
<answer>1</answer>

Question: {question}"""


def extract_math_answer(solution: str) -> str:
    """Extract final answer from MATH dataset solution (\\boxed{...})."""
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", solution)
    if boxed:
        return boxed[-1].strip()
    # GSM8K-style: #### 42 at end
    gsm8k = re.search(r"####\s*(.+?)(?:\s|$)", solution)
    if gsm8k:
        return gsm8k.group(1).strip()
    return solution.strip()


def convert_to_skyrl_format(question: str, answer: str, data_source: str = "math") -> dict:
    """Single row in SkyRL parquet schema.

    Schema mirrors searchR1/validation.parquet:
        prompt:        list[chat msg dict]
        data_source:   "math" / "gsm8k"
        ability:       "math"
        env_class:     "math_python"   ← key
        reward_spec:   {ground_truth: {target: [answer]}, style: "rule_math"}
        extra_info:    {}
        metadata:      {}
    """
    # Note: pyarrow can't serialize empty {}; use a dummy field to keep schema valid
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": SYSTEM_PROMPT.format(question=question)}],
        "ability": "math",
        "env_class": "math_python",
        "reward_spec": {
            "ground_truth": {"target": [answer]},
            "style": "rule_math",
        },
        "extra_info": {"_dummy": ""},
        "metadata": {"_dummy": ""},
        "uid": str(uuid.uuid4()),
    }


def _download_via_hf_hub(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    """Download a single file from HF mirror; returns local path."""
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)


def prep_gsm8k(out_path: str, n_rows: int = 5120):
    """GSM8K train: 5120 rows ≈ 10 steps × 512 batch (smoke test)."""
    print(f"Loading GSM8K from hf-mirror parquet directly...")
    # openai/gsm8k has parquet at main/train-00000-of-00001.parquet
    src = _download_via_hf_hub("openai/gsm8k", "main/train-00000-of-00001.parquet")
    df_src = pd.read_parquet(src)
    df_src = df_src.head(n_rows)
    rows = []
    for _, ex in df_src.iterrows():
        ans = extract_math_answer(ex["answer"])
        rows.append(convert_to_skyrl_format(ex["question"], ans, "gsm8k"))
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    print(f"  → {out_path} ({len(rows)} rows)")


def prep_math_train(out_path: str, n_rows: int = 26624):
    """MATH train: take all available train, then upsample to n_rows.

    Standard 'hendrycks/competition_math' has 7.5K train problems.
    Use 'lighteval/MATH' or fall back to 'qwedsacf/competition_math'.
    """
    print(f"Loading MATH train from hf-mirror parquet...")
    # Try lighteval/MATH first (newer, well-maintained)
    candidates = [
        ("lighteval/MATH", "default/train/0000.parquet"),
        ("DigitalLearningGmbH/MATH-lighteval", "data/train-00000-of-00001.parquet"),
        ("qwedsacf/competition_math", "data/train-00000-of-00001.parquet"),
    ]
    df_src = None
    for repo, fn in candidates:
        try:
            src = _download_via_hf_hub(repo, fn)
            df_src = pd.read_parquet(src)
            print(f"  loaded from {repo}/{fn} ({len(df_src)} rows)")
            break
        except Exception as e:
            print(f"  {repo}/{fn} failed: {type(e).__name__}")
    if df_src is None:
        raise RuntimeError("All MATH train sources failed; please add a working one")

    rows = []
    for _, ex in df_src.iterrows():
        # Different schemas: solution/answer field
        ans_field = ex.get("solution") if "solution" in df_src.columns else ex.get("answer")
        ans = extract_math_answer(ans_field)
        prob = ex.get("problem") or ex.get("question")
        rows.append(convert_to_skyrl_format(prob, ans, "math"))
    # If we don't have n_rows, repeat (with shuffle in launcher's data.shuffle_train_dataloader)
    base_rows = rows
    while len(rows) < n_rows:
        rows = rows + base_rows
    rows = rows[:n_rows]
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    print(f"  → {out_path} ({len(rows)} rows, base {len(base_rows)} unique)")


def prep_math_val(out_path: str, n_rows: int = 500):
    """MATH-500 val: standard MATH benchmark eval set."""
    print(f"Loading MATH-500 from hf-mirror parquet...")
    src = _download_via_hf_hub("HuggingFaceH4/MATH-500", "test.jsonl")
    # MATH-500 is jsonl
    import json
    with open(src) as f:
        items = [json.loads(line) for line in f]
    rows = []
    for ex in items[:n_rows]:
        ans = ex.get("answer") or extract_math_answer(ex.get("solution", ""))
        rows.append(convert_to_skyrl_format(ex["problem"], ans, "math"))
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    print(f"  → {out_path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/root/autodl-tmp/data/math")
    parser.add_argument("--skip-gsm8k", action="store_true")
    parser.add_argument("--skip-math", action="store_true")
    parser.add_argument("--skip-val", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.skip_gsm8k:
        prep_gsm8k(os.path.join(args.output_dir, "gsm8k_train_52step.parquet"))
    if not args.skip_math:
        prep_math_train(os.path.join(args.output_dir, "math_train_52step.parquet"))
    if not args.skip_val:
        prep_math_val(os.path.join(args.output_dir, "math_val.parquet"))

    print("✅ all parquet files ready")


if __name__ == "__main__":
    main()
