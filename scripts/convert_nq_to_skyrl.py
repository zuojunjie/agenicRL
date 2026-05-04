#!/usr/bin/env python3
"""Convert 4090 verl nq_search parquet → SkyRL-expected schema.

verl schema:    data_source, prompt, ability, reward_model, extra_info
SkyRL schema:   data_source, prompt, ability, env_class, reward_spec, extra_info, metadata

Adds:
  - env_class = "search"
  - reward_spec = reward_model (renamed)
  - metadata = None
  - extra_info += need_tools_kwargs, tools_kwargs (search config), question

Keeps prompt format (single user message) → matches 4090 baseline conditioning.
"""
import argparse
import os
import pandas as pd


def convert(in_path: str, out_path: str):
    df = pd.read_parquet(in_path)
    print(f"loaded {len(df)} rows from {in_path}")
    print(f"original columns: {list(df.columns)}")

    # Rename reward_model → reward_spec
    if "reward_model" in df.columns:
        df = df.rename(columns={"reward_model": "reward_spec"})

    # Add env_class
    df["env_class"] = "search"
    df["metadata"] = None

    # Augment extra_info with SkyRL search env's expected fields
    def augment(row):
        info = dict(row["extra_info"]) if row["extra_info"] is not None else {}
        info["need_tools_kwargs"] = True
        # extract question from the user prompt
        try:
            prompt = row["prompt"]
            user_msg = prompt[0]["content"] if isinstance(prompt, (list, tuple)) and len(prompt) > 0 else ""
            # NQ prompt ends with "Question: <q>?"
            q_start = user_msg.rfind("Question:")
            question = user_msg[q_start + len("Question:") :].strip().rstrip("?").strip() if q_start >= 0 else ""
        except Exception:
            question = ""
        info["question"] = question
        info["tools_kwargs"] = {
            "search": {
                "create_kwargs": {
                    "ground_truth": row["reward_spec"]["ground_truth"],
                    "question": question,
                    "data_source": row.get("data_source", "nq"),
                }
            }
        }
        return info

    df["extra_info"] = df.apply(augment, axis=1)

    # Reorder to SkyRL canonical order
    df = df[["data_source", "prompt", "ability", "env_class", "reward_spec", "extra_info", "metadata"]]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path)
    print(f"wrote {len(df)} rows to {out_path}")
    print(f"new columns: {list(df.columns)}")
    print(f"first row sample:")
    for col in df.columns:
        v = df.iloc[0][col]
        if isinstance(v, str):
            print(f"  {col}: {v[:200]!r}")
        else:
            print(f"  {col}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    convert(args.src, args.dst)
