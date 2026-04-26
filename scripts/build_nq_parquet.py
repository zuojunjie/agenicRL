"""
读 nq/{train,test,dev}.jsonl → 包成 Search-R1 prompt 格式 → 输出 parquet。

替代 external/Search-R1/scripts/data_process/nq_search.py（后者用 datasets.load_dataset
拉整个 RUC-NLPIR/FlashRAG_datasets repo 的复杂 metadata，在国内网络下会卡死）。

依赖（运行前要装好）：pandas, pyarrow

用法（云端 conda env 下）：
    python scripts/build_nq_parquet.py
"""
import json
import os
import pandas as pd

DATA_DIR = "/root/autodl-tmp/data/nq_search"

# 与 external/Search-R1/scripts/data_process/nq_search.py 的 prompt 完全一致
PROMPT_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between "
    "<information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer "
    "inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. Question: {question}\n"
)


def make_row(example, idx, split):
    q = example["question"].strip()
    if not q.endswith("?"):
        q += "?"
    return {
        "data_source": "nq",
        "prompt": [{"role": "user", "content": PROMPT_TEMPLATE.format(question=q)}],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": example["golden_answers"]},
        },
        "extra_info": {"split": split, "index": idx},
    }


def convert(jsonl_path, parquet_path, split):
    rows = []
    with open(jsonl_path) as f:
        for idx, line in enumerate(f):
            ex = json.loads(line)
            rows.append(make_row(ex, idx, split))
    pd.DataFrame(rows).to_parquet(parquet_path)
    print(f"  {split:<5}: {len(rows):>6} rows -> {parquet_path}")


if __name__ == "__main__":
    convert(f"{DATA_DIR}/train.jsonl", f"{DATA_DIR}/train.parquet", "train")
    convert(f"{DATA_DIR}/test.jsonl",  f"{DATA_DIR}/test.parquet",  "test")
    # dev 集留 jsonl，不转 parquet（不进训练管道）

    print("\n抽样:")
    df = pd.read_parquet(f"{DATA_DIR}/train.parquet")
    sample = df.iloc[0]
    print(f"  prompt[:200]: {sample['prompt'][0]['content'][:200]}...")
    print(f"  golden_answers: {sample['reward_model']['ground_truth']['target']}")
