"""
Phase -1 烟雾测试：验证本地 inference 通路通畅。

目标：
  1. 加载 Qwen2.5-0.5B-Instruct 到 MPS
  2. 复用 Search-R1 infer.py 的 agent loop 结构
  3. mock 掉 search() 函数（不依赖 retrieval_server）
  4. 跑通 1 条样本

不目标（这是烟雾测试，不是性能测试）：
  - 不要求 0.5B-Instruct 模型能正确生成 <search> 标签
  - 不要求最终答案正确
  - 只要不崩、能完整走完一次循环就算通过

用法：
  .venv/bin/python scripts/smoke_test.py
"""
from __future__ import annotations
import re
import time
import torch
import transformers


MODEL_PATH = "models/Qwen2.5-0.5B-Instruct"

# Qwen2.5 系列 EOS token id
QWEN_EOS = [151645, 151643]

# Search-R1 的 prompt 模板（与 infer.py 完全一致）
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

SEARCH_TEMPLATE = "\n\n{output_text}<information>{search_results}</information>\n\n"


def mock_search(query: str) -> str:
    """假装查到了几条相关文档，返回格式与真 retrieval_server 一致。"""
    return (
        f"Doc 1(Title: Mock document for '{query}') "
        f"This is a placeholder retrieved snippet about '{query}'. "
        "It pretends to contain relevant information.\n"
        f"Doc 2(Title: Another mock for '{query}') "
        "More placeholder content here for smoke-testing only.\n"
    )


def get_query(text: str) -> str | None:
    """从模型输出里提取最近一次 <search>...</search> 的内容。"""
    matches = re.findall(r"<search>(.*?)</search>", text, re.DOTALL)
    return matches[-1] if matches else None


class StopOnSequence(transformers.StoppingCriteria):
    """遇到任意目标序列就停止生成（与 infer.py 一致）。"""

    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [
            tokenizer.encode(s, add_special_tokens=False) for s in target_sequences
        ]
        self.target_lens = [len(ids) for ids in self.target_ids]

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < min(self.target_lens):
            return False
        for tids, tlen in zip(self.target_ids, self.target_lens):
            target = torch.as_tensor(tids, device=input_ids.device)
            if torch.equal(input_ids[0, -tlen:], target):
                return True
        return False


def main():
    # 1. 选 device：优先 MPS（Apple Silicon GPU），退化 CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16  # MPS 不完美支持 bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    print(f"[smoke] device={device}, dtype={dtype}")

    # 2. 加载 tokenizer + 模型
    t0 = time.time()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=dtype
    ).to(device)
    model.eval()
    print(f"[smoke] model loaded in {time.time() - t0:.1f}s")

    # 3. 准备 stopping criteria（停在 </search>）
    target_sequences = [
        "</search>", " </search>", "</search>\n", " </search>\n",
        "</search>\n\n", " </search>\n\n",
    ]
    stop = transformers.StoppingCriteriaList(
        [StopOnSequence(target_sequences, tokenizer)]
    )

    # 4. 构造 prompt（chat template wrap）
    question = "What is the capital of France?"
    raw = PROMPT_TEMPLATE.format(question=question)
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw}],
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        prompt = raw

    print("\n========== INITIAL PROMPT ==========")
    print(prompt)
    print("====================================\n")

    # 5. agent loop（最多 3 轮，防止 0.5B 失控）
    MAX_TURNS = 3
    for turn in range(MAX_TURNS):
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attn = torch.ones_like(ids)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                ids,
                attention_mask=attn,
                max_new_tokens=256,                  # 烟雾测试，给少点
                stopping_criteria=stop,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
            )
        dt = time.time() - t0

        new_tok = out[0][ids.shape[1]:]
        new_text = tokenizer.decode(new_tok, skip_special_tokens=True)
        last_id = out[0][-1].item()

        print(f"[turn {turn}] generated {len(new_tok)} tokens in {dt:.1f}s "
              f"({len(new_tok)/dt:.1f} tok/s), last_id={last_id}")
        print("=" * 8 + " GENERATED " + "=" * 8)
        print(new_text)
        print("=" * 27)

        # EOS：终止
        if last_id in QWEN_EOS:
            print("\n[smoke] hit EOS, agent loop terminated normally.")
            break

        # 否则尝试提取 search query
        query = get_query(tokenizer.decode(out[0], skip_special_tokens=True))
        if query is None:
            print(f"\n[smoke] no <search> tag found and no EOS — model probably "
                  f"didn't follow the schema. This is expected for an untrained "
                  f"0.5B model. Smoke test still PASSES (mechanism works).")
            break

        # 调 mock search
        results = mock_search(query.strip())
        print(f"\n[mock_search] query='{query.strip()}'")
        print(f"[mock_search] returned: {results[:120]}...")

        # 拼回 prompt 继续 loop
        prompt += SEARCH_TEMPLATE.format(
            output_text=new_text, search_results=results
        )
    else:
        print(f"\n[smoke] hit MAX_TURNS={MAX_TURNS} without finishing. OK for smoke test.")

    print("\n[smoke] ✅ PASS — agent loop mechanism works end-to-end.")


if __name__ == "__main__":
    main()
