# Search-R1 代码层心智模型

> **范围**：精读 `external/Search-R1/infer.py` (135 行) + `search_r1/search/retrieval_server.py` (393 行) 的笔记
> **来源**：master agent · `notes/agent-loop.md`
> **配套阅读**：先看《决策档案》了解为什么是 Search-R1；本篇是其代码骨架剖析

---

## 1. 整体架构（两个进程）

```
┌──────────────────────┐         POST /retrieve       ┌────────────────────────┐
│   Policy LLM 进程     │  ───────────────────────────▶│  Retrieval Server 进程  │
│   (vLLM / HF gen)    │  ◀─────────────────────────  │  (FastAPI + faiss + E5) │
│                      │      JSON {result: [...]}    │   :8000                │
└──────────────────────┘                              └────────────────────────┘
```

**关键**：训练时和 inference 时，policy LLM 和 retriever 始终是**两个独立进程**，通过 HTTP 通信。

## 2. Agent loop（infer.py 核心 60 行）

```python
prompt = f"""... <think>...</think> <search>...</search> <information>...</information> <answer>...</answer> ...
Question: {q}"""

while True:
    out = model.generate(
        prompt,
        max_new_tokens=1024,
        stopping_criteria=StopOnSequence(["</search>", " </search>", ...]),
    )

    # 终止条件：生成到 EOS（151645/151643 for Qwen2.5）
    if out[0][-1] in [151645, 151643]:
        print(decode(out)); break

    # 否则一定是停在了 </search>
    query = re.findall(r"<search>(.*?)</search>", decode(out))[-1]
    docs  = requests.post("http://127.0.0.1:8000/retrieve",
                          json={"queries": [query], "topk": 3, "return_scores": True}).json()
    snippets = format_docs(docs)        # "Doc 1(Title:...)...\nDoc 2..."

    prompt += f"\n\n{decode(out)}<information>{snippets}</information>\n\n"
```

### 2.1 Stopping criterion 的优雅之处

`StopOnSequence(["</search>", " </search>", "</search>\n", ...])`：

- 列举各种空格/换行组合，确保模型一旦生成完"求搜索"就立刻停
- 防止模型自己幻想出 `<information>` 内容（**重要！否则会自我中毒**）
- 这是把 **"工具调用边界"硬编码进 generation 中断**的简洁实现

### 2.2 Prompt template（行 20–24）

```
Answer the given question.
You must conduct reasoning inside <think> and </think> first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>
and it will return the top searched results between <information> and </information>.
You can search as many times as your want.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>.
For example, <answer> Beijing </answer>.
Question: {q}
```

未训练的 base/instruct 模型遇到这个 prompt 多半**不会**自发生成 `<search>` 标签 — 这正是 RL 要解决的：**通过奖励让模型学会用工具**。

## 3. Retrieval server（retrieval_server.py 核心抽象）

### 3.1 类层级

```
BaseRetriever (抽象)
├── BM25Retriever           # 用 pyserini.LuceneSearcher，CPU 即可
└── DenseRetriever          # 用 faiss + Encoder
    └── Encoder
        ├── E5      → "query: ..." / "passage: ..." 前缀
        ├── BGE     → "Represent this sentence ..." 前缀
        └── DPR/T5  → 不加前缀
```

### 3.2 FastAPI 路由

```python
POST /retrieve  →  retriever.batch_search(...)  →  {"result": [[doc, doc, ...], ...]}
```

请求体（来自 `infer.py.search()`）：

```json
{"queries": ["..."], "topk": 3, "return_scores": true}
```

响应（每个 query 一组 docs）：

```json
{"result": [[
    {"document": {"id": "0", "contents": "title\ntext..."},
     "score": 0.83},
    ...
]]}
```

### 3.3 ⚠️ Mac 不能直接跑

代码里有硬编码 CUDA：

- `model.cuda()` (L42)
- `inputs = {k: v.cuda() for ...}` (L97)
- `torch.cuda.empty_cache()` (L121, L266)

**要在 Mac 上跑，必须 patch 成 device-aware**。但本地不跑真服务器 — 直接 mock 即可（见 Phase -1 烟雾测试）。

## 4. 训练时 reward 是怎么算的（推测，待 Phase 0 验证）

`infer.py` 不涉及奖励计算，但根据论文（arxiv 2503.09516）：

```
trajectory = [<think>... <search> q1 </search>
              <information> docs </information>
              <think>... <search> q2 </search> ...
              <answer> A </answer>]

reward = 1 if exact_match(A, gold_answer) else 0
       (或 F1, 视 reward_model.style 而定)
```

**特点**：

- **Outcome reward**，对整条轨迹打一个分
- 没有过程奖励（PRM）— 这正是 GiGPO/ARPO 后面要解决的问题
- 检索到的 `<information>` token 在计算 loss 时**必须 mask 掉**（不是模型输出，是环境返回）

后一点是**复现的常见坑**：如果不 mask，模型会被迫"模仿"检索结果，胡乱地把搜索文本当成自己的输出来学。verl 的 multiturn loss masking 就是干这个的。

## 5. 与 GRPO 的连接（前瞻 Phase 1）

GRPO（DeepSeek-R1）的核心：

```
对同一个 prompt q，采样 G 条 trajectory τ₁..τ_G
计算每条的 reward r₁..r_G
归一化得 advantage Â_i = (r_i - mean) / std
按 token-level policy gradient 更新
```

把"trajectory"换成 Search-R1 的"含搜索的多轮轨迹"，GRPO 就直接迁移过来了。

> **Search-R1 = GRPO + 多轮工具调用 + outcome reward + token mask**

这就是为什么 5 篇论文路径里 R1 必须排第一 — 它是基础范式。

## 6. 后续阶段在哪里改这个 loop

| Phase | 在哪里加东西 |
|---|---|
| **P2 DAPO** | 改 GRPO 的 advantage 计算 / clip 系数 / token-level loss / sample 策略，**不动 agent loop 本身** |
| **P2 GSPO** | 改 importance sampling 粒度（token → sequence），同样不动 loop |
| **P3 ToRL** | 把 search 工具替换/扩展为 code interpreter，改 reward 设计 |
| **P4 GiGPO** | **在 loop 上加东西**：每次 `<search>...<information>` 段视为一个 step，分别打 advantage |
| **P5 ARPO** | 在 generation 时基于 entropy 决定 rollout 是否分支，改的是采样策略 |

---

**结论**：`infer.py` 这 60 行就是整个项目的"骨架"。后续 5 个 phase 的所有改动，都是在这个骨架上**做手术**。掌握了它，就掌握了 Search-R1 的全部。
