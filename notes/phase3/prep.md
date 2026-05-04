# Phase 3 — ToRL（Tool-integrated RL）预读

_2026-04-27 父 agent 在 Phase 1 训练时本地写_

## ToRL 核心一句话

把 search 当 **first-class tool**（不只是 reward 后验校验），引入 **multi-turn tool reward shaping**，让模型在使用工具的过程中也获得 RL 信号。

## 与 Phase 1 的算法 diff

| 维度 | Phase 1 (DeepSeek-R1) | Phase 3 (ToRL) |
|---|---|---|
| `max_turns` | 2 | **4** |
| Reward 信号粒度 | 终态（最后 answer + 全文 retrieval 召回） | 终态 + **每个 search turn 的局部信号** |
| 重复 query 惩罚 | ❌ | ✅ |
| 召回质量奖励 | ✅（全文一次性）| ✅（每 turn 累计） |

## 实施方案

### 改动 1：max_turns CLI override（无需 patch）
```bash
max_turns=4
```
直接在训练命令行覆盖，generation.py 已支持。

### 改动 2：扩展 reward function — `qa_em_format.compute_score_em` → `qa_em_torl.compute_score_em`

**新文件** `verl/utils/reward_score/qa_em_torl.py`，复用 Phase 1 DFA + 5 档分级，新增 2 项 shaping：

```python
# 在 compute_score_em 末尾加：
def compute_torl_shaping(solution_str, ground_truth):
    queries = extract_search_queries(solution_str)  # 抽 <search>...</search>
    info_blocks = extract_information_blocks(solution_str)

    # 重复 query 惩罚（同样的搜索词查多次没意义）
    repeat_penalty = 0
    seen = set()
    for q in queries:
        q_norm = normalize_answer(q)
        if q_norm in seen:
            repeat_penalty -= 0.05
        seen.add(q_norm)

    # 每 turn 召回奖励（这一轮的 information 是否包含 GT）
    per_turn_reward = 0
    for info in info_blocks:
        if any(normalize_answer(g) in normalize_answer(info)
               for g in ground_truth['target']):
            per_turn_reward += 0.05  # 每命中一次 +0.05

    return repeat_penalty + per_turn_reward
```

**最终 reward**：
```
final_reward = base_5_tier_reward + clamp(torl_shaping, -0.2, +0.2)
```

clamp 防止 shaping 主导（保持 base reward 仍是主导信号）。

### 改动 3：新建训练入口 `verl/trainer/main_ppo_torl.py`
复用 `main_ppo_format.py` 的 RewardManager，只把 `data_source → score_fn` 路由改到 `qa_em_torl.compute_score_em`。沿用 Phase 1+2 的 `_format` 后缀风格。

### 改动 4：新增 metrics
`generation.py` 已有 `valid_search_stats` → `env/number_of_valid_search`。我们再加：
- `env/number_of_unique_queries`（排重后的 query 数）
- `env/per_turn_retrieval_hit_rate`（每 turn 命中 GT 的占比）
- `env/repeat_query_count`

## Phase 3 实验设计

| 维度 | 值 |
|---|---|
| 起点 | Phase 2 final ckpt（取 DAPO/GSPO 中 val/nq 高的） |
| 入口 | `main_ppo_torl` |
| max_turns | 4 |
| reward | base 5 档 + ToRL shaping ±0.2 clamp |
| clip_ratio | 0.2 / 0.28（继承 Phase 2 DAPO patch） |
| loss_agg_mode | 取 Phase 2 胜出方（token or sequence） |
| 步数 | 50 |

## 看点

- **n_search 应明显上升**：Phase 1 max_turns=2 → 平均 1-1.5 search/sample；Phase 3 max_turns=4 应该到 2-3
- **unique_queries / total_queries 比率**：应接近 1（重复 query 惩罚生效）
- **per_turn_hit_rate 上升**：模型学会先粗搜后精搜
- **response_length 应再涨**：多 turn = 更长链
- **val/nq 提升**：最关键 — 工具使用得更好是否真翻译成最终答题准确？

## 风险 / 不确定性

1. **max_turns=4 可能让 response 长度爆炸 → OOM**：要看 Phase 1 平均 response_length 决定是否再调 max_response_length
2. **shaping 系数 ±0.05 可能太弱或太强**：Phase 3 跑完看 metrics 决定是否要 Phase 3.5 重调
3. **clamp ±0.2 是经验值**：理论上需要消融，但 50 步 phase 时间不够

## 时间 / 成本

- 50 步 × ~7-8 min/step（多 turn 慢）= **~6.5h** ≈ ¥75
- 这是 Phase 0-5 中除 Phase 5 之外最贵的，因为 multi-turn 计算量真的翻倍

## 与 Phase 4 GiGPO 的承接

Phase 4 GiGPO 改 advantage 计算（两层归一化），完全不依赖 ToRL 的 reward shaping。所以 Phase 3 失败也可以跳过直接 Phase 4。但**ToRL ckpt 是 Phase 5 ARPO 的最佳起点**（ARPO 强调 trajectory-level credit assignment，需要先有"会用工具"的模型）。

## 论文待读

- ToRL paper（具体哪篇论文，subagent 跑训练时去查）
- 比较 ToRL 与 ReAct / Toolformer 的差异（前者是 RL 训出来的，后者是 prompting/SFT）
