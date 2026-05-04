# Phase 3 — ToRL multi-turn tool reward shaping (cold-start, partial 40 步)

_2026-04-28 23:48 启动 → 2026-04-29 02:18 OOM → resume warm-start → 2026-04-29 17:08 user kill (partial)_

## 关键结果（含 shaping 校正）

| Phase | reward 函数 | step 40 val/nq | "raw EM" 估算 | Δ raw EM vs 0e |
|---|---|---|---|---|
| 0d | EM 二元 | 0.376 (step 40) | 0.376 | -10% |
| **0e** (baseline) | DFA 5 档 | **0.401** | **0.401** | (基线) |
| 2A DAPO | DFA 5 档 | 0.404 | 0.404 | -1% |
| **2B GSPO** | DFA 5 档 | **0.427** | **0.427** | **+6%** ⭐ |
| **3 ToRL** | **5档 + per-turn shaping** | 0.426 | **~0.380** | **-5%** ❌ |

⚠️ **重要纠正**：ToRL initial val 0.242 比 0e initial 0.196 高 0.046（shaping bonus 自带的"虚分"）。
**Apples-to-apples 比较应减掉 initial 差**：
- ToRL final 0.426 - shaping_premium 0.046 ≈ **raw EM 0.380**
- vs 0e raw EM 0.401 → **ToRL 实际比 0e 差 -5%**

## 🚨 关键洞察：shaping reward = 自我欺骗

per-turn shaping（重复惩罚 + 召回奖励）让模型**学会刷分**而不是**学会答题**：
- 模型 n_search 从 1.7 涨到 3.5（用满 max_turns=4）
- response_length 从 985 涨到 1937（接近 2K）
- 刷高 shaping bonus → val 数字涨
- **但 raw EM 准确率反而降了**（subtract shaping 后只有 ~0.380）

ToRL 论文成立的前提是 shaping coefficient 调好。我们 ±0.05 的设置在 50 步窗口下产生了 reward hacking 行为。

## Val 曲线（5 个 checkpoint）

| Step | val/nq | 来源 |
|---|---|---|
| 0 (initial) | 0.242 | recovrp3a (ToRL 初始 + shaping) |
| 10 | 0.353 | recovrp3a |
| 20 | 0.387 | recovrp3a |
| 30 | 0.391 | uyaxjdih (resume internal step 10) |
| **40** | **0.426** | **uyaxjdih (resume internal step 20)** ⭐ 终点 |

resume started from original step_20 ckpt, ppo_micro=16 防 OOM。

## 训练动力学（unified step 1-40）

| step | reward | resp_len | n_search | entropy |
|---|---|---|---|---|
| 1 | 0.254 | 663 | 1.01 | 0.957 |
| 10 | 0.343 | 709 | 1.15 | 0.907 |
| 20 | 0.408 | 961 | 1.64 | 0.885 |
| 24 | 0.429 | **1295** | 2.34 | 0.701 |
| 25-29 (resume start) | 0.41-0.45 | 985-1192 | 1.7-2.1 | 0.82-0.86 |
| 35 (unified) | 0.42 | 1700+ | 3.13 | 0.66 |
| **40 (unified)** | **0.41** | **1937** | **3.54** | **0.55** |

## 反常识发现

### 1. **Response_length 几乎翻 3x**（663 → 1937）
ToRL `+0.05/info_hit` 的 reward shaping **强烈鼓励长 trajectory**。模型学到"广搜深思"，每条 rollout 用 3+ search。

### 2. **n_search 翻 3.5x**（1.01 → 3.54）
模型用满 max_turns=4 的空间。但**用得多 ≠ 用得对**。

### 3. **OOM 在 step 22**
response_length 涨到 1294 时 ppo_micro=32 撑不住，actor backward OOM。修正用 ppo_micro=16 续训成功。

### 4. **训练慢 3x**（22 min/step vs 0e/2 的 7-8 min）
长 trajectory + multi-turn vLLM rollout = 计算量翻倍。

## 工程教训

### A. `max_obs_length` 是 per-turn cap，不是总 cap
我们设 `max_obs_length=800` 以为总 trajectory 不超 800。实际：4 turn × 800 = 3200 token 累积可能。**总 trajectory 没有 hard cap，模型能涨到接近 max_response × max_turns**。

### B. Shaping reward 必须有总 clamp
我们用 `clamp ±0.20`，但 base reward 0.1-1.0 已经够大，shaping 的边际收益让模型"贪婪刷分"。**未来需要乘性 cap**：`final = base × (1 + 0.05 × shaping)`。

### C. 多 turn 训练显存压力非线性增长
50 步内 response 不应让 KV cache 超 GPU 80%。**Phase 3 的下一版应该 max_response_length 350**（不是 500）+ `max_turns=3`（不是 4）。

### D. 续训的"分段配置"破坏了 single-variable
Phase 3 实际是：
- step 1-22: max_obs=800, ppo_micro=32
- step 23-40: max_obs=800, ppo_micro=16, warm-start

数学上 micro_batch 等价（gradient 同），但**整段训练失去了严格 single-variable 性质**。Phase 5 综合实验需注意。

## 结论：ToRL 实验"故事很好但数字没赢"

| 收获 | ✅/❌ |
|---|---|
| 验证了 multi-turn search 可以学（n_search 1.0→3.5） | ✅ |
| 验证了 shaping reward 鼓励长 trajectory | ✅ |
| **raw EM 准确率没赢 0e baseline** | ❌ **-5%** |
| 对未来 H800 硬件迁移有强需求（OOM + 慢） | ✅ |
| 经验：reward hacking 真实存在 | ✅ |

## 对 Phase 5 ARPO 的指导

ToRL 在 Phase 2-4 中**唯一输的实验**。Phase 5 综合时：
- ✅ **保留**：max_turns 控制 + multi-turn 探索
- ❌ **不保留**：per-turn shaping reward（hacking 太严重）
- 🔄 **替代方案**：把 retrieval reward 折进**最终 EM**而非每 turn 加分

## 已 finalize 资产

- `runs/phase3-torl-multitool-50/ckpts/global_step_20/` (~13G, recovrp3a 时代)
- `runs/phase3-torl-resume-step20/ckpts/global_step_20/` (~13G, resume final)
- `runs/phase3-torl-resume-step20/training.log`
- wandb：
  - `recovrp3a` (deleted by user)
  - `uyaxjdih` (resume run)
  - **`2bbe95xv` (phase3-torl-stitched-v2, 完整 unified)** ⭐

## 工程 cost

- 4×4090 跑 22h（2 段：5h OOM + 5h failed restart + 8h resume + 2h misc）
- 费用 ~¥250
- Phase 3 是这次实验里**算力 ROI 最差的一段**
