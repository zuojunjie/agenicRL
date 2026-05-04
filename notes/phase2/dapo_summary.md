# Phase 2A — DAPO clip-higher (50 steps cold-start)

_2026-04-28 09:00 启动 → 16:28 完成（实测 ~7.5h）_

## 🎯 核心结果：**DAPO 输给 Phase 0e**

| 维度 | Phase 0d (baseline) | Phase 0e | **Phase 2A DAPO** |
|---|---|---|---|
| reward 函数 | 二元 EM | 5档 DFA | 5档 DFA（同 0e） |
| clip range | 对称 0.2 / 0.2 | 对称 0.2 / 0.2 | **非对称 0.2 / 0.28** |
| Final val/nq | 0.376 (step 40 OOM) | **0.417** | **0.400** ⬇️ |
| 与 Phase 0e Δ | — | — | **-0.017 / -4%** ❌ |

**意外结论**：clip-higher 在我们 50 步小窗口下**反而比 vanilla GRPO 差**。

## Val 曲线（5 个数据点）

| Step | 0e | DAPO | Δ |
|---|---|---|---|
| 0 | 0.196 | 0.196 | 0 ✅ |
| 10 | 0.313 | 0.309 | -0.004 |
| 20 | 0.355 | 0.359 | +0.004 |
| 30 | 0.390 | 0.386 | -0.004 |
| 40 | 0.401 | 0.404 | +0.003 |
| 50 (final) | **0.417** | **0.400** | **-0.017** ❌ |

step 0-40 时 DAPO 与 0e 在 ±0.004 噪声内基本打平；**最后 10 步 DAPO 反而下滑了 -0.017**。

## 🚨 病理诊断：Entropy 爆炸

DAPO 的 actor/entropy_loss 演化：

```
1.00 (step 6) → 1.50 (step 28) → 4.92 (step 48) ← 指数失控
```

vs Phase 0e 的稳健爬升 0.96 → 1.85（线性）。

DAPO 后期 entropy 从 1.5 飙到 4.9，模型变得"过度随机"。

## 反直觉发现：DAPO 论文 promise 在 50 步窗口下不成立

| DAPO 论文承诺 | 我们 50 步实测 |
|---|---|
| Sample efficiency +5-10% pass@1 | ❌ -4% |
| Entropy preservation | ✅ 但**爆炸式过度**，反而 hurt |
| Stability via clip-higher | ❌ 后期 entropy 失控 = 不稳定 |
| KL drift 减少 | ⚠️ 早期低（step 18 = 0.119 vs 0e 0.184），中后期收敛 |

## 可能的原因（待验证）

1. **clip 上界 0.28 对小步数任务太激进**
   - DAPO 论文配方在 200+ 步训练中，前 50 步本来就是"warm up"，不指望优势
   - 我们整个训练就 50 步 = 全程都在"warm up"，没机会进入 clip-higher 真正发挥作用的稳态期
2. **5 档 DFA reward 与 clip-higher 互不相容**
   - DFA 给"format 合规但 answer 错"的样本 0.2-0.3 partial credit
   - clip-higher 让模型对这些 partial credit 行为放大太快
   - 模型学会"刷 format 合规"而不是"答对问题"
3. **熵爆炸来自 reward variance**
   - 5 档 reward 有 5 个值，标准差比二元 EM 大很多
   - clip-higher × 大方差 advantage = 大幅 update = entropy 上升
   - 50 步内未到 KL penalty 兜住的平衡点

## 训练动力学完整对比

| step | metric | Phase 0e | DAPO |
|---|---|---|---|
| 10 | reward | 0.319 | 0.309 |
| 10 | entropy | 0.96 | 1.00 |
| 10 | kl | 0.04 | 0.012 |
| 20 | reward | 0.358 | 0.359 |
| 20 | entropy | 1.04 | 1.11 |
| 20 | kl | 0.18 | 0.119 |
| 30 | reward | 0.394 | 0.386 |
| 30 | entropy | 1.18 | 1.50 |
| 30 | kl | 0.24 | 0.279 |
| 48 | reward | 0.410 | 0.416 |
| 48 | entropy | **1.85** | **4.92** ⚠️ |
| 48 | kl | 0.72 | 0.68 |

## 学习教训（写进 memory）

1. **DAPO 论文的优势是 200+ 步窗口现象**，50 步小窗口下风险大于收益
2. **clip-higher × 5档分级 reward 是个危险组合** —— 每个分级都成"局部最优"，模型来回横跳
3. **Entropy 爆炸是 RL 训练失败的明确信号**，应该用 entropy_coef regularization 兜住
4. **Phase 0e 这种"vanilla GRPO + DFA 5档 reward"在 50 步窗口下反而是甜蜜点**

## 后续 Phase 2B GSPO 的 implication

GSPO（sequence-level ratio）原本是为了稳定 long-sequence 训练，**应该比 DAPO 更稳**。预期：
- Final val/nq 接近或略高于 Phase 0e
- Entropy 不会像 DAPO 那样爆炸（sequence-level clipping 给所有 token 共享一个 ratio，避免单 token 过度更新）

如果 GSPO 也跑输 0e，说明在 50 步窗口下**改 PPO clip 不如改 reward 函数**，回到 Phase 0e 风格。

## 已 finalize 资产

- `runs/phase2-dapo-cliphigh-50/ckpts/global_step_45/` (~13G)
- `runs/phase2-dapo-cliphigh-50/training.log`
- `runs/phase2-dapo-cliphigh-50/metrics.csv`
- 飞书 wiki: 待推

## 工程 cost

- 实跑 ~7.5h
- 费用 ~¥86
- 学到的"反向证据"：clip-higher 不是 universal good
- 经济价值：避免在后续 phase 复用 clip-higher，节省试错成本
