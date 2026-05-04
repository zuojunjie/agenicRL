# Phase 2 — DAPO 预读（父 agent 在 Phase 1 训练时本地写）

_2026-04-27 _

## DAPO 4 大改进（按本 phase 实施优先级）

| # | 改进 | 改动量 | 本 phase 选 |
|---|---|---|---|
| 1 | **Clip-Higher**：PPO ratio 上界从对称 `[1-ε, 1+ε]` 改非对称 `[1-ε_low, 1+ε_high]` | 极小，5 行 | ✅ |
| 2 | **Dynamic Sampling**：动态过滤 reward 全 0 / 全 1 的 batch（梯度 = 0 浪费 compute） | 中等，~30 行 + sampler 改 | ⏸ 留 Phase 2.5 |
| 3 | **Token-level Loss**：长 response 的 mean → 总 token 上 sum-and-divide | 小，5 行 但要懂 batch math | ⏸ 留 Phase 2.5 |
| 4 | **Overlong Reward Shaping**：超 max_response_length 截断的 sample 给软惩罚而非硬截 | 中等 | ❌ |

**理由**：clip-higher 是单变量改动里最干净、收益最显著的。其余 3 项与 advantage normalization、batch 组织耦合，先看 #1 单独效果再叠加。

## Clip-Higher 代码定位（已读）

文件：`verl/trainer/ppo/core_algos.py:163 compute_policy_loss`

当前代码（line 190）：
```python
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
```

DAPO patch（拟）：
```python
def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask,
                        cliprange, cliprange_high=None):
    cliprange_high = cliprange_high if cliprange_high is not None else cliprange
    ...
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange_high)
```

调用点 `dp_actor.py:246-256` 也要传 `cliprange_high`：
```python
clip_ratio_high = self.config.get('clip_ratio_high', clip_ratio)
... compute_policy_loss(..., cliprange=clip_ratio, cliprange_high=clip_ratio_high)
```

CLI override：
```bash
+actor_rollout_ref.actor.clip_ratio_high=0.28  # baseline 0.2 → DAPO 0.28
```

## Phase 2 实验设计

| 配置 | Phase 1 | Phase 2 |
|---|---|---|
| Base model | Qwen2.5-3B-Instruct | **Phase 1 final ckpt（warm-start）** |
| Reward | format DFA | format DFA（不变） |
| Clip range | 对称 0.2 / 0.2 | 0.2 / **0.28** |
| Steps | 50 | 50 |
| 其他 | 不变 | 不变 |

⚠️ **依赖 Phase 1 ckpt 真落盘**。Phase 1 已配 SAVE_FREQ=5，subagent 已 arm step 5 ckpt 验证 monitor。如果 Phase 1 仍然没保住 ckpt（重蹈 Phase 0 覆辙），Phase 2 也只能 cold-start。

## 看点

- 📈 **entropy_loss**：clip-higher 给 explore 留窗口，entropy 应**回升或缓降**而非快速塌
- 📈 **val/nq**：因为 explore 更多，前期可能波动大，但应突破 Phase 1 final 值
- 📊 **pg_clipfrac**：clip 命中率，看上界放宽是否真的释放了梯度
- 🔍 **是否出现 "Aha moment"** 比 Phase 1 更明显（更多 explore = 更可能撞到反思链）

## 论文待读

- DAPO paper（ByteDance, 2025）§3.2 clip-higher 的消融数据（论文报告 +5-10% pass@1）
- 比较 Search-R1 baseline 和 DAPO 在 NQ 上的 reward curve

## 时间预算

50 步 × 6.7min ≈ 5.6h / ¥65（单 phase 上限按 ¥70 预留）
