# Phase 2 — DAPO vs GSPO A/B 对照设计

## 共同起点 & 终点

| 维度 | 值 |
|---|---|
| 起点 model | Phase 1 final ckpt (`global_step_50`) |
| Reward | 4 标签 DFA + 5 档（继承 Phase 1，不变） |
| 数据 | NQ 单跳 |
| 步数 | 50 + 50 |
| 评估 | val/test_score/nq @ step 10/20/30/40/50 |

## 唯一变量

| 项 | DAPO | GSPO |
|---|---|---|
| 入口 | main_ppo_format | main_ppo_format |
| `clip_ratio_low / high` | 0.20 / 0.28 | 0.20 / 0.28 |
| `loss_agg_mode` | **token** | **sequence** |
| 论文 | DAPO (ByteDance 2025) | GSPO (Qwen 2025) |

## 核心数学差异（一行）

| | ratio 公式 | clip 维度 | adv 取法 |
|---|---|---|---|
| DAPO | `exp(log_pi_new(t) - log_pi_old(t))` per token | per token | per token |
| GSPO | `exp(mean_t [log_pi_new(t) - log_pi_old(t)])` per seq | per seq | seq mean |

## 假设要验证

- **H1（GSPO 论文主张）**：sequence-level ratio 在长 response（>200 tokens）下方差更小，训练更稳，pg_clipfrac 应明显低于 DAPO
- **H2**：DAPO 的 token-level + clip-higher 给 explore 信号更细，前期 entropy 衰减更慢但波动更大
- **H3 (中性)**：50 步窗口太短，两者 final val/nq 差 < 1%（在 noise 内）；真正显著差异要看 reward 曲线斜率

## 看点（仪表盘）

每 phase 收集：

| 指标 | DAPO | GSPO | 解读 |
|---|---|---|---|
| `actor/pg_clipfrac` | A1 | A2 | clip 命中率 — H1 的直接证据 |
| `actor/pg_loss` | B1 | B2 | 数值范围 — sequence-level 一般小 1-2 量级 |
| `actor/entropy_loss` | C1 | C2 | explore 强度 |
| `critic/rewards/mean` | D1 | D2 | reward 上升斜率 |
| `val/test_score/nq @ 50` | E1 | E2 | 终点 |
| `response_length/mean` | F1 | F2 | 长度趋势 |

## 资源 & 时间

- 顺序模式：DAPO 50 (~2.5h, ¥30) → GSPO 50 (~2.5h, ¥30) = 5h, ¥60
- 不并行原因：单实例 4 GPU 全占
- 共享起点：Phase 1 final ckpt 成果加倍利用

## 风险

1. **Phase 1 ckpt 可能不存在**（之前 Phase 0 教训）
   - 防御：Phase 1 SAVE_FREQ=5，subagent 已 arm step 5 ckpt 验证
   - Plan B：DAPO/GSPO 双 cold-start（¥不变，但失去 warm-start 学习信号叠加叙事）
2. **GSPO patch 与 ulysses sequence parallel 兼容性未测**：当前训练没开 ulysses（sp_size=1），patch 应该 OK
3. **GSPO 论文公式可能简化太狠**：实际实现要仔细对齐，特别是 advantage 是否要做 length-normalize（我当前 patch 做了）。可能需要小迭代

## 上线前的 patch 验证清单

1. patch 应用无冲突：`patch -p1 --dry-run` 都通过
2. token mode 应该退化为 DAPO 行为（即 GSPO patch 的 `loss_agg_mode='token'` 路径 = DAPO patch 之后的状态）— 二者差异 = 0
3. sequence mode 上 sanity check：随机 batch shape 一致、loss 是 scalar、loss.backward() 不报错
4. 跑 1 step smoke test 看 wandb 出图正常

## 失败回退

- DAPO 跑崩 → 直接 abandon Phase 2，跳 Phase 3
- GSPO 跑崩 → DAPO 数据已收，仍可作为 Phase 2 单边产出
- 两个都顺利 → A/B 对比写飞书周报，决定 Phase 3 起点用 DAPO ckpt 还是 GSPO ckpt（取 final val/nq 高的）
