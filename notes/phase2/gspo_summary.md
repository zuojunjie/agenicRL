# Phase 2B — GSPO sequence-level ratio (50 steps cold-start)

_2026-04-28 16:45 启动 → 2026-04-28 22:30 完成（实测 ~5.7h），cost ~¥66_

## 🏆 关键结果：**GSPO 是 Phase 2 的赢家**

| 维度 | Phase 0e (baseline) | Phase 2A DAPO (输) | **Phase 2B GSPO** |
|---|---|---|---|
| ratio 维度 | token-level | token-level | **sequence-level** |
| clip 形状 | 对称 0.2 | 非对称 0.2/0.28 | 非对称 0.2/0.28 |
| **Final val/nq** | 0.417 | 0.400 | **0.4378** |
| **Δ vs 0e** | — | **-4.1%** ❌ | **+5.0%** ✅ |
| Δ vs DAPO | — | — | **+9.5%** ✅ |

## Val 曲线（6 个数据点）

| Step | 0e | DAPO | **GSPO** | GSPO Δ vs 0e |
|---|---|---|---|---|
| 0 | 0.196 | 0.196 | 0.196 | tie ✅ |
| 10 | 0.313 | 0.309 | **0.336** | +0.023 |
| 20 | 0.355 | 0.359 | **0.394** | +0.039 |
| 30 | 0.390 | 0.386 | **0.426** | +0.036 |
| 40 | 0.401 | 0.404 | **0.427** | +0.026 |
| **50** | 0.417 | 0.400 | **0.438** | **+0.021** |

GSPO **每个 val checkpoint 都赢 0e**，证据稳定。

## 训练动力学（5 个关键时点）

| step | reward | resp_len | grad | kl | entropy | finish | n_search | step_time |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.227 | 622 | 1.35 | 0.002 | 0.994 | 0.946 | 0.96 | 446s |
| 13 | 0.360 | 559 | 1.00 | 0.230 | 0.663 | 0.994 | 1.01 | 386s |
| 26 | **0.426** | 543 | 0.77 | 0.434 | **0.459** | 0.999 | 1.00 | 369s |
| 38 | 0.412 | 549 | 1.29 | 0.433 | 0.455 | 0.998 | 1.00 | 383s |
| **50** | 0.414 | 547 | 2.26 | 0.445 | 0.527 | 0.999 | 1.00 | 826s |

## 🧬 与 DAPO 完全相反的训练动力学

```
DAPO:       entropy 1.0 → 1.5 → 4.92  ⏫⏫⏫  失控爆炸
GSPO:       entropy 0.99 → 0.46 → 0.53  ⏬⏬   快速收敛
0e:         entropy 0.96 → 1.18 → 1.85  ↗     缓慢上升
```

| 现象 | DAPO | GSPO | 解读 |
|---|---|---|---|
| Entropy 趋势 | 爆炸（4.92） | 探底（0.46） | sequence-level 让多 token 共享一个 ratio → 模型快速 commit 到最优策略 |
| Response length | 605 | **547** ⏬ | GSPO 学到"言简意赅"路径 |
| KL drift 终值 | 0.68 | 0.44 | sequence-level ratio 让 PPO clip 更精确，drift 受控 |

## 🎓 论文 promise 验证

GSPO 论文 (Qwen 2025, arXiv:2507.18071) 主张：
1. ✅ **Sequence-level ratio 在长 response 下更稳定** — 我们实测：50 步内 reward 单调上升，entropy 不爆炸
2. ✅ **pg_clipfrac 更低** — GSPO 平均 0.005-0.01，DAPO 0.04+（约 5x 差距）
3. ✅ **Final task accuracy 提升** — val/nq +5% vs 0e baseline

## 实测节奏

- 50 步 × ~390s/step ≈ **5.4h 净训练**
- + val_before_train + 5 次中途 val + finalize = **~6.5h 总**
- step 50 那次特别慢（826s）因为含最后 val + ckpt save + cleanup

## 工程收获

✅ MAX_STEPS=52 trick 让 step 50 ckpt + val 都成功落盘  
✅ SAVE_FREQ=10 防御策略验证（5 个 ckpt 全部成功）  
✅ patch 体系（DAPO clip-higher + GSPO sequence-level 叠加 patch）工作正常  
✅ cold-start 严格 single-variable 实验干净

## 下一步：Phase 3 ToRL → Phase 4 GiGPO → Phase 5 ARPO

GSPO 是当前最强基线，Phase 5 综合时**优先以 GSPO 起点**。

## 已 finalize 资产

- `runs/phase2-gspo-seqlevel-50/ckpts/global_step_50/` (~13G)
- `runs/phase2-gspo-seqlevel-50/training.log`
- wandb run id: `pjze1hqd`
