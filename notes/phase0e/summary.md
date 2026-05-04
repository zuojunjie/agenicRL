# Phase 0e — GRPO + Format DFA Reward (50 steps cold-start)

_2026-04-27 19:18 启动 → 2026-04-28 02:18 完成（实测 7h），cost ~¥80_

## 关键结果

| 指标 | 值 | vs Phase 0d 0.376 |
|---|---|---|
| **Final val/nq** | **0.417** | **+0.041 / +11% 相对** ✅ |
| Initial val/nq | 0.196 | 与 Phase 0d 同起点 |
| Final train reward (step 48) | 0.410 | — |

## Val 曲线

| Step | val/test_score/nq |
|---|---|
| 0 (initial) | 0.196 |
| 10 | 0.313 (+60%) |
| 20 | 0.355 |
| 30 | 0.390 |
| 40 | 0.401 |
| Final (post-50) | **0.417** |

## 训练动力学（关键发现）

| 维度 | step 10 | step 20 | step 30 | step 40 | step 48 |
|---|---|---|---|---|---|
| reward mean | 0.319 | 0.358 | 0.394 | 0.422 | 0.410 |
| response_length | 624 | 612 | 600 | 590 | 586 |
| actor/grad_norm | 1.49 | 1.18 | 1.39 | 1.18 | 1.52 |
| actor/kl_loss | 0.04 | 0.18 | 0.24 | 0.51 | 0.72 ⏫ |
| **actor/entropy_loss** | 0.96 | 1.04 | 1.18 | 1.44 | **1.85** ⏫⏫ |
| env/finish_ratio | 0.97 | 0.99 | 0.99 | 0.99 | 1.00 |
| env/number_of_valid_search | 1.03 | 1.04 | 1.03 | 1.03 | 1.00 |

## 反常识发现：entropy + KL 双双单调上升

普通 RL 训练应该是 entropy 下降（模型变 confident）+ KL 收敛。**Phase 0e 反过来**：
- entropy 从 0.96 升到 **1.85**（+93%）
- KL 从 0.04 升到 **0.72**（+1700%）

**解释**：5 档分级 reward + 4 标签 DFA 给了**多条等效高分路径**（带不带 search、search query 风格、reasoning 文笔），模型不必收敛单一答案风格。R1 论文里"emergent diversity"现象在我们这里有清晰量化证据。

## response_length 趋势

624 → 586，**轻微下降**（-6%）。说明模型学会"言简意赅"——保留 DFA 合规结构但减少冗余推理。

## 实测节奏

- 50 步 × ~440s/step ≈ **6.1h 净训练**
- + val_before_train (~15min) + 5 次中途 val + finalize = **总 7h**
- 实际比 Phase 0d (5h, 44 步) 慢 ~40%，原因是 ppo_micro 64→32 + vllm util 0.4→0.3 的 OOM 防御

## 工程收获

✅ SAVE_FREQ=5 验证有效，9 个 ckpt 全部成功落盘  
✅ Phase 0d "ckpt 假死" 案没重演  
⚠️ 50 步刚好踩在 save_freq=5 节点上，但 step_50 ckpt 没落（verl 的逻辑问题），实际 warm-start 起点为 **step_45**（有效，损失 ~5 步训练效果）  
⚠️ search 返回的 information 频繁超 max_obs_length=500（实际 545-599）→ 信息丢失 ~10%，Phase 3 ToRL 应改 max_obs_length=800  
❌ 夜间接力 subagent 在 Phase 0e 完成 6h 后仍未触发 finalize，**失职** → 早晨手动接手

## 下一步：Phase 2A DAPO

- BASE_MODEL: `runs/phase0e-format-reward-50/ckpts/global_step_45`
- 唯一变量：`clip_ratio_high=0.28`（vs 对称 0.20）
- 入口仍 `main_ppo_format`（reward 不变）
- 50 步预计 ~5h
