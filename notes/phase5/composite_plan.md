# Phase 5 — ARPO 综合冲刺（200 步）

_最后一站：把前 4 phase 的胜出组件叠加 + 加 agentic-specific 增强_

## 待 Phase 2-4 跑完后才能确定的组件选择

### Phase 2 winner（ratio 维度）
| 候选 | 选择条件 |
|---|---|
| token-level (vanilla GRPO) | 0e/DAPO/GSPO 都接近 → 选最稳的 token |
| sequence-level (GSPO) | GSPO final val/nq > 0e + 0.02 |
| asymmetric clip (DAPO) | DAPO 赢（不太可能，目前 -4%）|

**截至 2026-04-28 22:00**：DAPO 输（0.400 vs 0.417），GSPO 跑中（step 20 = 0.394 领先 0e step 20）

### Phase 3 winner（reward 信号）
| 候选 | 选择条件 |
|---|---|
| 5档 DFA only (Phase 0e) | ToRL 失败/平 |
| **5档 + per-turn shaping (ToRL)** | val/nq > 0e + 0.02 |

### Phase 4 winner（advantage 计算）
| 候选 | 选择条件 |
|---|---|
| 单层 GRPO | GiGPO α 没影响 |
| **双层 GiGPO α=0.7** | GiGPO 训练曲线方差更小 |

## Phase 5 ARPO 新增组件（独立于 2-4 winner）

### 1. Trajectory-level credit assignment
**问题**：现在 GRPO 把整序列 reward 平摊到所有 token，每个 token 得到同一 advantage。但实际"答错"是因为某几个关键 token 错了，不是整序列错。

**做法**：用 implicit reward decomposition
```
α(token_t) = importance_at_t × global_reward
其中 importance_at_t = |log p_new(t) - log p_old(t)| / sum(|log p_diff|)
```
让 advantage 集中在"模型最不确定的 token"上。

### 2. Turn-level KL
**问题**：multi-turn 中，think/search/answer 三种 turn 的策略改动幅度不同，但单 KL 都用同一 β。

**做法**：把 KL 拆成 turn-type-conditional
```
β_think = 0.001
β_search = 0.005   # search 改动应更小（query design 容易 reward hack）
β_answer = 0.001
```

### 3. Adaptive entropy bonus
**问题**：GSPO 显示 entropy 会暴跌（mode collapse 风险）；DAPO 显示 entropy 会爆炸（探索过头）。需要自适应调节。

**做法**：每步监控 entropy，偏离 target_entropy=1.0 自动调 entropy_coef
```
if entropy < 0.5: bonus_coef *= 1.05
if entropy > 2.0: bonus_coef *= 0.95
```

## Phase 5 实验配置

| 项 | 值 | 说明 |
|---|---|---|
| 起点 | Phase 4 final ckpt | 站在巨人肩膀上 |
| 步数 | **200** | 比单 phase 50 步翻 4 倍，给综合方案足够 burn-in |
| 入口 | `main_ppo_arpo.py`（待写） | 综合 entry |
| save_freq | 20 | 每 ~3h 落盘一次 |
| test_freq | 20 | val 同步 |
| Reward | （取 Phase 3 winner） | |
| Ratio | （取 Phase 2 winner） | |
| Advantage | （取 Phase 4 winner） | |
| **新增**：trajectory_credit | 启用 | |
| **新增**：turn_level_kl | 启用 | |
| **新增**：adaptive_entropy | 启用 | |

## 时间 + 预算

- 200 步 × 7 min/step = 23h
- + 其他开销 = **24h**
- 费用 **¥276**
- 累计总预算（含前面 4 phase）：**~¥600** 

## 待写文件清单

- [ ] `verl/trainer/main_ppo_arpo.py`（综合 entry）
- [ ] `verl/trainer/ppo/core_algos.py` 改 `compute_grpo_outcome_advantage`：
  - [ ] trajectory_credit_assignment 函数
  - [ ] turn_level_kl 计算
  - [ ] adaptive_entropy_coef 控制器
- [ ] `scripts/phase5_train_arpo.sh`（综合训练脚本）
- [ ] `notes/phase5/results.md`（最终汇总）

## 风险

1. **200 步窗口暴露的问题** vs 50 步看不到的：
   - 可能 reward saturation（val/nq 在 0.5+ 上不去）
   - 可能 mode collapse 加剧
   - 可能 OOM 累积（KV cache 慢慢膨胀）
2. **多组件叠加 1+1<2 风险**：组件相互冲突
   - 如 trajectory_credit + GSPO sequence-level 可能冗余（都做"序列归约"）
3. **¥276 单 phase 投入大**：失败成本高，建议先跑 50 步预演

## 启动条件

只在以下都满足时才发射 Phase 5：
- ✅ Phase 2-4 全部完成
- ✅ 至少 2 phase 跑赢 0e baseline（否则综合方案没有素材）
- ✅ 用户拍板预算（¥276 是大数）

## 写作准备

在 Phase 2-4 跑的间隙，可以先写：
- ARPO 论文核心思想笔记
- main_ppo_arpo.py 骨架
- adaptive_entropy 控制器单测
