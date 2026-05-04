# AgenicRL 5-Phase 完整实验总结

_2026-04-30 更新 — Phase 5b 完成，全 7 个训练 phase 数据齐_
_硬件: 4×RTX 4090 vGPU-48GB (¥11.52/h) → 切 1×RTX PRO 6000 (¥5.98/h)_
_总投入: 约 ¥600, 约 100 GPU-小时_

## 🏆 终极结果表

| Phase | 算法变量 | val/nq | vs 0e | 学到了什么 |
|---|---|---|---|---|
| **0d** | GRPO + EM 二元 reward | 0.376 | -10% | RL 起点（baseline 起源） |
| **0e** ⭐ | + DFA 5档 reward | **0.417** | (基线) | reward 设计是最大杠杆（+11%） |
| **2A DAPO** | + clip-higher (asymmetric) | 0.400 | -4% ❌ | clip 形状改动→ entropy 失控 |
| **2B GSPO** ⭐⭐ | + sequence-level ratio | **0.4378** | **+5%** | **sequence 级 ratio = 唯一胜者** |
| **3 ToRL** | + multi-turn shaping | 0.426 raw / ~0.380 | -9% raw | reward shaping = hacking 陷阱 |
| **4 GiGPO** | + 双层 advantage (α=0.7) | 0.292 (kill@40) | -27% ❌❌ | outer adv → entropy 爆炸 |
| **5a ARPO** | + traj credit + adapt entropy | 0.357 | -14% ❌ | traj credit 是 hurt（即使 entropy 锁住） |
| **5b ARPO** | + turn-level KL only | 0.254 | -37% ❌❌ | search 段局部 anchor 救不了全局 entropy 失控 |

## 🎯 核心结论

> **GSPO sequence-level ratio 是 5 phase 实验中唯一可靠的算法改进 (+5% val/nq)**。
> 
> 其他改动（DAPO clip-higher、ToRL shaping、GiGPO 双层 advantage、ARPO trajectory credit、ARPO turn-level KL）**全部输给 vanilla GRPO + DFA 5 档 reward**。
> 
> 真正最大的杠杆不是 PPO 算法层，而是 **reward 函数设计**（0d→0e 的 +11% 远超后续任何 PPO 内部改进）。

## 📊 累积进步图

```
Qwen2.5-3B-Instruct (initial val 0.196, zero-shot NQ)
        │ +0.180  GRPO 算法本身（0d 验证 RL 可学）
   0d   0.376  🔻 baseline
        │ +0.041  reward 函数：EM → DFA 5档（0e）
   0e   0.417  ⭐ 主基线
        │ +0.021  PPO ratio 维度：token → sequence (GSPO 2B)
   GSPO 0.438  ⭐⭐ 当前最佳
```

**从 base model 到 GSPO 累计 +123% 相对提升**（0.196 → 0.438）。

## 🧬 失败模式深度复盘（5 phase）

### Phase 2A DAPO — clip-higher 让 entropy 暴
- 配置：clip_low=0.2, clip_high=0.28（非对称，鼓励 explore）
- 失败轨迹：entropy 0.96 (step 10) → 1.50 (step 28) → **4.92 (step 48)**
- 核心问题：clip 上界放宽 → policy update 步子变大 → 模型变随机
- 最终 val: 0.400 (-4%)

### Phase 3 ToRL — reward shaping = hacking
- 配置：max_turns 2→4，per-turn 召回奖励 +0.05，重复 query 惩罚 -0.05
- 看似收益：response_length 663→1937, n_search 1.0→3.5
- 真相：模型刷 shaping bonus（用工具拿"虚分"）→ raw EM 反降到 0.380（vs 0e 0.401）
- 中间 OOM @ step 22 因 KV cache 累积

### Phase 4 GiGPO — outer advantage 引入更多噪声
- 配置：α=0.7，70% inner + 30% outer (跨 prompt mean 归一化)
- 失败轨迹：step 25 entropy 1.84 → step 28 = 2.62（用户 kill 在 step 41）
- 核心问题：困难 prompt 的 outer adv 是大负值 → 推 policy 远离 base → entropy 爆
- 最终 val: 0.292 (-27%) 是所有 phase 最差

### Phase 5a ARPO (traj credit + adapt entropy) — adapt 兜住爆炸但 traj 损害
- adaptive entropy ✅ 锁住 entropy 在 0.86-1.0（**这一项设计成功**）
- trajectory credit ❌ per-token 加权让中等难度 prompt 学不动
- 最终 val: 0.357 (-14%)

### Phase 5b ARPO (turn-level KL only) — 局部 anchor 治标不治本
- 设计：search 段 KL β = 0.005（5x normal），其他段 0.001
- 前 40 步 OK：step 20 val=0.350（与 0e 0.355 几乎打平）
- step 40-50 entropy 暴：从 1.30 → **6.6**（比 DAPO 4.92 还高）
- val 从 step 40 的 0.371 → step 50 的 **0.254**（崩盘）
- 最终 val: 0.254 (-37%) 是 ARPO 系列最差

## 📈 全 phase val 曲线对照

| Step | 0e | DAPO | GSPO | GiGPO | 5a | 5b |
|---|---|---|---|---|---|---|
| 0 | 0.196 | 0.196 | 0.196 | 0.196 | 0.196 | 0.196 |
| 10 | 0.313 | 0.309 | 0.336 | 0.309 | 0.282 | 0.307 |
| 20 | 0.355 | 0.359 | 0.394 | 0.340 | 0.320 | 0.350 |
| 30 | 0.390 | 0.386 | 0.426 | 0.329 | 0.318 | 0.368 |
| 40 | 0.401 | 0.404 | 0.427 | 0.292 | 0.331 | 0.371 |
| **50** | **0.417** | 0.400 | **0.438** | (kill) | 0.357 | **0.254** |

## 💡 5 个工程级 takeaway

### 1. **Reward 函数设计 >> PPO 算法改进**
DFA 5 档 reward (0e) 比 EM 二元 (0d) 提升 +11%。后续所有 PPO 内部改动最多 +5%（GSPO）。
**"先把 reward 设计对，再调算法"**。

### 2. **Entropy 是 RL 训练的命脉**
DAPO/GiGPO/5a/5b 全部死于"entropy 失控"（不同方式）：
- DAPO: clip-higher 让 update 步子大
- GiGPO: outer adv 让"难题"产生大负 advantage
- 5b: search 段局部 KL 锚定挡不住整体 explore

**任何 advantage 信号增强机制都需要 entropy 控制器兜底**。

### 3. **Reward shaping 是 hacking 陷阱**
ToRL 的 per-turn 信号让模型学会"刷 shaping bonus"而不是"答对题"。
**Single-objective reward 比 multi-component shaping 安全**。

### 4. **Sequence-level 比 token-level 稳定**
GSPO 的 sequence-level ratio = "整序列共享一个 ratio"，不依赖 per-token noise。
**DAPO/GiGPO 通过 token-level 改动打破训练稳定性**。

### 5. **Single-variable 实验比综合 baseline 重要**
5 phase 都从 Qwen cold-start，每个 phase 只改一个变量。这个严格 single-variable 设计让我们能精确归因 +/- 5% 的来源。
**多变量实验在小窗口下产生混淆**。

## 📁 资产清单

### 飞书 wiki 节点
- 主路线图：https://my.feishu.cn/wiki/SZOVwhpCNiDdppkgCTgcRkeYnRb
- Phase 0e: https://my.feishu.cn/wiki/UURuwlRVwilOijk25r0caxmMnde
- Phase 2A DAPO: https://my.feishu.cn/wiki/Qad4wYt8bio5dnkV2I8cLNx8nLe
- Phase 2B GSPO: https://my.feishu.cn/wiki/BVjGwO2yiibguEkSOxLcu783nDe
- Phase 3 ToRL: https://my.feishu.cn/wiki/QEiowgxCuiz1VwkFKHncI7LDnAt
- Phase 4 GiGPO: 待推
- Phase 5a/b ARPO: 待推

### Wandb runs
- 0d: smzn6t38
- 0e: vzgvurae (历史命名 phase1-deepseek-format-reward-50-coldstart)
- DAPO: dddc2sz5
- GSPO: pjze1hqd
- ToRL stitched: 2bbe95xv
- GiGPO: 1bvs8zh9
- 5a: ybtcoznx
- 5b: u2aiddt4

### 云端 ckpts (已切 PRO 6000，数据盘跨实例继承)
```
/root/autodl-tmp/runs/
├── phase0d-baseline-attempt1/        (无 ckpt, OOM)
├── phase0e-format-reward-50/         step_45 ckpt ⭐
├── phase2-dapo-cliphigh-50/          step_45
├── phase2-gspo-seqlevel-50/          step_50 ⭐⭐ (winner)
├── phase3-torl-resume-step20/        step_20
├── phase4-gigpo-twolayer-50/         step_40
├── phase5a-arpo-trajcredit-adaptent-52/
└── phase5b-arpo-turnkl-only-52/
```

## 🚀 下一步规划

### 短期（已开始）
- ✅ 关 4×4090 实例（用户已操作）
- ✅ 开 1×RTX PRO 6000 (¥5.98/h)
- ✅ 数据盘已跨实例挂载（autodl-tmp 完整迁移）
- 🟡 验证 PRO 6000 上 vllm 0.6 vs 0.8 兼容性
- 🟡 在 PRO 6000 上重跑 Phase 0e 验证基线一致性

### 中期
- 在 PRO 6000 上跑 Phase 5（200 步）— 用 GSPO 配置作为最佳起点
- 可能升 7B 实验（2×PRO 6000）

### 长期
- ARPO 论文级别消融（adapt entropy 是否单独有效？需独立 phase 5c）
- 跨 task 验证（NQ → TriviaQA / HotpotQA / 2WikiMultihop）

## 🎓 学习目标完成度

| 目标 | 状态 |
|---|---|
| 端到端理解 Search-R1 + verl + GRPO | ✅ 完整 |
| GRPO vs PPO 直觉 + group baseline | ✅ |
| DFA 5档 reward 实现细节 + reward hacking | ✅ |
| GSPO sequence-level ratio 数学原理 + 验证 | ✅ |
| DAPO clip-higher 失败机制（entropy 爆炸） | ✅ 实证 |
| GiGPO 双层 advantage 失败机制 | ✅ 实证 |
| ARPO 4 个特性的工程实现 + 失败案例 | ✅ |
| Multi-turn vLLM rollout 内部机制 | ✅ |
| FSDP + CPU offload + cgroup 显存管理 | ✅ 踩过 |
| Wandb 误删恢复（binary patch + sync --append --id） | ✅ 实战 |
| 总累计费用 | **~¥600** |

## 一句话总结

**这次 5 phase 实验的最大收获不是任何一个新算法的 +5%，而是验证了"在 50 步小窗口下，PPO 内部改动很容易因 entropy 失控失败；唯一稳定可改的是 reward 函数和 ratio 计算维度（GSPO）"。这是只有跑过 5+ 个 phase 才能形成的工程直觉。**
