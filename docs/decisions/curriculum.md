# 学习路径 · 5 主干 + 2 增补

## 全景图覆盖矩阵

参照 *The Landscape of Agentic RL for LLMs: A Survey* (TMLR 2026)，17 篇必读论文，5 个能力维度。本项目主干覆盖 5/17 ≈ 30%：

| 维度 | 论文 | 是否主干 | 备注 |
|---|---|---|---|
| **Reasoning** | DeepSeek-R1 | ✅ | GRPO 算法核心 |
| | DAPO | ✅ | 算法改进的标准对照 |
| | **GSPO**（Qwen3） | 🆕 增补 | Qwen3 官方推荐配方，序列级 IS |
| | Satori | ❌ | 偏推理蒸馏，与搜索场景关系弱 |
| **Tool Use** | ToolRL | 🆕 预读 | 30 分钟，故事线开篇 |
| | ARTIST | ❌ | 已被 ToRL 吸收 |
| | ToRL | ✅ | code-tool RL 标杆 |
| | GiGPO | ✅ | 步级信用分配 |
| | ARPO | ✅ | 前沿，熵驱动分支 |
| **Planning** | RAP / LATS / VOYAGER / Planner-R1 | ❌ | Search-R1 不需要 MCTS |
| **Memory** | Memory-R1 / MemAgent | ❌ | 跨 episode 无记忆需求 |
| | ReSum | ⚠️ 待命 | 多跳长上下文撞墙时启用 |
| **Self-Improvement** | Reflexion / Absolute Zero / SiriuS | ❌ | outcome reward 直接，不需反思 |

## 路径偏置说明

本路径其实把全景图的"**能力维度**"框架悄悄换成了"**算法演进**"框架：

> - Landscape 问的是："agent 能做什么？" → 5 个能力维度
> - 我们的路径问的是："policy 怎么优化？" → GRPO 升级链

对 Search-R1 这个具体场景，这个偏置是合理的——Search-R1 = Reasoning 主干 ∩ Tool Use 主干，Planning/Memory/Self-Improvement 在此场景非核心。

## 论文 → 代码改动 映射

每读一篇论文，对应一次具体的代码改动：

```
Phase 0 │ 跑通 Search-R1 baseline                      │ 学 verl + GRPO 工程骨架
Phase 1 │ 读 DeepSeek-R1                               │ 注释 verl/grpo 代码，画 advantage 流
Phase 2 │ 读 DAPO + GSPO（双读双比）                    │ DAPO 4 项改进 / GSPO 序列级 IS A/B
Phase 3 │ 读 ToolRL（30min 前置）→ ToRL                │ 对比 search-as-tool 与 code-as-tool 奖励设计
Phase 4 │ 读 GiGPO                                     │ 改造为 step-level advantage
Phase 5 │ 读 ARPO                                      │ 引入熵驱动 rollout 分支
```

## 战略性决策记录

### 为什么把 GSPO 加进主干

- Qwen 系基座 + GSPO 是 Qwen3 官方对齐配方
- 序列级 importance sampling 在 agent 长序列场景比 DAPO 的 token-level 更稳
- Phase 2 改成 "DAPO + GSPO 双读双比"，是一次有研究价值的对照实验

### 为什么 ToolRL 只读不实现

- 直接从 DAPO 跳到 ToRL 会缺 "为什么 RL 能让模型自发学会调用工具" 的认知锚点
- ToolRL 是开篇文献，1 小时投入即可形成对照感
