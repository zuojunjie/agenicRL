# Phase 1 — DeepSeek-R1 论文笔记（0 步纯读）

_2026-04-27_
_Phase 1 在原计划里就是「读论文 0 步」，本文件 + `code_diff.md` 是全部交付物。_
_对应的真训练（GRPO + format reward 50 步）属于 **Phase 0e**，见 `../phase0e/`。_

## 1. 论文核心一句话

**用 rule-based composite reward + GRPO 替代 PPO + reward model**，无需 SFT cold-start，纯 RL 让 base model 自发涌现长 CoT 与"Aha moment"自反思。

## 2. 关键贡献（按 RL 视角）

| # | 贡献 | 在我们 5-Phase 路线里如何承接 |
|---|---|---|
| C1 | Rule-based reward (accuracy + format) 取代 RM | **Phase 0e** 复刻 |
| C2 | GRPO 替代 PPO（无 critic，组内归一化 advantage） | Phase 0 已用 |
| C3 | 不做 SFT cold-start，直接 RL on base model | Phase 0e cold-start 复刻 |
| C4 | "Aha moment" / 自发 self-reflection（emergent） | 观察项，非可控；Phase 0e 看 rollout 文本 |
| C5 | R1（vs R1-Zero）：两阶段 SFT→RL→SFT→RL | ❌ 留 Phase 5 综合时考虑 |

## 3. Reward 公式（Search-R1 fork 的 4 标签实现）

读 `verl/utils/reward_score/qa_em_format.py:154 compute_score_em`：

```
四标签 DFA: <think> → </think> → <search> → </search>
                              ↘ <information> → </information> → <think> ...
                              ↘ <answer> → </answer> = end
```

| 条件 | reward |
|---|---|
| answer 错 + format 错 | `final_format_score` = 0.1（pity 分） |
| answer 错 + format 对 + 检索没命中 | `structure_format_score` = 0.2 |
| answer 错 + format 对 + 检索命中 ground truth | `+ retrieval_score` = 0.3 |
| answer 对 + format 错 | `score - structure_format_score` = 0.8 |
| answer 对 + format 对 | `score` = 1.0 |

**默认 Phase 0e 系数（Search-R1 论文 v0.2）**：`structure=0.2 / final=0.1 / retrieval=0.1 / score=1.0`。

## 4. 与 Phase 0 baseline (`qa_em.py`) 的区别

```diff
- compute_score_em: 二分类 — answer 对=1 / 错=0（format 完全无视）
+ compute_score_em: 五档 — 1.0 / 0.8 / 0.3 / 0.2 / 0.1（DFA 状态机校验）
```

代码侧零文件改动，只切训练入口：
```bash
python -m verl.trainer.main_ppo  →  python -m verl.trainer.main_ppo_format
```

`main_ppo_format.py` 里 `_select_rm_score_fn(data_source)` 把 `'nq' → qa_em_format.compute_score_em`。其余 GRPO 算法不变。

## 5. R1-Zero vs R1 的两阶段

R1-Zero（我们 Phase 0e 对应的部分）：
- Step 1：base model + GRPO + rule reward → R1-Zero
- 问题：可读性差、language mixing

R1（论文最终版本，本路线 Phase 5 借鉴）：
- Step 1：少量 cold-start CoT 数据 SFT → 模型 v1
- Step 2：v1 + RL（含可读性 reward）→ v2
- Step 3：v2 + 大规模 SFT 数据 + RL → v3 = R1
- 一共 4 阶段，远比 R1-Zero 复杂

## 6. 局限 & 后续 phase 接力

- **format 是个二元信号**：DFA 只 accept/reject，不会教模型"为什么错"。
- **retrieval reward 是后验**（看 information 块里有没有 GT），无法引导 search query 设计。**Phase 3 ToRL 把 search 升为 first-class tool**。
- **GRPO 的 token-level ratio 长序列方差大**。**Phase 2B GSPO 改 sequence-level**。

## 7. 阅读清单

- [x] DeepSeek-R1 paper §2.2 (reward modeling) §2.3 (GRPO 算法)
- [x] DeepSeek-R1 paper §3 R1-Zero vs R1 训练流程
- [x] Search-R1 paper §3（4 标签 DFA 的应用）
- [ ] DAPO paper（Phase 2A 准备）
- [ ] GSPO paper（Phase 2B 准备）
- [ ] R1-Zero training curves（论文 fig 2/3）vs 我们 Phase 0e 的曲线对比

## 8. Phase 1 交付清单

- ✅ `notes/phase1/paper.md` — 本文件（论文核心 1 页）
- ✅ `notes/phase1/code_diff.md` — main_ppo vs main_ppo_format 代码差异分析
- ✅ Phase 1 在飞书路线图中的章节定位（0 步，纯学习成果）
