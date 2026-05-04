# Phase 1 — Reward 代码定位报告

## 重大利好：DeepSeek-R1 风格 format reward 已在 Search-R1 上游实现

不需要自己写 reward 函数。Search-R1 仓库已经提供了完整的 format reward 路径，与 baseline 是**双轨制**：

| 轨道 | 入口 | reward 函数 |
|---|---|---|
| Baseline (EM only) | `verl.trainer.main_ppo` | `verl/utils/reward_score/qa_em.py::compute_score_em` |
| **DeepSeek-R1 format** | `verl.trainer.main_ppo_format` | `verl/utils/reward_score/qa_em_format.py::compute_score_em` |

云端绝对路径：
- `/root/autodl-tmp/external/Search-R1/verl/trainer/main_ppo_format.py`
- `/root/autodl-tmp/external/Search-R1/verl/utils/reward_score/qa_em_format.py`

## qa_em_format 的实际 reward 设计（比父 agent 描述的更严格）

不是简单的"含 `<think>` 和 `<answer>` 就 +0.1"，而是**严格的状态机**：
- 检查 `<think>` / `<search>` / `<information>` / `<answer>` 四种 tag 的开闭配平
- 必须按 DFA `start → think → (search → information → think)* → answer → end` 完整匹配
- 标签外只允许空白
- 任意违规 → format_score=0；完全合规 → format_score>0 与 EM accuracy 复合

这意味着 Phase 1 同时在评估 **multi-turn 工具调用结构合规性**，而非仅 think/answer 包裹。

## Phase 1 实施方案（极简 + 完全可逆）

**不改任何代码**。在训练命令里把入口从 `verl.trainer.main_ppo` 换成 `verl.trainer.main_ppo_format` 即可。

需要在 `scripts/phase0_train_grpo.sh` 加一个 `REWARD_VARIANT` env 开关：
- `REWARD_VARIANT=em`（默认）→ `python -m verl.trainer.main_ppo`
- `REWARD_VARIANT=em_format` → `python -m verl.trainer.main_ppo_format`

零文件改动；回退就是不传 env。
