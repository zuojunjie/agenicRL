# Phase 0e — GRPO + Format Reward（50 步对照实验）

_2026-04-27 重命名自 phase1-deepseek-format-reward-50-coldstart_

## 编号修正说明

按你最早确认的原计划：
- **Phase 1 = 读 R1 论文，0 步训练**（成果在 `notes/phase1/paper.md` + `code_diff.md`）
- 当前云端跑的"50 步 + format reward"训练**不属于** Phase 1，重命名为 **Phase 0e**

Phase 0e 的角色：
- Phase 0 baseline (EM 二元 reward) 的对照实验
- 验证"加 format reward DFA + 5 档分级"对 base model 的提升
- 同时为 Phase 2 DAPO/GSPO 提供 warm-start 起点

## Wandb run 名

**保留** `phase1-deepseek-format-reward-50-coldstart`（不改），保证 wandb 曲线连续性。
仅 `runs/` 目录在 finalize 后从 `phase1-...` 改名为 `phase0e-format-reward-50`。

## 看点

跟原 Phase 1 笔记一致（`notes/phase0e/progress.md` 实时更新）：
- val/nq 是否突破 Phase 0 baseline 0.376
- format 合规率上升曲线
- response_length / n_search 演化
