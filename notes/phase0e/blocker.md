# Phase 1 BLOCKER — 在启动训练前停下，等父 agent 决策

时间：2026-04-27（subagent 第一次 SSH 检查后）
GPU 状态：4×vGPU-48GB 全部空闲（0 MiB 占用）—— 还**没有**起训练，¥0 已花。

## 阻塞原因：Phase 0d ckpt 25 实际不存在，无法 warm-start

父 agent 任务书声明：
> ckpt 路径（云端）：`/root/autodl-tmp/runs/phase0d-baseline-50/ckpts/global_step_25/`

实际现状（SSH 验证）：

1. **目录名不一致**：实际是 `phase0d-baseline-attempt1`（不是 `phase0d-baseline-50`）。
2. **ckpts 目录是空的**：
   ```
   /root/autodl-tmp/runs/phase0d-baseline-attempt1/ckpts/   # 空
   /root/autodl-tmp/external/Search-R1/verl_checkpoints/phase0-nq-grpo-qwen2.5-3b-it-em/actor/   # 空
   ```
   全盘 `find /root/autodl-tmp -name "global_step_*" -type d` 无结果。
3. **原因可推断**：从 `training.log` 末尾的 hydra overrides 看到 attempt1 跑的实际配置是 `trainer.save_freq=25 trainer.test_freq=20`（SAVE_FREQ=10 的 patch 当时还**没**应用）。OOM 发生在 step 44，按 save_freq=25 应该有 step 25 的 ckpt 落盘，但落盘后疑似被清理或写入失败；现在已经无法恢复。
4. metrics.csv / summary.md / training.log 都在，**只有权重不在**。Phase 0 final val/nq=0.376（step 40）这个数字仍然有效，可作对比基线。

## 选项（请父 agent 选一个）

### A. Cold-start Phase 1 — 从 Qwen2.5-3B-Instruct 原始 base model 起，跑 50 步带 format reward
- 优点：起点干净，与 Phase 0d attempt1（也是 cold-start）严格可比，验证 format reward 单变量影响。
- 缺点：浪费了 Phase 0d 4.9h 训练成果；Phase 1 等于从 0 开始的对照实验，不是论文 R1-Zero 的"在已经会用 search 的模型上加 format"叙事。
- 时间：~4.9h × ¥11.5 ≈ **¥56**（外加 OOM 风险，因为 attempt1 OOM 在 step 44，Phase 1 跑 50 步会再次撞墙）。

### B. 先重跑 Phase 0d ckpt 救援 — 只跑 25 步、save_freq=5、ppo_micro=32 拿到稳定 ckpt，再启 Phase 1
- 优点：保持原计划"warm-start from Phase 0 ckpt"叙事；ppo_micro=32 也顺便验证 OOM patch。
- 缺点：双倍时间。
- 时间：救援 ~2.5h + Phase 1 ~2.5h ≈ **¥58**。

### C. 调整 Phase 1 范围：直接 cold-start 但只跑 25 步带 format reward
- 优点：单次 ~2.5h，¥29，能快速看到 format_score 是否生效。
- 缺点：步数少，val/nq 提升信号弱。
- 时间：~2.5h ≈ **¥29**。

### D. 跳过 Phase 1，直接 Phase 0e — 用 SAVE_FREQ=5 + ppo_micro=32 重跑 50 步 baseline，把 ckpt 保住，再做 Phase 1
- 我个人推荐：路径最稳，下次 Phase 1 就有真正的 ckpt 25 起点。
- 时间：~5h ≈ **¥58**，但这次 ckpt 一定保得住。

## 一个独立的好消息

DeepSeek-R1 风格 format reward **不需要写代码**。Search-R1 上游已经实现，详见
`/Users/cv/code/agenicRL/notes/phase1/reward_code_location.md`。
入口直接从 `verl.trainer.main_ppo` 切到 `verl.trainer.main_ppo_format` 即可，
完全可逆，无需改文件。

## 我的等待动作

等父 agent 在四个选项中选一个回复后再继续。GPU 当前空闲，不计费。
