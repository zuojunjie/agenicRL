# Phase 1 — 训练进度日志

EXPERIMENT: phase1-deepseek-format-reward-50-coldstart
Strategy: cold-start from Qwen2.5-3B-Instruct, main_ppo_format entry, 50 steps, save_freq=5, ppo_micro=32, vllm_util=0.3, LOGGER=wandb

## T+0min — 启动

- retrieval_server PID 97702（faiss-gpu sharded 4 GPU），健康（带 return_scores=true 时正常返回）
- training PID 98221，main_task pid 101597
- launcher: `/root/autodl-tmp/agenicRL/scripts/phase1_train_grpo_format.sh`（从 phase0 副本，3 处 sed patch：main_ppo→main_ppo_format / ppo_micro 64→32 / vllm_util 0.4→0.3）
- val_before_train=true 正在跑，看到 model 已经会用 `<search>...</search>` 和 `<answer>...</answer>`，但 think tag 缺失 — 这正是 format reward 要学的信号
- 4×GPU 28-29 GB / 49 GB，util 78-80%

## 计划检查点

- step 0: initial val/test_score/nq baseline（应 ≈ 0.196，与 Phase 0d 一致）
- step 5: 第一个 ckpt 落盘验证（>100MB 才算成功）
- step 10/20/30/40/50: val/test_score/nq + reward
- step 50 finalize → wandb URL + 飞书推送
