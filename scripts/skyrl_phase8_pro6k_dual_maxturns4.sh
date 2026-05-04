#!/usr/bin/env bash
# Phase 8 = Phase 7.8c (or 7.8d) base + max_turns 2 → 4
#
# 单变量改动: generator.max_turns: 2 → 4
#   - 允许 agent 最多 4 轮 search-generate 推理
#   - 对 multi-hop QA（HotpotQA 等）更友好
#   - 当前 NQ 数据集大多 1 turn 即可答，4 turn 上限对单跳影响小
#
# Base 选择: 用 7.8c (regular + sequence_mean + mini=64) 还是 7.8d (paper GSPO + asym clip)
#   - 待 7.8c 与 7.8d final val 数据出来后决定
#   - 默认本 launcher 基于 7.8c (验证过的稳定配置)
#
# 风险:
#   - max_turns=4 → 单 trajectory token 数最多 4× (4 turns × 500 = 2000 tokens)
#   - generate 时长可能 +30-60%
#   - 单 step wall time 估 5.5 → 7-8 min
#   - 52 步总时长 ≈ 6-7 h
#
# 期望:
#   - 训练 reward 持平或略低 (NQ 单跳为主，多 turn 不一定加分)
#   - val reward 持平或略升
#   - 真正受益是 HotpotQA 多跳数据集（Phase 11 + multi-dataset）

set -ex
ulimit -n 65535

DATA_DIR="/root/autodl-tmp/data/searchR1"
RUN_NAME="phase8-skyrl-pro6k-dual-maxturns4"
MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-3B-Instruct"
PROJECT="agentic-rl-search"

cd /root/autodl-tmp/external/SkyRL

source /etc/network_turbo 2>/dev/null
export no_proxy="${no_proxy},hf-mirror.com,pypi.tuna.tsinghua.edu.cn,127.0.0.1,localhost"
export NO_PROXY="$no_proxy"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export PATH=/root/.local/bin:$PATH

# Sky/Ray memory budget
export _SKYRL_USE_NEW_INFERENCE=0
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export RAY_object_store_memory=5000000000

# NCCL only (PYTORCH_CUDA_ALLOC_CONF=expandable_segments 与 vllm 冲突)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

curl -sf -m 5 -X POST http://127.0.0.1:8000/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"query":"smoke","topk":1,"return_scores":true}' > /dev/null \
    || { echo "❌ retrieval :8000 not ready"; exit 1; }
echo "[phase8] ✅ retrieval up"

uv run --isolated --frozen --extra fsdp -m skyrl.train.entrypoints.main_base \
    data.train_data="['${DATA_DIR}/train_52steps.parquet']" \
    data.val_data="['${DATA_DIR}/validation.parquet']" \
    \
    `# === Algorithm: 4090 对齐 (regular + sequence_mean) ===` \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.algorithm.policy_loss_type="regular" \
    trainer.algorithm.loss_reduction="sequence_mean" \
    trainer.algorithm.eps_clip_low=0.2 \
    trainer.algorithm.eps_clip_high=0.2 \
    trainer.algorithm.use_kl_loss=true \
    trainer.algorithm.kl_loss_coef=0.001 \
    trainer.algorithm.kl_estimator_type="k3" \
    trainer.algorithm.use_kl_in_reward=false \
    \
    `# === Optimizer (4090 等价) ===` \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.policy.optimizer_config.max_grad_norm=1.0 \
    trainer.policy.optimizer_config.num_warmup_steps=15 \
    \
    `# === Model + FSDP (4090 等价) ===` \
    trainer.policy.model.path="$MODEL_PATH" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp2 \
    trainer.policy.fsdp_config.cpu_offload=false \
    trainer.ref.fsdp_config.cpu_offload=false \
    trainer.gradient_checkpointing=true \
    trainer.use_sample_packing=true \
    trainer.flash_attn=true \
    \
    `# === GPU dist: dual ===` \
    trainer.placement.policy_num_gpus_per_node=2 \
    trainer.placement.ref_num_gpus_per_node=2 \
    \
    `# === Inference: dual GPU TP=2 ===` \
    generator.inference_engine.num_engines=1 \
    generator.inference_engine.tensor_parallel_size=2 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.gpu_memory_utilization=0.50 \
    generator.inference_engine.async_engine=true \
    generator.inference_engine.enforce_eager=false \
    generator.inference_engine.max_num_batched_tokens=16384 \
    \
    `# === Batch: mini=64 (二分搜索锁定的关键修复)，micro=32 (4090 速度) ===` \
    trainer.epochs=1 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=512 \
    trainer.policy_mini_batch_size=64 \
    trainer.micro_forward_batch_size_per_gpu=32 \
    trainer.micro_train_batch_size_per_gpu=32 \
    trainer.max_prompt_length=4096 \
    \
    `# === Generation (4090 等价) ===` \
    generator.max_input_length=4096 \
    generator.sampling_params.max_generate_length=500 \
    generator.batched=false \
    generator.use_conversation_multi_turn=false \
    generator.n_samples_per_prompt=5 \
    generator.max_turns=4 \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=1.0 \
    generator.sampling_params.stop='["</search>", "</answer>"]' \
    \
    `# === Search environment ===` \
    environment.env_class="search" \
    environment.skyrl_gym.max_env_workers=16 \
    environment.skyrl_gym.search.log_requests=false \
    environment.skyrl_gym.search.search_url="http://127.0.0.1:8000/retrieve" \
    environment.skyrl_gym.search.topk=3 \
    \
    `# === Logging + ckpt: production, save every 10 ===` \
    trainer.logger="wandb" \
    trainer.project_name="$PROJECT" \
    trainer.run_name="${RUN_NAME}" \
    trainer.ckpt_interval=10 \
    trainer.hf_save_interval=50 \
    trainer.max_ckpts_to_keep=3 \
    trainer.resume_mode=latest \
    trainer.ckpt_path="/root/autodl-tmp/runs/${RUN_NAME}" \
    \
    `# === Eval: 跳过 val@0，每 10 step eval ===` \
    trainer.eval_batch_size=256 \
    trainer.eval_before_train=false \
    trainer.eval_interval=10 \
    generator.eval_sampling_params.temperature=0 \
    generator.eval_sampling_params.stop='["</search>", "</answer>"]' \
    generator.eval_sampling_params.max_generate_length=500 \
    \
    trainer.export_path="/root/autodl-tmp/runs/${RUN_NAME}/exports" \
    \
    "$@" 2>&1 | tee "/tmp/${RUN_NAME}.log"
