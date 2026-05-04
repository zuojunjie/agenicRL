#!/usr/bin/env bash
# Phase 7 = Phase 6 = E (PRO 6000 #1) + 3 changes:
#   1. policy/ref_num_gpus_per_node: 1 → 2
#   2. tensor_parallel_size: 1 → 2
#   3. ckpt_interval: 50 → 10  (4090 等价)
#
# Other config inherits PRO 6000 #1 launcher verbatim:
#   - policy_loss_type=gspo (NOT aligned to verl vanilla — by design)
#   - loss_reduction default token_mean (NOT aligned)
#   - ppo_mini_batch=64 / micro=8 (NOT aligned to 4090 256/32)
#   - flash_attn=false / use_sample_packing=false (NOT aligned)
#   - epochs=15 + train.parquet (full)
#   - gpu_memory_utilization=0.18

set -ex
ulimit -n 65535

DATA_DIR="/root/autodl-tmp/data/searchR1"
RUN_NAME="phase7-skyrl-pro6k-dual-inherit-maxturn2"
MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-3B-Instruct"
PROJECT="agentic-rl-search"

cd /root/autodl-tmp/external/SkyRL

source /etc/network_turbo 2>/dev/null
export no_proxy="${no_proxy},hf-mirror.com,pypi.tuna.tsinghua.edu.cn,127.0.0.1,localhost"
export NO_PROXY="$no_proxy"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export PATH=/root/.local/bin:$PATH

# Sky/Ray memory budget controls (proven from earlier smoke v23 work)
export _SKYRL_USE_NEW_INFERENCE=0
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export RAY_object_store_memory=5000000000

# Dual-GPU NCCL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Verify retrieval up
curl -sf -m 5 -X POST http://127.0.0.1:8000/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"query":"smoke","topk":1,"return_scores":true}' > /dev/null \
    || { echo "❌ retrieval :8000 not ready"; exit 1; }
echo "[phase7] ✅ retrieval up"

# Phase 7 launcher (3-line diff vs Phase 6 = E PRO 6000 #1):
#   - policy/ref_num_gpus_per_node = 2
#   - tensor_parallel_size = 2
#   - ckpt_interval = 10  (was 50)

uv run --isolated --frozen --extra fsdp -m skyrl.train.entrypoints.main_base \
    data.train_data="['${DATA_DIR}/train.parquet']" \
    data.val_data="['${DATA_DIR}/validation.parquet']" \
    \
    `# === Algorithm: GSPO native (inherits PRO 6000 #1, NOT aligned to verl) ===` \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.algorithm.policy_loss_type="gspo" \
    trainer.algorithm.eps_clip_low=0.2 \
    trainer.algorithm.eps_clip_high=0.2 \
    trainer.algorithm.use_kl_loss=true \
    trainer.algorithm.kl_loss_coef=0.001 \
    trainer.algorithm.kl_estimator_type="k3" \
    \
    `# === Optimizer ===` \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.policy.optimizer_config.max_grad_norm=1.0 \
    trainer.policy.optimizer_config.num_warmup_steps=15 \
    \
    `# === Model + FSDP (inherit) ===` \
    trainer.policy.model.path="$MODEL_PATH" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp2 \
    trainer.policy.fsdp_config.cpu_offload=false \
    trainer.ref.fsdp_config.cpu_offload=false \
    trainer.use_sample_packing=false \
    trainer.flash_attn=false \
    \
    `# === GPU dist (Phase 7 changes #1, #2) ===` \
    trainer.placement.policy_num_gpus_per_node=2 \
    trainer.placement.ref_num_gpus_per_node=2 \
    \
    `# === Inference: dual GPU TP=2 ===` \
    generator.inference_engine.num_engines=1 \
    generator.inference_engine.tensor_parallel_size=2 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.gpu_memory_utilization=0.18 \
    generator.inference_engine.async_engine=true \
    \
    `# === Batch (inherit PRO 6000 #1) ===` \
    trainer.epochs=15 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=512 \
    trainer.policy_mini_batch_size=64 \
    trainer.micro_forward_batch_size_per_gpu=8 \
    trainer.micro_train_batch_size_per_gpu=8 \
    trainer.max_prompt_length=4096 \
    \
    `# === Generation (inherit) ===` \
    generator.max_input_length=4096 \
    generator.sampling_params.max_generate_length=500 \
    generator.batched=false \
    generator.use_conversation_multi_turn=false \
    generator.n_samples_per_prompt=5 \
    generator.max_turns=2 \
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
    `# === Logging + ckpt (Phase 7 change #3: ckpt_interval=10) ===` \
    trainer.logger="wandb" \
    trainer.project_name="$PROJECT" \
    trainer.run_name="${RUN_NAME}" \
    trainer.ckpt_interval=10 \
    trainer.hf_save_interval=999 \
    trainer.max_ckpts_to_keep=3 \
    trainer.resume_mode=latest \
    trainer.ckpt_path="/root/autodl-tmp/runs/${RUN_NAME}" \
    \
    `# === Eval ===` \
    trainer.eval_batch_size=256 \
    trainer.eval_before_train=true \
    trainer.eval_interval=10 \
    generator.eval_sampling_params.temperature=0 \
    generator.eval_sampling_params.stop='["</search>", "</answer>"]' \
    generator.eval_sampling_params.max_generate_length=500 \
    \
    trainer.export_path="/root/autodl-tmp/runs/${RUN_NAME}/exports" \
    \
    "$@" 2>&1 | tee "/tmp/${RUN_NAME}.log"
