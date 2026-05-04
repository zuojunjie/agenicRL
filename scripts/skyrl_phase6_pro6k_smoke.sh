#!/usr/bin/env bash
# SkyRL smoke test on PRO 6000 — 10 step run to verify stack end-to-end.
# Uses SkyRL's `uv run --isolated --frozen --extra fsdp`.
#
# Pre-requisites:
#   - SkyRL .venv built at /root/autodl-tmp/external/SkyRL/.venv (via uv sync --extra fsdp)
#   - retrieval_server :8000 alive (chunked-add path, faiss sm_120 patched)
#   - NQ→SkyRL parquet at /root/autodl-tmp/data/searchR1/
#   - Patched faiss in conda searchr1 env (or equivalently in SkyRL venv)
#   - PyTorch 2.8.0+cu128 (Blackwell-aware)

set -ex
ulimit -n 65535

DATA_DIR="/root/autodl-tmp/data/searchR1"
RUN_NAME="phase6-skyrl-smoke-pro6k-1gpu"
MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-3B-Instruct"
PROJECT="agentic-rl-search"

cd /root/autodl-tmp/external/SkyRL

source /etc/network_turbo 2>/dev/null
export no_proxy="${no_proxy},hf-mirror.com,pypi.tuna.tsinghua.edu.cn,127.0.0.1,localhost"
export NO_PROXY="$no_proxy"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export PATH=/root/.local/bin:$PATH

# Memory budget controls (proven safe from earlier smoke v23 work)
export _SKYRL_USE_NEW_INFERENCE=0
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export RAY_object_store_memory=5000000000

# Verify retrieval up
curl -sf -m 5 -X POST http://127.0.0.1:8000/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"queries":["smoke"],"topk":1,"return_scores":true}' > /dev/null \
    || { echo "❌ retrieval :8000 not ready"; exit 1; }
echo "[smoke] ✅ retrieval up"

# Smoke configuration: short, small batch, no val, cold-start GSPO vanilla clip
uv run --isolated --frozen --extra fsdp -m skyrl.train.entrypoints.main_base \
    data.train_data="['${DATA_DIR}/train.parquet']" \
    data.val_data="['${DATA_DIR}/validation.parquet']" \
    \
    `# Algorithm: GSPO vanilla clip (Phase 6 = E)` \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.algorithm.policy_loss_type="gspo" \
    trainer.algorithm.eps_clip_low=0.2 \
    trainer.algorithm.eps_clip_high=0.2 \
    trainer.algorithm.use_kl_loss=true \
    trainer.algorithm.kl_loss_coef=0.001 \
    trainer.algorithm.kl_estimator_type="k3" \
    \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.policy.optimizer_config.max_grad_norm=1.0 \
    trainer.policy.optimizer_config.num_warmup_steps=2 \
    \
    trainer.policy.model.path="$MODEL_PATH" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp2 \
    trainer.policy.fsdp_config.cpu_offload=false \
    trainer.ref.fsdp_config.cpu_offload=false \
    trainer.placement.policy_num_gpus_per_node=1 \
    trainer.placement.ref_num_gpus_per_node=1 \
    trainer.use_sample_packing=false \
    trainer.flash_attn=false \
    \
    generator.inference_engine.num_engines=1 \
    generator.inference_engine.tensor_parallel_size=1 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.gpu_memory_utilization=0.18 \
    generator.inference_engine.async_engine=true \
    \
    `# Smoke: small batches, 10 steps total, no eval` \
    trainer.epochs=1 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=64 \
    trainer.policy_mini_batch_size=16 \
    trainer.micro_forward_batch_size_per_gpu=4 \
    trainer.micro_train_batch_size_per_gpu=4 \
    trainer.max_prompt_length=4096 \
    \
    generator.max_input_length=4096 \
    generator.sampling_params.max_generate_length=500 \
    generator.batched=false \
    generator.use_conversation_multi_turn=false \
    generator.n_samples_per_prompt=2 \
    generator.max_turns=2 \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=1.0 \
    generator.sampling_params.stop='["</search>", "</answer>"]' \
    \
    environment.env_class="search" \
    environment.skyrl_gym.max_env_workers=8 \
    environment.skyrl_gym.search.log_requests=false \
    environment.skyrl_gym.search.search_url="http://127.0.0.1:8000/retrieve" \
    environment.skyrl_gym.search.topk=3 \
    \
    trainer.logger="console" \
    trainer.project_name="$PROJECT" \
    trainer.run_name="${RUN_NAME}" \
    trainer.ckpt_interval=999 \
    trainer.hf_save_interval=999 \
    trainer.max_ckpts_to_keep=2 \
    trainer.resume_mode=latest \
    trainer.ckpt_path="/root/autodl-tmp/runs/${RUN_NAME}" \
    \
    trainer.eval_batch_size=64 \
    trainer.eval_before_train=false \
    trainer.eval_interval=999 \
    generator.eval_sampling_params.temperature=0 \
    generator.eval_sampling_params.stop='["</search>", "</answer>"]' \
    generator.eval_sampling_params.max_generate_length=500 \
    \
    trainer.export_path="/root/autodl-tmp/runs/${RUN_NAME}/exports" \
    \
    "$@" 2>&1 | tee "/tmp/${RUN_NAME}.log"
