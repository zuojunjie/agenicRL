#!/usr/bin/env bash
# SkyRL Search-R1 GSPO repro on 2×PRO 6000 (Blackwell sm_120)
# Adapted from SkyRL examples/train/search/run_search.sh (8GPU → 2GPU)

set -ex
ulimit -n 65535

DATA_DIR="/root/autodl-tmp/data/searchR1"
RUN_NAME="skyrl-gspo-pro6k-2gpu-52steps"
MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-3B-Instruct"

# TIS = Token Importance Sampling, gives sequence-level-ish behavior
TIS_TYPE=token
TIS_IMP_RATIO_CAP=2.0

cd /root/autodl-tmp/external/SkyRL

# Source env (academic accel + no_proxy + HF mirror)
source /etc/network_turbo 2>/dev/null
export no_proxy="${no_proxy},hf-mirror.com,pypi.tuna.tsinghua.edu.cn,127.0.0.1,localhost"
export NO_PROXY="$no_proxy"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/.cache/huggingface

source .venv/bin/activate

# Verify retrieval_server alive
curl -sf -m 5 -X POST http://127.0.0.1:8000/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"queries":["smoke"],"topk":1}' > /dev/null \
    || { echo "❌ retrieval_server :8000 not ready"; exit 1; }
echo "[skyrl-gspo] ✅ retrieval up"

uv run --frozen --extra fsdp -m skyrl.train.entrypoints.main_base \
    data.train_data="['${DATA_DIR}/train.parquet']" \
    data.val_data="['${DATA_DIR}/validation.parquet']" \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.policy.optimizer_config.max_grad_norm=0.5 \
    trainer.policy.optimizer_config.num_warmup_steps=15 \
    trainer.algorithm.use_kl_loss=true \
    trainer.algorithm.kl_loss_coef=0.001 \
    trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_TYPE \
    trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TIS_IMP_RATIO_CAP \
    trainer.policy.model.path="$MODEL_PATH" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp2 \
    trainer.policy.fsdp_config.cpu_offload=false \
    trainer.ref.fsdp_config.cpu_offload=true \
    trainer.placement.policy_num_gpus_per_node=2 \
    trainer.placement.ref_num_gpus_per_node=2 \
    generator.inference_engine.num_engines=1 \
    generator.inference_engine.tensor_parallel_size=1 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.gpu_memory_utilization=0.5 \
    trainer.epochs=1 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=512 \
    trainer.policy_mini_batch_size=256 \
    trainer.micro_forward_batch_size_per_gpu=4 \
    trainer.micro_train_batch_size_per_gpu=4 \
    trainer.max_prompt_length=2048 \
    generator.max_input_length=4096 \
    generator.sampling_params.max_generate_length=500 \
    generator.inference_engine.async_engine=true \
    generator.batched=false \
    generator.use_conversation_multi_turn=false \
    generator.n_samples_per_prompt=5 \
    generator.max_turns=4 \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=1.0 \
    generator.sampling_params.stop='["</search>", "</answer>"]' \
    environment.env_class="search" \
    environment.skyrl_gym.max_env_workers=16 \
    environment.skyrl_gym.search.log_requests=false \
    environment.skyrl_gym.search.search_url="http://127.0.0.1:8000/retrieve" \
    environment.skyrl_gym.search.topk=3 \
    trainer.logger="wandb" \
    trainer.project_name="agentic-rl-search" \
    trainer.run_name="${RUN_NAME}" \
    trainer.ckpt_interval=10 \
    trainer.hf_save_interval=999 \
    trainer.max_ckpts_to_keep=3 \
    trainer.resume_mode=latest \
    trainer.ckpt_path="/root/autodl-tmp/runs/${RUN_NAME}" \
    trainer.eval_batch_size=256 \
    trainer.eval_before_train=false \
    generator.eval_sampling_params.temperature=0 \
    generator.eval_sampling_params.stop='["</search>", "</answer>"]' \
    generator.eval_sampling_params.max_generate_length=500 \
    trainer.export_path="/root/autodl-tmp/runs/${RUN_NAME}/exports" \
    trainer.eval_interval=10 \
    +trainer.total_steps=52 \
    "$@"
