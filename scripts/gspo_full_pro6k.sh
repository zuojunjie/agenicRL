#!/usr/bin/env bash
# GSPO 52-step reproduction on PRO 6000 (1 GPU) — built on v23 working stack.
#
# Algorithm: SkyRL-canonical Search-R1 with TIS=2.0 (token importance sampling)
# This is SkyRL's GSPO-equivalent (off-policy correction at sequence level).
# Key delta vs original 4090 verl run (which used loss_agg_mode=sequence):
#   - Framework: verl 0.7 → SkyRL 0.19 (Blackwell sm_120 compat)
#   - Algorithm: GRPO + loss_agg=sequence → GRPO + TIS=2.0 (similar variance reduction)
#   - GPU: 4090×4 → PRO 6000 × 1 (4× less, but each is bigger)
#
# Memory fixes from v23 (all required):
#   - mmap retrieval (-27 GB CPU)
#   - num_workers=0 patched in trainer_utils.py (-40 GB CPU)
#   - RAY_object_store_memory=5GB (-20 GB)
#   - vllm gpu_util=0.30
#
# Expected: ~5-6h for 52 steps at batch=512. ETA visible in dashboard.

set -ex
ulimit -n 65535

DATA_DIR="/root/autodl-tmp/data/searchR1"
RUN_NAME="gspo-pro6k-1gpu-52steps"
MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-3B-Instruct"

cd /root/autodl-tmp/external/SkyRL

source /etc/network_turbo 2>/dev/null
export no_proxy="${no_proxy},hf-mirror.com,pypi.tuna.tsinghua.edu.cn,127.0.0.1,localhost"
export NO_PROXY="$no_proxy"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export PATH=/root/.local/bin:$PATH

source .venv/bin/activate

# v23 memory budget controls
export _SKYRL_USE_NEW_INFERENCE=0
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export RAY_object_store_memory=5000000000

# Verify retrieval up
curl -sf -m 5 -X POST http://127.0.0.1:8000/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"queries":["smoke"],"topk":1}' > /dev/null \
    || { echo "❌ retrieval not ready"; exit 1; }
echo "[gspo] ✅ retrieval up"

# Per-process RAM trace + step trace
TRACE=/tmp/gspo_full_ram_trace.log
echo "ts,total_GB,step,reward,p_train_s" > "$TRACE"
(
    while true; do
        TOTAL=$(awk '{s+=$2} END {printf "%.1f", s/1024/1024}' <(ps -eo pid,rss --no-headers))
        # parse last metric values
        STEP=$(grep -c "timing/policy_train" /tmp/gspo_full.log 2>/dev/null || echo 0)
        REWARD=$(grep "loss/avg_final_rewards" /tmp/gspo_full.log 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" | tail -1)
        TIMING=$(grep "timing/policy_train" /tmp/gspo_full.log 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" | tail -1)
        echo "$(date +%H:%M:%S),$TOTAL,${STEP:-0},${REWARD:-0},${TIMING:-0}" >> "$TRACE"
        sleep 30
    done
) &
TRACER_PID=$!
trap "kill $TRACER_PID 2>/dev/null" EXIT

# GSPO full config — based on SkyRL canonical run_search.sh + 4090 verl baseline + v23 mem fixes
python -m skyrl.train.entrypoints.main_base \
    data.train_data="['${DATA_DIR}/train.parquet']" \
    data.val_data="['${DATA_DIR}/validation.parquet']" \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.policy.optimizer_config.max_grad_norm=0.5 \
    trainer.policy.optimizer_config.num_warmup_steps=15 \
    trainer.algorithm.use_kl_loss=true \
    trainer.algorithm.kl_loss_coef=0.001 \
    trainer.algorithm.off_policy_correction.tis_ratio_type=token \
    trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=2.0 \
    trainer.policy.model.path="$MODEL_PATH" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp2 \
    trainer.policy.fsdp_config.cpu_offload=false \
    trainer.ref.fsdp_config.cpu_offload=false \
    trainer.placement.policy_num_gpus_per_node=1 \
    trainer.placement.ref_num_gpus_per_node=1 \
    trainer.use_sample_packing=false \
    trainer.flash_attn=false \
    generator.inference_engine.num_engines=1 \
    generator.inference_engine.tensor_parallel_size=1 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.gpu_memory_utilization=0.30 \
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
