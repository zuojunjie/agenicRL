#!/usr/bin/env bash
# GSPO repro on 2×PRO 6000 (Blackwell sm_120) using:
#   - verl 0.7.1 (upstream)
#   - vllm 0.9.2 (Blackwell-compatible)
#   - search_multiturn_grpo.yaml (Search-R1 native config)
#   - DFA 5-tier reward (qa_em_format, ports our 4090 baseline 0.4378 setup)
#   - GSPO via loss_agg_mode=seq-mean-token-mean

set -ex
ulimit -n 65535

source /etc/network_turbo 2>/dev/null
unset http_proxy https_proxy
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/searchr1-v2

VERL_DIR="/root/autodl-tmp/external/verl"
DATA_DIR="/root/autodl-tmp/data/nq_search"
MODEL="/root/autodl-tmp/models/Qwen2.5-3B-Instruct"
TOOL_CONFIG="$VERL_DIR/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml"
CONFIG_PATH="$VERL_DIR/examples/sglang_multiturn/config"

export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-0,1}
N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
EXP_NAME=${EXPERIMENT_NAME:-gspo-pro6k-2gpu-52}

# vllm + sm_120 specific
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=16

# Verify retrieval up
curl -sf -m 10 -X POST http://127.0.0.1:8000/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"queries":["smoke"],"topk":1}' > /dev/null \
    || { echo "❌ retrieval_server :8000 not ready"; exit 1; }
echo "[gspo] ✅ retrieval up"

cd $VERL_DIR

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.policy_loss.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='agentic-rl-search' \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=52 \
    trainer.default_local_dir="/root/autodl-tmp/runs/$EXP_NAME" \
    2>&1 | tee /tmp/${EXP_NAME}.log
