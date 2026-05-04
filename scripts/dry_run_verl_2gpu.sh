#!/usr/bin/env bash
# verl 1-step dry-run on PRO 6000 (2 GPU) to validate:
#   - verl + Qwen2.5-3B FSDP init on 2 GPUs
#   - vllm rollout works on PRO 6000
#   - GSPO patch loads
#   - First training step completes
#
# Uses small batch to keep retrieval cost low (with CPU Flat ~5s/q,
# batch=8 × n_agent=2 × 2 turns = ~32 queries × 5s = ~3 min retrieval).
# Total dry-run estimated 5-10 min.

set -e

source /etc/network_turbo 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-0,1}
N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export DATA_DIR='/root/autodl-tmp/data/nq_search'
export BASE_MODEL='/root/autodl-tmp/models/Qwen2.5-3B-Instruct'

export EXPERIMENT_NAME=dry-run-pro6k-${N_GPUS}gpu
export WAND_PROJECT='agentic-rl-search'

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=XFORMERS
export OMP_NUM_THREADS=16

echo "[dry-run] CUDA=$CUDA_VISIBLE_DEVICES (N=$N_GPUS), batch=tiny, MAX_STEPS=1"

# Verify retrieval_server alive
if ! curl -sf -m 3 -X POST http://127.0.0.1:8000/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"queries":["smoke"],"topk":1}' > /dev/null 2>&1; then
    echo "❌ retrieval_server not on :8000 — start one first"
    exit 1
fi
echo "[dry-run] ✅ retrieval_server alive"

mkdir -p verl_checkpoints

# Tiny batch + skip val_before_train to minimize retrieval cost
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_format \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=16 \
    data.train_batch_size=8 \
    data.val_batch_size=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    +actor_rollout_ref.actor.clip_ratio_high=0.28 \
    +actor_rollout_ref.actor.loss_agg_mode=sequence \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=2 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=[console] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=999 \
    trainer.test_freq=999 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee /tmp/dry_run_$N_GPUS.log
