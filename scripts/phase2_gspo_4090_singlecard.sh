#!/usr/bin/env bash
# Phase 2 GSPO on 4090 vGPU 48GB single card.
# Adapted from phase2_train_gspo.sh (4 GPU → 1 GPU).
#
# Key deltas vs original:
#   - CUDA_VISIBLE_DEVICES=0 (single GPU)
#   - n_gpus_per_node=1
#   - FSDP_MODE=full (params + grads + optim all offloaded to CPU)
#   - ppo_micro_batch=8 (down from 32, single GPU)
#   - log_prob_micro_batch=32 (down from 128)
#   - retriever uses IVF_PQ (CPU only, no GPU contention)
#
# Memory budget on 48GB GPU:
#   - vllm gpu_memory_utilization=0.5 = 24 GB
#   - FSDP with full offload: ~12 GB (params on GPU during forward)
#   - Activations: ~5 GB
#   - Total: ~41 GB / 48 → 7 GB headroom

set -e
source /etc/network_turbo 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

export CUDA_VISIBLE_DEVICES=0
N_GPUS=1
export DATA_DIR='/root/autodl-tmp/data/nq_search'
export BASE_MODEL='/root/autodl-tmp/models/Qwen2.5-3B-Instruct'

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-phase2-gspo-1gpu-ada48-52-VALONLY}
export WAND_PROJECT='agentic-rl-search'

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
LOSS_AGG_MODE=sequence

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=XFORMERS

MAX_STEPS=${MAX_STEPS:-52}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
LOGGER=${LOGGER:-wandb}

# Single GPU: must offload everything to CPU
PARAM_OFF=true
GRAD_OFF=true
OPTIM_OFF=true

echo "[phase2-gspo-1gpu] CUDA=$CUDA_VISIBLE_DEVICES, single Ada 48GB"
echo "[phase2-gspo-1gpu] FSDP_MODE=full (param+grad+optim all offloaded)"

VERL_ROOT="/root/autodl-tmp/external/Search-R1"
if ! grep -q "loss_agg_mode" "$VERL_ROOT/verl/trainer/ppo/core_algos.py"; then
    echo "❌ GSPO patch 未应用！"
    exit 1
fi
echo "[phase2-gspo-1gpu] ✅ GSPO patch 已生效"

if ! curl -sf -m 3 -X POST http://127.0.0.1:8000/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"queries":["smoke"],"topk":1,"return_scores":true}' > /dev/null 2>&1; then
    echo "❌ retrieval_server 没响应 :8000，先启动 launch_retrieval_ivfpq_4090.sh"
    exit 1
fi
echo "[phase2-gspo-1gpu] ✅ retrieval up"
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export RAY_object_store_memory=2000000000

cd $VERL_ROOT
mkdir -p verl_checkpoints

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_format \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
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
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO_LOW \
    +actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    +actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFF \
    actor_rollout_ref.actor.fsdp_config.grad_offload=$GRAD_OFF \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIM_OFF \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=$PARAM_OFF \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=3 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=[$LOGGER] \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=$MAX_STEPS \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log
