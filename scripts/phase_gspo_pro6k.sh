#!/usr/bin/env bash
# GSPO repro on PRO 6000 (2 GPUs), validating that 4090 baseline 0.4378 holds.
#
# Single-variable diff vs phase2_train_gspo.sh:
#   - TRAIN_GPUS default: 0,1 (was 0,1,2,3 on 4090)
#   - gpu_memory_utilization: 0.5 (was 0.3 on 24GB 4090; PRO 6000 has 96GB so we can give vllm more)
#   - EXPERIMENT_NAME: phase-gspo-pro6k-2gpu-52
#
# Everything else (loss_agg=sequence, clip_high=0.28, etc.) IDENTICAL to
# 4090 GSPO winner that scored 0.4378.
#
# Pre-reqs:
#   1. retrieval_server with IVF_PQ index running on :8000 (launch_retrieval_ivfpq.sh)
#   2. verl DAPO + GSPO patches applied
#   3. /root/autodl-tmp/data/nq_search/{train,test}.parquet present

set -e

source /etc/network_turbo 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-0,1}    # 2 GPU default for PRO 6000
N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export DATA_DIR='/root/autodl-tmp/data/nq_search'
export BASE_MODEL='/root/autodl-tmp/models/Qwen2.5-3B-Instruct'

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-phase-gspo-pro6k-2gpu-52}
export WAND_PROJECT='agentic-rl-search'

CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-sequence}    # GSPO key flag

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=XFORMERS

# OMP threads (PRO 6000 host has 208 cores; cap to avoid oversubscription)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

MAX_STEPS=${MAX_STEPS:-52}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
LOGGER=${LOGGER:-wandb}

# PRO 6000 has 96GB VRAM, plenty for full FSDP without offload
FSDP_MODE=${FSDP_MODE:-minimal}    # 'full' = all offload (slow), 'minimal' = optim only (faster on big VRAM)
case "$FSDP_MODE" in
    full)    PARAM_OFF=true;  GRAD_OFF=true;  OPTIM_OFF=true ;;
    minimal) PARAM_OFF=false; GRAD_OFF=false; OPTIM_OFF=true ;;
    none)    PARAM_OFF=false; GRAD_OFF=false; OPTIM_OFF=false ;;
esac

echo "[gspo-pro6k] CUDA=$CUDA_VISIBLE_DEVICES (N=$N_GPUS), clip=[$CLIP_RATIO_LOW, $CLIP_RATIO_HIGH], loss_agg=$LOSS_AGG_MODE"
echo "[gspo-pro6k] FSDP=$FSDP_MODE (param_off=$PARAM_OFF, grad_off=$GRAD_OFF, optim_off=$OPTIM_OFF)"

test -d "$BASE_MODEL"             || { echo "❌ BASE_MODEL 不在"; exit 1; }
test -f "$DATA_DIR/train.parquet" || { echo "❌ NQ train.parquet 不在"; exit 1; }

VERL_ROOT="/root/autodl-tmp/external/Search-R1"
if ! grep -q "loss_agg_mode" "$VERL_ROOT/verl/trainer/ppo/core_algos.py"; then
    echo "❌ GSPO patch 未应用！"
    exit 1
fi
echo "[gspo-pro6k] ✅ GSPO patch 已生效"

# Verify retrieval_server up
if ! curl -sf -m 3 -X POST http://127.0.0.1:8000/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"queries":["smoke"],"topk":1}' > /dev/null 2>&1; then
    echo "❌ retrieval_server 没响应 :8000，先 launch_retrieval_ivfpq.sh"
    exit 1
fi
echo "[gspo-pro6k] ✅ retrieval_server alive"

mkdir -p verl_checkpoints

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_format \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512 \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO_LOW \
    +actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    +actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFF \
    actor_rollout_ref.actor.fsdp_config.grad_offload=$GRAD_OFF \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIM_OFF \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=$PARAM_OFF \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=[$LOGGER] \
    +trainer.val_only=false \
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
