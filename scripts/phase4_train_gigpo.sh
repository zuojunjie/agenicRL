#!/usr/bin/env bash
# Phase 4 — GiGPO: 两层 group advantage（inner + outer）
#
# 唯一变量 vs Phase 0e:
#   advantage 计算从单层 GRPO → 双层（α·inner + (1-α)·outer）
#
# 起点：默认 warm-start from Phase 3 ToRL final ckpt（如有）；否则 Phase 0e；否则 cold-start
#
# 前置:
#   bash /root/autodl-tmp/agenicRL/patches/phase4_gigpo_deploy.sh

set -e

source /etc/network_turbo 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-0,1,2,3}
N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export DATA_DIR='/root/autodl-tmp/data/nq_search'

# ⚠️ 默认 cold-start（与所有其他 phase 严格 single-variable 对比）
# 显式 export WARM_START_CKPT=/path 才会 warm-start
if [ -n "${WARM_START_CKPT:-}" ] && [ -d "$WARM_START_CKPT" ]; then
    export BASE_MODEL="$WARM_START_CKPT"
    echo "[phase4-gigpo] warm-start (opt-in) from $BASE_MODEL"
else
    export BASE_MODEL='/root/autodl-tmp/models/Qwen2.5-3B-Instruct'
    echo "[phase4-gigpo] cold-start (default) from $BASE_MODEL"
fi

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-phase4-gigpo-twolayer-50}
export WAND_PROJECT='agentic-rl-search'

# Phase 4 核心变量：α 控制 inner vs outer 比例
export GIGPO_ALPHA=${GIGPO_ALPHA:-0.7}    # 0.7=70% inner, 30% outer (论文推荐)

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=XFORMERS

MAX_STEPS=${MAX_STEPS:-52}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
LOGGER=${LOGGER:-wandb}

FSDP_MODE=${FSDP_MODE:-full}
case "$FSDP_MODE" in
    full)    PARAM_OFF=true;  GRAD_OFF=true;  OPTIM_OFF=true ;;
    minimal) PARAM_OFF=false; GRAD_OFF=false; OPTIM_OFF=true ;;
    none)    PARAM_OFF=false; GRAD_OFF=false; OPTIM_OFF=false ;;
esac

# 训练入口：默认 main_ppo_format（与 Phase 0e/DAPO/GSPO 同 reward）
# 唯一变量是 GIGPO_ALPHA（advantage 计算）
ENTRY="verl.trainer.main_ppo_format"
echo "[phase4-gigpo] GIGPO_ALPHA=$GIGPO_ALPHA, entry=$ENTRY"

# Patch 校验
VERL_ROOT="/root/autodl-tmp/external/Search-R1"
if ! grep -q "GIGPO_ALPHA" "$VERL_ROOT/verl/trainer/ppo/ray_trainer.py"; then
    echo "❌ GiGPO patch 未应用！先跑 phase4_gigpo_deploy.sh"
    exit 1
fi
echo "[phase4-gigpo] ✅ GiGPO patch 已生效"

mkdir -p verl_checkpoints

PYTHONUNBUFFERED=1 python3 -m $ENTRY \
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
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFF \
    actor_rollout_ref.actor.fsdp_config.grad_offload=$GRAD_OFF \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIM_OFF \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
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
