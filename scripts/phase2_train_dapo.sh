#!/usr/bin/env bash
# Phase 2 — DAPO clip-higher (warm-start from Phase 1 final ckpt)
#
# 改动 vs phase0_train_grpo.sh:
#   1. 入口仍是 main_ppo_format（继承 Phase 1 的 4 标签 DFA reward — 单变量只换 clip）
#   2. BASE_MODEL = Phase 1 final ckpt（如果存在）；否则 cold-start
#   3. 新增 +actor_rollout_ref.actor.clip_ratio_high=0.28 (DAPO baseline 0.28, vanilla 0.20)
#   4. EXPERIMENT_NAME 默认 phase2-dapo-cliphigh-50
#
# 前置条件：
#   1. patch 已 apply 到 Search-R1：
#      cd /root/autodl-tmp/external/Search-R1
#      patch -p1 < /root/agenicRL/patches/phase2_dapo_clip_higher.patch
#   2. 验证 patch 生效：
#      grep -n cliprange_high verl/trainer/ppo/core_algos.py  # 应 4 处命中
#   3. retrieval_server 在跑（端口 8000 health 200）
#
# 用法：
#   bash scripts/phase2_train_dapo.sh
#   PHASE1_CKPT=/root/autodl-tmp/runs/phase1-.../ckpts/global_step_50 bash scripts/phase2_train_dapo.sh

set -e

source /etc/network_turbo 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

# ============================================================
# 配置
# ============================================================
export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-0,1,2,3}
N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export DATA_DIR='/root/autodl-tmp/data/nq_search'

# Phase 2 起点：默认 cold-start（与 Phase 0e single-variable 三方对比，都从 Qwen 开始）
# 显式 `export PHASE1_CKPT=/path` 才会 warm-start
if [ -n "${PHASE1_CKPT:-}" ] && [ -d "$PHASE1_CKPT" ] && [ "$(ls -A $PHASE1_CKPT 2>/dev/null)" ]; then
    export BASE_MODEL="$PHASE1_CKPT"
    echo "[phase2-dapo] warm-start (opt-in) from $BASE_MODEL"
else
    export BASE_MODEL='/root/autodl-tmp/models/Qwen2.5-3B-Instruct'
    echo "[phase2-dapo] cold-start (default) from $BASE_MODEL"
fi

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-phase2-dapo-cliphigh-50}
export WAND_PROJECT='agentic-rl-search'

# DAPO 核心超参
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}    # ← Phase 2 唯一变量

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=XFORMERS

MAX_STEPS=${MAX_STEPS:-52}    # 让 step 50 ckpt + val 完整存在
SAVE_FREQ=${SAVE_FREQ:-10}    # 2026-04-27 Phase 0e 后调整：5 太碎，10 足够（损失上界 10 步 ≈ ¥10）
TEST_FREQ=${TEST_FREQ:-10}
LOGGER=${LOGGER:-wandb}

FSDP_MODE=${FSDP_MODE:-full}
case "$FSDP_MODE" in
    full)    PARAM_OFF=true;  GRAD_OFF=true;  OPTIM_OFF=true ;;
    minimal) PARAM_OFF=false; GRAD_OFF=false; OPTIM_OFF=true ;;
    none)    PARAM_OFF=false; GRAD_OFF=false; OPTIM_OFF=false ;;
esac

echo "[phase2] CUDA=$CUDA_VISIBLE_DEVICES (n_gpus=$N_GPUS), clip=[$CLIP_RATIO_LOW, $CLIP_RATIO_HIGH], FSDP=$FSDP_MODE"

test -d "$BASE_MODEL"             || { echo "❌ BASE_MODEL 不在 $BASE_MODEL"; exit 1; }
test -f "$DATA_DIR/train.parquet" || { echo "❌ NQ train.parquet 不在 $DATA_DIR"; exit 1; }

# Patch 校验：clip_ratio_high 关键字必须在 verl 源码里
VERL_ROOT="/root/autodl-tmp/external/Search-R1"
if ! grep -q "clip_ratio_high\|cliprange_high" "$VERL_ROOT/verl/trainer/ppo/core_algos.py"; then
    echo "❌ DAPO patch 未应用！先跑：cd $VERL_ROOT && patch -p1 < /root/agenicRL/patches/phase2_dapo_clip_higher.patch"
    exit 1
fi
echo "[phase2] ✅ DAPO patch 已生效"

mkdir -p verl_checkpoints

# ============================================================
# 训练
# ============================================================
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
