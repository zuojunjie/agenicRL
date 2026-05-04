#!/usr/bin/env bash
# Phase 5b — ARPO: trajectory credit + adaptive entropy + turn-level KL（叠加在 5a 上）
# 与 5a 严格 single-variable 对比（相对 5a 唯一新增 turn-level KL）
#
# 前置:
#   1. dp_actor.py 已 apply 5a + 5b patch（按顺序）
#   2. retrieval_server 在跑

set -e

source /etc/network_turbo 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-0,1,2,3}
N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export DATA_DIR='/root/autodl-tmp/data/nq_search'
export BASE_MODEL='/root/autodl-tmp/models/Qwen2.5-3B-Instruct'

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-phase5b-arpo-turnkl-only-52}
export WAND_PROJECT='agentic-rl-search'

# === Phase 5b: ONLY turn-level KL（不带 5a 特性，与 0e single-variable 对比）===
export ARPO_TRAJ_CREDIT=0
export ARPO_ADAPT_ENTROPY=0

export ARPO_TURN_KL=1
export ARPO_BETA_THINK=0.001
export ARPO_BETA_SEARCH=0.005      # search 段更紧 anchor
export ARPO_BETA_ANSWER=0.001
export ARPO_BETA_INFO=0.0          # information 是 env 注入，不计 KL
export ARPO_BETA_OTHER=0.001
export ARPO_TOKENIZER_PATH=/root/autodl-tmp/models/Qwen2.5-3B-Instruct

# 不带 GSPO / GiGPO / DAPO
unset GIGPO_ALPHA

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

echo "[phase5b] ARPO 5a (traj_credit + adapt_entropy) + 5b (turn-level KL)"
echo "[phase5b] β: think=$ARPO_BETA_THINK search=$ARPO_BETA_SEARCH answer=$ARPO_BETA_ANSWER info=$ARPO_BETA_INFO"

# patch 验证
VERL=/root/autodl-tmp/external/Search-R1
if ! grep -q "ARPO_TURN_KL" "$VERL/verl/workers/actor/dp_actor.py"; then
    echo "❌ Phase 5b patch 未应用！先跑 phase5b_apply.py（确保 5a 已先应用）"
    exit 1
fi
if ! grep -q "ARPO_TRAJ_CREDIT" "$VERL/verl/workers/actor/dp_actor.py"; then
    echo "❌ Phase 5a patch 未应用！先跑 phase5a_apply.py"
    exit 1
fi
echo "[phase5b] ✅ 5a + 5b patches 都已生效"

mkdir -p verl_checkpoints

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_format \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
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
