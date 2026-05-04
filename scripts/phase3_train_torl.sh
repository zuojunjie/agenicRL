#!/usr/bin/env bash
# Phase 3 — ToRL: multi-turn tool reward shaping
#
# 唯一变量 vs Phase 0e：
#   1. 入口 main_ppo_format → main_ppo_torl
#   2. reward 函数 qa_em_format → qa_em_torl（叠加 per-turn shaping）
#   3. max_turns 2 → 4
#   4. max_obs_length 500 → 800（解决 Phase 0e/2 频繁 OBSERVATION TOO LONG warning）
#
# 关于起点：
#   默认 warm-start from Phase 0e step_45（叠加 ToRL on top of "已会用 format 的模型"）
#   如要 cold-start，PHASE0E_CKPT="" 即可
#
# 前置条件：
#   1. 云端已存在 verl/utils/reward_score/qa_em_torl.py
#   2. 云端已存在 verl/trainer/main_ppo_torl.py
#   3. retrieval_server 在跑（如 GSPO 跑完后已停，需先 launch_retrieval_server.sh）

set -e

source /etc/network_turbo 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-0,1,2,3}
N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export DATA_DIR='/root/autodl-tmp/data/nq_search'

# ⚠️ 所有 Phase 都从 Qwen2.5-3B-Instruct cold-start（与 0e/DAPO/GSPO 严格 single-variable 对比）
# 显式 export PHASE0E_CKPT=/path 才会 warm-start
if [ -n "${PHASE0E_CKPT:-}" ] && [ -d "$PHASE0E_CKPT" ] && [ "$(ls -A $PHASE0E_CKPT 2>/dev/null)" ]; then
    export BASE_MODEL="$PHASE0E_CKPT"
    echo "[phase3-torl] warm-start (opt-in) from $BASE_MODEL"
else
    export BASE_MODEL='/root/autodl-tmp/models/Qwen2.5-3B-Instruct'
    echo "[phase3-torl] cold-start (default) from $BASE_MODEL"
fi

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-phase3-torl-multitool-50}
export WAND_PROJECT='agentic-rl-search'

# ToRL 核心超参
MAX_TURNS=${MAX_TURNS:-4}                       # ← Phase 3 主要变量
MAX_OBS_LENGTH=${MAX_OBS_LENGTH:-800}           # 截断阈值放宽
REPEAT_PENALTY=${REPEAT_PENALTY:--0.05}
PER_TURN_BONUS=${PER_TURN_BONUS:-0.05}
SHAPING_CLAMP=${SHAPING_CLAMP:-0.20}

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

echo "[phase3-torl] CUDA=$CUDA_VISIBLE_DEVICES n_gpus=$N_GPUS, max_turns=$MAX_TURNS, max_obs=$MAX_OBS_LENGTH"
echo "[phase3-torl] shaping: repeat_pen=$REPEAT_PENALTY per_turn=$PER_TURN_BONUS clamp=$SHAPING_CLAMP"

test -d "$BASE_MODEL"             || { echo "❌ BASE_MODEL 不在 $BASE_MODEL"; exit 1; }
test -f "$DATA_DIR/train.parquet" || { echo "❌ NQ train.parquet 不在"; exit 1; }

VERL_ROOT="/root/autodl-tmp/external/Search-R1"
test -f "$VERL_ROOT/verl/utils/reward_score/qa_em_torl.py" || { echo "❌ qa_em_torl.py 不存在"; exit 1; }
test -f "$VERL_ROOT/verl/trainer/main_ppo_torl.py" || { echo "❌ main_ppo_torl.py 不存在"; exit 1; }
echo "[phase3-torl] ✅ ToRL 文件已就绪"

mkdir -p verl_checkpoints

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_torl \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=$MAX_OBS_LENGTH \
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
    max_turns=$MAX_TURNS \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log
