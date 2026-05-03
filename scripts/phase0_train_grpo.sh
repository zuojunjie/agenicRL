#!/usr/bin/env bash
# Phase 0 baseline 训练脚本
# 改自 external/Search-R1/train_grpo.sh
# 适配：4×A100-80G + Qwen2.5-3B-Instruct + 本地预下载的模型/数据
#
# 用法（在 GPU 模式下，retrieval_server 已起动后）：
#   bash scripts/phase0_train_grpo.sh
#
# 改动 vs 原版：
#   1. CUDA_VISIBLE_DEVICES: 8 GPU → 4 GPU
#   2. trainer.n_gpus_per_node: 8 → 4
#   3. BASE_MODEL: HF repo id → 本地预下载路径
#   4. 修上游 bug: $TRAIN_DATA_DIR/$TEST_DATA_DIR 都没定义，原脚本会崩；改成 $DATA_DIR
#   5. data.train_batch_size: 先保持 512，OOM 再降到 256
#   6. 加一个 mkdir -p verl_checkpoints

set -e

# ============================================================
# 必备环境（每次新 SSH session 都要 source）
# ============================================================
source /etc/network_turbo 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

# ============================================================
# 配置
# ============================================================
# GPU 选择：默认 4 卡共享 retrieval，可以改 TRAIN_GPUS=0,1,2 把 retrieval 独占到 GPU 3
export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-0,1,2,3}
N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export DATA_DIR='/root/autodl-tmp/data/nq_search'
export BASE_MODEL='/root/autodl-tmp/models/Qwen2.5-3B-Instruct'
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-phase0-nq-grpo-qwen2.5-3b-it-em}
export WAND_PROJECT='agentic-rl-search'

# 4090 无 NVLink — 强制 NCCL 走 socket 而不是 P2P
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 减少 GPU 显存碎片（OOM 错误推荐的设置）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# vllm 与 qwen2 的 flash_attn 兼容性问题，强制 XFORMERS
export VLLM_ATTENTION_BACKEND=XFORMERS

# 允许命令行覆盖 max_steps 用作 smoke test
MAX_STEPS=${MAX_STEPS:-50}
# ⚠️ SAVE_FREQ 默认 10（不再 100）——Phase 0d 教训：OOM 时损失太多 step
# 详见 memory/ckpt_save_lessons.md
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}

# Logger: console (默认，无 wandb 依赖) 或 wandb (需 WANDB_API_KEY env var)
LOGGER=${LOGGER:-console}

# FSDP offload 模式：
#   full     - param + grad + optim 全 offload 到 CPU（默认，4 卡共享 retrieval 时用）
#   minimal  - 只 optim offload（params/grads 在 GPU，actor update 提速 30-50%，需 dedicated retrieval 留出显存）
#   none     - 全在 GPU（最快但显存吃紧）
FSDP_MODE=${FSDP_MODE:-full}
case "$FSDP_MODE" in
    full)    PARAM_OFF=true;  GRAD_OFF=true;  OPTIM_OFF=true ;;
    minimal) PARAM_OFF=false; GRAD_OFF=false; OPTIM_OFF=true ;;
    none)    PARAM_OFF=false; GRAD_OFF=false; OPTIM_OFF=false ;;
    *) echo "❌ unknown FSDP_MODE=$FSDP_MODE"; exit 1 ;;
esac
echo "[config] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (n_gpus=$N_GPUS), FSDP_MODE=$FSDP_MODE"

# ============================================================
# 启动前检查
# ============================================================
test -d "$BASE_MODEL"             || { echo "❌ 模型不在 $BASE_MODEL"; exit 1; }
test -f "$DATA_DIR/train.parquet" || { echo "❌ NQ train.parquet 不在 $DATA_DIR"; exit 1; }
test -f "$DATA_DIR/test.parquet"  || { echo "❌ NQ test.parquet 不在 $DATA_DIR"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -4
mkdir -p verl_checkpoints

# 5 分钟烟雾测试可以单独跑（max_steps 改成 5）
# 这里是真训练版

# ============================================================
# 训练（参数大体对齐 Search-R1 论文 v0.2 配方）
# ============================================================
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
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
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFF \
    actor_rollout_ref.actor.fsdp_config.grad_offload=$GRAD_OFF \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIM_OFF \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
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
