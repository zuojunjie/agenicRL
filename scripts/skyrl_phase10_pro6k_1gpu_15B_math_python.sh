#!/usr/bin/env bash
# Phase 10 = 2× PRO 6000 + Qwen2.5-Math-7B-Instruct + MATH+Python (GSPO baseline)
#
# 与 Phase 9 的关键差异:
#   1. 模型: Qwen2.5-7B-Instruct → Qwen2.5-Math-7B-Instruct (paper 实证 +10-15% 在数学上)
#   2. 数据: NQ → MATH (后续 GSM8K smoke + MATH 全程)
#   3. 工具: search retrieval_server → Python sandbox (CPU 进程, 不占 GPU)
#   4. GPU: 3 卡 → 2 卡 (省 1 卡 retrieval, 无 cpu_offload 仍稳)
#   5. env_class: search → math_python
#   6. max_turns: 2 → 8 (Math CoT 多次 Python 调用)
#   7. max_response: 500 → 1024 (Math 答案推理更长)
# 算法继承 7.8d/Phase 9 验证最优:
#   - paper-faithful GSPO (sequence-level IS)
#   - asymmetric clip [0.2, 0.28]
#   - mini=64, micro=16 (从 32 减半留余地; 7B + 长 CoT activations 更大)
#   - flash_attn + sample_packing + enforce_eager=false
# 内存预算 (per training GPU 0/1, 96 GB):
#   FSDP 7B sharded (no offload):       ~53 GB
#   all-gather 临时:                    ~30 GB
#   activations (longer CoT, micro=16):  ~8 GB
#   buffer:                             ~5 GB
#   合计 train peak:                    ~96 GB ⚠️ 卡边界 (要监控)
# 期望:
#   单 step:  ~10-12 min (vs Phase 9 8.4 min, 因 max_response 长 + max_turns 多)
#   52 步:    ~9-11 h
#   val (MATH-500 pass@1): 30-45% (paper ToRL-7B-Instruct + GSPO 应在此区间)

set -ex
ulimit -n 65535
DATA_DIR="/root/autodl-tmp/data/math"
RUN_NAME="phase10-skyrl-pro6k-1gpu-15B-Math-Inst"
MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-Math-1.5B-Instruct"
PROJECT="agenic-rl-A6000New"
cd /root/autodl-tmp/external/SkyRL
# Wandb key from ~/.netrc
export WANDB_API_KEY=$(awk '/api.wandb.ai/{getline; getline; print $2}' /root/.netrc)
source /etc/network_turbo 2>/dev/null
export no_proxy="${no_proxy},hf-mirror.com,pypi.tuna.tsinghua.edu.cn,127.0.0.1,localhost"
export NO_PROXY="$no_proxy"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export PATH=/root/.local/bin:$PATH
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=GRAPH,INIT
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# Sky/Ray memory budget
export _SKYRL_USE_NEW_INFERENCE=0
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export RAY_object_store_memory=5000000000
# NCCL only
# === Phase 10: 2 GPU 训练 + Python sandbox 在 CPU 进程 ===
export CUDA_VISIBLE_DEVICES=0
# Verify model + data ready
test -d "$MODEL_PATH" || { echo "❌ Model not found: $MODEL_PATH"; exit 1; }
test -f "${DATA_DIR}/math_train_52step.parquet" || { echo "❌ Data not found: ${DATA_DIR}/math_train_52step.parquet"; exit 1; }
echo "[phase10] ✅ model + data ready"
UV_PROJECT_ENVIRONMENT=/root/autodl-tmp/.skyrl-venv uv run --frozen --extra fsdp -m skyrl.train.entrypoints.main_base \
    data.train_data="['${DATA_DIR}/math_train_52step.parquet']" \
    data.val_data="['${DATA_DIR}/math_val.parquet']" \
    \
    `# === Algorithm: Paper-faithful GSPO + asym clip (Phase 9 验证最优) ===` \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.algorithm.policy_loss_type="gspo" \
    trainer.algorithm.loss_reduction="sequence_mean" \
    trainer.algorithm.eps_clip_low=0.2 \
    trainer.algorithm.eps_clip_high=0.28 \
    trainer.algorithm.use_kl_loss=true \
    trainer.algorithm.kl_loss_coef=0.001 \
    trainer.algorithm.kl_estimator_type="k3" \
    trainer.algorithm.use_kl_in_reward=false \
    `# === Optimizer ===` \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.policy.optimizer_config.max_grad_norm=1.0 \
    trainer.policy.optimizer_config.num_warmup_steps=15 \
    `# === Model + FSDP (no cpu_offload, 2 卡足够) ===` \
    trainer.policy.model.path="$MODEL_PATH" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp2 \
    trainer.policy.fsdp_config.cpu_offload=false \
    trainer.ref.fsdp_config.cpu_offload=false \
    trainer.gradient_checkpointing=true \
    trainer.use_sample_packing=true \
    trainer.flash_attn=true \
    `# === GPU dist: 2 卡 ===` \
    trainer.placement.policy_num_gpus_per_node=1 \
    trainer.placement.ref_num_gpus_per_node=1 \
    `# === Inference: TP=2 ===` \
    generator.inference_engine.num_engines=1 \
    generator.inference_engine.tensor_parallel_size=1 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.gpu_memory_utilization=0.80 \
    generator.inference_engine.async_engine=true \
    generator.inference_engine.enforce_eager=false \
    generator.inference_engine.max_num_batched_tokens=16384 \
    generator.inference_engine.max_num_seqs=512 \
    `# === Batch (mini=64 二分锁定, micro=16 给 long CoT 留余地) ===` \
    trainer.epochs=1 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=512 \
    trainer.policy_mini_batch_size=64 \
    trainer.micro_forward_batch_size_per_gpu=64 \
    trainer.micro_train_batch_size_per_gpu=32 \
    trainer.max_prompt_length=2048 \
    `# === Generation (long CoT for math reasoning) ===` \
    generator.max_input_length=2048 \
    generator.sampling_params.max_generate_length=768 \
    generator.batched=false \
    generator.use_conversation_multi_turn=false \
    generator.n_samples_per_prompt=8 \
    generator.max_turns=4 \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=1.0 \
    generator.sampling_params.stop='["</python>", "</answer>"]' \
    `# === Math + Python environment (新 env class) ===` \
    environment.env_class="math_python" \
    environment.skyrl_gym.max_env_workers=16 \
    `# === Logging + ckpt: production, save every 10 ===` \
    trainer.logger="wandb" \
    trainer.project_name="$PROJECT" \
    trainer.run_name="${RUN_NAME}" \
    trainer.ckpt_interval=10 \
    trainer.hf_save_interval=50 \
    trainer.max_ckpts_to_keep=3 \
    trainer.resume_mode=latest \
    trainer.ckpt_path="/root/autodl-tmp/runs/${RUN_NAME}" \
    `# === Eval: 跳过 val@0, 每 10 step eval ===` \
    trainer.eval_batch_size=128 \
    trainer.eval_before_train=false \
    trainer.eval_interval=10 \
    generator.eval_sampling_params.temperature=0 \
    generator.eval_sampling_params.stop='["</python>", "</answer>"]' \
    generator.eval_sampling_params.max_generate_length=768 \
    trainer.export_path="/root/autodl-tmp/runs/${RUN_NAME}/exports" \
    "$@" 2>&1 | tee "/tmp/${RUN_NAME}.log"
