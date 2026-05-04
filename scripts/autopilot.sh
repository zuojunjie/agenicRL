#!/usr/bin/env bash
# 整夜自主作战 controller
# 调度顺序：retrieval_server → smoke test → real training
# 每步失败有 fallback；任何阶段产出都会写到 morning_briefing.md
#
# 用法：在 cloud 上 nohup bash autopilot.sh > /tmp/autopilot.log 2>&1 &
# 监控：tail -f /tmp/autopilot.log
set -uo pipefail

REPO=/root/autodl-tmp/agenicRL
BRIEF=/tmp/morning_briefing.md
LOG=/tmp/autopilot.log

# 时间戳工具
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a $LOG; }

write_brief() {
    cat > $BRIEF <<EOF
# 🌅 Morning Briefing — agenicRL Phase 0
_Generated $(ts) by autopilot.sh_

$1
EOF
}

append_brief() { echo -e "$1" >> $BRIEF; }

write_brief "## 状态: 进行中
启动时间: $(ts)
预算上限: ¥60 (~10h GPU)
"

# ============================================================
# 阶段 1：retrieval_server
# ============================================================
say "===== Phase 1: 启动 retrieval_server ====="
append_brief "\n## Phase 1: retrieval_server"

bash $REPO/scripts/launch_retrieval_server.sh 2>&1 | tee -a $LOG
RS_OK=$?

if [ $RS_OK -ne 0 ] || ! curl -s -m 5 -X POST http://127.0.0.1:8000/retrieve \
   -H 'Content-Type: application/json' \
   -d '{"queries":["test"],"topk":1}' > /dev/null 2>&1; then
    append_brief "❌ retrieval_server 启动失败 — 训练阶段无法继续"
    append_brief "诊断: \`tail -50 /tmp/retrieval_server.log\`"
    say "FATAL: retrieval_server unreachable, aborting"
    exit 1
fi
append_brief "✅ retrieval_server 上线 (PID $(cat /tmp/retrieval.pid))"

# ============================================================
# 阶段 2：smoke test (max_steps=2, 5-10 min)
# ============================================================
say "===== Phase 2: smoke test (max_steps=2) ====="
append_brief "\n## Phase 2: smoke test (max_steps=2)"
SMOKE_START=$(date +%s)

cd $REPO/.. # 确保 verl_checkpoints 落在合理位置
cd /root/autodl-tmp/external/Search-R1
MAX_STEPS=2 TEST_FREQ=1 SAVE_FREQ=2 \
    EXPERIMENT_NAME=phase0-smoke \
    bash $REPO/scripts/phase0_train_grpo.sh 2>&1 | tee /tmp/smoke.log | tail -200

SMOKE_RC=${PIPESTATUS[0]}
SMOKE_DUR=$(( $(date +%s) - SMOKE_START ))

# hydra 错误也走 exit 0，所以单看 exit code 不够
# 也要检查 log 里是否有 "Error executing job" 或 "Traceback"
SMOKE_ERR=0
if [ $SMOKE_RC -ne 0 ]; then SMOKE_ERR=1; fi
if grep -qE "Error executing job|^Traceback|RuntimeError|OutOfMemoryError" /tmp/smoke.log 2>/dev/null; then SMOKE_ERR=1; fi
# 训练正常会跑 ≥30s（rollout + grad），如果 <30s 必失败
if [ $SMOKE_DUR -lt 30 ]; then SMOKE_ERR=1; fi

if [ $SMOKE_ERR -ne 0 ]; then
    append_brief "❌ smoke test 失败 (exit=$SMOKE_RC, ${SMOKE_DUR}s, err markers detected)"
    append_brief "log tail:\n\`\`\`\n$(tail -40 /tmp/smoke.log)\n\`\`\`"
    say "smoke test failed, aborting real training"

    # 收集诊断信息
    append_brief "\n### 诊断快照"
    append_brief "GPU 状态:\n\`\`\`\n$(nvidia-smi 2>&1)\n\`\`\`"
    exit 2
fi

append_brief "✅ smoke test 通过 (${SMOKE_DUR}s, exit=0)"

# 提取 smoke test 的关键 metric
SMOKE_VAL=$(grep -E "val.*reward|response.*length" /tmp/smoke.log | head -5)
[ -n "$SMOKE_VAL" ] && append_brief "smoke metrics:\n\`\`\`\n$SMOKE_VAL\n\`\`\`"

# ============================================================
# 阶段 3：真训练 baseline
# ============================================================
say "===== Phase 3: real training baseline ====="
append_brief "\n## Phase 3: real training (max_steps=300, 估 4-5h)"
REAL_START=$(date +%s)

# 不跑 1005 步那么多——300 步够看趋势
# 4090 无 NVLink + GRPO rollout = 估 1-1.5 min/step
# 300 步 ≈ 5h，吃完夜间窗口
MAX_STEPS=300 TEST_FREQ=20 SAVE_FREQ=50 \
    EXPERIMENT_NAME=phase0-baseline \
    bash $REPO/scripts/phase0_train_grpo.sh 2>&1 | tee /tmp/training.log

TRAIN_RC=${PIPESTATUS[0]}
TRAIN_DUR=$(( $(date +%s) - REAL_START ))

if [ $TRAIN_RC -eq 0 ]; then
    append_brief "✅ 真训练完成 (${TRAIN_DUR}s = $((TRAIN_DUR / 60)) min, exit=0)"
else
    append_brief "⚠️ 真训练 exit=$TRAIN_RC after ${TRAIN_DUR}s ($((TRAIN_DUR / 60)) min)"
    append_brief "log tail:\n\`\`\`\n$(tail -40 /tmp/training.log)\n\`\`\`"
fi

# ckpt 现状
CKPT_DIR=/root/autodl-tmp/external/Search-R1/verl_checkpoints/phase0-baseline
if [ -d $CKPT_DIR ]; then
    append_brief "checkpoints:\n\`\`\`\n$(ls -la $CKPT_DIR | head)\n\`\`\`"
fi

# 抽 wandb 关键指标（如果有）
WANDB_LOG=/root/autodl-tmp/external/Search-R1/phase0-baseline.log
if [ -f $WANDB_LOG ]; then
    LAST_REWARD=$(grep -oE "reward[^,]*?[0-9.]+" $WANDB_LOG | tail -1)
    LAST_STEP=$(grep -oE "step.*?[0-9]+" $WANDB_LOG | tail -1)
    append_brief "最后训练状态: $LAST_STEP, $LAST_REWARD"
fi

# ============================================================
# 收工
# ============================================================
say "===== Autopilot 收工 ====="
append_brief "\n---\n_收工时间: $(ts), 总耗时: $(($(date +%s) - SMOKE_START))s_"

# 关 retrieval_server 省 GPU 电
if [ -f /tmp/retrieval.pid ]; then
    kill $(cat /tmp/retrieval.pid) 2>/dev/null && say "retrieval_server 已停"
fi

say "DONE. 看 $BRIEF 总结。"
