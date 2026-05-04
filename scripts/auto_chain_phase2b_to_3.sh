#!/usr/bin/env bash
# 云端自治 watchdog：等 Phase 2B GSPO 结束 → finalize → 起 Phase 3 ToRL
#
# 用法（云端）：
#   nohup bash /root/autodl-tmp/agenicRL/scripts/auto_chain_phase2b_to_3.sh > /root/autodl-tmp/auto_chain.log 2>&1 &
#
# 不依赖 Claude session。即使 Mac 关机也继续。

set -uo pipefail

LOG=/root/autodl-tmp/auto_chain.log
PID_2B=${PID_2B:-302130}
GSPO_RUN=phase2-gspo-seqlevel-50
PHASE3_RUN=phase3-torl-multitool-50

ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

echo "$(ts) ===== auto_chain start ====="
echo "$(ts) watching PID $PID_2B (Phase 2B GSPO)"

# ============================================================
# 1) 等 Phase 2B 退出
# ============================================================
while kill -0 $PID_2B 2>/dev/null; do
    STEP=$(grep -E "epoch 0, step" /tmp/phase2_gspo.log 2>/dev/null | tail -1 | grep -oE "step [0-9]+")
    echo "$(ts) Phase 2B alive — $STEP"
    sleep 300  # 5 min poll
done
echo "$(ts) Phase 2B exited"

# 等 5s 让最后一次 wandb sync 完成
sleep 10

# ============================================================
# 2) Phase 2B finalize
# ============================================================
echo "$(ts) ===== finalize Phase 2B ====="
cd /root/autodl-tmp/external/Search-R1
EXPERIMENT_NAME=$GSPO_RUN \
  bash /root/autodl-tmp/agenicRL/scripts/finalize_run.sh $GSPO_RUN \
  "Phase 2B GSPO sequence-level cold-start MAX_STEPS=52" 2>&1 | tail -20

# ============================================================
# 3) 抓 GSPO final val/nq 写到 marker file（让 Mac 端拉取）
# ============================================================
GSPO_FINAL=$(grep -oE "Final validation metrics:.*nq.: [0-9.]+" /tmp/phase2_gspo.log | tail -1 | grep -oE "[0-9]+\.[0-9]+")
echo "$(ts) GSPO final val/nq = $GSPO_FINAL"
echo "phase2b_final_val=$GSPO_FINAL" > /root/autodl-tmp/runs/$GSPO_RUN/final_val.txt

# ============================================================
# 4) 起 Phase 3 ToRL
# ============================================================
echo "$(ts) ===== launch Phase 3 ToRL ====="
# 验证 retrieval_server 还活着
if ! curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "$(ts) ⚠️ retrieval_server 没响应，尝试重启"
    nohup bash /root/autodl-tmp/agenicRL/scripts/launch_retrieval_server.sh > /tmp/retrieval.log 2>&1 &
    sleep 30
fi

# 验证 Phase 0e ckpt（Phase 3 warm-start 起点）存在
if [ ! -d /root/autodl-tmp/runs/phase0e-format-reward-50/ckpts/global_step_45 ]; then
    echo "$(ts) ❌ Phase 0e ckpt 不存在，Phase 3 改 cold-start"
    export PHASE0E_CKPT=""
fi

cd /root/autodl-tmp/external/Search-R1
EXPERIMENT_NAME=$PHASE3_RUN \
  nohup bash /root/autodl-tmp/agenicRL/scripts/phase3_train_torl.sh > /tmp/phase3_torl.log 2>&1 &
PHASE3_PID=$!
sleep 10
echo "$(ts) Phase 3 launched, PID=$PHASE3_PID"

# 验证起来了
if ps -p $PHASE3_PID > /dev/null 2>&1; then
    echo "$(ts) ✅ Phase 3 进程存活"
else
    echo "$(ts) ❌ Phase 3 启动失败，看 /tmp/phase3_torl.log"
    exit 1
fi

# ============================================================
# 5) 写最终状态文件
# ============================================================
cat > /root/autodl-tmp/auto_chain_status.txt <<EOF
auto_chain.sh completed at $(date)
Phase 2B GSPO: completed, final_val=$GSPO_FINAL
Phase 3 ToRL: launched as PID $PHASE3_PID, log=/tmp/phase3_torl.log
EOF
echo "$(ts) ===== auto_chain done ====="
