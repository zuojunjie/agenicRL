#!/usr/bin/env bash
# 云端 watchdog：监控 Phase 4 GiGPO → 自动 finalize → 写状态文件
# 不接 Phase 5（要等用户拍板）
#
# 用法（用 setsid 真脱离 SSH）:
#   setsid nohup bash auto_chain_phase4_finalize.sh > /root/autodl-tmp/auto_chain_p4.log 2>&1 < /dev/null &

set -uo pipefail
PID_4=${PID_4:-491831}
PHASE4_RUN=phase4-gigpo-twolayer-50

ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

echo "$(ts) ===== auto_chain Phase 4 watchdog start ====="
echo "$(ts) watching PID $PID_4"

# 1. 等 Phase 4 退出
while kill -0 $PID_4 2>/dev/null; do
    LOG=/root/autodl-tmp/external/Search-R1/$PHASE4_RUN.log
    [ -f "$LOG" ] || LOG=/tmp/phase4_gigpo.log
    STEP=$(grep -E "epoch 0, step" "$LOG" 2>/dev/null | tail -1 | grep -oE "step [0-9]+")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    echo "$(ts) Phase 4 alive — $STEP, gpu_peak=${GPU_MEM} MiB"
    sleep 300
done
echo "$(ts) Phase 4 exited"
sleep 15

# 2. Finalize Phase 4
echo "$(ts) ===== finalize Phase 4 ====="
cd /root/autodl-tmp/external/Search-R1
EXPERIMENT_NAME=$PHASE4_RUN \
  bash /root/autodl-tmp/agenicRL/scripts/finalize_run.sh $PHASE4_RUN \
  "Phase 4 GiGPO (alpha=0.7) cold-start MAX_STEPS=52" 2>&1 | tail -20

# 3. 抓 Phase 4 final val/nq
LOG_PATH=/root/autodl-tmp/runs/$PHASE4_RUN/training.log
P4_FINAL=$(grep -oE "Final validation metrics:.*nq.: [0-9.]+" "$LOG_PATH" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+")
[ -z "$P4_FINAL" ] && P4_FINAL=$(grep -oE "Final validation metrics:.*nq.: [0-9.]+" /tmp/phase4_gigpo.log 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+")
echo "$(ts) Phase 4 final val/nq = $P4_FINAL"
echo "phase4_final_val=$P4_FINAL" > /root/autodl-tmp/runs/$PHASE4_RUN/final_val.txt

# 4. 写最终状态
cat > /root/autodl-tmp/auto_chain_p4_status.txt <<EOF
auto_chain Phase 4 done at $(date)
Phase 4 GiGPO: completed
Final val/nq: $P4_FINAL
Run dir: /root/autodl-tmp/runs/$PHASE4_RUN/
等用户拍板：
  - 推 Phase 4 summary 飞书（Mac 端跑 push_summary_to_feishu.sh）
  - 决定 Phase 5 ARPO 是否跑（200 步 ¥276，需用户确认预算）
EOF

echo "$(ts) ===== auto_chain Phase 4 done ====="
