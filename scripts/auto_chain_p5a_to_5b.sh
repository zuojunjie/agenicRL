#!/usr/bin/env bash
# 链式 watchdog v3: Phase 5a → 5b（接续，Phase 4 已 finalize）
# 监控当前 Phase 5a PID，跑完后 apply 5b patch + 起 5b

set -uo pipefail
PID_5A=${PID_5A:-542940}
PHASE5A_RUN=phase5a-arpo-trajcredit-adaptent-52
PHASE5B_RUN=phase5b-arpo-traj-adaptent-turnkl-52

ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

echo "$(ts) ===== auto_chain v3 watchdog: 5a → 5b ====="

# 1. 等 Phase 5a
while kill -0 $PID_5A 2>/dev/null; do
    LOG=/root/autodl-tmp/external/Search-R1/$PHASE5A_RUN.log
    [ -f "$LOG" ] || LOG=/tmp/phase5a_arpo.log
    STEP=$(grep -E "epoch 0, step" "$LOG" 2>/dev/null | tail -1 | grep -oE "step [0-9]+")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    echo "$(ts) Phase 5a alive — $STEP, gpu_peak=${GPU_MEM} MiB"
    sleep 300
done
echo "$(ts) Phase 5a exited"
sleep 15

# 2. Finalize 5a
echo "$(ts) ===== finalize Phase 5a ====="
cd /root/autodl-tmp/external/Search-R1
EXPERIMENT_NAME=$PHASE5A_RUN \
  bash /root/autodl-tmp/agenicRL/scripts/finalize_run.sh $PHASE5A_RUN \
  "Phase 5a ARPO traj_credit + adapt_entropy cold-start 52" 2>&1 | tail -10

P5A_LOG=/root/autodl-tmp/runs/$PHASE5A_RUN/training.log
P5A_FINAL=$(grep -oE "Final validation metrics:.*nq.: [0-9.]+" "$P5A_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+")
echo "$(ts) Phase 5a final val/nq = $P5A_FINAL"
echo "phase5a_final_val=$P5A_FINAL" > /root/autodl-tmp/runs/$PHASE5A_RUN/final_val.txt

# 3. Apply 5b patch
echo "$(ts) ===== apply Phase 5b patch ====="
python3 /root/autodl-tmp/agenicRL/patches/phase5b_apply.py 2>&1 | tail -10

# 4. Verify retrieval
if ! curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "$(ts) ⚠️ retrieval_server 没响应，重启"
    nohup bash /root/autodl-tmp/agenicRL/scripts/launch_retrieval_server.sh > /tmp/retrieval.log 2>&1 &
    sleep 30
fi

# 5. Launch Phase 5b
echo "$(ts) ===== launch Phase 5b ====="
cd /root/autodl-tmp/external/Search-R1
EXPERIMENT_NAME=$PHASE5B_RUN \
  nohup bash /root/autodl-tmp/agenicRL/scripts/phase5b_train_arpo.sh > /tmp/phase5b_arpo.log 2>&1 &
sleep 15
PID_5B=$(ps -ef | grep "experiment_name=$PHASE5B_RUN" | grep -v grep | head -1 | awk '{print $2}')
if [ -z "$PID_5B" ]; then
    echo "$(ts) ❌ Phase 5b 启动失败"
    tail -30 /tmp/phase5b_arpo.log
    exit 1
fi
echo "$(ts) ✅ Phase 5b launched PID=$PID_5B"

# 6. Wait Phase 5b
while kill -0 $PID_5B 2>/dev/null; do
    LOG=/root/autodl-tmp/external/Search-R1/$PHASE5B_RUN.log
    [ -f "$LOG" ] || LOG=/tmp/phase5b_arpo.log
    STEP=$(grep -E "epoch 0, step" "$LOG" 2>/dev/null | tail -1 | grep -oE "step [0-9]+")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    echo "$(ts) Phase 5b alive — $STEP, gpu_peak=${GPU_MEM} MiB"
    sleep 300
done
echo "$(ts) Phase 5b exited"
sleep 15

# 7. Finalize 5b
echo "$(ts) ===== finalize Phase 5b ====="
cd /root/autodl-tmp/external/Search-R1
EXPERIMENT_NAME=$PHASE5B_RUN \
  bash /root/autodl-tmp/agenicRL/scripts/finalize_run.sh $PHASE5B_RUN \
  "Phase 5b ARPO traj+adapt+turn_kl cold-start 52" 2>&1 | tail -10

P5B_LOG=/root/autodl-tmp/runs/$PHASE5B_RUN/training.log
P5B_FINAL=$(grep -oE "Final validation metrics:.*nq.: [0-9.]+" "$P5B_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+")
echo "$(ts) Phase 5b final val/nq = $P5B_FINAL"
echo "phase5b_final_val=$P5B_FINAL" > /root/autodl-tmp/runs/$PHASE5B_RUN/final_val.txt

# 8. 写最终状态
cat > /root/autodl-tmp/auto_chain_v3_status.txt <<EOF
auto_chain v3 (5a→5b) done at $(date)
Phase 5a final_val: $P5A_FINAL
Phase 5b final_val: $P5B_FINAL
EOF

echo "$(ts) ===== ALL DONE ====="
