#!/usr/bin/env bash
# 链式 watchdog: 等 Phase 4 → finalize → apply 5a patch → 起 Phase 5a
# 不接 Phase 5b（要等用户验 5a 结果再决定 5b 启动 + 5b patch 是否要写）

set -uo pipefail
PID_4=${PID_4:-491831}
PHASE4_RUN=phase4-gigpo-twolayer-50
PHASE5A_RUN=phase5a-arpo-trajcredit-adaptent-52

ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

echo "$(ts) ===== auto_chain p4→5a watchdog start ====="

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
  "Phase 4 GiGPO alpha=0.7 cold-start 52 steps" 2>&1 | tail -10

# 3. 抓 P4 final val
LOG_PATH=/root/autodl-tmp/runs/$PHASE4_RUN/training.log
P4_FINAL=$(grep -oE "Final validation metrics:.*nq.: [0-9.]+" "$LOG_PATH" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+")
echo "$(ts) Phase 4 final val/nq = $P4_FINAL"
echo "phase4_final_val=$P4_FINAL" > /root/autodl-tmp/runs/$PHASE4_RUN/final_val.txt

# 4. Apply Phase 5a patch
echo "$(ts) ===== apply Phase 5a patch ====="
python3 /root/autodl-tmp/agenicRL/patches/phase5a_apply.py 2>&1 | tail -10

# 5. 验证 retrieval_server
if ! curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "$(ts) ⚠️ retrieval_server 未响应，重启"
    nohup bash /root/autodl-tmp/agenicRL/scripts/launch_retrieval_server.sh > /tmp/retrieval.log 2>&1 &
    sleep 30
fi

# 6. 启动 Phase 5a
echo "$(ts) ===== launch Phase 5a ARPO ====="
cd /root/autodl-tmp/external/Search-R1
EXPERIMENT_NAME=$PHASE5A_RUN \
  nohup bash /root/autodl-tmp/agenicRL/scripts/phase5a_train_arpo.sh > /tmp/phase5a_arpo.log 2>&1 &
PHASE5A_PID=$!
sleep 15

if ps -p $PHASE5A_PID > /dev/null 2>&1; then
    echo "$(ts) ✅ Phase 5a launched, PID=$PHASE5A_PID"
else
    REAL_PID=$(ps -ef | grep "experiment_name=$PHASE5A_RUN" | grep -v grep | head -1 | awk '{print $2}')
    if [ -n "$REAL_PID" ]; then
        echo "$(ts) ✅ Phase 5a python PID=$REAL_PID"
        PHASE5A_PID=$REAL_PID
    else
        echo "$(ts) ❌ Phase 5a 启动失败"
        tail -30 /tmp/phase5a_arpo.log
        exit 1
    fi
fi

cat > /root/autodl-tmp/auto_chain_p4_to_5a_status.txt <<EOF
auto_chain_p4_to_5a done at $(date)
Phase 4 GiGPO: completed, final_val=$P4_FINAL
Phase 5a ARPO: launched as PID $PHASE5A_PID
等用户拍板 5b 是否跑 + 5b turn-level KL patch 是否写
EOF

echo "$(ts) ===== chain done ====="
