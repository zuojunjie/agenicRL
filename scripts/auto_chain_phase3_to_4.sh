#!/usr/bin/env bash
# 云端自治 watchdog：等 Phase 3 ToRL 结束 → finalize → apply GiGPO patch → 起 Phase 4
#
# 用法:
#   nohup bash /root/autodl-tmp/agenicRL/scripts/auto_chain_phase3_to_4.sh > /root/autodl-tmp/auto_chain_3to4.log 2>&1 &

set -uo pipefail

LOG_TAG=auto_chain_3to4
PID_3=${PID_3:-371115}
PHASE3_RUN=phase3-torl-multitool-50
PHASE4_RUN=phase4-gigpo-twolayer-50

ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

echo "$(ts) ===== $LOG_TAG start ====="
echo "$(ts) watching PID $PID_3 (Phase 3 ToRL)"

# 1) 等 Phase 3 退出
while kill -0 $PID_3 2>/dev/null; do
    STEP=$(grep -E "epoch 0, step" /tmp/phase3_torl.log 2>/dev/null | tail -1 | grep -oE "step [0-9]+")
    echo "$(ts) Phase 3 alive — $STEP"
    sleep 300
done
echo "$(ts) Phase 3 exited"
sleep 10

# 2) Phase 3 finalize
echo "$(ts) ===== finalize Phase 3 ====="
cd /root/autodl-tmp/external/Search-R1
EXPERIMENT_NAME=$PHASE3_RUN \
  bash /root/autodl-tmp/agenicRL/scripts/finalize_run.sh $PHASE3_RUN \
  "Phase 3 ToRL multi-turn shaping cold-start MAX_STEPS=52 max_turns=4" 2>&1 | tail -20

# 3) 抓 Phase 3 final val/nq
P3_FINAL=$(grep -oE "Final validation metrics:.*nq.: [0-9.]+" /tmp/phase3_torl.log | tail -1 | grep -oE "[0-9]+\.[0-9]+")
echo "$(ts) Phase 3 final val/nq = $P3_FINAL"
echo "phase3_final_val=$P3_FINAL" > /root/autodl-tmp/runs/$PHASE3_RUN/final_val.txt

# 4) Apply GiGPO patch + ray_trainer 修改
echo "$(ts) ===== apply GiGPO patch ====="
cd /root/autodl-tmp/external/Search-R1

# 检查是否已 apply
if grep -q "GIGPO_ALPHA" verl/trainer/ppo/ray_trainer.py 2>/dev/null; then
    echo "$(ts) ⚠️ GiGPO patch 已应用过，跳过"
else
    bash /root/autodl-tmp/agenicRL/patches/phase4_gigpo_deploy.sh 2>&1 | tail -10
fi

# 5) 起 Phase 4 GiGPO cold-start
echo "$(ts) ===== launch Phase 4 GiGPO ====="
# 检查 retrieval_server
if ! curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "$(ts) ⚠️ retrieval_server 没响应，重启"
    nohup bash /root/autodl-tmp/agenicRL/scripts/launch_retrieval_server.sh > /tmp/retrieval.log 2>&1 &
    sleep 30
fi

cd /root/autodl-tmp/external/Search-R1
GIGPO_ALPHA=0.7 \
EXPERIMENT_NAME=$PHASE4_RUN \
  nohup bash /root/autodl-tmp/agenicRL/scripts/phase4_train_gigpo.sh > /tmp/phase4_gigpo.log 2>&1 &
PHASE4_PID=$!
sleep 15

if ps -p $PHASE4_PID > /dev/null 2>&1; then
    echo "$(ts) ✅ Phase 4 进程存活, PID=$PHASE4_PID"
else
    # 实际 PID 可能是 python3 子进程
    REAL_PID=$(ps -ef | grep "experiment_name=phase4-gigpo" | grep -v grep | head -1 | awk '{print $2}')
    if [ -n "$REAL_PID" ]; then
        echo "$(ts) ✅ Phase 4 python PID=$REAL_PID"
        PHASE4_PID=$REAL_PID
    else
        echo "$(ts) ❌ Phase 4 启动失败"
        tail -30 /tmp/phase4_gigpo.log
        exit 1
    fi
fi

cat > /root/autodl-tmp/auto_chain_status.txt <<EOF
auto_chain_3to4 done at $(date)
Phase 3 ToRL: completed, final_val=$P3_FINAL
Phase 4 GiGPO: launched as PID $PHASE4_PID
EOF
echo "$(ts) ===== $LOG_TAG done ====="
