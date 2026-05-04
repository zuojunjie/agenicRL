#!/usr/bin/env bash
# 链式 watchdog v2: Phase 4 → 5a → 5b 全自动
#
# 流程:
#   1. 等 Phase 4 → finalize → apply 5a patch → 起 Phase 5a
#   2. 等 Phase 5a → finalize → apply 5b patch (5b 在 5a 上叠加) → 起 Phase 5b
#   3. 等 Phase 5b → finalize → 写状态文件
#   4. 退出，等用户回来推飞书 + 决策切硬件

set -uo pipefail
PID_4=${PID_4:-491831}
PHASE4_RUN=phase4-gigpo-twolayer-50
PHASE5A_RUN=phase5a-arpo-trajcredit-adaptent-52
PHASE5B_RUN=phase5b-arpo-traj-adaptent-turnkl-52

ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

echo "$(ts) ===== auto_chain v2 watchdog: P4 → 5a → 5b ====="

# ============================================================
# 阶段 1: 等 Phase 4 退出
# ============================================================
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

# ============================================================
# 阶段 2: Finalize Phase 4 + 起 Phase 5a
# ============================================================
echo "$(ts) ===== finalize Phase 4 ====="
cd /root/autodl-tmp/external/Search-R1
EXPERIMENT_NAME=$PHASE4_RUN \
  bash /root/autodl-tmp/agenicRL/scripts/finalize_run.sh $PHASE4_RUN \
  "Phase 4 GiGPO alpha=0.7 cold-start 52 steps" 2>&1 | tail -10

P4_LOG=/root/autodl-tmp/runs/$PHASE4_RUN/training.log
P4_FINAL=$(grep -oE "Final validation metrics:.*nq.: [0-9.]+" "$P4_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+")
echo "$(ts) Phase 4 final val/nq = $P4_FINAL"
echo "phase4_final_val=$P4_FINAL" > /root/autodl-tmp/runs/$PHASE4_RUN/final_val.txt

echo "$(ts) ===== apply Phase 5a patch ====="
python3 /root/autodl-tmp/agenicRL/patches/phase5a_apply.py 2>&1 | tail -10

if ! curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "$(ts) ⚠️ retrieval_server 没响应，重启"
    nohup bash /root/autodl-tmp/agenicRL/scripts/launch_retrieval_server.sh > /tmp/retrieval.log 2>&1 &
    sleep 30
fi

echo "$(ts) ===== launch Phase 5a ====="
cd /root/autodl-tmp/external/Search-R1
EXPERIMENT_NAME=$PHASE5A_RUN \
  nohup bash /root/autodl-tmp/agenicRL/scripts/phase5a_train_arpo.sh > /tmp/phase5a_arpo.log 2>&1 &
sleep 15
PID_5A=$(ps -ef | grep "experiment_name=$PHASE5A_RUN" | grep -v grep | head -1 | awk '{print $2}')
if [ -z "$PID_5A" ]; then
    echo "$(ts) ❌ Phase 5a 启动失败"
    tail -30 /tmp/phase5a_arpo.log
    exit 1
fi
echo "$(ts) ✅ Phase 5a launched PID=$PID_5A"

# ============================================================
# 阶段 3: 等 Phase 5a 退出
# ============================================================
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

# ============================================================
# 阶段 4: Finalize Phase 5a + apply 5b patch + 起 Phase 5b
# ============================================================
echo "$(ts) ===== finalize Phase 5a ====="
cd /root/autodl-tmp/external/Search-R1
EXPERIMENT_NAME=$PHASE5A_RUN \
  bash /root/autodl-tmp/agenicRL/scripts/finalize_run.sh $PHASE5A_RUN \
  "Phase 5a ARPO traj_credit + adapt_entropy cold-start 52" 2>&1 | tail -10

P5A_LOG=/root/autodl-tmp/runs/$PHASE5A_RUN/training.log
P5A_FINAL=$(grep -oE "Final validation metrics:.*nq.: [0-9.]+" "$P5A_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+")
echo "$(ts) Phase 5a final val/nq = $P5A_FINAL"
echo "phase5a_final_val=$P5A_FINAL" > /root/autodl-tmp/runs/$PHASE5A_RUN/final_val.txt

echo "$(ts) ===== apply Phase 5b patch (additive on 5a) ====="
python3 /root/autodl-tmp/agenicRL/patches/phase5b_apply.py 2>&1 | tail -10

if ! curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "$(ts) ⚠️ retrieval_server 没响应，重启"
    nohup bash /root/autodl-tmp/agenicRL/scripts/launch_retrieval_server.sh > /tmp/retrieval.log 2>&1 &
    sleep 30
fi

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

# ============================================================
# 阶段 5: 等 Phase 5b 退出
# ============================================================
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

# ============================================================
# 阶段 6: Finalize Phase 5b + 写最终状态
# ============================================================
echo "$(ts) ===== finalize Phase 5b ====="
cd /root/autodl-tmp/external/Search-R1
EXPERIMENT_NAME=$PHASE5B_RUN \
  bash /root/autodl-tmp/agenicRL/scripts/finalize_run.sh $PHASE5B_RUN \
  "Phase 5b ARPO traj+adapt+turn_kl cold-start 52" 2>&1 | tail -10

P5B_LOG=/root/autodl-tmp/runs/$PHASE5B_RUN/training.log
P5B_FINAL=$(grep -oE "Final validation metrics:.*nq.: [0-9.]+" "$P5B_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+")
echo "$(ts) Phase 5b final val/nq = $P5B_FINAL"
echo "phase5b_final_val=$P5B_FINAL" > /root/autodl-tmp/runs/$PHASE5B_RUN/final_val.txt

cat > /root/autodl-tmp/auto_chain_v2_status.txt <<EOF
auto_chain v2 (P4→5a→5b) done at $(date)
Phase 4 GiGPO:  final_val=$P4_FINAL
Phase 5a ARPO:  final_val=$P5A_FINAL
Phase 5b ARPO:  final_val=$P5B_FINAL

下一步（用户手动）：
  - 推 P4/5a/5b summary 飞书
  - 五 phase 总对比
  - 备份 ckpt
  - 切 RTX PRO 6000 实例
EOF

echo "$(ts) ===== ALL DONE ====="
