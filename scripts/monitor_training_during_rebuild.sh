#!/bin/bash
# 在 IVF_PQ 重建期间监控训练健康度
# 每 5 min 抓一次 wandb step time，degrade > 15% 报警
#
# 用法: nohup bash monitor_training_during_rebuild.sh > /tmp/training_monitor.log 2>&1 &

set -uo pipefail
LOG=/tmp/training_monitor.log
TRAIN_PID=437458    # Phase 3 续训
BASELINE_STEP_TIME=1200  # 实测约 9 min/step (550s)，degrade 阈值

source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

while true; do
    TS=$(date +"%H:%M:%S")

    # 1. 训练 PID 还活吗？
    if ! kill -0 $TRAIN_PID 2>/dev/null; then
        echo "$TS ⚠️ 训练 PID $TRAIN_PID 不在了！停止监控" >> $LOG
        break
    fi

    # 2. 系统 CPU/RAM/IO 抓帧
    LOAD=$(awk '{print $1}' /proc/loadavg)
    FREE_RAM=$(free -g | awk '/^Mem:/ {print $7}')
    TRAIN_RSS=$(ps -o rss= -p $TRAIN_PID 2>/dev/null | awk '{print int($1/1024/1024)}')

    # 3. wandb 拉最新 timing_s/step
    STEP_TIME=$(python3 -c "
import wandb
api = wandb.Api()
runs = list(api.runs('agentic-rl-search', filters={'display_name': 'phase3-torl-resume-step20', 'state':'running'}))
if not runs:
    print(-1)
else:
    h = runs[0].history(keys=['timing_s/step'], samples=20)
    if len(h) > 0:
        print(int(h.iloc[-1]['timing_s/step']))
    else:
        print(0)
" 2>/dev/null)

    # 4. 判断 degrade
    if [ "$STEP_TIME" -gt "$((BASELINE_STEP_TIME + BASELINE_STEP_TIME / 6))" ]; then
        STATUS="⚠️ DEGRADE"
    else
        STATUS="✅ OK"
    fi

    echo "$TS load=$LOAD free_ram=${FREE_RAM}G train_rss=${TRAIN_RSS}G step_time=${STEP_TIME}s $STATUS" >> $LOG

    sleep 300  # 5 min
done
