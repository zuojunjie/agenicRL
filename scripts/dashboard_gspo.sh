#!/usr/bin/env bash
# GSPO 52-step run dashboard with progress bar, reward curve, ETA.
set +e

LOG=${1:-/tmp/gspo_flat.log}
TRACE=${2:-/tmp/gspo_flat_ram_trace.log}
TOTAL_STEPS=52

SPID=$(pgrep -f "skyrl.train.entrypoints" | head -1)
RPID=$(pgrep -f "retrieval_server.py" | head -1)
S_RT=$([ -n "$SPID" ] && ps -o etime= -p "$SPID" | tr -d " " || echo "—")
R_RT=$([ -n "$RPID" ] && ps -o etime= -p "$RPID" | tr -d " " || echo "—")
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
CG=$(awk '{printf "%.1f", $1/1024/1024/1024}' /sys/fs/cgroup/memory.current 2>/dev/null)

# Step count + last reward + last timing
# Count completed steps via "Finished: 'policy_train'" events (works in both wandb + console mode)
STEPS=$(grep -c "Finished: 'policy_train'" "$LOG" 2>/dev/null)
# extract reward (after "avg_final_rewards: ") not from timestamp ms
LAST_REWARD=$(grep "avg_final_rewards" "$LOG" 2>/dev/null | tail -1 | grep -oE "avg_final_rewards: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+")
LAST_TIMING=$(grep "Finished: 'policy_train'" "$LOG" 2>/dev/null | tail -1 | grep -oE "time cost: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+")
LAST_REWARD=${LAST_REWARD:-—}
LAST_TIMING=${LAST_TIMING:-—}
[ -z "$STEPS" ] && STEPS=0

# Progress bar (40 wide)
PBW=40
pfill=$(awk -v s=$STEPS -v t=$TOTAL_STEPS -v w=$PBW 'BEGIN{printf "%d", s*w/t}')
ppad=$((PBW - pfill))
pbar=$(printf "%${pfill}s" | tr ' ' '#')$(printf "%${ppad}s" | tr ' ' '.')
ppct=$(awk -v s=$STEPS -v t=$TOTAL_STEPS 'BEGIN{printf "%4.1f", s*100/t}')

# Avg step time + ETA — use "Finished: 'step'" wallclock (full step) if available, else 1.5x policy_train
AVG_TIME=$(grep "Finished: 'policy_train'" "$LOG" 2>/dev/null | tail -20 \
           | grep -oE "time cost: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" \
           | awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n; else print 0}')

# Reward sparkline (unicode blocks)
spark() {
    grep "avg_final_rewards" "$LOG" 2>/dev/null | tail -52 \
        | grep -oE "avg_final_rewards: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | tail -52 \
        | awk '
        BEGIN{ blocks="▁▂▃▄▅▆▇█" }
        { v[++n]=$1; if(v[n]>max) max=v[n] }
        END {
            if(n==0 || max==0) { print "(no data yet)"; exit }
            for(i=1; i<=n; i++) {
                idx = int(v[i] / max * 7)
                if(idx<0) idx=0; if(idx>7) idx=7
                printf "%s", substr(blocks, idx*3+1, 3)
            }
        }'
}

# RAM sparkline
ram_spark() {
    awk -F, 'NR>1 {v[++n]=$2; if(v[n]>max) max=v[n]}
    END {
        blocks="▁▂▃▄▅▆▇█"
        if(n==0 || max==0) { print "(no data)"; exit }
        start = (n>40 ? n-40 : 0)
        for(i=start+1; i<=n; i++) {
            idx = int(v[i] / max * 7); if(idx<0) idx=0; if(idx>7) idx=7
            printf "%s", substr(blocks, idx*3+1, 3)
        }
    }' "$TRACE" 2>/dev/null
}

# ETA
if [ "$STEPS" -gt 0 ] && [ "$AVG_TIME" != "0" ]; then
    REMAINING_STEPS=$((TOTAL_STEPS - STEPS))
    # rough: each step = train_time + ~10x for rollout
    EST_PER_STEP=$(awk -v t=$AVG_TIME 'BEGIN{printf "%d", t * 10}')
    ETA_SEC=$((REMAINING_STEPS * EST_PER_STEP))
    ETA_H=$((ETA_SEC / 3600))
    ETA_M=$(((ETA_SEC % 3600) / 60))
    ETA_STR="${ETA_H}h${ETA_M}m"
else
    ETA_STR="—"
fi

# Latest event
LAST_EVENT=$(grep -E "Started:|Finished:|loss/avg" "$LOG" 2>/dev/null \
             | grep -vE "trainer\.|generator\.|environment\." | tail -1 \
             | sed -E 's/.*\[36m\([^)]*\)\[0m //; s/\x1b\[[0-9;]*m//g' \
             | head -c 75)
[ -z "$LAST_EVENT" ] && LAST_EVENT="(initializing)"

# === Render ===
HR=$(printf '%76s' | tr ' ' '=')
echo "+${HR}+"
printf "| %-74s |\n" "GSPO 52-step run · PRO 6000 · $(date +%H:%M:%S) · target val/nq=0.4378"
echo "+${HR}+"
printf "| smoke PID %-6s rt %-8s | retrieval PID %-6s rt %-8s         |\n" "${SPID:-DEAD}" "$S_RT" "${RPID:-—}" "$R_RT"
echo "+${HR}+"
printf "| Step  %3s / %d  [%s]  %s%%                  |\n" "$STEPS" "$TOTAL_STEPS" "$pbar" "$ppct"
printf "| Last reward:  %-8s    Last train timing:  %-6s s                |\n" "$LAST_REWARD" "$LAST_TIMING"
printf "| Avg train time (last 20):  %-6s s    ETA:  %-10s              |\n" "$AVG_TIME" "$ETA_STR"
echo "+${HR}+"

# Reward sparkline
echo "| Reward curve (last 52 steps):                                              |"
SPARK_R=$(spark)
printf "|   %-72s |\n" "$SPARK_R"

# RAM sparkline
echo "| RAM curve (last 40 × 30s):                                                 |"
SPARK_M=$(ram_spark)
printf "|   %-72s |\n" "$SPARK_M"

# Bars
GPU_PCT=$(awk -v u=$GPU_USED 'BEGIN{printf "%.1f", u*100/98304}')
CPU_PCT=$(awk -v c=$CG 'BEGIN{printf "%.1f", c*100/110}')
gpu_fill=$(awk -v u=$GPU_USED -v b=30 'BEGIN{printf "%d", u*b/98304}')
cpu_fill=$(awk -v c=$CG -v b=30 'BEGIN{printf "%d", c*b/110}')
gpu_bar=$(printf "%${gpu_fill}s" | tr ' ' '#')$(printf "%$((30-gpu_fill))s" | tr ' ' '.')
cpu_bar=$(printf "%${cpu_fill}s" | tr ' ' '#')$(printf "%$((30-cpu_fill))s" | tr ' ' '.')

echo "+${HR}+"
printf "| GPU [%s] %5s MiB util %3s%% (%s%%)|\n" "$gpu_bar" "$GPU_USED" "${GPU_UTIL:-?}" "$GPU_PCT"
printf "| CPU [%s] %6s GB             (%s%%)|\n" "$cpu_bar" "$CG" "$CPU_PCT"
echo "+${HR}+"
printf "| Last event:                                                                |\n"
printf "|   %-72s |\n" "$LAST_EVENT"
echo "+${HR}+"
