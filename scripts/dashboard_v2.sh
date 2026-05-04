#!/usr/bin/env bash
# agenicRL training dashboard v2 — richer process timeline view.
# Renders:
#   - current state bars (GPU / CPU)
#   - RAM history sparkline (unicode blocks, last 30 samples)
#   - ASCII line-graph of CPU RAM over time
#   - top processes
#   - timeline of training milestones (Started/Finished events with timing)
set +e

LOG=${1:-/tmp/skyrl_smoke23.log}
TRACE=${2:-/tmp/skyrl_smoke23_ram_trace.log}

# ─── current snapshot ─────────────────────────────────────────────────
SPID=$(pgrep -f "skyrl.train.entrypoints" | head -1)
RPID=$(pgrep -f "retrieval_server.py" | head -1)
S_RT=$([ -n "$SPID" ] && ps -o etime= -p "$SPID" | tr -d " " || echo "—")
R_RT=$([ -n "$RPID" ] && ps -o etime= -p "$RPID" | tr -d " " || echo "—")
CPU_TOTAL=$(awk '{s+=$2} END {printf "%.1f", s/1024/1024}' <(ps -eo pid,rss --no-headers))
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)

# ─── helpers ──────────────────────────────────────────────────────────
hbar() { awk -v u=$1 -v t=$2 -v b=$3 'BEGIN{f=int(u*b/t); for(i=0;i<f;i++) printf "#"; for(i=f;i<b;i++) printf "."}'; }

# Sparkline over last 30 RAM samples (unicode blocks)
spark() {
    awk -F, '
    BEGIN{ blocks="▁▂▃▄▅▆▇█"; max=0; n=0 }
    NR>1 { v[++n]=$2; if(v[n]>max) max=v[n] }
    END {
        start = (n>30 ? n-30 : 0)
        for(i=start+1; i<=n; i++) {
            idx = int(v[i] / max * 7); if(idx<0) idx=0; if(idx>7) idx=7
            printf "%s", substr(blocks, idx*3+1, 3)  # 3 bytes per unicode glyph
        }
    }
    ' "$TRACE" 2>/dev/null
}

# ASCII line-graph of last N samples (height H rows)
line_graph() {
    local W=${1:-50}
    local H=${2:-6}
    awk -F, -v w=$W -v h=$H '
    BEGIN { max=0; n=0 }
    NR>1 { v[++n]=$2; if(v[n]>max) max=v[n] }
    END {
        if(n==0 || max==0) { print "(no data)"; exit }
        start = (n>w ? n-w : 0)
        cnt = n - start
        for(row=h-1; row>=0; row--) {
            ymin = max * row / h
            ymax = max * (row+1) / h
            label = (row==h-1 ? sprintf("%4.0f", max) : (row==0 ? "   0" : "    "))
            printf "%s |", label
            for(i=start+1; i<=n; i++) {
                if(v[i] >= ymin) {
                    if(v[i] >= ymax) printf "█"; else printf "▄"
                } else printf " "
            }
            print ""
        }
        # x axis
        printf "     +"; for(i=0; i<cnt; i++) printf "-"; print ""
        printf "      "; for(i=0; i<cnt; i++) printf " "; print ""
    }
    ' "$TRACE" 2>/dev/null
}

# Extract training timeline events
timeline_events() {
    grep -E "Started:|Finished:|avg_final_rewards|Initialize" "$LOG" 2>/dev/null \
        | grep -vE "trainer\.|generator\.|environment\." \
        | sed -E 's/.*\[36m\([^)]*\)\[0m //; s/\x1b\[[0-9;]*m//g; s/^ *//' \
        | grep -E "^[0-9]" | tail -10 \
        | awk '
        {
            # Extract timestamp (first column)
            ts = substr($0, 12, 8)
            # Extract event message after the last "|"
            n = split($0, parts, "|")
            msg = parts[n]
            # Trim
            sub(/^[ \t]+/, "", msg)
            printf "  %s  %s\n", ts, msg
        }'
}

# ─── render ───────────────────────────────────────────────────────────
HR=$(printf '%76s' | tr ' ' '─')
echo "┌${HR}┐"
printf "│ %-74s │\n" "agenicRL · PRO 6000 · v23 · $(date +%H:%M:%S)"
echo "├${HR}┤"
printf "│ retrieval (mmap)  PID %-6s  runtime %-10s                          │\n" "${RPID:-—}" "$R_RT"
printf "│ smoke v23         PID %-6s  runtime %-10s                          │\n" "${SPID:-DEAD}" "$S_RT"
echo "├${HR}┤"

# Current bars
GPU_PCT=$(awk -v u=$GPU_USED 'BEGIN{printf "%.1f", u*100/98304}')
CPU_PCT=$(awk -v c=$CPU_TOTAL 'BEGIN{printf "%.1f", c*100/110}')
gpu_bar=$(hbar $GPU_USED 98304 30)
cpu_bar=$(hbar $(awk -v c=$CPU_TOTAL 'BEGIN{printf "%d", c*1000}') 110000 30)
printf "│ GPU [%s] %5s MiB  util %3s%%  (%5s%%)│\n" "$gpu_bar" "$GPU_USED" "${GPU_UTIL:-?}" "$GPU_PCT"
printf "│ CPU [%s] %6s GB             (%5s%%)│\n" "$cpu_bar" "$CPU_TOTAL" "$CPU_PCT"
echo "├${HR}┤"

# Sparkline (last 30)
SPARK=$(spark)
printf "│ RAM sparkline (last 30 × 10s):                                              │\n"
printf "│   %-72s │\n" "$SPARK"
echo "├${HR}┤"

# ASCII line graph
echo "│ RAM line graph (last 50 samples, GB):                                       │"
line_graph 60 6 | while IFS= read -r line; do
    printf "│ %-74s │\n" "$line"
done
echo "├${HR}┤"

# Top processes
echo "│ Top 5 RAM consumers:                                                        │"
ps -eo pid,rss,comm --sort=-rss --no-headers 2>/dev/null | head -5 | while read pid rss comm; do
    gb=$(awk -v r=$rss 'BEGIN{printf "%5.1f", r/1024/1024}')
    printf "│   %5s  %s GB  %-50s   │\n" "$pid" "$gb" "$comm"
done
echo "├${HR}┤"

# Training timeline
echo "│ Training timeline (last 10 events):                                         │"
EVENTS=$(timeline_events)
if [ -z "$EVENTS" ]; then
    printf "│   %-72s │\n" "(no train events yet — still in init)"
else
    echo "$EVENTS" | head -10 | while IFS= read -r line; do
        printf "│ %-74s │\n" "$(echo "$line" | head -c 74)"
    done
fi

# Counters
POSTS=$(grep -c "POST /retrieve" /root/autodl-tmp/runs/retrieval/server_mmap.log 2>/dev/null || echo 0)
TRACE_LINES=$(($(wc -l < "$TRACE" 2>/dev/null || echo 1) - 1))
echo "├${HR}┤"
printf "│ POSTs to retrieval: %-6s  ·  RAM samples: %-6s                          │\n" "$POSTS" "$TRACE_LINES"
echo "└${HR}┘"
