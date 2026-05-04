#!/usr/bin/env bash
# Compact agenicRL training dashboard — render in <1s on remote.
set +e

SPID=$(pgrep -f "skyrl.train.entrypoints" | head -1)
RPID=$(pgrep -f "retrieval_server.py" | head -1)
S_RT=$([ -n "$SPID" ] && ps -o etime= -p "$SPID" | tr -d " " || echo "—")
R_RT=$([ -n "$RPID" ] && ps -o etime= -p "$RPID" | tr -d " " || echo "—")
CPU_TOTAL=$(awk '{s+=$2} END {printf "%.1f", s/1024/1024}' <(ps -eo pid,rss --no-headers))
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
GPU_TOTAL=98304

# bars (awk arithmetic, no bc)
BAR=24
gpu_fill=$(awk -v u=$GPU_USED -v t=$GPU_TOTAL -v b=$BAR 'BEGIN{printf "%d", u*b/t}')
cpu_fill=$(awk -v c=$CPU_TOTAL -v b=$BAR 'BEGIN{printf "%d", c*b/110}')
gpu_pad=$((BAR-gpu_fill))
cpu_pad=$((BAR-cpu_fill))
gpu_bar=$(printf "%${gpu_fill}s" | tr ' ' '#')$(printf "%${gpu_pad}s" | tr ' ' '.')
cpu_bar=$(printf "%${cpu_fill}s" | tr ' ' '#')$(printf "%${cpu_pad}s" | tr ' ' '.')

# percent (gpu)
gpu_pct=$(awk -v u=$GPU_USED -v t=$GPU_TOTAL 'BEGIN{printf "%4.1f", u*100/t}')
cpu_pct=$(awk -v c=$CPU_TOTAL 'BEGIN{printf "%4.1f", c*100/110}')

# latest event
LAST=$(grep -E "Started:|Finished:|avg_final_rewards|sync_weights|policy_train|fwd_logprobs|FSDP|Initialized" /tmp/skyrl_smoke23.log 2>/dev/null \
        | grep -vE "^\+|trainer\.|generator\.|environment\.|skyrl_entrypoint pid.*[a-z_]+:" \
        | tail -1 \
        | sed -E 's/.*INFO[[:space:]]*\| //; s/\x1b\[[0-9;]*m//g' \
        | head -c 110)
[ -z "$LAST" ] && LAST="(no train events yet)"

# RAM trace last 10
TRACE=/tmp/skyrl_smoke23_ram_trace.log
TRACE_LINES=$(wc -l < "$TRACE" 2>/dev/null || echo 0)
SPARK=$(tail -10 "$TRACE" 2>/dev/null | awk -F, 'NR>1 {printf "%.0f ", $2}')

# retrieval POSTs
POSTS=$(grep -c "POST /retrieve" /root/autodl-tmp/runs/retrieval/server_mmap.log 2>/dev/null || echo 0)

# top 3 RAM
TOP3=$(ps -eo pid,rss,comm --sort=-rss --no-headers 2>/dev/null | head -3 \
       | awk '{printf "    %5s  %5.1f GB  %s\n", $1, $2/1024/1024, $3}')

printf "+%s+\n" "$(printf '%72s' | tr ' ' '-')"
printf "| %-70s |\n" "agenicRL  ·  PRO 6000  ·  v23  dashboard  ·  $(date +%H:%M:%S)"
printf "+%s+\n" "$(printf '%72s' | tr ' ' '-')"
printf "| retrieval (mmap)  PID %-5s  runtime %-7s                            |\n" "${RPID:-N/A}" "${R_RT:-—}"
printf "| smoke v23         PID %-5s  runtime %-7s                            |\n" "${SPID:-DEAD}" "${S_RT:-—}"
printf "|                                                                        |\n"
printf "| GPU [%s]  %5s MiB / 96 GB   util %3s%%   (%s%%)|\n" "$gpu_bar" "$GPU_USED" "${GPU_UTIL:-?}" "$gpu_pct"
printf "| CPU [%s]  %6s GB / 110 GB                  (%s%%)|\n" "$cpu_bar" "$CPU_TOTAL" "$cpu_pct"
printf "|                                                                        |\n"
printf "| retrieval POSTs (累计 rollouts):  %-6s                                |\n" "$POSTS"
printf "| RAM trace samples:                %-6s                                |\n" "$TRACE_LINES"
printf "| RAM history (last 10 × 10s):                                           |\n"
printf "|   %-69s |\n" "$SPARK"
printf "|                                                                        |\n"
printf "| Top 3 RAM consumers:                                                   |\n"
echo "$TOP3" | while read line; do printf "|%s|\n" "$(printf '%-72s' "$line")"; done
printf "|                                                                        |\n"
printf "| Last train event:                                                      |\n"
printf "|   %-69s |\n" "$LAST"
printf "+%s+\n" "$(printf '%72s' | tr ' ' '-')"
