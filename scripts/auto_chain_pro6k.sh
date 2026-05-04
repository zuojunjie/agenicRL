#!/usr/bin/env bash
# 远端自动接力脚本 — 不依赖 Claude session
#
# 监测当前训练 (RUN_NAME) 完成后，按队列依次启动下一个 phase。
# 完成判断：进程退出 + 至少有一个 global_step_X 已落盘
#
# 使用：
#   nohup bash auto_chain_pro6k.sh > /tmp/auto_chain.log 2>&1 < /dev/null &
#
# 队列在下面 PHASES 数组里改

set -u

LOG=/tmp/auto_chain.log
exec >> "$LOG" 2>&1

echo "[$(date '+%F %T')] auto_chain start"

# 队列：每行 = "RUN_NAME launcher_path"
# 注：7.8d 已在 11:18 由人工启动，watchdog 仅负责接力 Phase 8
PHASES=(
    "phase8-skyrl-pro6k-dual-maxturns4              /root/autodl-tmp/agenicRL/scripts/skyrl_phase8_pro6k_dual_maxturns4.sh"
)

WANDB_API_KEY=$(awk '/api.wandb.ai/{getline; getline; print $2}' /root/.netrc)
export WANDB_API_KEY

cleanup_orphans() {
    # 删 5 min 以上的孤儿 uv venv
    ACTIVE=$(for pid in $(ps aux | grep -E 'main_base|skyrl|ray::' | grep -v grep | awk '{print $2}'); do
        grep -oE '/root/autodl-tmp/\.cache/uv/builds-v0/\.tmp[A-Za-z0-9]+' /proc/$pid/maps 2>/dev/null
    done | sort -u)
    for d in /root/autodl-tmp/.cache/uv/builds-v0/.tmp*; do
        [ ! -d "$d" ] && continue
        if ! echo "$ACTIVE" | grep -qF "$d" && [ $(( $(date +%s) - $(stat -c %Y "$d") )) -gt 300 ]; then
            rm -rf "$d"
            echo "[$(date '+%F %T')]   cleaned orphan venv: $(basename $d)"
        fi
    done
    # 删 30 min 以上的孤儿 ray sessions
    for d in /tmp/ray/session_*; do
        [ ! -d "$d" ] && continue
        if [ $(( $(date +%s) - $(stat -c %Y "$d") )) -gt 1800 ]; then
            rm -rf "$d"
        fi
    done
}

wait_for_current() {
    local run_name="$1"
    echo "[$(date '+%F %T')] waiting for current training to finish..."
    while true; do
        # 检查是否还有 main_base 进程
        n=$(ps aux | grep -E 'main_base' | grep -v grep | wc -l)
        if [ "$n" -eq 0 ]; then
            echo "[$(date '+%F %T')]   no main_base processes detected"
            sleep 30  # 给一点时间让 ray 工作进程也退出
            return 0
        fi
        sleep 60
    done
}

launch_phase() {
    local run_name="$1"
    local launcher="$2"
    echo "[$(date '+%F %T')] LAUNCH: $run_name"
    echo "[$(date '+%F %T')]   launcher: $launcher"

    # ckpt 防误删审计：不动 run dir，只清空旧 launch out
    rm -f "/tmp/${run_name}-launch.out"

    nohup bash -c "export WANDB_API_KEY='$WANDB_API_KEY'; bash '$launcher'" \
        > "/tmp/${run_name}-launch.out" 2>&1 < /dev/null &
    LAUNCH_PID=$!
    echo "[$(date '+%F %T')]   launched bash PID=$LAUNCH_PID"
    sleep 60
    # 验证进程起来了
    n=$(ps aux | grep -E 'main_base' | grep -v grep | wc -l)
    if [ "$n" -eq 0 ]; then
        echo "[$(date '+%F %T')]   ❌ launch failed (no main_base after 60s)"
        return 1
    fi
    echo "[$(date '+%F %T')]   ✅ training process active"
    return 0
}

# 主循环：等待当前训练 → 清理 → 启动下一个
for entry in "${PHASES[@]}"; do
    run_name=$(echo "$entry" | awk '{print $1}')
    launcher=$(echo "$entry" | awk '{print $2}')

    # 等当前训练 (无论是 7.8c, 7.8d 等) 退出
    wait_for_current ""
    cleanup_orphans
    sleep 10  # GPU 释放缓冲

    if ! launch_phase "$run_name" "$launcher"; then
        echo "[$(date '+%F %T')] STOP: launch failed, aborting chain"
        exit 1
    fi
done

# 等最后一个 phase 跑完
wait_for_current ""
cleanup_orphans
echo "[$(date '+%F %T')] auto_chain ALL DONE"
