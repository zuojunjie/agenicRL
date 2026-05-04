#!/usr/bin/env bash
# 远端 watchdog：监测 Phase 7.8e 完成 → 自动关机
#
# 启用: nohup bash auto_shutdown_pro6k.sh > /tmp/auto_shutdown.log 2>&1 < /dev/null &
# 取消: pkill -f auto_shutdown_pro6k.sh
#
# 触发关机条件:
#   - main_base 进程消失 (训练结束 / 崩溃)
#   - 连续 3 分钟 (3 次 60s 检查) 都无进程
#   - 给 SkyRL 留时间存最终 ckpt + dump eval
# 然后:
#   - 执行 shutdown -h now (AutoDL 容器即停)

set -u
LOG=/tmp/auto_shutdown.log
exec >> "$LOG" 2>&1

echo "[$(date '+%F %T')] auto_shutdown watchdog started, PID=$$"
echo "[$(date '+%F %T')] will trigger shutdown -h now after 3 consecutive 60s checks of 0 main_base processes"

# 给训练至少 30 min 启动时间，避免装包阶段误判
echo "[$(date '+%F %T')] grace period 30 min for warmup..."
sleep 1800

zero_count=0
while true; do
    n=$(ps aux | grep -E 'main_base' | grep -v grep | wc -l)
    if [ "$n" -eq 0 ]; then
        zero_count=$((zero_count + 1))
        echo "[$(date '+%F %T')] no main_base ($zero_count/3 consecutive)"
        if [ "$zero_count" -ge 3 ]; then
            echo "[$(date '+%F %T')] 🛑 confirmed training ended (3 consecutive checks). Final cleanup + shutdown."
            # 给 ckpt save 留 60s 余地
            sleep 60
            # 关闭 retrieval_server
            pkill -f retrieval_server.py 2>/dev/null
            # 关机
            echo "[$(date '+%F %T')] 🛑 calling shutdown -h now"
            sync
            shutdown -h now
            exit 0
        fi
    else
        if [ "$zero_count" -gt 0 ]; then
            echo "[$(date '+%F %T')] main_base recovered (n=$n), reset counter"
        fi
        zero_count=0
    fi
    sleep 60
done
