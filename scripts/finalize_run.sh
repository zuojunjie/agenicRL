#!/usr/bin/env bash
# 训练结束后的归档脚本：把 ckpt + log + summary 组织到 /root/autodl-tmp/runs/<run-name>/
#
# 用法（云端）：
#   bash scripts/finalize_run.sh <run_name> "<notes>"
#
# 例如：
#   bash scripts/finalize_run.sh phase0d-baseline-50 "GRPO baseline, Qwen2.5-3B-Instruct, 50 steps"
#
# 做的事：
#   1. 把 verl_checkpoints/<EXPERIMENT_NAME>/actor/global_step_* 移到 runs/<run_name>/ckpts/
#   2. 拷贝 /tmp/training.log 到 runs/<run_name>/training.log
#   3. 跑 generate_summary.py 生成 metrics.csv + summary.md

set -e

RUN_NAME="${1:?usage: finalize_run.sh <run_name> [notes]}"
NOTES="${2:-}"

# 默认 EXPERIMENT_NAME 与 phase0_train_grpo.sh 默认值一致（脚本内可被 env override）
EXPERIMENT_NAME="${EXPERIMENT_NAME:-phase0-nq-grpo-qwen2.5-3b-it-em}"
CKPT_SRC="/root/autodl-tmp/external/Search-R1/verl_checkpoints/${EXPERIMENT_NAME}/actor"
LOG_SRC="/tmp/training.log"
RUNS_DIR="/root/autodl-tmp/runs"
OUT="${RUNS_DIR}/${RUN_NAME}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

mkdir -p "$OUT/ckpts"

echo "===== 检查 ckpts 源目录 ====="
if [ ! -d "$CKPT_SRC" ]; then
    echo "⚠️ $CKPT_SRC 不存在 — 训练可能没产生 ckpt"
else
    echo "$(ls -1 $CKPT_SRC | wc -l) 个 ckpt 在 $CKPT_SRC"
    ls -1 "$CKPT_SRC"
fi

echo
echo "===== 移动 ckpts 到 $OUT/ckpts/ ====="
if [ -d "$CKPT_SRC" ] && [ -n "$(ls -A $CKPT_SRC 2>/dev/null)" ]; then
    # 用 mv 而不是 cp，避免 ~6GB ckpt 占双份磁盘
    mv "$CKPT_SRC"/* "$OUT/ckpts/" 2>/dev/null || true
    echo "✅ moved"
    du -sh "$OUT/ckpts/"
else
    echo "(skipped, no ckpts)"
fi

echo
echo "===== 剪枝中间 ckpts，仅保留最大 step（节省磁盘） ====="
# 策略：训练成功完成后，只有最后一个 ckpt 有用（作为下一 phase warm-start 起点）
# 中间 ckpt 可以删除，每个 ~13G。set KEEP_ALL_CKPTS=1 跳过剪枝
if [ "${KEEP_ALL_CKPTS:-0}" != "1" ] && [ -d "$OUT/ckpts" ]; then
    # 找最大 step 的 ckpt 目录
    LAST_CKPT=$(ls -d "$OUT/ckpts/global_step_"* 2>/dev/null | sort -t_ -k3 -n | tail -1)
    if [ -n "$LAST_CKPT" ]; then
        echo "保留: $(basename $LAST_CKPT)"
        for d in "$OUT/ckpts/global_step_"*; do
            if [ "$d" != "$LAST_CKPT" ]; then
                echo "  删除中间 ckpt: $(basename $d) ($(du -sh $d | cut -f1))"
                rm -rf "$d"
            fi
        done
        echo "剪枝后 ckpts 总占用: $(du -sh $OUT/ckpts | cut -f1)"
    fi
fi

echo
echo "===== 拷训练 log ====="
if [ -f "$LOG_SRC" ]; then
    cp "$LOG_SRC" "$OUT/training.log"
    echo "✅ log: $(wc -l < $OUT/training.log) lines, $(du -h $OUT/training.log | cut -f1)"
else
    echo "❌ $LOG_SRC 不存在"
fi

echo
echo "===== 生成 metrics.csv + summary.md ====="
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
python "$SCRIPT_DIR/generate_summary.py" \
    --log "$OUT/training.log" \
    --output-dir "$OUT" \
    --run-name "$RUN_NAME" \
    --notes "$NOTES"

echo
echo "===== 归档完成 ====="
ls -la "$OUT"
echo
echo "summary.md 前几行预览："
head -20 "$OUT/summary.md"
echo
echo "下一步: rsync $OUT/summary.md 回 Mac，用 lark-cli 推飞书"
