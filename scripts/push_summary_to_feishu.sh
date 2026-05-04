#!/usr/bin/env bash
# 把指定 run 的 summary.md 推到飞书云文档（Mac 端运行）
#
# 用法：
#   bash scripts/push_summary_to_feishu.sh phase0d-baseline-50
#
# 做的事：
#   1. rsync 云端 /root/autodl-tmp/runs/<run_name>/summary.md 到本地 notes/runs/<run_name>/
#   2. 用 lark-cli docs +create 推到飞书"我的空间"
#   3. 输出文档 URL

set -e

RUN_NAME="${1:?usage: push_summary_to_feishu.sh <run_name>}"

REPO=/Users/cv/code/agenicRL
LOCAL_DIR="$REPO/notes/runs/$RUN_NAME"
mkdir -p "$LOCAL_DIR"

echo "===== rsync 云端 summary.md ====="
rsync -av "autodl-agenicrl:/root/autodl-tmp/runs/$RUN_NAME/summary.md" \
          "autodl-agenicrl:/root/autodl-tmp/runs/$RUN_NAME/metrics.csv" \
          "$LOCAL_DIR/"

echo
echo "===== 推飞书 ====="
SUMMARY_FILE="$LOCAL_DIR/summary.md"
test -f "$SUMMARY_FILE" || { echo "❌ $SUMMARY_FILE 不存在"; exit 1; }

# 走 stdin 管道（per memory note，绝对路径模式不可用）
RESPONSE=$(cat "$SUMMARY_FILE" | lark-cli docs +create \
    --api-version v2 \
    --content - \
    --doc-format markdown 2>&1)

echo "$RESPONSE" | tail -30
echo
URL=$(echo "$RESPONSE" | grep -oE "https://[^ \"]+\.feishu\.[^ \"]+" | head -1)
if [ -n "$URL" ]; then
    echo "✅ 飞书文档地址: $URL"
else
    echo "⚠️ 没解析到 URL，看上面 RESPONSE"
fi
