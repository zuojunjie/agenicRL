#!/usr/bin/env bash
# sync_with_server.sh — 服务器/本地 双向 rsync（git gateway 模式）
#
# 模型：服务器是真源，本地是 git 中转。
# 关键约定：服务器上所有 code 必须放在 /root/autodl-tmp/agenicRL/ 下，
#            才能被本脚本同步。临时丢到 /tmp、/root/foo/ 等位置一律会丢失。
#
# 用法:
#   bash scripts/sync_with_server.sh pull <host> <port>          # server → local（典型，git 上传前）
#   bash scripts/sync_with_server.sh push <host> <port>          # local → server (bootstrap / hotfix)
#   bash scripts/sync_with_server.sh pull <host> <port> --mirror # 严格镜像，删 local 多余文件
#
# 典型 git 上传流程：
#   bash scripts/sync_with_server.sh pull connect.westd.seetacloud.com 12233
#   git status && git diff
#   git add -A && git commit -m "..." && git push
set -euo pipefail

if [ $# -lt 3 ]; then
    cat <<EOF
Usage: $0 <pull|push> <ssh_host> <ssh_port> [--mirror]
  pull   server → local (默认安全，不删 local)
  push   local → server (bootstrap 或推紧急修改)
  --mirror  严格镜像 dst 端 (删多余文件)，默认关

Examples:
  $0 pull connect.westd.seetacloud.com 12233
  $0 push connect.westd.seetacloud.com 12233 --mirror
EOF
    exit 1
fi

DIRECTION=$1
HOST=$2
PORT=$3
MIRROR_FLAG=""
if [ "${4:-}" = "--mirror" ]; then
    MIRROR_FLAG="--delete"
fi
LOCAL_ROOT=$(cd "$(dirname "$0")/.." && pwd)
REMOTE_ROOT=/root/autodl-tmp/agenicRL

case $DIRECTION in
    pull) SRC="root@$HOST:$REMOTE_ROOT/"; DST="$LOCAL_ROOT/" ;;
    push) SRC="$LOCAL_ROOT/"; DST="root@$HOST:$REMOTE_ROOT/" ;;
    *) echo "❌ direction must be 'pull' or 'push'"; exit 1 ;;
esac

echo "=== rsync $DIRECTION: $SRC → $DST $MIRROR_FLAG ==="

rsync -av $MIRROR_FLAG \
    --exclude='.git/' \
    --exclude='runs/' \
    --exclude='ckpts/' \
    --exclude='ckpts_*/' \
    --exclude='wandb/' \
    --exclude='*.log' \
    --exclude='*.pyc' \
    --exclude='__pycache__/' \
    --exclude='.cache/' \
    --exclude='.skyrl-venv/' \
    --exclude='venvs/' \
    --exclude='data/' \
    --exclude='models/' \
    --exclude='external/' \
    --exclude='*.parquet' \
    --exclude='*.safetensors' \
    --exclude='*.bin' \
    --exclude='*.pth' \
    --exclude='*.pt' \
    --exclude='*.index' \
    --exclude='*.faiss' \
    --exclude='*.tar' \
    --exclude='*.tar.gz' \
    --exclude='*.zip' \
    --exclude='*.bak' \
    --exclude='*.bak_*' \
    --exclude='nohup.out' \
    --exclude='.DS_Store' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$SRC" "$DST"

echo
if [ "$DIRECTION" = "pull" ]; then
    cd "$LOCAL_ROOT"
    echo "=== git status (review changes) ==="
    git status --short
    echo
    echo "Hint:"
    echo "  git diff && git add -A && git commit -m '...' && git push"
fi
