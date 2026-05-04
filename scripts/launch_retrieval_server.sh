#!/usr/bin/env bash
# Launch retrieval_server (background, GPU-accelerated faiss)
# 在 4 卡 GPU 模式下，把 64GB E5 index 加载到 GPU 0 上的 faiss
# 服务器监听 :8000，训练时 retriever.url=http://127.0.0.1:8000/retrieve

set -e

# 基础环境
source /etc/network_turbo 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

# 4090 没 NVLink，关 P2P fallback to socket
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

INDEX=/root/autodl-tmp/data/wikipedia_index/e5_Flat.index
CORPUS=/root/autodl-tmp/data/wikipedia_index/wiki-18.jsonl
TARGET=/root/autodl-tmp/external/Search-R1/search_r1/search/retrieval_server.py

test -f $INDEX  || { echo "❌ E5 index not found"; exit 1; }
test -f $CORPUS || { echo "❌ corpus not found"; exit 1; }
test -f $TARGET || { echo "❌ retrieval_server not found"; exit 1; }

# 4 卡模式：faiss_gpu + shard，64GB fp16 → 8GB/卡。比 CPU 模式快 ~80x。
# 1 卡模式回退到 CPU faiss（64GB 单卡装不下）。
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 | xargs)
if [ "${GPU_COUNT:-0}" -ge 2 ]; then
    GPU_FLAG="--faiss_gpu"
    export CUDA_VISIBLE_DEVICES=0,1   # 让 faiss 看到 4 卡 shard
    echo "[launch] using faiss-gpu sharded across 2 GPUs (PRO 6000 dual, fp16) (16GB fp16 each, ~8GB used)"
else
    GPU_FLAG=""
    echo "[launch] using faiss-cpu (1TB RAM mode for 64GB index, ~1m22s/query)"
fi

cd /root/autodl-tmp/external/Search-R1
echo "[launch] starting retrieval_server on port 8000 (background)..."
nohup python search_r1/search/retrieval_server.py \
    --index_path $INDEX \
    --corpus_path $CORPUS \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model /root/autodl-tmp/models/e5-base-v2 \
    $GPU_FLAG \
    > /tmp/retrieval_server.log 2>&1 &

echo "$!" > /tmp/retrieval.pid
echo "[launch] PID=$(cat /tmp/retrieval.pid)"
echo "[launch] log: /tmp/retrieval_server.log"
echo "[launch] 等待启动 (CPU faiss 加载 64GB 索引 + uvicorn 启动，估 1-3 min)..."

# 等待 server 上线（180 个 5s = 15 min 超时，给 CPU mode 足够时间）
READY=0
for i in $(seq 1 180); do
    if curl -s -m 2 -X POST http://127.0.0.1:8000/retrieve \
        -H 'Content-Type: application/json' \
        -d '{"queries":["test"],"topk":1}' > /dev/null 2>&1; then
        echo "[launch] ✅ server ready @ +$((i*5))s"
        READY=1
        break
    fi
    sleep 5
done

if [ $READY -eq 0 ]; then
    echo "[launch] ❌ TIMEOUT — server didn't come up in 15 min"
    echo "[launch] log tail:"
    tail -30 /tmp/retrieval_server.log
    exit 1
fi

echo
echo "===== 测试一条查询 ====="
curl -s -X POST http://127.0.0.1:8000/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"queries":["What is the capital of France"],"topk":2,"return_scores":true}' \
    | python -c "import json, sys; d=json.load(sys.stdin); r=d['result'][0]; [print(f'  doc {i+1} (score={x.get(\"score\",\"?\"):.3f}): {x[\"document\"][\"contents\"][:120]}...') for i,x in enumerate(r)]"

echo
echo "===== 启动完成 ====="
echo "PID: $(cat /tmp/retrieval.pid)"
echo "log: tail -f /tmp/retrieval_server.log"
echo "kill: kill \$(cat /tmp/retrieval.pid)"
