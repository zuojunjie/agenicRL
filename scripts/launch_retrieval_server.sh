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

# 让 retrieval_server 只用 GPU 0（训练用 GPU 0-3，但 retrieval 只占用 1 张时主要在初始化时争）
# 实际上 retrieval_server 内部 faiss-gpu 自己挑 GPU，不冲突
export CUDA_VISIBLE_DEVICES=3   # 给 retrieval 留 GPU 3，训练用 0-2 (实测 vllm 占 1 卡，actor 占 2 卡)
# 注：上面 CUDA_VISIBLE_DEVICES 仅对当前 shell 有效，训练脚本会重新 export

INDEX=/root/autodl-tmp/data/wikipedia_index/e5_Flat.index
CORPUS=/root/autodl-tmp/data/wikipedia_index/wiki-18.jsonl
TARGET=/root/autodl-tmp/external/Search-R1/search_r1/search/retrieval_server.py

test -f $INDEX  || { echo "❌ E5 index not found"; exit 1; }
test -f $CORPUS || { echo "❌ corpus not found"; exit 1; }
test -f $TARGET || { echo "❌ retrieval_server not found"; exit 1; }

# 决策：试 faiss_gpu，失败 fallback 到 cpu
HAS_GPU_FAISS=$(python -c "import faiss; print(faiss.get_num_gpus() > 0)" 2>/dev/null || echo "False")
echo "[launch] faiss-gpu available: $HAS_GPU_FAISS"

if [ "$HAS_GPU_FAISS" = "True" ]; then
    GPU_FLAG="--faiss_gpu"
else
    GPU_FLAG=""
    echo "[launch] using faiss-cpu (RAM mode for 64GB index)"
fi

cd /root/autodl-tmp/external/Search-R1
echo "[launch] starting retrieval_server (background)..."
nohup python search_r1/search/retrieval_server.py \
    --index_path $INDEX \
    --corpus_path $CORPUS \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    $GPU_FLAG \
    > /tmp/retrieval_server.log 2>&1 &

echo "$!" > /tmp/retrieval.pid
echo "[launch] PID=$(cat /tmp/retrieval.pid)"
echo "[launch] log: /tmp/retrieval_server.log"
echo "[launch] 等待启动 (faiss 加载 64GB 大概 30-60s)..."

# 等待 server 上线
for i in $(seq 1 60); do
    if curl -s -m 2 -X POST http://127.0.0.1:8000/retrieve \
        -H 'Content-Type: application/json' \
        -d '{"queries":["test"],"topk":1}' > /dev/null 2>&1; then
        echo "[launch] ✅ server ready @ +${i}s"
        break
    fi
    sleep 5
done

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
