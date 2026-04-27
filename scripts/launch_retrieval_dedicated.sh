#!/usr/bin/env bash
# 单卡独占模式 retrieval_server
# 把 64GB faiss index (fp16 → 32GB) 整放在 GPU 3 一张卡上
# 训练独占 GPU 0/1/2 (n_gpus_per_node=3)
#
# 优势：
#   - 训练 GPU 完全独占（没 faiss 抢显存）
#   - 可以关 FSDP offload，actor update 提速 30-50%
#   - faiss 单卡查询无 NCCL 同步开销
#
# 用法：
#   bash scripts/launch_retrieval_dedicated.sh
#
# 与 launch_retrieval_server.sh 的区别：
#   后者 auto-detect 4 卡 → shard
#   这个永远把 retrieval 锁在 GPU 3

set -e

# 基础环境
source /etc/network_turbo 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

# 4090 没 NVLink，关 P2P
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 关键：retrieval 独占 GPU 3
export CUDA_VISIBLE_DEVICES=${RETRIEVAL_GPU:-3}

INDEX=/root/autodl-tmp/data/wikipedia_index/e5_Flat.index
CORPUS=/root/autodl-tmp/data/wikipedia_index/wiki-18.jsonl
TARGET=/root/autodl-tmp/external/Search-R1/search_r1/search/retrieval_server.py

test -f $INDEX  || { echo "❌ E5 index not found"; exit 1; }
test -f $CORPUS || { echo "❌ corpus not found"; exit 1; }
test -f $TARGET || { echo "❌ retrieval_server not found"; exit 1; }

echo "[launch] dedicated retrieval on GPU $CUDA_VISIBLE_DEVICES"
echo "[launch] faiss IndexFlat 64GB → fp16 32GB single card"

cd /root/autodl-tmp/external/Search-R1
nohup python search_r1/search/retrieval_server.py \
    --index_path $INDEX \
    --corpus_path $CORPUS \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model /root/autodl-tmp/models/e5-base-v2 \
    --faiss_gpu \
    > /tmp/retrieval_server.log 2>&1 &

echo "$!" > /tmp/retrieval.pid
echo "[launch] PID=$(cat /tmp/retrieval.pid)"
echo "[launch] log: /tmp/retrieval_server.log"
echo "[launch] 等待启动 (faiss 32GB 单卡加载 ~5-10 min)..."

# 等待 server 上线
READY=0
for i in $(seq 1 180); do
    if curl -s -m 2 -X POST http://127.0.0.1:8000/retrieve \
        -H 'Content-Type: application/json' \
        -d '{"queries":["test"],"topk":1,"return_scores":true}' > /dev/null 2>&1; then
        echo "[launch] ✅ server ready @ +$((i*5))s"
        READY=1
        break
    fi
    sleep 5
done

if [ $READY -eq 0 ]; then
    echo "[launch] ❌ TIMEOUT after 15 min"
    tail -30 /tmp/retrieval_server.log
    exit 1
fi

# 测速：单卡 faiss 比 4 卡 shard 快否？
echo
echo "===== 单卡 faiss 测速 (3 次连续查询) ====="
for i in 1 2 3; do
    time curl -s -m 30 -X POST http://127.0.0.1:8000/retrieve \
        -H 'Content-Type: application/json' \
        -d '{"queries":["What is the capital of France"],"topk":3,"return_scores":true}' \
        > /dev/null
done

echo
echo "===== GPU 3 显存占用 ====="
nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 3

echo
echo "===== 启动完成 ====="
echo "PID: $(cat /tmp/retrieval.pid)"
echo "训练时设 CUDA_VISIBLE_DEVICES=0,1,2  (训练独占)"
