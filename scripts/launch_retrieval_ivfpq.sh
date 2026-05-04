#!/usr/bin/env bash
# Launch retrieval_server using IVF_PQ index (CPU mode, fast, ~1.3GB)
# vs old launch_retrieval_server.sh which loads 64GB Flat
#
# Why CPU: IVF_PQ on CPU is already <10ms/query for 21M docs; GPU faiss-gpu
# 1.12 has no sm_120 kernels anyway.

set -e

source /etc/network_turbo 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

INDEX=/root/autodl-tmp/data/wikipedia_index/e5_IVFPQ.index
CORPUS=/root/autodl-tmp/data/wikipedia_index/wiki-18.jsonl
TARGET=/root/autodl-tmp/external/Search-R1/search_r1/search/retrieval_server.py

test -f $INDEX  || { echo "❌ E5 IVF_PQ index not found at $INDEX"; exit 1; }
test -f $CORPUS || { echo "❌ corpus not found"; exit 1; }
test -f $TARGET || { echo "❌ retrieval_server not found"; exit 1; }

cd /root/autodl-tmp/external/Search-R1
echo "[launch-ivfpq] starting retrieval_server with IVF_PQ on CPU..."
nohup python search_r1/search/retrieval_server.py \
    --index_path $INDEX \
    --corpus_path $CORPUS \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model /root/autodl-tmp/models/e5-base-v2 \
    > /tmp/retrieval_ivfpq.log 2>&1 &

PID=$!
echo $PID > /tmp/retrieval_ivfpq.pid
echo "[launch-ivfpq] PID=$PID, log=/tmp/retrieval_ivfpq.log"
echo "[launch-ivfpq] 等待启动 (IVF_PQ ~1.3GB load + uvicorn, 估 30s)..."

for i in $(seq 1 60); do
    if curl -s -m 2 -X POST http://127.0.0.1:8000/retrieve \
        -H 'Content-Type: application/json' \
        -d '{"queries":["test"],"topk":1}' > /dev/null 2>&1; then
        echo "[launch-ivfpq] ✅ ready @ +$((i*5))s"
        break
    fi
    [ $((i % 6)) -eq 0 ] && echo "  …waiting ${i}/60 × 5s"
    sleep 5
done

echo
echo "=== quick benchmark (single query) ==="
time curl -s -m 5 -X POST http://127.0.0.1:8000/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"queries":["who is the president of the united states"],"topk":3}' \
    | head -c 300
echo

echo "=== batch=10 query benchmark ==="
time curl -s -m 30 -X POST http://127.0.0.1:8000/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"queries":["q1 united states president","q2 capital france","q3 isaac newton","q4 photosynthesis","q5 quantum mechanics","q6 albert einstein","q7 boiling point water","q8 mount everest","q9 great wall china","q10 lithium battery"],"topk":3}' \
    > /tmp/bench_ivfpq.json
echo "result size: $(wc -c </tmp/bench_ivfpq.json) bytes"
