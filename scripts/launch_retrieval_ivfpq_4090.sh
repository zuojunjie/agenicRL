#!/usr/bin/env bash
# 4090 single-card setup: IVF_PQ retrieval on CPU (no GPU contention with training)
set -e

source /etc/network_turbo 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

INDEX=/root/autodl-tmp/data/wikipedia_index/e5_IVFPQ.index
CORPUS=/root/autodl-tmp/data/wikipedia_index/wiki-18.jsonl
TARGET=/root/autodl-tmp/external/Search-R1/search_r1/search/retrieval_server.py

test -f $INDEX  || { echo "❌ IVF_PQ index not found"; exit 1; }
test -f $CORPUS || { echo "❌ corpus not found"; exit 1; }
test -f $TARGET || { echo "❌ retrieval_server.py not found"; exit 1; }

echo "[launch] IVF_PQ retrieval on CPU (no GPU)"
cd /root/autodl-tmp/external/Search-R1
nohup python search_r1/search/retrieval_server.py \
    --index_path $INDEX \
    --corpus_path $CORPUS \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model /root/autodl-tmp/models/e5-base-v2 \
    > /tmp/retrieval_ivfpq.log 2>&1 &

echo "$!" > /tmp/retrieval.pid
echo "[launch] PID=$(cat /tmp/retrieval.pid)"

for i in $(seq 1 60); do
    if curl -sf -m 2 -X POST http://127.0.0.1:8000/retrieve \
        -H 'Content-Type: application/json' \
        -d '{"queries":["test"],"topk":1}' > /dev/null 2>&1; then
        echo "[launch] ✅ server ready @ +$((i*5))s"
        exit 0
    fi
    sleep 5
done
echo "[launch] ❌ TIMEOUT"
tail -20 /tmp/retrieval_ivfpq.log
exit 1
