#!/usr/bin/env bash
# bootstrap_after_reboot.sh — single-shot recovery after PRO 6000 reboot
# Usage (from cloud after SSH back):
#   bash /root/autodl-tmp/agenicRL/scripts/bootstrap_after_reboot.sh

set -e

source /etc/network_turbo 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1

echo "=== 1. quick env sanity ==="
python -c "
import torch, vllm, faiss
print('torch:', torch.__version__, 'sm:', torch.cuda.get_device_capability(0))
print('vllm:', vllm.__version__)
print('faiss:', faiss.__version__, 'has_GPU:', hasattr(faiss, 'StandardGpuResources'))
"

echo
echo "=== 2. apply retrieval_server GPU fp16 patch ==="
python /root/autodl-tmp/agenicRL/patches/retrieval_server_gpu_fp16.py

echo
echo "=== 3. patch launch script: enable --faiss_gpu on 1 GPU ==="
LAUNCH=/root/autodl-tmp/agenicRL/scripts/launch_retrieval_server.sh
# Replace the GPU_COUNT≥4 condition with ≥1 (single GPU is now valid via fp16 direct path)
python -c "
path = '$LAUNCH'
src = open(path).read()
old = '    if [ \"\${GPU_COUNT:-0}\" -ge 4 ]; then'
new = '    if [ \"\${GPU_COUNT:-0}\" -ge 1 ]; then'
if old in src:
    open(path, 'w').write(src.replace(old, new))
    print('PATCHED launch script: 1 GPU = faiss_gpu mode')
elif new in src:
    print('launch script already patched')
else:
    print('launch script pattern not found, manual check needed')
"

echo
echo "=== 4. relaunch retrieval_server (GPU fp16 mode) ==="
# Kill any stale retrieval_server first
pkill -9 -f retrieval_server.py 2>/dev/null || true
sleep 2
nohup bash /root/autodl-tmp/agenicRL/scripts/launch_retrieval_server.sh > /tmp/retrieval_v3.log 2>&1 < /dev/null &
disown
sleep 10
echo "launched. Checking log…"
tail -20 /tmp/retrieval_v3.log
tail -10 /tmp/retrieval_server.log 2>/dev/null

echo
echo "=== 5. server up? wait up to 5 min for GPU index load ==="
for i in $(seq 1 60); do
    if curl -s -m 3 -X POST http://127.0.0.1:8000/retrieve \
        -H 'Content-Type: application/json' \
        -d '{"queries":["test"],"topk":1}' > /dev/null 2>&1; then
        echo "✅ server ready @ +$((i*5))s"
        break
    fi
    [ $((i % 6)) -eq 0 ] && echo "  …still loading (${i}/60 × 5s = $((i*5))s)"
    sleep 5
done

echo
echo "=== 6. benchmark batch=10 query ==="
time curl -s -m 30 -X POST http://127.0.0.1:8000/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"queries":["q1 united states president","q2 capital france","q3 isaac newton","q4 photosynthesis","q5 quantum mechanics","q6 albert einstein","q7 boiling point water","q8 mount everest","q9 great wall china","q10 lithium battery"],"topk":3}' \
    > /tmp/bench_batch10.json

echo "result size: $(wc -c </tmp/bench_batch10.json) bytes"
echo
echo "=== 7. final status ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
free -g | head -2
echo
echo "✅ Bootstrap done. Next: verl 1-step dry-run."
