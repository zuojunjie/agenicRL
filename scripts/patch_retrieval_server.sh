#!/usr/bin/env bash
# Patch external/Search-R1/search_r1/search/retrieval_server.py 让它 device-aware
# 目的：
#   1. 无卡模式可以 import / 启动测试（虽然真跑还是要 GPU + faiss-gpu）
#   2. 未来想在小机器 sanity check 时不用改源码
#
# 改动（5 处）：
#   - load_model() 里 model.cuda() → model.to(_DEFAULT_DEVICE)
#   - encode() 里 inputs.cuda() → inputs.to(_DEFAULT_DEVICE)
#   - encode() 末尾 torch.cuda.empty_cache() → 条件化
#   - 顶部加一个 _DEFAULT_DEVICE 推断
#
# 用法（在 cloud 上）：
#   bash scripts/patch_retrieval_server.sh
#
# 幂等：再跑一次会跳过已改的内容
set -e

# 确保 python3 在 PATH（conda env or 系统）
if ! command -v python3 > /dev/null; then
    source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate searchr1 || true
fi
command -v python3 > /dev/null || { echo "❌ python3 不在 PATH"; exit 1; }

TARGET=/root/autodl-tmp/external/Search-R1/search_r1/search/retrieval_server.py
test -f "$TARGET" || { echo "❌ $TARGET 不存在"; exit 1; }

# 已 patch 过则跳过（看顶部有没有我们的标记）
if grep -q "_DEFAULT_DEVICE" "$TARGET"; then
    echo "✓ already patched, skipping"
    exit 0
fi

# 备份
cp "$TARGET" "$TARGET.bak.$(date +%s)"

# 1. 顶部 import 后插入 device 推断
python3 <<PY
import re
p = "$TARGET"
src = open(p).read()

# (1) 在 "import faiss" 后面加 _DEFAULT_DEVICE 推断
inject = '''
# ---- patch (agenicRL): device-aware encoder for non-GPU sanity testing ----
_DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)
print(f"[retrieval_server] using device: {_DEFAULT_DEVICE}")
'''
src = src.replace(
    "import faiss\nimport torch",
    "import faiss\nimport torch" + inject,
    1,
)

# (2) load_model: model.cuda() -> model.to(_DEFAULT_DEVICE)
src = src.replace(
    "model.cuda()",
    "model.to(_DEFAULT_DEVICE)",
    1,
)

# (3) encode: inputs cuda → DEVICE
src = src.replace(
    'inputs = {k: v.cuda() for k, v in inputs.items()}',
    'inputs = {k: v.to(_DEFAULT_DEVICE) for k, v in inputs.items()}',
    1,
)

# (4) torch.cuda.empty_cache() 条件化（出现 2 次，缩进可能不同 — 用 regex 保留各自缩进）
src = re.sub(
    r"^(\s+)torch\.cuda\.empty_cache\(\)$",
    lambda m: f"{m.group(1)}if _DEFAULT_DEVICE == 'cuda':\n{m.group(1)}    torch.cuda.empty_cache()",
    src,
    flags=re.MULTILINE,
)

open(p, "w").write(src)
print("✓ patched")
PY

echo
echo "===== Diff (head) ====="
diff <(head -130 "$TARGET.bak."* 2>/dev/null | tail -30) <(head -130 "$TARGET") | head -30 || true

echo
echo "===== 验证 import 不崩 ====="
source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1
cd /root/autodl-tmp/external/Search-R1
python -c "
import sys
sys.path.insert(0, 'search_r1/search')
# 不能直接 import retrieval_server，因为它会启动 FastAPI app；只测语法
import ast
with open('search_r1/search/retrieval_server.py') as f:
    ast.parse(f.read())
print('✓ syntax valid')
print('✓ patch applied successfully')
"
