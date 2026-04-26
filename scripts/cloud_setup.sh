#!/usr/bin/env bash
# AutoDL 实例环境搭建脚本
# 跑在云端实例上（不是本地 Mac），把 conda env 装好。
# 用法：在 AutoDL ssh 进入后，cd /root/autodl-tmp/agenicRL && bash scripts/cloud_setup.sh
set -euo pipefail

# ============================================================
# 关键：AutoDL 学术加速代理
# 国内访问 pypi / github / huggingface 直连慢得离谱，
# AutoDL 自家代理 10.37.1.23:12798 是唯一正确姿势。
# 这条是 AutoDL 几乎所有教程不写但必须做的事。
# ============================================================
if [ -f /etc/network_turbo ]; then
    source /etc/network_turbo
    echo "[setup] AutoDL 学术加速 已启用"
else
    echo "[setup] WARNING: /etc/network_turbo 不存在，可能不在 AutoDL 实例上"
fi

# ============================================================
# 激活 conda env
# ============================================================
source /root/miniconda3/etc/profile.d/conda.sh
ENV_NAME="searchr1"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[setup] conda env '${ENV_NAME}' 已存在，激活"
else
    echo "[setup] 创建 conda env '${ENV_NAME}' (python=3.9)"
    conda create -n "${ENV_NAME}" python=3.9 -y
fi
conda activate "${ENV_NAME}"

# ============================================================
# Search-R1 官方依赖（严格按 README 顺序）
# 注意：vllm / flash-attn / faiss-gpu 真正运行需要 GPU
# 但 pip install 阶段可以在无卡模式完成
# 例外：flash-attn 安装时会编译 CUDA 内核，需要 GPU 在场，因此延后
# ============================================================
echo "[setup] 1/4: torch 2.4.0+cu121"
python -c "import torch" 2>/dev/null && echo "  (already installed: $(python -c 'import torch; print(torch.__version__)'))" \
    || pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

echo "[setup] 2/4: vllm 0.6.3"
python -c "import vllm" 2>/dev/null && echo "  (already installed)" \
    || pip install vllm==0.6.3

echo "[setup] 3/4: verl + 通用工具"
# verl: 安装 Search-R1 仓库本身（它把 verl 作为 package 安装）
SEARCH_R1_DIR="/root/autodl-tmp/external/Search-R1"
if [ -d "${SEARCH_R1_DIR}" ]; then
    pip install -e "${SEARCH_R1_DIR}"
else
    echo "[setup] ERROR: ${SEARCH_R1_DIR} 不存在，先 clone Search-R1"
    exit 1
fi
pip install wandb 'transformers<4.48' datasets accelerate

echo "[setup] 4/4: flash-attn 推迟（要 GPU 才能编译）"
echo "  → 切到 GPU 模式后再跑：pip install flash-attn --no-build-isolation"

# ============================================================
# 验证
# ============================================================
echo "[setup] 验证 import..."
python -c "
import torch, transformers, datasets
print(f'torch        {torch.__version__}, cu={torch.version.cuda}')
print(f'transformers {transformers.__version__}')
print(f'datasets     {datasets.__version__}')
"

# vllm 在无卡模式下 import 会警告但不应该崩
python -c "import vllm; print(f'vllm         {vllm.__version__}')" 2>&1 | tail -1 || echo "vllm import 失败（无卡模式预期，切 GPU 模式后再验证）"

echo
echo "[setup] ✅ DONE"
echo
echo "下一步："
echo "  - 仍在无卡模式：可以 pip 装更多包、下载模型/数据"
echo "  - 切 GPU 模式后：bash scripts/install_flash_attn.sh"
