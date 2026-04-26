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
# HuggingFace 下载关键环境：禁用 xet 协议（cas-server.xethub.hf.co 401）
# 用国内 hf-mirror.com 镜像（更稳，AutoDL 学术加速对它特别友好）
# ============================================================
export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT=https://hf-mirror.com

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
if python -c "import torch" 2>/dev/null; then
    echo "  (already installed: $(python -c 'import torch; print(torch.__version__)'))"
else
    # 走 R2 不稳定，用 Plan B：triton 从 PyPI Aliyun 镜像，torch 用 --no-deps
    # 然后单独补 11 个 nvidia-cu12 wheel
    pip install triton==3.0.0
    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 --no-deps \
        --resume-retries 100 --timeout 120
    pip install \
        nvidia-cublas-cu12==12.1.3.1 \
        nvidia-cuda-cupti-cu12==12.1.105 \
        nvidia-cuda-nvrtc-cu12==12.1.105 \
        nvidia-cuda-runtime-cu12==12.1.105 \
        nvidia-cudnn-cu12==9.1.0.70 \
        nvidia-cufft-cu12==11.0.2.54 \
        nvidia-curand-cu12==10.3.2.106 \
        nvidia-cusolver-cu12==11.4.5.107 \
        nvidia-cusparse-cu12==12.1.0.106 \
        nvidia-nccl-cu12==2.20.5 \
        nvidia-nvtx-cu12==12.1.105 \
        nvidia-nvjitlink-cu12 \
        "filelock>=3.13" "typing-extensions>=4.8" "sympy>=1.13" \
        "networkx>=3.0" "jinja2>=3.1" "fsspec>=2024"
fi

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
