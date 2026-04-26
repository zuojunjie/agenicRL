# 云端环境搭建踩坑记录

## 1. AutoDL 学术加速代理 (`/etc/network_turbo`)

**症状**：直接 `pip install torch==2.4.0 --index-url download.pytorch.org/whl/cu121` 卡 9 分钟无进展，env 大小一直 194M，pip 进程 CPU 接近 0。

**根因**：国内直连 `download.pytorch.org` 经常被限速到几 KB/s 或直接超时。

**解法**：每次新 SSH session 装东西前必须执行：
```bash
source /etc/network_turbo
```

这设置 `http_proxy=http://10.37.1.23:12798`，AutoDL 内部专为加速 PyPI/GitHub/HuggingFace 设计。

**实测**：source 之前 9 分钟未动；source 之后 HEAD 请求 2.3s 返回 200，wheel 下载也跟着活了。

**已固化**：`scripts/cloud_setup.sh` 第一行就 source。

## 2. AutoDL 镜像不是 PEP 503 兼容的索引

**症状**：试 `pip install torch==2.4.0 --index-url https://mirrors.aliyun.com/pytorch-wheels/cu121/`：
```
ERROR: Could not find a version that satisfies the requirement torch==2.4.0 (from versions: none)
```

**根因**：Aliyun 的 `pytorch-wheels/` 是普通 HTTP 文件目录（HTML 列表），不是 pip 期望的 PEP 503 简单索引格式。pip 解析不到 wheel。

**解法**：不要把这个 URL 当 `--index-url`。要么 wget 单个 wheel 然后 `pip install ./xxx.whl`，要么用别的 mirror。

## 3. PyTorch 的 `download-r2.pytorch.org` 域名学术加速不稳

**症状**：source 学术加速后，主 torch wheel (799MB) 下成功，但 `triton==3.0.0` (209MB) 在 `download-r2.pytorch.org`（PyTorch 的 Cloudflare R2 备份 CDN）反复超时，6 次重试只下了 2.9MB。

**根因**：AutoDL 学术加速代理对 `download.pytorch.org` 主域名 OK，但对 R2 备份子域名（CDN 节点不同）连接不稳。

**解法**（绕过去）：
```bash
# 步骤 1：triton 从普通 PyPI（走 /etc/pip.conf 的 Aliyun 镜像）装
pip install triton==3.0.0

# 步骤 2：torch 用 --no-deps，主 wheel 用 cache，不再重装依赖
pip install torch==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --no-deps

# 步骤 3：补回 torch 的轻量依赖（在 PyPI 通用镜像里都有）
pip install "filelock>=3.13" "typing-extensions>=4.8" \
    "sympy>=1.13" "networkx>=3.0" "jinja2>=3.1" "fsspec>=2024"
```

**经验**：当一个 wheel 反复 R2 失败，先看它在不在通用 PyPI（`pip search` 不工作了，直接 `pip index versions <package>` 或 google）。
PyTorch wheel 因为带 cu121 后缀只在 PT 自家源；
但 triton、各种 nvidia-* 大多数其实在 PyPI 上也有。

### 3.1 `--no-deps` 之后的代价：手动补 11 个 nvidia-cu12 库

`pip install torch==2.4.0+cu121 --no-deps` 装完后 `import torch` 会报：
```
ImportError: libcudnn.so.9: cannot open shared object file: No such file or directory
```

torch 实际上依赖 11 个独立打包的 nvidia-cu12 wheel（每个对应一个 CUDA 库）。`--no-deps` 把它们全跳了。

**补救清单**（版本严格匹配 torch 2.4.0 metadata）：
```bash
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
    nvidia-nvjitlink-cu12
```

这些在 PyPI 上都有（NVIDIA 自家上传），走 Aliyun 镜像很快。

## 4. SSH 端口在切模式后会变

**症状**：从 GPU 模式切到无卡模式（或反之），原 SSH 命令 `ssh -p 47992 ...` 不通了。

**根因**：AutoDL 切模式 = 销毁旧容器 + 新建容器，新容器映射到不同端口。

**解法**：
- 每次切模式后从 AutoDL 控制台拷新的 SSH 命令，更新 `~/.ssh/config` 里的 `Port` 字段
- 持久化盘 `/root/autodl-tmp/` 内容**不变**（这就是它的意义）
- 系统盘 `/root/`（包括 `/root/miniconda3`）多数情况也保留，但官方说"重置系统"会清

## 5. pkill 误杀自己

**症状**：在 SSH 远程脚本里写 `pkill -9 -f "pip"` 试图清理之前卡住的 pip，结果整个脚本静默退出，输出为空。

**根因**：`pkill -9 -f "pip"` 按命令行匹配，**当前正在运行的 bash 脚本 cmdline 里有 "pip" 字符串**（因为脚本后面要 pip install），所以 pkill 把自己也杀了。

**解法**：用更精确的匹配：
```bash
pkill -9 -f "pip install.*torch==2.4.0"  # 精确到具体的 pip 命令
```

或者直接 `pgrep` 拿 PID 再 `kill`。

## 6. pip `--quiet` 在 SSH 后台时让监控失效

**症状**：SSH 后台跑 `pip install --quiet ... 2>&1 | tail -25`，输出文件 9 分钟全是空，根本不知道是卡住还是在工作。

**根因**：
- `--quiet` 把进度全省了
- `tail -25` 只在 stdin 关闭后输出
- 两者叠加：完全黑盒，只在最后一刻 dump 25 行

**解法**：
- 不加 `--quiet`，让 pip 输出 Downloading 进度
- 用 `grep --line-buffered` 流式过滤而不是 `tail`
- 或者在 Mac 端开 Monitor 工具盯 `Successfully|ERROR|Downloading` 关键词

## 7. HuggingFace xet (xethub.hf.co) 401 拒绝

**症状**：`huggingface_hub.snapshot_download(...)` 立刻报错：
```
RuntimeError: Data processing error: File reconstruction error:
CAS Client Error: Request error: HTTP status client error (401 Unauthorized),
domain: https://cas-server.xethub.hf.co/v1/reconstructions/...
```

**根因**：新版 huggingface_hub (≥0.25 左右) 默认对大文件用 xet 协议（HF 收购的去重 CAS 后端）下载提速。
但 `cas-server.xethub.hf.co` 这个域名走 AutoDL 学术加速代理时被 401 拒绝（认证逻辑或代理转发问题）。

**解法**：禁用 xet，强制走传统 HTTP：
```bash
export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT=https://hf-mirror.com   # 顺手用国内 HF 镜像，更稳
```

或在 Python 里 `os.environ['HF_HUB_DISABLE_XET'] = '1'` 在 `from huggingface_hub import ...` 之前。

**实测**：禁用 xet 后 Qwen2.5-3B (~6GB) 立刻开始正常下载。

## 7.5. `hf_transfer` 与 AutoDL 学术加速代理不兼容

**症状**：开 `HF_HUB_ENABLE_HF_TRANSFER=1` 后，`hf_hub_download` 直接报：
```
RuntimeError: An error occurred while downloading using `hf_transfer`.
Consider disabling HF_HUB_ENABLE_HF_TRANSFER for better error handling.
```

**根因**：`hf_transfer` 是 HF 官方 Rust 写的并行下载器，号称 5-10x 加速，但它走自己的 HTTP 客户端逻辑、不读 `http_proxy` 环境变量。AutoDL 学术加速代理因此被旁路，hf_transfer 直连 hf.co 失败。

**解法**：**不要开** `HF_HUB_ENABLE_HF_TRANSFER`。接受默认下载器的 ~2 MB/s 速度。

**含义**：80GB Wikipedia index 通过 AutoDL 代理 + 默认下载器 ≈ 11 小时。**这是过夜任务**，nohup 启动后睡觉。

## 8. SSH heredoc 引号嵌套陷阱

**症状**：在 ssh remote 上想跑 `python -c "..."` 嵌套写在 bash 里，f-string 里的单引号被 outer shell 抢走解析：
```
/bin/bash: eval: line 71: syntax error near unexpected token `('
```

**根因**：bash → ssh → bash → python -c "..."，三层嵌套，单/双引号优先级混乱。

**解法**：**绝不**在 ssh 命令里直接嵌入 python -c "..."。改成：
1. 用 heredoc 写 `.py` 文件到 remote `/tmp/`
2. 然后 `python /tmp/that_file.py`

模板：
```bash
ssh host 'bash -s' <<'OUTER_EOF'
cat > /tmp/myscript.py <<'PY_EOF'
# 这里随便写 python，引号无忧
print(f'time={time.time()}')
PY_EOF
python /tmp/myscript.py
OUTER_EOF
```

## 总结：cloud_setup.sh 的"五个绝不"

1. **绝不**忘 `source /etc/network_turbo`
2. **绝不**用 `pkill -9 -f` 模糊匹配
3. **绝不**对长流程加 `--quiet`
4. **绝不**假设 R2 备份域名稳——准备 `--no-deps + 单独装 triton` 的 Plan B
5. **绝不**忘 `export HF_HUB_DISABLE_XET=1` 在 hf 下载前
6. **绝不**在 ssh 命令里直接 `python -c`，写到 `/tmp/*.py` 再调
