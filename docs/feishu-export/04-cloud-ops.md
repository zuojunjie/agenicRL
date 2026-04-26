# 云端运维大全 · 踩坑与传输实测

> **范围**：AutoDL 4×A100 实例上做 Search-R1 训练时遇到的所有"非教程"问题
> **来源**：
> - §1–§9 由 master agent 在 Phase -1 设置过程中沉淀
> - §10 由 subAgent 1（worktree `sweet-archimedes-7376d4`）实测补充
> - 末尾"七个绝不"由 master + subAgent 1 共同维护
> **维护节奏**：每次踩到新坑就追加一节；每凿出一条新规则就追加到"绝不"

---


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

## 9. 突破：hf-mirror.com + 8 路并发 = 6× 加速

**症状**：AutoDL 学术加速代理对 hf.co 限速 ~430 KB/s，单连接无法提速。

**关键发现**：
- 走 `/etc/network_turbo` 代理：430 KB/s
- 直连 hf.co：被墙
- **直连 hf-mirror.com**：2 MB/s
- **直连 hf-mirror.com + 8 路并发 range 请求**：**12 MB/s ⭐**

**为什么只有 hf-mirror.com 有效**：
- AutoDL 网络对 hf-mirror.com (国内 GitHub Pages 代理) 走专线优化
- hf-mirror 透传 hf.co file URL（同样的 `/datasets/USER/REPO/resolve/main/FILE` 格式）
- 单连接被 hf-mirror 限速 ~2 MB/s，但允许多并发

**实操**：写一个 Python 多线程 range 下载器（见 `scripts/parallel_dl.py`），不走代理，直接 hf-mirror.com，8 个 worker 拆分 byte range 并行 GET。

**性能数据**（80GB Wiki + 索引）：
- 之前 hf_hub_download + 代理：估算 11h 完成
- 改用 parallel_dl.py + hf-mirror：**~55 分钟完成** 6×

**重要**：metadata API（list-files / repo-info）走 hf-mirror 不稳，但**直接 file URL 完全 OK**。所以混合策略：metadata 用 hf.co 拿，文件下载用 hf-mirror 自己拼 URL。

## 10. 本地→云上传通道实测（SCP / xftp / "文件存储网页")

**触发场景**：评论区流传"AutoDL 直连 HF 慢，先本地下完再 xftp 传上去更快"。
2026-04-26 实测验证此说法是否成立。

### 测试方法

```bash
# 1. 造 1GB 随机数据（避免 SSH 压缩协议美化结果）
dd if=/dev/urandom of=/tmp/test_1g.bin bs=1m count=1024
# 1.88s 写完，569 MB/s（本地 SSD，无瓶颈）

# 2. SCP 上传到 AutoDL 实例硬盘
time scp /tmp/test_1g.bin autodl-agenicrl:/root/autodl-tmp/test_1g.bin
# real 3m12.167s
```

测试环境：
- 本地：MacBook，国内家用千兆光纤（具体上行套餐未知）
- AutoDL 实例：西区 `connect.westd.seetacloud.com:45432`
- 时间：周日晚 21:43，避开了工作时段拥塞

### 实测结果

```
1024 MB / 192 s = 5.33 MB/s ≈ 42.7 Mbps
```

这个数字**恰好等于运营商默认对千兆光纤上行的限速**（30–50 Mbps 区间），说明瓶颈在**家用宽带上行**，不是 AutoDL 那一端。AutoDL 实例下行能力远超 5.33 MB/s（见 section 9 的 14 MB/s 实测）。

### 与现有方案对比（HF→云路径）

| 路径 | 速度 | 80GB 耗时 | 是否推荐 |
|---|---|---|---|
| HF → 云直连（parallel_dl.py + hf-mirror，section 9） | **12–14 MB/s** | ~55 min | 🟢 当前方案 |
| HF → 本地 → 云（SCP/xftp） | 上限 ~5.3 MB/s | ≥4.3h（仅上传段） | 🔴 反而更慢 |
| HF → 本地 → 云（AutoDL 文件存储网页上传） | 估 10–20 MB/s（未测） | 估 1–2.5h（仅上传段） | 🟡 仅当 SCP 不够时考虑 |

**关键结论**：对于"HF 上的公开数据集 → AutoDL"这条路径，**parallel_dl.py 全面碾压本地中转**。评论的建议在 section 9 突破之前是对的，突破之后就过时了。

### xftp 与 SCP 是同一通道（破除误解）

评论里"xftp 传可能更快"是**错的**：
- xftp = NetSarang 出品的 GUI 工具，底层协议是 **SFTP**（SSH File Transfer Protocol）
- SFTP 跑在 **SSH 通道**上，与 `scp` 共用同一加密、压缩、TCP 连接
- 速度差异最多 ±5%（取决于客户端的 pipeline 实现），不存在数量级提升

真正快的应该是 **AutoDL "文件存储"（autodl-fs）的网页多线程上传** —— 它走 HTTPS + AutoDL 自家 CDN，不经 SSH。但只能浏览器手动拖文件，无法脚本化。

### SCP 通道仍然是这些场景的正解

虽然 HF→云 不该走它，但 **本地 artifact → 云** 的几类场景**只能**用 SCP：

1. **训练 checkpoint 反向拖回本地**（云→本地，约 5–10 MB/s 实测预期）
2. **自己改造的中等数据集上传**（< 30GB，HF 上没有）
3. **代码同步**（git push/pull 走的就是 SSH，同通道）
4. **临时调试文件**（< 1GB，一次性的）

→ 此类场景按 5.33 MB/s 估算耗时即可。

### 后续可优化路径（未实施）

- **AutoDL "文件存储"（autodl-fs）** 网页上传：声称带 CDN 加速，理论 10–20 MB/s。挂载点 `/root/autodl-fs/` 跨实例共享、断机不丢数据，比 `/root/autodl-tmp/` 更适合大件素材。如未来需要传 ≥30GB 的本地素材，先试这条路。
- **rsync 增量+压缩**：`rsync -avzP --partial` 替代 scp，断点续传更稳，但单连接上限同样卡在家用上行。
- **运营商上行升级**：联通/电信支持解锁千兆对称（一般要换企业宽带），上行可到 100+ Mbps，那 SCP 路径才会真正快起来。

### 操作要点 / 易踩坑

```bash
# ✅ 正确：测试用随机数据，避免压缩偏差
dd if=/dev/urandom of=test.bin bs=1m count=1024

# ❌ 错误：用 /dev/zero 测，SSH 压缩会假冒 100MB/s
dd if=/dev/zero of=test.bin bs=1m count=1024

# ✅ 正确：清理两端
rm /tmp/test_1g.bin
ssh autodl-agenicrl 'rm /root/autodl-tmp/test_1g.bin'

# ❌ 错误：忘了清理云端，1GB 占着 350GB 持久盘空间
```

## 总结：cloud_setup.sh 的"七个绝不"

1. **绝不**忘 `source /etc/network_turbo`
2. **绝不**用 `pkill -9 -f` 模糊匹配
3. **绝不**对长流程加 `--quiet`
4. **绝不**假设 R2 备份域名稳——准备 `--no-deps + 单独装 triton` 的 Plan B
5. **绝不**忘 `export HF_HUB_DISABLE_XET=1` 在 hf 下载前
6. **绝不**在 ssh 命令里直接 `python -c`，写到 `/tmp/*.py` 再调
7. **绝不**把"本地下完再 SCP/xftp"当作 HF→云加速方案——上行带宽是硬瓶颈，已被 parallel_dl.py 碾压
