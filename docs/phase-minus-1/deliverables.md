# Phase -1 · 本地 CPU 预热清单

> 零云费用阶段。在本地完成所有可以脱离 GPU 的事，云上每分钟才能高效。

## 8 项交付物

| # | 交付物 | 状态 | 验收标准 |
|---|---|---|---|
| 1 | Python 环境（uv + venv，Python 3.10） | ✅ **完成** | `import torch, transformers, datasets, requests` 通过；MPS 可用 |
| 2 | Search-R1 仓库 clone + 代码通读 | ✅ **完成** | 笔记落盘于 `notes/agent-loop.md`，掌握 infer.py + retrieval_server.py |
| 3 | Wikipedia corpus + 预构建 retrieval index 下载 | ⏸ 待启动（夜间）| `e5_Flat.index` + `wiki-18.jsonl` 落盘 |
| 4 | QA 数据集（NQ / HotpotQA dev） | ⏸ 待启动 | jsonl 落盘 |
| 5 | CPU/MPS 烟雾测试：Qwen2.5-0.5B 跑通 agent loop | ✅ **完成** | mock-search 版本 PASS, MPS 9.7 tok/s |
| 6 | AutoDL 账号 + 镜像选定 + 网盘开通 | ⏸ 待启动（用户操作）| 账号充值，网盘 100GB 挂载 |
| 7 | 一键启动脚本 `bootstrap.sh` | ⏸ 待写 | 新实例可达就绪状态 |
| 8 | wandb 账号 + project + 命名规范 | ⏸ 待启动 | `wandb login` 通过 |

**进度**：4/8 完成，剩 4 项里 #6 是用户实名操作、#3 是大下载，#4 #7 #8 都是 30 分钟内能搞定的轻活。

## 决策 A 已定锤 🟢

Search-R1 官方提供预构建索引下载（HuggingFace），命令：

```bash
python external/Search-R1/scripts/download.py --save_path data/wikipedia_index
cat data/wikipedia_index/part_* > data/wikipedia_index/e5_Flat.index
gzip -d data/wikipedia_index/wiki-18.jsonl.gz
```

数据来源：
- `PeterJinGo/wiki-18-e5-index` (`part_aa` + `part_ab`，合并为 `e5_Flat.index`)
- `PeterJinGo/wiki-18-corpus` (`wiki-18.jsonl.gz`)

## ⚠️ 本地 vs 云端版本鸿沟（关键备忘）

| 工具 | 本地（学习/读代码） | 云端（必须严格匹配） |
|---|---|---|
| Python | 3.10.20 | 3.9（Search-R1 主环境） |
| torch | 2.11.0 | **2.4.0** + cu121 |
| transformers | 5.6.2 | **<4.48** |
| vllm | ❌ 不装 | **0.6.3** |
| flash-attn | ❌ 不装 | 必装 |
| faiss | 不装（暂） | **faiss-gpu 1.8.0** |

本地新版本对"读代码 + 简单 inference"无碍，但 **bootstrap.sh 必须严格按官方版本号装**，否则 verl 会崩。

## Agent loop 心智模型（来自 `infer.py`）

```python
# 60 行就完整描述了 Search-R1 的核心交互
loop:
    1. model.generate(...) 直到遇到 </search> 或 EOS
    2. 若 EOS → 输出 <answer>，break
    3. 若 </search>:
        - 正则提取 <search>(.*?)</search>
        - POST http://localhost:8000/retrieve {queries, topk=3, return_scores}
        - 把结果格式化成 "Doc 1(Title:...) ...\nDoc 2..."
        - 拼接到 prompt: <information>{results}</information>
        - 继续 loop
```

这就是整个范式。Phase 1 读 DeepSeek-R1 时回头看这段代码，会发现 GRPO 是在 **整条** 这种轨迹上算 advantage 的。

## 下一步候选动作

按优先级：

1. **精读 `infer.py` + `search_r1/search/retrieval_server.py`** — 30 min，完成交付物 #2
2. **后台启动索引下载**（~50–80GB，看网速 1–8h） — 决策 A 已定，可以开了
3. **下载 Qwen2.5-0.5B 跑 mock 烟雾测试** — 验证本地 inference 通路，不需要真检索服务
4. **AutoDL 账号注册** — 5 分钟的事，可以随时插着做

## 关键决策（待拍板）

### 决策 A：retrieval index 怎么解决

| 方案 | 大小 | CPU 可行 | 复现度 | 推荐度 |
|---|---|---|---|---|
| 下载 Search-R1 官方预构建索引 | ~50–80GB | ✅ 仅下载 | 100% | 🟢 强烈推荐 |
| 用 BM25 (pyserini) | ~5GB | ✅ 还能 CPU 构建 | ~80%（精度低于 E5） | 🟡 备选 |
| 自己 GPU 上重建 E5 索引 | ~80GB | ❌ | 100% | 🔴 不必要 |

### 决策 B：本地磁盘空间

预算约 96GB：

```
Wikipedia corpus + index   ~80 GB
Qwen2.5-0.5B + 3B 权重     ~10 GB
QA 数据集                   ~1 GB
代码 + 缓存                 ~5 GB
─────────────────────────────────
合计                       ~96 GB
```

如果本地不够，方案 B：直接把 index 上传 AutoDL 持久化网盘。

### 决策 C：bootstrap.sh 草案

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp                    # 持久化网盘挂载点
git -C agenicRL pull                   # 拉本地推上来的代码
source venv/bin/activate
wandb login --relogin "$WANDB_KEY"
nvidia-smi                              # 确认 4 卡都在
bash smoke_test.sh                      # 5 分钟烟雾测试
echo "Ready. Launch training when ready."
```
