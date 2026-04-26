# Phase -1 · 本地 CPU 预热清单

> 零云费用阶段。在本地完成所有可以脱离 GPU 的事，云上每分钟才能高效。

## 8 项交付物

| # | 交付物 | 本地可行 | 预计耗时 | 验收标准 |
|---|---|---|---|---|
| 1 | Python 环境（uv 或 conda） | ✅ | 30 min | `python -c "import torch, transformers, vllm"` 通过 |
| 2 | Search-R1 仓库 clone + 代码通读 | ✅ | 2–4 h | 能口述 agent loop（`<search>` → 检索 → `<answer>`）经过哪些代码文件 |
| 3 | Wikipedia corpus + 预构建 retrieval index 下载 | ✅ | 看带宽，~50–80GB | 文件落盘、md5 校验通过 |
| 4 | QA 数据集下载（NQ / HotpotQA dev） | ✅ | 30 min | jsonl 落盘 |
| 5 | CPU 烟雾测试：Qwen2.5-0.5B 跑通一次 inference + search 调用 | ✅（慢） | 1–2 h | 1 条 NQ 样本完整跑完 agent loop |
| 6 | AutoDL 账号 + 镜像选定 + 持久化网盘开通 | ✅ | 30 min | 账号充值 ¥100，网盘 100GB 已挂载 |
| 7 | 一键启动脚本 `bootstrap.sh` | ✅ | 1 h | 在新实例上跑一遍可达"等价于本地预热完毕"状态 |
| 8 | wandb 账号 + project + 命名规范 | ✅ | 15 min | `wandb login` 通过 |

**总计**：1–2 个完整晚上。

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
