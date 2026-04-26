# Phase -1 · 本地预热与里程碑

> **范围**：进入烧 GPU 之前的"零云费用预热阶段"的 8 项交付物 + 执行历史 + 镜像保存策略
> **来源**：master agent · `docs/phase-minus-1/deliverables.md` + `docs/journal/2026-04-26-kickoff.md` 摘要 + 本对话补充
> **前提认知**：见姊妹篇《决策档案》§3 算力约束

---

## 1. 8 项交付物清单

| # | 交付物 | 状态（2026-04-26 晚） | 验收标准 |
|---|---|---|---|
| 1 | Python 环境（本地 uv + 云端 conda searchr1） | ✅ **完成** | 本地：torch+MPS 可用；云端：torch 2.4.0+cu121 + vllm 0.6.3 + verl 全栈 import 通过 |
| 2 | Search-R1 仓库 clone + 代码通读 | ✅ **完成** | 本地+云端都有，笔记落盘（见《Search-R1 心智模型》子 doc） |
| 3 | Wikipedia corpus + 预构建 retrieval index 下载 | 🟢 **接近完成** | `e5_Flat.index` (80GB)；得益于 parallel_dl.py 突破，预计 < 1h 完成 |
| 4 | QA 数据集（NQ / HotpotQA dev） | ⏸ 等 #3 完事再做 | jsonl 落盘 |
| 5 | CPU/MPS 烟雾测试：Qwen2.5-0.5B 跑通 agent loop | ✅ **完成** | 本地 mock-search 版本 PASS, MPS 9.7 tok/s |
| 6 | AutoDL 账号 + 镜像选定 + 网盘开通 | ✅ **完成** | 4×A100 实例已开（无卡模式中），350GB 持久化盘挂载 |
| 7 | 一键启动脚本 `bootstrap.sh` | ✅ **完成** | `scripts/cloud_setup.sh` 集成所有踩坑经验 |
| 8 | wandb 账号 + project + 命名规范 | ✅ **完成** | 用户已配置（key 在 ~/.netrc） |

**总进度**：7/8 完成 + #3 接近完成。

## 2. 决策 A 已定锤：用 Search-R1 官方预构建索引

```bash
python external/Search-R1/scripts/download.py --save_path data/wikipedia_index
cat data/wikipedia_index/part_* > data/wikipedia_index/e5_Flat.index
gzip -d data/wikipedia_index/wiki-18.jsonl.gz
```

数据来源：
- `PeterJinGo/wiki-18-e5-index`（`part_aa` + `part_ab`，合并为 `e5_Flat.index`）
- `PeterJinGo/wiki-18-corpus`（`wiki-18.jsonl.gz`）

> 详见《云端运维大全》§9 — parallel_dl.py + hf-mirror 把下载从 11h 压到 ~55min。

## 3. ⚠️ 本地 vs 云端版本鸿沟（关键备忘）

| 工具 | 本地（学习/读代码） | 云端（必须严格匹配） |
|---|---|---|
| Python | 3.10.20 | **3.9**（Search-R1 主环境） |
| torch | 2.11.0 | **2.4.0** + cu121 |
| transformers | 5.6.2 | **<4.48** |
| vllm | ❌ 不装 | **0.6.3** |
| flash-attn | ❌ 不装 | 必装 |
| faiss | 不装（暂） | **faiss-gpu 1.8.0** |

本地新版本对"读代码 + 简单 inference"无碍，但 **bootstrap.sh 必须严格按官方版本号装**，否则 verl 会崩。

---

## 4. 执行历史 · 2026-04-26 kickoff 摘要

### 4.1 三个前置变量

| 变量 | 决定 |
|---|---|
| 算力预算 | 4×A100 |
| 场景偏好 | 搜索（Search-R1） |
| 学习深度 | 复现 + 论文递进 |

### 4.2 路径调整：从 5 篇到 7 篇

最初提案：DeepSeek-R1 → DAPO → ToRL → GiGPO → ARPO（5 主干）

经过和全景图对比后调整：

- 🆕 **GSPO** 加入 Phase 2，与 DAPO 双读双比
- 🆕 **ToolRL** 作为 Phase 3 前置阅读（30min，不实现）
- ⚠️ **ReSum** 作为多跳长上下文备选弹药

### 4.3 算力约束的关键转折

用户披露本地无 GPU，需租云。工作流从"本地长跑"切到"本地 + 云端冲刺"：

| 维度 | 决定 |
|---|---|
| 提供商 | AutoDL |
| 基座 | **Qwen2.5-3B-Instruct（全程不升 7B）** |
| 节奏 | 晚间常规型 |
| 训练原子性 | **每 run 一次跑完，不跨夜** |

用户**明确反对**"跨夜续训"的过度工程化设计。

### 4.4 系统体检（12:13）

- Mac arm64（Apple Silicon），236GB 可用磁盘
- 系统 Python 3.9.6 太旧，brew 是 Intel 版（pour ruby 失败），改用 uv 官方 curl 安装
- git 2.50.1 可用

### 4.5 uv + Python 3.10（12:14）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # → ~/.local/bin/uv
uv python install 3.10                             # → 3.10.20
uv venv --python 3.10 .venv
```

### 4.6 烟雾测试通过（12:23）

`scripts/smoke_test.py` mock 掉 `search()`，跑 Qwen2.5-0.5B-Instruct on MPS：

```
[smoke] device=mps, dtype=torch.float16
[smoke] model loaded in 4.0s
[turn 0] generated 79 tokens in 8.1s (9.7 tok/s), last_id=151645
[smoke] hit EOS, agent loop terminated normally.
[smoke] ✅ PASS
```

**值得注意**：未训练的 0.5B-Instruct 模型直接回答 "Paris"，**没有**生成 `<search>` 标签 — 这正是 RL 要解决的问题。烟雾测试验证的是 **机制**，不是 **行为**。

---

## 5. 镜像保存策略（云端持久化清单）

AutoDL 有 **三层独立的"持久化"**：

| 层 | 路径 | 保存什么 | 怎么生效 | 风险 |
|---|---|---|---|---|
| **系统盘** | `/`、`/root`、`~` 下一切 | conda env、apt 包、ssh key、shell 配置 | **保存镜像**（手动） | 实例释放 = 系统盘消失，没保存就重装一遍 |
| **数据盘** | `/root/autodl-tmp/` | 模型权重、数据集、代码 | 自动持久（**只要不释放实例**） | 实例**释放**就没了；**关机**没事 |
| **文件存储** | `/root/autodl-fs/` | 跨实例共享数据 | 自动持久，**与实例无关** | 单独按 GB 计费 |
| **百度网盘** | 挂载点 | 长期归档备份 | 永久（百度那边的事） | 速度慢，不能直接训练读 |

> **"保存实例" ≠ 接百度网盘**。前者是环境快照，后者是数据备份，互补不替代。

### 5.1 推荐保存的 3 个镜像版本（免费槽位 3 个，正好用满）

| 镜像名 | 存的时间点 | 作用 |
|---|---|---|
| `searchr1-phase-1-base` | **Phase -1 收尾后** | 装好的 vllm/verl/flash-attn 全栈环境，灾难恢复用 |
| `searchr1-phase-1-trained` | 第一次 GRPO 跑通后 | 有跑通过的训练脚本配置；checkpoint 不在镜像，单独走数据盘+网盘 |
| `searchr1-phase-3-tool` | 切到 ToRL 阶段（code interpreter）后 | 工具链改造完的环境 |

### 5.2 保存镜像的操作步骤

```
1. 等关键作业彻底跑完（下载/训练）
   ↓
2. 清理系统盘垃圾（释放镜像大小）：
      conda clean -ay
      pip cache purge
      rm -rf /tmp/* ~/.cache/pip
      rm -rf ~/.cache/huggingface/hub/datasets--*  ← 这些已落 /root/autodl-tmp/
   ↓
3. 控制台 → 实例列表 → 操作列「关机」（注意：是关机，不是释放！）
   ↓
4. 等实例状态变成「已关机」（约 30s-2min）
   ↓
5. 同一行 → 「更多」→ 「保存镜像」
   → 名字：searchr1-phase-1-base
   → 描述：torch 2.4.0+cu121 / vllm 0.6.3 / verl 0.1 / flash-attn / hf cache cleared
   ↓
6. 等 5–30 分钟（按系统盘占用大小）
```

⚠️ 保存期间实例锁死，所以选**完成阶段性里程碑**时做。

### 5.3 五个常见坑

| 坑 | 表现 | 防御 |
|---|---|---|
| 没关机就保存 | 控制台拒绝 | 必须先关机 |
| 误点"释放"代替"关机" | 数据盘 80GB 全没 | **关机 ≠ 释放**，看清按钮 |
| 系统盘满（默认 30–50GB）保存失败 | 容量不足 | `du -sh ~/* /root/* | sort -h` 找大头，搬到 `/root/autodl-tmp/` |
| 在 `/root` 下放大模型权重 | 镜像爆 | 大文件全部 `/root/autodl-tmp/`，用软链 `ln -s` 暴露给系统 |
| 把镜像当数据备份 | 镜像专门排除大数据 | 数据走数据盘 + 不释放，或传百度网盘归档 |

### 5.4 推荐的"完整持久化策略"

```
[ 系统环境 ]   →  保存镜像（关键里程碑）
[ Wiki 数据 ]  →  数据盘 /root/autodl-tmp/data/  (实例不释放就行)
[ 训练代码 ]   →  GitHub 推送（本地+云双向 git push/pull）
[ checkpoint ] →  训练完先放数据盘，重要的传百度网盘归档
[ wandb log ]  →  自动云端，不用管
```

**百度网盘的角色 = "checkpoint 长期归档"**，不是首选实时存储（速度慢、训练时不能直接读）。
