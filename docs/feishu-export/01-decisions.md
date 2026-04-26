# 决策档案 · 为什么这样做

> **范围**：Search-R1 场景选择 / 论文路径 / 算力配置 — 三大根决策的"为什么"和"备选权衡"
> **来源**：master agent · `docs/decisions/{scenario,curriculum,compute}.md`
> **更新原则**：决策一旦敲定不轻易动；变更时在文末追加"修订记录"

---

## 1. 场景选择 · Search-R1

### 1.1 决策

**场景**：Search-R1 路径 — 训一个 LLM 交错进行推理与搜索引擎工具调用，奖励 = QA outcome 正确性（EM/F1）。

### 1.2 备选与权衡

| 类别 | 代表论文 | 优点 | 痛点 | 评价 |
|---|---|---|---|---|
| **工具调用**（搜索/计算器/code） | ToRL, ARPO, GiGPO | 奖励信号干净、单 GPU 可跑 | "不够 agentic" | ✅ **选中** |
| Web/GUI Agent | WebArena 系 | 最像真 agent | env 工程量巨大 | ❌ env 成本太高 |
| 长程推理 | DeepSeek-R1, DAPO | 基础设施最成熟 | 已被刷满 | ❌ 与本目标错位 |

### 1.3 为什么是 Search-R1（而非 ToRL/ToolRL）

Search-R1 处在全景图的精确交叉点：

- **算法骨架** = DeepSeek-R1 的 GRPO（推理路径）
- **工具范式** = ToRL/ToolRL（工具使用路径）
- **奖励信号** = QA outcome（最干净的一类）

更关键的是：

- 官方代码基于 **verl**，verl 又是 DAPO/GSPO 的官方实现框架
- 后续读 DAPO/GSPO 论文时，改进可以直接 patch 进同一份代码
- A/B 对比成本极低

### 1.4 核心技术栈默认值

| 决策 | 默认 | 备选 |
|---|---|---|
| 基座模型 | **Qwen2.5-3B-Instruct** | Qwen2.5-7B / Llama-3.2-3B |
| 检索后端 | 本地 Wikipedia + E5 dense + FAISS | SerpAPI / BM25 |
| 数据集 | NQ + HotpotQA + 2Wiki + Musique + Bamboogle | 单一数据集 |
| 奖励 | 纯 outcome (EM 主) | + format / + step PRM |
| 算法 | GRPO（vanilla）→ DAPO/GSPO → GiGPO → ARPO | PPO |
| 训练框架 | **verl** | OpenRLHF / TRL |
| 生成引擎 | **vLLM** | SGLang |

---

## 2. 学习路径 · 5 主干 + 2 增补

### 2.1 全景图覆盖矩阵

参照 *The Landscape of Agentic RL for LLMs: A Survey* (TMLR 2026)，17 篇必读论文，5 个能力维度。本项目主干覆盖 **5/17 ≈ 30%**：

| 维度 | 论文 | 是否主干 | 备注 |
|---|---|---|---|
| **Reasoning** | DeepSeek-R1 | ✅ | GRPO 算法核心 |
| | DAPO | ✅ | 算法改进的标准对照 |
| | **GSPO**（Qwen3） | 🆕 增补 | Qwen3 官方推荐配方，序列级 IS |
| | Satori | ❌ | 偏推理蒸馏，与搜索场景关系弱 |
| **Tool Use** | ToolRL | 🆕 预读 | 30 分钟，故事线开篇 |
| | ARTIST | ❌ | 已被 ToRL 吸收 |
| | ToRL | ✅ | code-tool RL 标杆 |
| | GiGPO | ✅ | 步级信用分配 |
| | ARPO | ✅ | 前沿，熵驱动分支 |
| **Planning** | RAP / LATS / VOYAGER / Planner-R1 | ❌ | Search-R1 不需要 MCTS |
| **Memory** | Memory-R1 / MemAgent | ❌ | 跨 episode 无记忆需求 |
| | ReSum | ⚠️ 待命 | 多跳长上下文撞墙时启用 |
| **Self-Improvement** | Reflexion / Absolute Zero / SiriuS | ❌ | outcome reward 直接，不需反思 |

### 2.2 路径偏置说明

本路径其实把全景图的"**能力维度**"框架悄悄换成了"**算法演进**"框架：

> - Landscape 问的是："agent 能做什么？" → 5 个能力维度
> - 我们的路径问的是："policy 怎么优化？" → GRPO 升级链

对 Search-R1 这个具体场景，这个偏置是合理的 — Search-R1 = Reasoning 主干 ∩ Tool Use 主干，Planning/Memory/Self-Improvement 在此场景非核心。

### 2.3 论文 → 代码改动 映射

每读一篇论文，对应一次具体的代码改动：

```
Phase 0 │ 跑通 Search-R1 baseline                      │ 学 verl + GRPO 工程骨架
Phase 1 │ 读 DeepSeek-R1                               │ 注释 verl/grpo 代码，画 advantage 流
Phase 2 │ 读 DAPO + GSPO（双读双比）                    │ DAPO 4 项改进 / GSPO 序列级 IS A/B
Phase 3 │ 读 ToolRL（30min 前置）→ ToRL                │ 对比 search-as-tool 与 code-as-tool 奖励设计
Phase 4 │ 读 GiGPO                                     │ 改造为 step-level advantage
Phase 5 │ 读 ARPO                                      │ 引入熵驱动 rollout 分支
```

### 2.4 战略性决策记录

#### 为什么把 GSPO 加进主干

- Qwen 系基座 + GSPO 是 Qwen3 官方对齐配方
- 序列级 importance sampling 在 agent 长序列场景比 DAPO 的 token-level 更稳
- Phase 2 改成 "DAPO + GSPO 双读双比"，是一次有研究价值的对照实验

#### 为什么 ToolRL 只读不实现

- 直接从 DAPO 跳到 ToRL 会缺 "为什么 RL 能让模型自发学会调用工具" 的认知锚点
- ToolRL 是开篇文献，1 小时投入即可形成对照感

---

## 3. 算力约束 · AutoDL + 4× RTX 4090 48GB + 3B

> ⚠️ **2026-04-26 晚订正**：原决策按 `4×A100 80G` 估算，实际开机看 `nvidia-smi` 是 `RTX 4090 / 49140 MiB` × 4。AutoDL 的 "vGPU-48GB" 商品名实际是**物理改装版 4090**（AD102 核心 + 48GB GDDR6X，无 NVLink），不是虚拟切分。预算因此从 ¥8K 调到 **¥1.5K**。

### 3.1 决策清单

| 维度 | 决定 | 备注 |
|---|---|---|
| 提供商 | **AutoDL** | 国内、便宜、持久化网盘成熟 |
| 卡型 | **4× RTX 4090 48GB** | AutoDL "vGPU-48GB"，物理改装版 4090，无 NVLink |
| 基座 | Qwen2.5-3B-Instruct | **全程不升 7B**，48GB 单卡 + rollout 显存吃紧时容错小 |
| 节奏 | 晚间常规型 | 大多数晚上不开机 |
| 训练原子性 | **每 run 一次跑完** | 不跨夜续训，避免过度工程 |
| 预算硬上限 | **¥1.5K**（按 4090 重估） | 比原估算便宜约 5× |

### 3.1' 卡型差异速查

| 维度 | 原假设 A100 80G | 实际 4090 48GB |
|---|---|---|
| 单卡显存 | 80GB | 48GB（够 3B，**7B 边缘**） |
| 4 卡总显存 | 320GB | 192GB |
| FP16 算力 | 312 TFLOPS | **330 TFLOPS** ✅ 反而更快 |
| 显存带宽 | 2 TB/s HBM2e | 1 TB/s GDDR6X |
| 多卡通信 | NVLink 600 GB/s | **仅 PCIe 5.0** ❌ DDP/rollout 同步慢 |
| 价格 | ~¥45/h | **~¥4–8/h** |

**净结论**：算力够用、更快、更便宜，但多卡通信是新瓶颈，3B 上限更必要。

### 3.2 本地 vs 云：工作区分裂

| | **本地（笔记本）** | **云 GPU 实例** |
|---|---|---|
| 干什么 | 写/改代码、读论文、看日志、画图、git | **只做训练** — 拉代码、训、存网盘、关机 |
| 算力 | CPU 即可 | 4× RTX 4090 48GB |
| 持续时间 | 全天候 | 短爆发（4–8h/次） |
| 同步方式 | git push | git pull + 持久化网盘 |

**核心准则**：**实例上永远不写新代码**。

### 3.3 训练节奏的两类晚上

| 类型 | 是否租 GPU | 时长 | 内容 |
|---|---|---|---|
| 研发夜（多数） | ❌ | 1–3h | 写代码、读论文、看上次日志、设计实验 |
| 训练夜（少数） | ✅ | 一次性 4–8h | 启动一次完整实验，跑完关机 |

### 3.4 ⚡ AutoDL 学术加速（关键技巧）

> 这条几乎所有 AutoDL 教程不写但**必须做**。

国内直连 `download.pytorch.org` / `huggingface.co` / `pypi.org` 有时极慢甚至超时。AutoDL 自带学术加速代理：

```bash
source /etc/network_turbo
# 这一行设置 http_proxy=http://10.37.1.23:12798
# 之后所有 pip / git / hf_hub_download / curl 都自动走加速
```

**实测教训**：第一次装 torch 没 source 这个，pip 卡了 9 分钟无进展。source 之后秒级响应。

**含义**：
- `cloud_setup.sh` 第一行就 `source /etc/network_turbo`
- 每次新开 SSH session 装东西，都要先 source（不持久）
- 仅限学术用途、AutoDL 不承诺稳定性，但实测对 PyPI/GitHub/HF 都生效

### 3.5 无卡模式 = 省钱设置阶段

AutoDL 实例可以在两个模式间切换：

| 模式 | 价格 | 用途 |
|---|---|---|
| 无卡模式 | ~¥0.5/h | 装环境、下数据、写代码、debug |
| 4× RTX 4090 48GB 满载 | ~¥4–8/h | **只在真训练时切过去** |

**关键流程**：
1. 创建实例时进入无卡模式
2. 装好环境、下完模型 + 数据集 + index 到持久化盘
3. 关机 → 改"开机（带卡）" → 训练
4. 训练完 → 关机 → 切回无卡模式分析日志

**SSH 端点会变**（实测三次：`47992 → 45432 → westd:45432 → westb:35726`）：每次切模式后从 AutoDL 控制台拷新 SSH 命令，更新 `~/.ssh/config` 的 `HostName` 和 `Port`。

### 3.6 持久化盘 vs 系统盘

| 路径 | 类型 | 是否持久 | 放什么 |
|---|---|---|---|
| `/root/`、`/root/miniconda3/` | 系统盘 | ✅ 跨停启 / ⚠️ 重置系统会丢 | conda env、个人配置 |
| `/root/autodl-tmp/` | 持久化网盘（350GB） | ✅ **任何情况都不丢** | 代码、模型权重、数据集、index、ckpt |
| `/autodl-pub/` | 公开数据集（7.3TB 只读） | 共享 | 不动 |

工作区 `/root/autodl-tmp/agenicRL/` 在持久化盘上，干净。

### 3.7 烟雾测试制度

每次"真训练"前，**强制 5 分钟 smoke test**：

```yaml
smoke_test:
  model: Qwen2.5-0.5B（或 3B 子集）
  gpus: 1
  max_steps: 10
  目标: 不求收敛，只求"不崩"
```

5 分钟 ≈ ¥0.5 的成本，挡掉很多 4 卡 24 小时翻车。

### 3.8 成本预估（3B 配置）

按 AutoDL 4× RTX 4090 48GB ≈ **¥6/h**（取 ¥4–8/h 中位数）。"PCIe 慢系数 1.3" 是估算，因为 4090 没 NVLink，DDP all-reduce + GRPO rollout 同步比 A100+NVLink 慢约 30%；Phase 0 跑通后用墙钟时间替换：

| Phase | 单次时长（×1.3） | 跑几次 | GPU-小时 | 成本 |
|---|---|---|---|---|
| P0 baseline | 5–8h | 3–4 | 20–32 | ¥0.12K–0.19K |
| P2 DAPO+GSPO | 5–8h | 4–5 | 25–40 | ¥0.15K–0.24K |
| P3 ToRL | 6–10h | 2–3 | 14–30 | ¥0.08K–0.18K |
| P4 GiGPO | 8–10h | 2–3 | 20–30 | ¥0.12K–0.18K |
| P5 ARPO | 8–13h | 1–2 | 13–26 | ¥0.08K–0.16K |
| **小计** | | | **92–158h** | **¥0.55K–0.95K** |
| 翻车冗余 ×1.3 | | | | **¥0.7K–1.25K** |

实际预算 **¥1K 上下**，硬上限 **¥1.5K**（原按 A100 估算的 ¥5K / ¥8K 已下调）。
