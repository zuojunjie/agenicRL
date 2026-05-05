# agenicRL — 端到端 Agentic RL 学习项目

> 通过真实训练场景把 Agentic RL 技术栈学透。<br/>
> 过程式记录、决策可追溯、每一次实验对照一篇论文。

---

## 当前状态（2026-05-06）

**NQ 路线（Phase 0-9）✅ 已收敛**

| Phase | 主题 | val/nq | Δ vs 0e | 结论 |
|---|---|---|---|---|
| 0d | GRPO + EM 二元 reward | 0.376 | -10% | baseline 起源 |
| **0e** | + DFA 5 档 reward | **0.417** | (基线) | ✅ 主基线 |
| 1 | 读 R1 论文（0 步） | — | — | ✅ |
| 2A | DAPO clip-higher | 0.400 | -4% ❌ | entropy 4.92 爆炸 |
| **2B** | GSPO sequence-level | **0.4378** | +5% ✅ | 3B 第一个赢家 |
| 3 | ToRL multi-turn shaping | 0.380 | -9% ❌ | reward hacking |
| 4 | GiGPO 双层 advantage | 0.292 (kill@40) | -27% ❌❌ | 全 phase 最差 |
| 5a/5b | ARPO 双子实验 | 0.357 / 0.254 | -14% / -37% ❌ | adapt entropy ✅，其他 ❌ |
| 6 | 4090→PRO 6000 迁移调通 | ~0.42 | 等价 | smoke OK |
| 7 系列 | 4090→PRO6K 调优（9 子实验） | — | — | 二分定位 mini_batch 元凶 |
| **7.8d** | paper GSPO + asym clip | **0.4490** | +7.7% ✅ | ⭐ **3B SOTA** |
| 7.8e | + format reward | 0.4296 | +3.0% | NQ 上反而 hurt |
| 8 | max_turns=4 实验 | 0.4475 | +7.3% | 0 收益（NQ 单跳本质）|
| **9** | **Qwen2.5-7B + dedicated retrieval (3 GPU)** | **0.4701** | **+12.7% ⭐⭐** | **7B SOTA, +140% from base** |

**Tool Use 路线（Phase 10-15）🟡 启动中** — NQ 已到顶，切换到 MATH + Python sandbox（agentic 本色）

| Phase | 主题 | 核心改动 | 状态 |
|---|---|---|---|
| **10** | MATH + Python (ToRL/TIR) | Qwen2.5-Math-7B-Instruct + paper GSPO + max_turns=8 + Python 沙盒 | 🟡 代码+数据就绪，等 GPU |
| 11 | ToRL R1-Zero | Math-Base + per-turn shaping | ⏸ |
| 12 | ARTIST | outcome-based + multi-turn 信用分配 | ⏸ |
| 13 | GiGPO 重试 | task-level 多跳让 outer adv 有意义 | ⏸ |
| 14 | ARPO entropy-adaptive rollout | 继承 5a 验证有效的 adapt entropy | ⏸ |
| 15 | ToolRL multi-tier reward | 多层级工具 reward 设计 | ⏸ |

---

## 累积进步图

```
Qwen2.5-3B-Instruct (val 0.196 zero-shot)
     │ +0.180  GRPO 算法本身
0d   0.376
     │ +0.041  DFA 5档 reward
0e   0.417   ⭐
     │ +0.021  GSPO sequence-level
2B   0.4378
     │ +0.011  paper GSPO + asym clip
7.8d 0.4490  ⭐⭐ 3B SOTA
     │ +0.021  3B → 7B
9    0.4701  ⭐⭐⭐ 7B SOTA (+140% from base)
```

---

## 飞书文档导航

完整知识库：https://my.feishu.cn/wiki/Rjmqwq3jNiIXG8kp9gtcOghQnpb （AgenicRL · 项目知识中枢）

### 总览（3 篇）

- [项目知识中枢](https://my.feishu.cn/wiki/Rjmqwq3jNiIXG8kp9gtcOghQnpb) — 入口
- [学习路线图（Phase 0 → Phase 9）](https://my.feishu.cn/wiki/SZOVwhpCNiDdppkgCTgcRkeYnRb)
- [5-Phase 完整实验总结](https://my.feishu.cn/wiki/RMKAw01Jnify8xkFVnXcYRv0ngb)

### Phase 单篇（10 篇）

| Phase | wiki | 一句话 |
|---|---|---|
| 0e | [link](https://my.feishu.cn/wiki/UURuwlRVwilOijk25r0caxmMnde) | GRPO + Format DFA Reward；主基线 0.417 |
| 2A | [link](https://my.feishu.cn/wiki/Qad4wYt8bio5dnkV2I8cLNx8nLe) | DAPO clip-higher；entropy 4.92 爆炸 |
| 2B | [link](https://my.feishu.cn/wiki/BVjGwO2yiibguEkSOxLcu783nDe) | GSPO sequence-level；3B 第一个赢家 |
| 3 | [link](https://my.feishu.cn/wiki/QEiowgxCuiz1VwkFKHncI7LDnAt) | ToRL multi-turn shaping；reward hacking |
| 4 | [link](https://my.feishu.cn/wiki/LiHZwgr5tiy1bCkBlprcT5H4npZ) | GiGPO 双层 advantage；kill@40 全 phase 最差 |
| 5 | [link](https://my.feishu.cn/wiki/EJzIwAtS7i9cXCkGuLjcchZin5d) | ARPO 5a + 5b；adapt entropy ✅，traj credit ❌ |
| 6 | [link](https://my.feishu.cn/wiki/NDMewxHVkiVuQHkvLYqc5b6Nnuf) | 4090→PRO 6000 迁移调通 |
| 7 | [link](https://my.feishu.cn/wiki/MvZHwhdvLivewjkYHnuc5D0Dnhe) | PRO6K 调优大表；7.8d 3B SOTA |
| 8 | [link](https://my.feishu.cn/wiki/PU23wyY53idJcOkOPguc5HBCnjd) | max_turns=4 实验；NQ 单跳证伪 |
| 9 | [link](https://my.feishu.cn/wiki/QdUTwoGEUiYEkMkZYwHcG3L5nVh) | Qwen2.5-7B + dedicated retrieval；7B SOTA 0.4701 |

### 路线规划

- [Tool Use 优化链复现计划（Phase 10-15）](https://my.feishu.cn/docx/CF15d0YYVoK563xskSfcC2cTnDd)

---

## 已部署线上服务

| 服务 | 地址 | 用途 |
|---|---|---|
| Wikipedia 检索 HTTP | `https://u983007-b268-7846df3b.westd.seetacloud.com:8443/retrieve` | 给本地 talkAgent 项目用；e5-base-v2 + IVFPQ + Wikipedia 21M passages，~600ms/query |
| 持久化 uv venv | `source /root/autodl-tmp/venvs/activate.sh` | 跨实例复用；用 SkyRL `uv.lock` 严格 pin |

---

## 五大工程级 takeaway（Phase 0-9 总结）

1. **Reward 函数设计 >> PPO 算法改进** — DFA 5档（0e）比 EM（0d）+11%；后续所有 PPO 内部改动最多 +5%（GSPO）
2. **Entropy 是 RL 训练命脉** — DAPO/GiGPO/5a/5b 全部死于 entropy 失控；GSPO sequence-level 自然控制
3. **Reward shaping 是 hacking 陷阱** — ToRL 的 per-turn shaping 让模型学会刷虚分
4. **硬件迁移要 single-variable 校准 baseline** — Phase 7.5 改 9 个变量直接挂；二分定位 mini_batch_size 是元凶
5. **Capacity scale 是稳定杠杆** — 算法定型后 3B → 7B 是性能突破最直接路径（+2.1% vs 算法 +1.1%）

---

## 硬件演进时间线

```
Phase 0-5 (4090 时代)     2026-04-22 → 04-30
  4× RTX 4090 vGPU-48GB    ¥11.52/h
  Qwen2.5-3B-Instruct
  IVFPQ 索引 (1.5 GB)
  vllm 0.6.x + verl
     ↓ 切硬件 + 数据盘跨实例继承
Phase 6-8 (PRO 6000 单卡)  2026-05-01 → 05-03
  1× RTX PRO 6000 96GB     ¥5.98/h
  Qwen2.5-3B-Instruct
  Flat 索引 (60 GB GPU)
  vllm 0.8.x v1 async + SkyRL
     ↓ 升模型规模
Phase 9 (PRO 6000 三卡)    2026-05-03 → 05-04
  3× RTX PRO 6000 96GB     ¥17.94/h
  Qwen2.5-7B-Instruct
  GPU 0,1 = train + vllm
  GPU 2 = dedicated retrieval HTTP service
     ↓ 切 task: NQ → MATH+Python
Phase 10 (PRO 6000 双卡)   2026-05-05 → 进行中
  2× RTX PRO 6000 96GB     ¥11.96/h（无卡装环境时 ¥0.5/h）
  Qwen2.5-Math-7B-Instruct
  Python sandbox + sympy verifier
```

**总投入 ≈ ¥1250**（截至 2026-05-05）

---

## 仓库布局

```
agenicRL/
├── README.md                          # 本文件
├── notes/                             # 论文 + 学习笔记
│   ├── all_phases_summary.md          # 5-Phase 总结底稿
│   ├── agent_rl_trl_reading_d1.md     # Agent RL + TRL 阅读笔记
│   ├── phase{0e,1,2,3,5}/             # 各 phase 论文/diff/对比
│   └── runs/                          # phase 运行档案
├── scripts/                           # 训练 / 部署 / 工具脚本
│   ├── skyrl_phase{6,7,7.5,...,9,10}*.sh    # SkyRL 时代 launcher
│   ├── phase{0,1,2,3,4,5a,5b}_train_*.sh    # verl 时代 launcher
│   ├── launch_retrieval_*.sh                # FAISS retrieval server 启动
│   ├── auto_chain_*.sh / auto_shutdown_*.sh # 自动化链路
│   ├── prep_data.py                         # 数据 prep（GSM8K + MATH + MATH-500）
│   ├── cloud_setup.sh                       # AutoDL 实例 env 一键搭建（旧 conda）
│   └── feishu_uploader.py                   # 飞书 OpenAPI 上传（已弃用，留作参考）
├── skyrl_patches_phase10/             # Phase 10 (MATH+Python) 代码包
│   ├── deploy.sh                            # SCP 部署到远端
│   ├── prep_data.py                         # GSM8K + MATH + MATH-500 → SkyRL parquet
│   ├── python_sandbox.py                    # Python 沙盒工具（subprocess + 资源限制）
│   └── math_python/                         # math_python skyrl-gym env
│       ├── env.py                                 # MathPythonEnv
│       └── utils.py                               # sympy verifier reward
├── patches/                           # verl 时代 patch (DAPO/GSPO/ToRL/GiGPO/ARPO)
└── docs/                              # 项目早期文档（mkdocs 可渲染）
    ├── decisions/  architecture/  journal/  feishu-export/
    └── ...
```

---

## 快速开始（云端）

### 一次性环境搭建（持久化 uv venv）

```bash
# 1. SSH 到 AutoDL 实例
ssh -p <port> root@connect.westd.seetacloud.com

# 2. 装 uv
curl -fsSL https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH

# 3. 用 SkyRL 自带 uv.lock 严格 pin 安装到 autodl-tmp（持久化）
source /etc/network_turbo
cd /root/autodl-tmp/external/SkyRL
UV_PROJECT_ENVIRONMENT=/root/autodl-tmp/.skyrl-venv \
UV_CACHE_DIR=/root/autodl-tmp/.cache/uv \
    uv sync --frozen --extra fsdp

# 4. 写一键 activate（已在 /root/autodl-tmp/venvs/activate.sh）
source /root/autodl-tmp/venvs/activate.sh
```

### 跨实例复用（autodl-tmp 已带 venv 和 cache）

```bash
source /root/autodl-tmp/venvs/activate.sh
# python / pip / vllm 全部就绪，0 重装
```

### 启动 Phase 10（MATH + Python，2 卡 PRO 6000）

```bash
source /root/autodl-tmp/venvs/activate.sh
bash /root/autodl-tmp/agenicRL/scripts/skyrl_phase10_pro6k_2gpu_math_python.sh
```

---

## 关键算法术语

| 术语 | 一句话 |
|---|---|
| **GRPO** | DeepSeek-R1 核心：同 prompt 采 G 条轨迹，按 reward 归一化得 advantage，token-level 更新 |
| **GSPO** | Qwen IS 粒度改进：token → sequence level（已实证 +5%，最稳赢家）|
| **DAPO** | GRPO 工程化改进 4 件套：动态采样、token-level loss、双 clip、长度衰减（NQ 上失败）|
| **GiGPO** | 双层 advantage（inner + outer 跨 prompt）；NQ 上失败，可能 task-level 多跳能救 |
| **ARPO** | 熵驱动 rollout 分支 + 4 个特性。adapt entropy 单测有效，其他 hurt |
| **ToRL/TIR** | Tool-Integrated Reasoning：把工具调用当 first-class，per-turn shaping reward |
| **SkyRL** | NovaSky 出的 vllm v1 async + ray + FSDP-2 RL 框架（Phase 6+ 用）|
| **verl** | volcengine RL 训练框架（Phase 0-5 用）；Phase 6+ 已切 SkyRL |

更详细见 [飞书项目知识中枢 § Glossary](https://my.feishu.cn/wiki/Rjmqwq3jNiIXG8kp9gtcOghQnpb)。
