# agenicRL — 端到端 Agentic RL 学习项目

> 通过一个真实场景的训练（**Search-R1**），把 Agentic RL 的技术栈学透。
> 过程式记录、决策可追溯、每一次实验对照一篇论文。

## 项目快览

- **场景**：Search-R1 风格——LLM 交错进行推理与搜索引擎工具调用，奖励 = QA EM
- **数据集**：NQ (Natural Questions) — 79168 train / 3610 val
- **基座模型**：Qwen2.5-3B-Instruct
- **算法路径**：DeepSeek-R1 → DAPO + GSPO → ToolRL/ToRL → GiGPO → ARPO
- **训练框架**：verl (Search-R1 fork) + SkyRL（PRO 6000 适配）
- **评估**：val/test_score/nq (binary EM)

## 已完成阶段（Phase 0-5 on 4090×4）

| Phase | 算法 | 主要变更 | 终点 val/nq |
|-------|------|---------|--------------|
| 0d | GRPO baseline | base | 0.196 (val_before_train) |
| 0e | GRPO + format reward | DFA 4层格式 | ~0.40 |
| 1 | DAPO clip-higher | clip_high=0.28 (token) | -1.7pt vs 0d |
| 2A/2B | GSPO sequence | clip=[0.2, 0.28] + sequence agg | **0.438** ⭐ |
| 3 | ToRL shaping | +per_turn_hit_bonus | 0.380 (real EM, 含 shaping) |
| 4 | GiGPO step-credit | alpha=0.7 step-level adv | ~0.41 |
| 5a/5b | ARPO trajectory credit | adapt_ent | ~0.42 |

> 详见 [飞书 Phase 0-5 复盘文档](https://my.feishu.cn/docx/SDH3djw3Gou01nx9XBhcFYTOnLb)

## Phase 6 = E (vanilla clip ablation) 三机并跑

实验目标：删除 GSPO 的 `clip_high=0.28` asymmetry，看 vanilla clip [0.2, 0.2] 是否影响最终分。

| 机器 | Stack | step time | 状态 |
|---|---|---|---|
| **4090×4** (autodl-pro6k-v2) | verl Search-R1 fork | 7.4 min | ✅ **完成 final val=0.4314**（vs baseline 0.438，-0.7pt 中性）|
| **3090×4** (autodl-4090-v3) | verl Search-R1 fork | 16.9 min | 🔄 step 25+ 跑中 |
| **PRO 6000 ×1** (autodl-pro6k-blackwell) | SkyRL | 17.8 min | ⏸ 4 步采样后 OOM (reward 0.14→0.30 healthy) |
| **PRO 6000 ×2 dual** (autodl-pro6k-dual) | SkyRL Plan A | ~5-7 min（估）| ⏳ 装机中 |

### Phase 6 = E 结论

**clip_high asymmetry 实质中性**（4090 0.4314 vs baseline 0.438，-0.7pt 在 NQ 噪声范围内）。GSPO 的 `clip_high=0.28` 不是关键改动。

## 多机部署经验

### Hardware variance（同 model 同 data 跨硬件）
- 4090 (sm_89): val_before_train = **0.196**
- 3090 (sm_86, AutoDL 改装 48GB 版): val_before_train = **0.2143**（+1.87pt）
- PRO 6000 (sm_120): val_before_train = 0.143（SkyRL stack 不同，跨 stack 不可比）

bf16/fp16 浮点非结合律 + 不同 GPU kernel 导致 logits 末位差异 → token argmax 翻转 → 轨迹发散。**结论**：跨硬件比较看 delta 不看 absolute。

### PRO 6000 Blackwell 装机踩坑（5 大发现）

1. **`cudaMalloc` 单次硬限 30 GiB**（驱动级别，非显存不足）
   - 60GB FP32 e5_Flat 索引装不进 → OOM
   - **修复**: faiss `StandardGpuResources::allocMemory` patch — `cudaMalloc` 失败时 fallback 到 `cudaMallocManaged`
2. **PyPI 直连慢 / 抽风** → 切换清华镜像 `UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple`
3. **megatron-bridge 要 cublas 13** wheel 不存在 → `--no-install-package megatron-*`
4. **`/root/.cache/uv` 填爆系统盘 30GB** → 挪到 `/root/autodl-tmp/.cache/uv`
5. **AutoDL 自定义服务 默认只映射 :6006 / :8008** 公网（不是 :8000）→ retrieval_server.py 改监听 :6006

### Phase 6 OOM 调试历史（SkyRL on PRO 6000）
- `flash_attn=true`: 收益仅 -7%，加 OOM 风险，弃
- `PYTORCH_ALLOC_CONF=expandable_segments`: 容器 seccomp 屏蔽 `pidfd_getfd` 系统调用，弃
- baseline (`micro_batch=8`) 反复 OOM 于 `chunked_entropy_from_logits` 4.64GB alloc
- `micro_batch=4`: entropy chunk 减半至 2.3GB，**4 步稳定跑过**（step 5 仍偶发 OOM）

### 双 PRO 6000 方案 A（Plan A）

```
GPU 0 (96GB)              GPU 1 (96GB)
├ retrieval shard 0       ├ retrieval shard 1
│  e5_Flat FP16 ~15GB     │  e5_Flat FP16 ~15GB
├ vllm (TP=2 part 0)      ├ vllm (TP=2 part 1)
└ FSDP shard (actor+ref)  └ FSDP shard (actor+ref)
```

每 GPU 占 ~50GB / 96GB，余 46GB headroom 充足。每卡 15GB FP16 alloc 远低于 30GiB cap，**不需要 cudaMallocManaged 补丁**。

## 4090 ↔ PRO 6000 SkyRL 超参对齐表（subagent 验证）

### 完全等价
| 4090 verl | SkyRL key | 状态 |
|---|---|---|
| `policy_loss.loss_mode=vanilla` | `algorithm.policy_loss_type=regular` | ✅ |
| `loss_agg_mode=sequence` | `algorithm.loss_reduction=sequence_mean` | ✅ |
| `kl_loss_type=low_var_kl` | `algorithm.kl_estimator_type=k3` | ✅ Schulman k3 |
| `clip_ratio_low=0.2`, `clip_ratio_high=0.2` | `eps_clip_low/high=0.2` | ✅ |
| `model.enable_gradient_checkpointing=true` | `trainer.gradient_checkpointing=true` (默认) | ✅ |
| `n_agent=5` | `generator.n_samples_per_prompt=5` | ✅ |
| `lr=1e-6, lr_warmup_steps_ratio=0.285` | `lr=1e-6, num_warmup_steps=15` | ✅ |
| `train_batch_size=512, ppo_mini_batch=256` | 同 | ✅ |
| `state_masking=true` (mask `<information>`) | env 自动 mask obs_ids | ✅ 隐式 |

### 等价但 key 名不同
| 4090 verl | SkyRL key | 备注 |
|---|---|---|
| `model.use_remove_padding=True` | `trainer.use_sample_packing=true` (要 `flash_attn=true`) | 不同机制同效果 |
| `total_training_steps=52` | `epochs=1 + train_data 子集 26624 行` | workaround |
| `max_start_length=2048` | `trainer.max_prompt_length=4096 + generator.max_input_length=4096` | 分两个 key |
| `VLLM_ATTENTION_BACKEND=XFORMERS` | `+generator.engine_init_kwargs.attention_backend=xformers` | 配置传透 |

### 不可完全对齐
| 4090 verl | SkyRL 状态 | 影响 |
|---|---|---|
| `data.max_obs_length=500` (search obs token cap) | ❌ **无原生**，需写补丁 | retrieval doc 不截断 → reward 行为可能差 1-5pt |
| `algorithm.no_think_rl=false` (强制 `<think>` tag) | ❌ 无强制 | 但 4090 baseline 实际用 binary EM，无 think 格式分 → **0pt 影响** |

## 仓库导航

```
agenicRL/
├── docs/                    # 决策、架构、阶段产物文档
├── notes/                   # 跨阶段总结、夜跑日志
├── patches/                 # GSPO/DAPO/ARPO 等 verl 算法补丁
├── scripts/                 # 各阶段训练 launcher
│   ├── phase0_train_grpo.sh
│   ├── phase1_train_grpo_format.sh
│   ├── phase2_train_gspo.sh         # ⭐ Phase 6 = E 4090 launcher（CLIP_HIGH=0.2）
│   ├── phase3_train_torl.sh
│   ├── phase4_train_gigpo.sh
│   ├── phase5a_train_arpo.sh
│   ├── skyrl_phase6_pro6k_full_index.sh   # SkyRL on PRO 6000 #1
│   ├── skyrl_phase6_pro6k_smoke.sh
│   ├── convert_nq_to_skyrl.py             # NQ→SkyRL parquet 转换
│   └── auto_chain_*.sh                    # 阶段间自动接力
├── data/                    # 训练数据（gitignore）
└── external/                # verl / Search-R1 / SkyRL 子模块（gitignore）
```

## 关键飞书文档
- [Phase 0-5 复盘](https://my.feishu.cn/docx/SDH3djw3Gou01nx9XBhcFYTOnLb)（含 Eval 公平性 audit）
- [PRO 6000 训练复盘](https://my.feishu.cn/docx/K2dfdv1RUofmycxG8BjcA7MinXf)（含 Stack A/B/C/D 多版尝试）

## 当前进度

- [x] Phase 0–5：全部跑通（4090×4，verl Search-R1 fork）
- [x] 飞书复盘文档（含 eval 公平性、PRO 6000 多版尝试）
- [x] **Phase 6 = E（4090）**：完成，0.4314 vs baseline 0.438（中性结论）
- [ ] **Phase 6 = E（3090）**：step 25+/52 跑中
- [ ] **Phase 6 = E（PRO 6000 dual）**：装机中
- [ ] Phase 7：max_turns 2→4
- [ ] Phase 8：max_obs 500→800
- [ ] Phase 9：6/7/8 胜者组合

## 本地预览文档站（可选）

```bash
pip install mkdocs-material
mkdocs serve   # http://127.0.0.1:8000
```
