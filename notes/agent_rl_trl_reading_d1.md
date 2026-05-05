# TRL 代码精读 Day 1 — GRPO 主线 + PPO 对照

> 日期:2026-05-05  
> 仓库:`/Users/cv/code/agenicRL/external/trl/` (浅 clone)  
> 知识基线:`~/Downloads/agent_rl_notes.md`(Phase A/B/C/D 已通)  
> 模式:精读 GRPO 主线,PPO 作对照

---

## 0. 关键发现:TRL 已把 PPO 移入 `experimental/`

| 文件 | 行数 | 状态 |
|---|---|---|
| `trl/trainer/grpo_trainer.py` | 2732 | ✅ 主线 |
| `trl/trainer/rloo_trainer.py` | 1535 | 主线(GRPO 近亲) |
| `trl/experimental/ppo/ppo_trainer.py` | 1037 | ⚠️ 降级 |

**信号**:无 value model 路线(GRPO/RLOO)正取代 PPO 成为社区主流。原因之一是 PPO 自己手写 train loop,跟 HF Trainer 接口对不上,工程成本高。

---

## 1. GRPOTrainer 调用链总图

```
training_step()                          line 1127     ← HF Trainer 钩子入口
  └─ _prepare_inputs()                   line 1139
       └─ _generate_and_score_completions()  line 1819   ★核心准备阶段
            ├─ env.reset / 拼 prompt(可选)   line 1827
            ├─ _generate()                              line 1693  → vLLM/HF 生成
            │    └─ _tool_call_loop()                   line 1486  → 多轮工具
            ├─ padding(prompt 左,completion 右)
            ├─ no_grad forward × 2:
            │    ├─ old_per_token_logps(条件)  line 2051
            │    └─ ref_per_token_logps(KL≠0)  line 2097
            ├─ _calculate_rewards()                     line 1196
            └─ advantage = (r - mean) / std            line 2146-2168

compute_loss()                           line 2355     ← HF Trainer 钩子
  └─ _compute_loss()                     line 2437     ★算法本身
       ├─ _get_per_token_logps_and_entropies()  → 算 π_θ
       ├─ coef_1 = exp(new_logp - old_logp)         → ratio
       ├─ KL k3 = exp(...) - (...) - 1              → KL 惩罚
       ├─ coef_2 = clamp(coef_1, 1-ε_low, 1+ε_high) → clip
       └─ -min(coef_1*A, coef_2*A) + β*KL           → loss
```

---

## 2. 概念 → 代码位置映射(对照笔记 4 phase)

| 笔记概念 | 文件位置 | 关键变量 |
|---|---|---|
| Phase A · group-relative A | grpo_trainer.py:2146-2168 | `mean_grouped_rewards`, `std_rewards` |
| Phase A · num_generations=G | :551 | 注释 `# = G in the GRPO paper` |
| Phase C · state mask | :2444, :2568 | `loss_mask = completion_mask × tool_mask` |
| Phase C · max_turns | :552, :1486 | `max_tool_calling_iterations` |
| Phase D · ratio | :2497-2509 | `coef_1 = exp(log_importance_weights)` |
| Phase D · clip + min | :2527-2534 | `coef_2 = clamp(...)`, `-min(coef_1*A, coef_2*A)` |
| Phase D · num_iterations=μ | :626 | `# = 𝜇 in the GRPO paper` |
| Phase B · KL k3 | :2514-2516 | `exp(ref - new) - (ref - new) - 1` |
| Rollout · agentic loop | :1486-1691 | `_tool_call_loop` |
| Rollout · soft failure | :1528-1531 | `try/except → {"error": str(e)}` |

---

## 3. 三个工程关键洞察

### 3.1 两个 mask 必须相乘(state mask 实现)

```python
loss_mask = completion_mask × tool_mask
```

- `completion_mask`:挡 padding(右 pad 出的假 token)
- `tool_mask`:挡 tool 返回 token(rollout 时动态构建,line 1665)
- 相乘 = 既不是 padding 又不是 tool 返回的才算 loss

**重要**:`tool_mask` 的 padding 用 1(line 1915),靠 `completion_mask` 把关。两个 mask 的责任分得很干净。

`mask_truncated_completions=True` 时还会把整条截断 trajectory 置零(line 1925)。

### 3.2 `old_per_token_logps` 的"懒计算"+ `.detach()` fallback

```python
# _compute_loss line 2479-2480
old_per_token_logps = inputs.get("old_per_token_logps")
old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
```

当 `num_iterations==1`(每次 rollout 只更新 1 步)时,old_logp == new_logp。代码用 `.detach()` 切梯度但保数值,让:
- `coef_1 = exp(0) = 1`(数值上)
- 梯度 `∂coef_1/∂θ = ∂per_token_logps/∂θ`(只有左边带梯度)
- `-coef_1·A` 退化成纯 policy gradient `-A·∇log π_θ`

→ **同一段代码同时处理 on-policy 和 off-policy**,clip/IS 自动变成无害恒等运算。

→ 但**用了 vLLM 一定要算 old_logp**(vLLM 推理和训练 forward 浮点精度有差,即使权重相同 logp 也差几个 nat)。

### 3.3 GRPO 多卡 advantage 必须先 all-gather 再切片

```python
# line 2188-2194
all_process_advantages = advantages.clone()  # 全 batch 算 mean/std
advantages = advantages[process_slice]       # 切回本进程
```

**原因**:group-relative 必须在全 batch 上算 mean/std,否则单进程凑不齐组。  
**代价**:reward all-gather → 算 advantage → scatter,通信开销。  
**对比**:PPO 用 value model 算 advantage,每进程独立,无通信。

→ GRPO **用通信换显存**。

---

## 4. `_tool_call_loop` 关键点

骨架:while 循环,每轮"执行工具 → append 结果 → 长度检查 → 再生成 → 更新 tool_mask"。

| 你笔记 6.5 节 | TRL 实现 |
|---|---|
| ① 工具协议:tag vs JSON | 全用 OpenAI tools API JSON(line 1518),要求模型 chat template 支持 tool calling |
| ② 停止条件 4 种 | 实现了全部 4 种,且**长度爆了不粗暴 truncate,而是 rollback 这一轮 tool 消息**(line 1599-1604) |
| ③ 软失败 > 硬失败 | ✅ try/except → `{"error": str(e)}` 喂回模型(line 1528-1531) |
| ④ 并发控制 | 同 sample 内 async 工具用 asyncio.gather 并发,**跨 sample 串行**(line 1506) → 大 batch + 重 retrieval 时是瓶颈 |

**核心设计哲学**: `tool_mask` 在 rollout 时动态构建(append 0 或 1),不在训练时用正则反推。优势:不依赖模型输出格式、多模态友好、零成本。

---

## 5. PPO vs GRPO 五大差异对照

| 维度 | PPO | GRPO |
|---|---|---|
| **value model** | ✅ `PolicyAndValueWrapper`(显存翻倍) | ❌ 砍掉 |
| **advantage 算法** | GAE(per-token,时间维度折扣) | (r-mean)/std(per-trajectory,batch 维度对比) |
| **KL 加在哪** | reward 里(`reward = score - β·KL`) | loss 里(`loss + β·KL`) |
| **clip 数量** | 两套(policy ratio + value pred) | 一套(policy ratio) |
| **代码组织** | 手写 train loop(experimental/) | HF Trainer 子类化(主线) |
| **超参** | gamma, lam, vf_coef, cliprange, cliprange_value | num_generations, ε_low, ε_high |
| **跨步信用分配** | ✅ 有(GAE) | ❌ 无(整条共享 A) |
| **跨题难度归一** | ❌ 无(value 估不准就崩) | ✅ 有(组内归一) |
| **去掉 KL 的代价** | 改 reward 计算,易出 bug | β=0,改一行 |

**核心 trade-off**:PPO 用 value 网络自我引导(精致但脆弱),GRPO 用并行采样自我对比(朴素但鲁棒)。本质是**训练算力 vs 推理算力**的转换。

---

## 6. Value Model 是什么 + 怎么训(case)

### 6.1 价值

数学题 trajectory:`"我先减3" → "得到 2x = 8" → "再除以2" → "x = 4"`,r=1.0

- **GRPO**:所有 4 个 step 共享同一个 advantage,**学不到 step 2 算错了**
- **PPO**:value model `V(s_t)` 给每个状态打分 → 看出"step 2 的 value 突然下降" → 精确定位错误

```
trajectory A (成功):  V = [0.5, 0.7, 0.85, 0.95]  → r=1.0
trajectory B (失败):  V = [0.5, 0.7, 0.40, 0.30]  → r=0.0
                                ↑ step 2 之后 value 网络看出走偏
```

→ **per-token credit assignment** 是 PPO 的核心优势,但代价是要同步训练 value 网络。

### 6.2 Value 怎么训(bootstrap,不是循环)

关键澄清:`values` 和 `vpred` 是**两个不同时刻的 value 网络快照**:

| 名字 | 何时算 | 是否带梯度 |
|---|---|---|
| `values` | rollout 时,no_grad,SAVE | ❌ 冻结 |
| `returns = advantages + values` | 紧接着算,no_grad,SAVE | ❌ 冻结(target) |
| `vpred` | 训练循环里 forward,带梯度 | ✅ 更新 |

```python
# 阶段 1(no_grad,冻结)
values = forward(value_model, trajectory)         # SAVE
# 阶段 2(no_grad,冻结)
advantages = GAE(rewards, values, γ, λ)
returns = advantages + values                     # SAVE,作为 target
# 阶段 3(带梯度,反复 forward)
for ppo_epoch in ...:
    vpred = forward(value_model, trajectory)      # 新预测
    vf_loss = (vpred - returns) ** 2              # target 已冻结
    backward()
```

→ **不是循环,是 bootstrap**:target 包含"老 value 预测 + 真实 reward 修正",**比纯老 value 更靠谱**(真实 reward 是新信息)。

### 6.3 数值案例(trajectory A,γ=1, λ=0.95)

```
values  = [0.5, 0.7, 0.85, 0.95]   (老 value)
rewards = [0,    0,   0,    1.0]
GAE → advantages = [0.476, 0.290, 0.1475, 0.05]
returns = advantages + values = [0.976, 0.990, 0.9975, 1.0]

→ 老 V 全估低了!vf_loss = MSE(vpred, returns) → 把所有预测往上推
→ 下次类似 prompt:V 预测 [0.55, 0.75, 0.88, 0.97](更准)
```

trajectory B (失败,r=0):

```
values  = [0.5, 0.7, 0.4, 0.3]
GAE → advantages = [-0.433, -0.666, -0.385, -0.3]
returns = [0.067, 0.034, 0.015, 0.0]

→ 老 V 全估高了!vf_loss → 把所有预测往下推
→ 下次类似 prompt:V 预测更低
```

---

## 7. GAE 数值案例(同一个 trajectory)

```
values  = [0.5, 0.7, 0.85, 0.95]
rewards = [0,    0,   0,    1.0]
γ = 1.0, λ = 0.95

倒序循环:
t=3: δ = 1.0 + 0 - 0.95 = 0.05,    A_3 = 0.05
t=2: δ = 0   + 0.95 - 0.85 = 0.10,  A_2 = 0.10 + 0.95·0.05 = 0.1475
t=1: δ = 0   + 0.85 - 0.7  = 0.15,  A_1 = 0.15 + 0.95·0.1475 = 0.290
t=0: δ = 0   + 0.7  - 0.5  = 0.20,  A_0 = 0.20 + 0.95·0.290  = 0.476

advantages = [0.476, 0.290, 0.1475, 0.05]
```

**直觉**:离终点越远,advantage 越大(开局对了路功劳大);离终点越近,越小(value 已预见到了)。

**对比 GRPO**:同条 trajectory,group mean=0.5, std=0.5 → A = +1.0,**所有 token 都是 +1.0**,完全不区分谁的贡献大。这就是 GRPO 失去的 credit assignment,也是 GSPO/Token-level GRPO 想补上的。

---

## 8. 反向阅读:_compute_loss 的 inputs 字典

```python
inputs = {
    "prompt_ids", "completion_ids",
    "prompt_mask", "completion_mask",  # ← 怎么"看见"
    "advantages",                       # ← 学习信号
    "old_per_token_logps",              # ← 学得多猛
    "ref_per_token_logps",              # ← 别学跑偏
    "tool_mask",                        # ← 哪些位置算
}
```

**这 7 个字段 = 4 phase 的全部物质载体**。  
任何 RL 变种(DAPO/GSPO/DrGRPO)改的都是**怎么用这 7 个字段算 loss**,不是这 7 个字段本身。

→ `_generate_and_score_completions` 是**基础设施**,`_compute_loss` 才是**算法本身**。读 PPO/GRPO 对比时看的就是 `_compute_loss` 的差异。

---

## 9. 未读但已挖出的进阶话题(留给 Day 2+)

| 概念 | 在哪 | 是什么 |
|---|---|---|
| `get_high_entropy_mask` | grpo_trainer.py:1007 | top-entropy quantile,token-level credit assignment 补丁 |
| `get_off_policy_mask` | :2366 | vLLM-训练精度差异检测,off-policy mask |
| `get_gamma_weights` | :2389 | vespo 的 gamma 加权 |
| `use_bias_correction_kl` | :2518 | KL 的 IS 修正 |
| `loss_type=cispo/sapo/vespo/dr_grpo/luspo` | :2523-2586 | 5 种 GRPO 变种 loss |
| `ε_low ≠ ε_high` | :2527 | DAPO 不对称 clip(对正/负 advantage 不同容忍度) |
| `vllm_importance_sampling_mode` | :2068-2092 | 4 种 vLLM IS 修正策略 |
| `delta` 双向 clip | :2529 | DAPO 加的上下两侧裁剪 |

这些都是"GRPO → 真实 agentic 训练"的工程补丁。基线 vanilla 通了之后,可以单点击破。

---

## 10. Day 2+ 可能方向

- (A) **PPO experimental 完整精读**:`train()` 主循环、reward shaping with KL、value loss/clip 细节
- (B) **verl 框架对比**:看工业级 framework 怎么解决 TRL 的串行 tool 瓶颈(`AsyncBatchTool`)、FSDP 分片细节、vLLM 集成
- (C) **DAPO 论文 + 代码**:scale_rewards="none"、ε 不对称、双向 clip、去 KL 这几个改动的实证依据
- (D) **动手实验**:0.5B 模型 + Mac CPU 跑 minimal GRPO,改 ε / β 看 advantage / ratio 实际数值,对照理论
- (E) **GSPO / DrGRPO**:序列级 ratio、token-level credit assignment 的修补思路

---

## 附录 · 一句话总结

> **PPO** 用 value 网络做"沿时间维度的差分"算 per-token advantage(GAE),信号精细但工程沉重(显存翻倍 + value 估不准会全崩)。  
> **GRPO** 砍掉 value 网络,改用"沿 batch 维度的多组采样对比"做近似,工程轻盈但丢了 token 级 credit assignment。  
> 两条路本质都在解决 reward 太稀疏 —— 一个用模型预测填空,一个用多次采样填空。
