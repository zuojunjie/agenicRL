# Agent RL 学习笔记 — GRPO / PPO Clip / State Masking / KL Penalty

> 一份从工程动机出发、逐层推导算法设计的 Agent RL 自学笔记。  
> 学习路径：**A (GRPO 核心) → C (State Masking) → D (PPO Clip) → B (KL Penalty)**。

---

## 目录

- [一、学习计划与疑问清单](#一学习计划与疑问清单)
- [二、Phase A：GRPO 核心](#二phase-agrpo-核心)
- [三、Phase C：State Masking](#三phase-cstate-masking)
- [四、Phase D：PPO Clip + Ratio](#四phase-dppo-clip--ratio)
- [五、Phase B：KL Penalty](#五phase-bkl-penalty)
- [六、Rollout 系统设计](#六rollout-系统设计)
- [七、疑问清单 · 全部回答](#七疑问清单--全部回答)
- [八、关键概念速查表](#八关键概念速查表)
- [九、TRL 代码精读 Day 1](#九trl-代码精读-day-1--grpo-主线--ppo-对照)

---

## 一、学习计划与疑问清单

### 学习顺序

| 阶段 | 内容 | 必学理由 |
|------|------|---------|
| A | GRPO 核心 | 后续所有 phase 的基础 |
| C | State Masking | agentic RL 与传统 RLHF 的最大差异 |
| D | PPO Clip | 理解 DAPO/GSPO 的前置 |
| B | KL Penalty | 理解 reward hacking 的工程对策 |

### 原始疑问清单

**Rollout**：
1. `max_turns=2` 中 `max_turns` 含义？
2. rollout 必须用基线模型吗？能用更大的模型（如 1000B）吗？
3. `retrieval_server` 在 rollout 中的作用？本质是工具调用吗？
4. 整体 prompt 怎么设计？

**Reward**：
1. reward 如何标记为 0 or 1？

**Group-Relative Advantage**：
1. `r_{i,1}` 是怎么来的？
2. 所有 trajectory 都纳入均值方差计算吗？

**Reference Log-Prob**：
1. KL 散度 loss 的具体形式是什么？

**Policy Update (PPO-style, FSDP 4 卡分片)**：
1. 含义拆解？
2. ratio 如何设计？

---

## 二、Phase A：GRPO 核心

### 2.1 从 Policy Gradient 说起

**目标**：用 reward 信号去更新模型参数 θ。

**核心直觉**：答案对 → 增大 π_θ(o|q)；答案错 → 减小或不变。

**朴素 policy gradient**：

$$\nabla_\theta J = \mathbb{E}\left[ r \cdot \nabla_\theta \log \pi_\theta(o|q) \right]$$

- `∇_θ log π_θ(o|q)`：让模型更倾向生成 o 的方向
- 乘上 r：r=1 鼓励，r=0 不变

### 2.2 朴素方法的缺陷：绝对 reward 的尺度问题

**反例**：

| 题目 | rewards | 直接当梯度权重的问题 |
|------|---------|-------------------|
| 简单题 A | [0.9, 0.95, 1.0, 0.85] | 全部正梯度，0.85 那个差答案也被鼓励 |
| 困难题 B | [0.0, 0.1, 0.0, 0.05] | 全部接近 0，0.1 那个好答案学不到 |

**问题本质**：绝对 reward 只反映"题目难易"，不反映"在这道题里哪个答案相对更好"。

### 2.3 解决方案：Advantage = r − baseline

减去一个 baseline `b`，得到 **advantage** `A = r − b`。

| 题目 | rewards | baseline | advantage |
|------|---------|----------|-----------|
| A | [0.9, 0.95, 1.0, 0.85] | 0.925 | [-0.025, +0.025, +0.075, -0.075] |
| B | [0.0, 0.1, 0.0, 0.05] | 0.0375 | [-0.0375, +0.0625, -0.0375, +0.0125] |

**新公式**：

$$\nabla_\theta J = \mathbb{E}\left[ A \cdot \nabla_\theta \log \pi_\theta(o|q) \right]$$

### 2.4 baseline 怎么算？PPO vs GRPO

#### 方案 1（PPO）：训一个 value model

`A = r − V_φ(q)`，V_φ 是单独训练的价值网络。

- ✅ 优点：可处理稀疏 reward；泛化性好
- ❌ 缺点：显存翻倍；多一个模型要训；估不准会传染 policy

#### 方案 2（GRPO）：组内采样均值

对同一 prompt q 采样 G 个答案，用组内统计量做 baseline：

$$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

- ✅ 不需要 value model → 显存减半
- ✅ 用推理算力换训练算力（推理便宜可并行）
- ⚠️ 对 cold start 敏感（详见 Phase B & Rollout）

### 2.5 GRPO 名字解读

- **G**roup：一组答案（同 prompt 采样 G 次）
- **R**elative：advantage 是组内相对量
- **P**olicy **O**ptimization：还是 policy gradient

### 2.6 关键事实

- `r_i` 来自外部 verifier（数学题用 SymPy 比对、代码题跑测试用例），不是来自神经网络
- 这就是 **RLVR (Reinforcement Learning with Verifiable Rewards)** 的核心
- "组"的边界 = **同一个 prompt 的多次采样**，不同 prompt 不混

---

## 三、Phase C：State Masking

### 3.1 问题场景

agentic trajectory 里 token 有不同来源：

```text
[user]      Question: 北京今天天气？
[assistant] 我需要查一下。<tool_call>search("北京天气")</tool_call>
[tool]      <tool_result>晴，26°C</tool_result>
[assistant] 北京今天晴，26°C。
```

| Token 段 | 来源 | 是 policy 生成的吗？ |
|---------|------|------------------|
| Question 部分 | user | ❌ |
| 我需要查一下…</tool_call> | **assistant** | ✅ |
| <tool_result>...</tool_result> | tool | ❌ |
| 北京今天晴，26°C | **assistant** | ✅ |

### 3.2 错误做法：所有 token 都算 loss

后果：模型会被错误地"教着去生成"工具返回内容，产生幻觉（觉得自己能控制环境）。

> 类比：训狗坐下时给零食，但门铃响时也给零食 → 狗会以为门铃响是自己能控制的，开始疯狂叫。

### 3.3 正确做法：State Mask

只在 **assistant 自生成的 token** 上算 loss：

$$\nabla_\theta J = \mathbb{E}\left[ A_i \cdot \sum_{t \in \mathcal{M}_i} \nabla_\theta \log \pi_\theta(o_{i,t}) \right]$$

其中 `M_i` = trajectory i 中 assistant 生成 token 的位置集合。

### 3.4 工程实现

```python
loss_mask = torch.zeros(seq_len)
for span in trajectory.assistant_spans:
    loss_mask[span.start : span.end] = 1.0

per_token_logprob = compute_logprob(model, tokens)
loss = -(advantage * per_token_logprob * loss_mask).sum() / loss_mask.sum()
```

**关键**：分母用 `loss_mask.sum()`（被算 loss 的 token 数），不是 `seq_len`，否则 tool 返回多的 trajectory 会被错误稀释。

### 3.5 vanilla RLHF vs agentic RL

| | Vanilla RLHF | Agentic RL |
|--|--------------|------------|
| trajectory 结构 | prompt → 一段连续回复 | prompt → assistant → tool → assistant → tool → ... |
| 需要 mask 的环境 token | 无（除 prompt） | **每次 tool 返回** |
| state masking 重要性 | 不关键 | **决定训练成败** |

### 3.6 衍生概念：On-Policy

- **on-policy 定义**：用于计算梯度的 token 必须是当前 policy π_θ 自己采样出来的
- assistant 自生成 token = on-policy ✅
- tool 返回 token = 不是 on-policy ❌
- state mask 本质 = "只保留 on-policy 的 token 算 loss"

### 3.7 max_turns 的含义

`max_turns` = trajectory 中 **assistant turn 的最大数量**。

`max_turns = 2` 示例：

```text
Turn 1: [assistant] 我先搜一下。<search>...</search>
        [tool] <result>...</result>
Turn 2: [assistant] 根据结果，答案是 XX。  ← 必须给出最终答案
```

**为什么需要上限**：
- 训练稳定（无限轮次会显存爆炸）
- 截断处理（达到上限未答 → reward = 0）
- 避免无限循环（早期模型可能反复调工具不答题）

---

## 四、Phase D：PPO Clip + Ratio

### 4.1 动机：rollout 太贵，想多次复用

朴素流程：

```
1. rollout (用 π_θ 生成 trajectory)
2. 算 advantage
3. 更新 θ
4. 回到 1
```

问题：rollout 占总训练时间 50%–80%，每更新一步就重新 rollout 浪费 GPU。

**目标**：一次 rollout 的数据，做多次梯度更新。

### 4.2 引入 Importance Sampling

数据是 θ_old 采的，参数已更新到 θ → off-policy。

数学修正：

$$\mathbb{E}_{o \sim \pi_\theta}[f(o)] = \mathbb{E}_{o \sim \pi_{\theta_{old}}}\left[\frac{\pi_\theta(o)}{\pi_{\theta_{old}}(o)} f(o)\right]$$

修正系数 ρ = π_θ(o) / π_θ_old(o) 就是 **PPO ratio**。

| ρ 值 | 含义 |
|------|------|
| ρ = 1 | 新旧 policy 看法一致 |
| ρ > 1 | 新 policy 更倾向此样本 |
| ρ < 1 | 新 policy 更不倾向此样本 |
| ρ → 0 或 ∞ | **分布偏离过远 → 训练崩溃** |

### 4.3 Token 级 ratio

LLM 是 token-by-token 生成的，所以 ratio 也是 per-token 算：

$$\rho_{i,t} = \exp\left(\log \pi_\theta(o_{i,t}|\cdot) - \log \pi_{\theta_{old}}(o_{i,t}|\cdot)\right)$$

```python
logp_new = compute_logprob(model, tokens)
logp_old = compute_logprob(old_model, tokens)
ratio = (logp_new - logp_old).exp()
```

### 4.4 Clip：给 ratio 戴安全帽

直接给 ratio 设上下限：

$$\mathcal{L}_{i,t} = \min\Big(\rho_{i,t} \cdot A_i, \; \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \cdot A_i\Big)$$

- ε 通常取 0.1 或 0.2
- 在原 ratio 和 clip 后的 ratio 之间取**小**的（min）

### 4.5 四种场景的 clip 行为（核心表）

| 场景 | A | ρ vs 1±ε | 不 clip 的 loss | clip 后 loss | min 选谁 | 训练含义 |
|------|---|---------|---------------|-------------|---------|---------|
| 1 | + | ρ > 1+ε | ρA 大 | (1+ε)A 小 | clip | **限制鼓励**（好样本但过头） |
| 2 | + | ρ < 1-ε | ρA 小 | (1-ε)A 大 | 原值 | **不限制**（好样本远离，让它回来） |
| 3 | − | ρ > 1+ε | ρA 很负 | (1+ε)A 略负 | clip | **限制抑制**（坏样本过度拥抱，温和打压） |
| 4 | − | ρ < 1-ε | ρA 略负 | (1-ε)A 较负 | 原值 | **不限制**（坏样本远离，继续远离） |

**一句话哲学**：PPO 永远不会用极大的梯度做"过激更新"，但会用合理的梯度做"必要修正"。

### 4.6 Epoch 数限制

ratio 累积偏离 1 → clip 触发频率增加 → clip 后梯度变常数 → **梯度消失**。

```text
∂[clip(ρ, 1-ε, 1+ε) · A] / ∂θ = 0    （当 clip 触发时）
```

**实践**：一次 rollout 通常做 2–4 个 epoch，再多就大量 token 无梯度，是浪费算力。

### 4.7 不能用 1000B 训 7B

如果 rollout 用 1000B、训练 7B：
- ratio = π_7B / π_1000B 反映的是**模型能力差**，不是**策略更新差**
- 大概率严重偏离 1 → 全部触发 clip → 梯度被 clip 压成常数 → 训练信号失效

✅ rollout/训练必须用**同结构、同初始化、参数差距小**的模型（实际是同一模型在不同时刻的快照）。

### 4.8 完整 Policy Update 循环

```text
1. ROLLOUT      用 θ_old 对 G 个 prompt 各采样 K 条 trajectory
2. REWARD       每条 trajectory 调 verifier 拿 r_i
3. ADVANTAGE    每个 prompt 内做 group-relative 归一化
4. STATE MASK   只在 assistant token 上算 loss
5. POLICY UPDATE
   for epoch in 1..N:        # N 通常 2-4
     ratio  = π_θ / π_θ_old
     loss   = -min(ρA, clip(ρ)A)
     loss  *= loss_mask
     θ.backward()
6. SYNC θ → θ_old，回到 1
```

**FSDP 4 卡分片** 是工程实现：模型参数分片到 4 张卡，各卡算梯度后 all-reduce。算法本身无关，只是分布式框架。

### 4.9 严格 on-policy 是否能避免 ratio 爆炸？

理论上可以（每次更新前重新 rollout，ratio ≡ 1）。

但 LLM agentic RL 实际**不可能严格 on-policy**：

| 工程现实 | 影响 |
|---------|------|
| rollout 占 50%–80% 训练时间 | 每步重 rollout 不可行 |
| 推理引擎（vLLM）vs 训练引擎（FSDP） | 切换开销巨大 |
| 多卡分布式不可能严格同步 | 物理上做不到 on-policy |
| vLLM 与训练框架精度差异 | 即使权重相同也有数值误差 |

→ **ratio + clip 在 LLM RL 中是必然存在的**。

---

## 五、Phase B：KL Penalty

### 5.1 动机：Reward Hacking

**反例**：verifier 漏洞 = "答案含 42 就 reward=1"。

训练后模型学到的不是"做对题"，而是"任何答案都塞 42"——这就是 reward hacking。

### 5.2 PPO Clip 防不住

clip 只保证**单步更新幅度小**。但 1000 步合规小步累积 → policy 仍可漂移到极端。

**结论**：clip 保证局部稳定，不保证全局合理。

### 5.3 解决方案：Reference Model + KL Penalty

保存训练前的 SFT checkpoint 作为 **reference model π_ref**（参数冻结）。

新 loss：

$$\mathcal{L}_{\text{total}} = \underbrace{-\mathbb{E}\left[\min(\rho A, \text{clip}(\rho)A)\right]}_{\text{PPO loss}} + \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

**直觉**：把 policy 用一根皮筋拴在 SFT 模型附近。

### 5.4 KL 散度的三种估计形式

#### 形式 1（朴素）

$$\hat{D}_{\text{KL}} = \log \pi_\theta - \log \pi_{\text{ref}}$$

- ✅ 简单、无偏
- ❌ 方差大、可能为负

#### 形式 2（k2 平方）

$$\hat{D}_{\text{KL}} = \frac{1}{2}(\log \pi_\theta - \log \pi_{\text{ref}})^2$$

- ✅ 总是非负
- ❌ 有偏

#### 形式 3（k3，**GRPO 标准**）

$$\hat{D}_{\text{KL}} = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log \frac{\pi_{\text{ref}}}{\pi_\theta} - 1$$

- ✅ 无偏、低方差、总是非负
- ✅ DeepSeekMath / GRPO 论文采用

```python
log_ratio = logp_ref - logp_theta
kl = log_ratio.exp() - log_ratio - 1   # 逐 token
loss = ppo_loss + beta * (kl * loss_mask).sum() / loss_mask.sum()
```

### 5.5 β 的调节

| β 值 | 行为 |
|------|------|
| 0 | 无 KL，纯 PPO，容易 reward hacking |
| 0.001–0.01 | 实践常用范围 |
| > 0.05 | KL 主导，policy 几乎不动 |

### 5.6 进阶：DAPO 干脆去掉 KL

DAPO（ByteDance 2024）发现：在数学推理上 KL **反而有害**——因为推理任务需要大幅偏离 SFT 分布。

教训：**KL 不是教条，是工程权衡**。任务越需要颠覆 SFT 的习惯，KL 越应该弱甚至去掉。

---

## 六、Rollout 系统设计

### 6.1 retrieval_server 是什么

本质就是**工具调用 HTTP 服务**：

```text
POST http://retrieval-server:8000/search
{ "query": "比亚迪 2024 净利润", "top_k": 5 }
→ { "results": ["...", "...", "..."] }
```

模型生成 `<search>...</search>` → rollout 框架 POST 给 server → 拿到结果填入 `<result>...</result>` → 继续生成。

### 6.2 为什么单独搞个 server

| 好处 | 说明 |
|------|------|
| 数据隔离 | 检索语料库几十 GB，不想加载到每个 worker |
| 资源解耦 | rollout 用 GPU，检索用 CPU+大内存 |
| 缓存复用 | 多 worker 命中同一 query 可缓存 |
| 替换灵活 | 想换检索引擎只改 server |

### 6.3 工具谱（agentic RL 的所有工具都是 HTTP 服务）

| 工具类型 | 服务形式 | 代表工作 |
|---------|---------|---------|
| 检索 (RAG) | retrieval server | search-R1, R1-Searcher |
| 代码执行 | sandbox (Docker/E2B) | ReTool, ToRL |
| 计算器 | sympy/wolframalpha | ToolRL |
| 网页浏览 | headless browser | WebGPT, WebShaper |
| API 调用 | HTTP proxy | API-Bank, ToolBench |
| SQL | DB gateway | Spider-RL |

### 6.4 retrieval_server vs sympy executor 的本质差异

| 维度 | retrieval_server | sympy executor |
|------|----------------|----------------|
| 结果稳定性 | 有不确定性 | 完全可复现 |
| 结果空间 | 开放（自然语言） | 封闭（表达式/错误） |
| 延迟 | 高（毫秒~秒级） | 低（毫秒） |
| **可验证性** | **弱**（结果对错难判断） | **强**（直接判对错） |
| 对训练的影响 | 引入噪声 | 信号干净 |

**核心洞察**：工具的确定性 → 决定 reward 的可验证性 → 决定 RL 能不能 work。这就是为什么数学/代码 RL 比开放问答 RL 更早成熟。

### 6.5 Agentic Search 的 4 个设计决策

#### ① 工具协议

简单标签 vs JSON：

```text
<search>北京天气</search>
```

```json
<tool_call>{"name": "search", "arguments": {"query": "..."}}</tool_call>
```

经验：单工具用标签，多工具/复杂参数用 JSON。

#### ② 停止条件

```python
if "<answer>" in output:        # 模型给最终答案
    return DONE
elif "<search>" in output:       # 还要调工具
    result = call_tool(...)
    continue
elif turn_count >= max_turns:    # 强制截断
    return TRUNCATED
elif total_tokens > max_tokens:  # 上下文要爆
    return TRUNCATED
```

#### ③ 失败处理：软失败 > 硬失败

- 硬失败：trajectory 终止，reward = 0
- 软失败：把错误信息作为 `<result>` 返回让模型重试

经验：**软失败更利于学习**——模型可以学到"看到 timeout 就重试"。

#### ④ 并发控制

```text
GPU rollout (256 并行) → 异步 client → retrieval_server (1000 QPS)
                                              ↓
                                          Redis 缓存
                                              ↓
                                        Elasticsearch
```

### 6.6 Prompt 设计

#### 模板骨架

```text
[System]
你是一个能调用搜索工具的助手。

工具协议：
- 调用搜索：<search>查询关键词</search>
- 给最终答案：<answer>答案内容</answer>

工作流程：
1. 思考问题（用 <think>...</think>）
2. 如需信息，调用搜索工具
3. 收到结果后继续思考
4. 直到能给出答案

最多调用搜索 {max_turns} 次。

[User]
{question}

[Assistant]
<think>
```

#### 关键设计要点

**① 强制 `<think>` 标签**

不只是格式要求——它是给 RL 探索"扩容"：

- 没有 `<think>`：trajectory 短，梯度只更新少数 token，学到"答案是 X"
- 有 `<think>`：trajectory 长，梯度更新整个推理链，学到"通过这种推理 → 得到 X"

`<think>` 给模型一个**安全的内部探索空间**，可以试错、自我纠错。这就是为什么 R1 系列模型的 think 越训越长。

**② 工具协议必须 system prompt 显式说**：模型不会自己猜协议。

**③ 必须用 chat template**：

```python
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
```

利用 SFT 模型已经学会的格式。

**④ 解决 Cold Start 问题**：

GRPO 特有的失败模式 → 一组 trajectory 全 r=0 → mean=std=0 → **advantage 归零** → 梯度消失。

PPO 用 value model 还能有点信号，GRPO 完全靠组内对比，所以 cold start 时彻底学不到。

解决方案：

| 方案 | 做法 |
|------|------|
| **SFT warm-up** | 先用高质量 trajectory SFT，让模型先会基本格式 |
| **Format reward** | reward 加项 "格式正确 +0.1" 引导先学协议 |
| **Curriculum** | 从简单题（1 次工具就能答）开始 |

R1-Zero 跳过 SFT 是例外，靠 base model 已经很强。**对中小模型几乎都需要 SFT warm-up**。

---

## 七、疑问清单 · 全部回答

### Rollout

**Q1: max_turns 含义？**  
trajectory 中 assistant turn 最大数量。每个 turn 通常 = 一次"思考 + 决策（调工具或给答案）"。

**Q2: 能用更大模型 rollout 吗（如 1000B）？**  
不能。ratio = π_7B / π_1000B 反映的是模型能力差，不是策略更新差，会大幅偏离 1 → 全部触发 clip → 训练信号失效。rollout/训练必须同结构同初始化。

**Q3: retrieval_server 作用？**  
本质是工具调用 HTTP 服务，可以替换成任何工具（代码执行、计算器、浏览器等）。所有 agentic 工具的统一抽象 = "input 字符串 → output 字符串"。

**Q4: prompt 设计？**  
system 写工具协议 + chat template + 强制 `<think>` 标签 + 明确 max_turns + cold start 用 SFT warm-up。

### Reward

**Q1: reward 怎么标 0/1？**  
完整轨迹生成完后由外部 verifier 判断（数学题用 SymPy、代码题跑测试、写作用 reward model）。每个完整 trajectory 一个 r_i。**整条轨迹的每个 token 共享同一个 r_i**（即 r_{i,t} = r_i）。

### Group-Relative Advantage

**Q1: r_{i,1} 来源？**  
来自外部 verifier 对完整 trajectory 的判定。下标 t 不代表 reward 在 token 间分配，所有 t 的 r_{i,t} 都等于该 trajectory 的整体 r_i。

**Q2: 所有 trajectory 都纳入均值方差吗？**  
不是。**组的边界 = 同一个 prompt 的多次采样**。一个 batch 里 4 道题各采 8 个答案 = 4 个独立的组，每组单独算 mean/std。

### Reference Log-Prob

**Q1: KL 散度具体形式？**  
GRPO 用 k3 estimator：

```text
KL = exp(logπ_ref - logπ_θ) - (logπ_ref - logπ_θ) - 1
```

特性：无偏、低方差、总是非负。逐 token 算后乘 loss_mask 再求平均。

### Policy Update

**Q1: 含义拆解？**  
完整循环 = ROLLOUT → REWARD → ADVANTAGE → STATE MASK → 多 epoch (ratio + clip + KL) → SYNC。FSDP 4 卡分片只是分布式工程实现。

**Q2: ratio 怎么设计？**  
ρ = exp(logπ_θ - logπ_θ_old)，逐 token，clip 到 [1-ε, 1+ε]，loss 取 min(ρA, clip(ρ)A)。梯度只通过 logπ_θ，logπ_θ_old detach。

---

## 八、关键概念速查表

### 公式

| 概念 | 公式 |
|------|------|
| Policy Gradient | `∇J = E[A · ∇log π_θ(o)]` |
| Advantage (GRPO) | `A_i = (r_i - mean) / std` |
| Importance Ratio | `ρ = π_θ(o) / π_θ_old(o)` |
| PPO Clip Loss | `L = min(ρA, clip(ρ, 1-ε, 1+ε)·A)` |
| KL k3 Estimator | `KL = exp(logπ_ref - logπ_θ) - (logπ_ref - logπ_θ) - 1` |
| Total Loss | `L = -PPO_loss + β·KL` |

### 工程超参

| 参数 | 典型值 | 说明 |
|------|--------|------|
| group size G | 8 ~ 16 | 同 prompt 采样数 |
| clip ε | 0.1 ~ 0.2 | ratio 容忍范围 |
| epochs per rollout | 2 ~ 4 | 旧数据复用次数 |
| KL β | 0.001 ~ 0.05 | KL 惩罚强度 |
| max_turns | 2 ~ 8 | 工具调用最大轮次 |

### 失败模式速查

| 现象 | 可能原因 | 对策 |
|------|---------|------|
| 一组 trajectory 全 r=0 | cold start，模型没学会工具协议 | SFT warm-up / format reward |
| ratio 频繁触发 clip | epoch 太多，θ 偏离 θ_old 太远 | 减少 epoch，重新 rollout |
| 模型胡乱塞特定关键词 | reward hacking，verifier 有漏洞 | 加强 KL，修 verifier |
| trajectory 越来越长不收敛 | max_turns 太大或无限制 | 设置硬上限 |
| 训练 loss 下降但 reward 不涨 | state mask 写错，在 tool token 上算了 loss | 检查 mask 逻辑 |
| 梯度爆炸 / NaN | 没用 clip，或 ratio 计算精度问题 | 检查 logp 数值范围 |

### 算法对比

| 算法 | baseline 来源 | 特点 |
|------|-------------|------|
| REINFORCE | 无（或全局均值） | 严格 on-policy，sample 效率低 |
| PPO | value model V_φ(q) | clip + value，标准 RLHF |
| GRPO | 组内均值 | 无 value model，省显存 |
| DAPO | 组内均值，去 KL | 数学推理优化 |
| GSPO | 序列级 ratio 而非 token 级 | 长 trajectory 更稳 |

---

> **学习状态**：4 个 phase 全部完成 ✅，10 条疑问全部清空 ✅。  
> **下一步可选方向**：实战看 verl/OpenRLHF 代码 / 进阶 DAPO / GSPO / DrGRPO / 综合考题检验。
## 九、TRL 代码精读 Day 1 — GRPO 主线 + PPO 对照

_日期:2026-05-05 · 仓库:trl@HEAD(experimental/ppo + trainer/grpo)_

> 日期:2026-05-05  
> 仓库:`/Users/cv/code/agenicRL/external/trl/` (浅 clone)  
> 知识基线:`~/Downloads/agent_rl_notes.md`(Phase A/B/C/D 已通)  
> 模式:精读 GRPO 主线,PPO 作对照

---

### 0. 关键发现:TRL 已把 PPO 移入 `experimental/`

| 文件 | 行数 | 状态 |
|---|---|---|
| `trl/trainer/grpo_trainer.py` | 2732 | ✅ 主线 |
| `trl/trainer/rloo_trainer.py` | 1535 | 主线(GRPO 近亲) |
| `trl/experimental/ppo/ppo_trainer.py` | 1037 | ⚠️ 降级 |

**信号**:无 value model 路线(GRPO/RLOO)正取代 PPO 成为社区主流。原因之一是 PPO 自己手写 train loop,跟 HF Trainer 接口对不上,工程成本高。

---

### 1. GRPOTrainer 调用链总图

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

### 2. 概念 → 代码位置映射(对照笔记 4 phase)

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

### 3. 三个工程关键洞察

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

### 4. `_tool_call_loop` 关键点

骨架:while 循环,每轮"执行工具 → append 结果 → 长度检查 → 再生成 → 更新 tool_mask"。

| 你笔记 6.5 节 | TRL 实现 |
|---|---|
| ① 工具协议:tag vs JSON | 全用 OpenAI tools API JSON(line 1518),要求模型 chat template 支持 tool calling |
| ② 停止条件 4 种 | 实现了全部 4 种,且**长度爆了不粗暴 truncate,而是 rollback 这一轮 tool 消息**(line 1599-1604) |
| ③ 软失败 > 硬失败 | ✅ try/except → `{"error": str(e)}` 喂回模型(line 1528-1531) |
| ④ 并发控制 | 同 sample 内 async 工具用 asyncio.gather 并发,**跨 sample 串行**(line 1506) → 大 batch + 重 retrieval 时是瓶颈 |

**核心设计哲学**: `tool_mask` 在 rollout 时动态构建(append 0 或 1),不在训练时用正则反推。优势:不依赖模型输出格式、多模态友好、零成本。

---

### 5. PPO vs GRPO 五大差异对照

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

### 6. Value Model 是什么 + 怎么训(case)

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

### 7. GAE 数值案例(同一个 trajectory)

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

### 8. 反向阅读:_compute_loss 的 inputs 字典

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

### 9. 未读但已挖出的进阶话题(留给 Day 2+)

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

### 10. Day 2+ 可能方向

- (A) **PPO experimental 完整精读**:`train()` 主循环、reward shaping with KL、value loss/clip 细节
- (B) **verl 框架对比**:看工业级 framework 怎么解决 TRL 的串行 tool 瓶颈(`AsyncBatchTool`)、FSDP 分片细节、vLLM 集成
- (C) **DAPO 论文 + 代码**:scale_rewards="none"、ε 不对称、双向 clip、去 KL 这几个改动的实证依据
- (D) **动手实验**:0.5B 模型 + Mac CPU 跑 minimal GRPO,改 ε / β 看 advantage / ratio 实际数值,对照理论
- (E) **GSPO / DrGRPO**:序列级 ratio、token-level credit assignment 的修补思路

---

### 附录 · 一句话总结

> **PPO** 用 value 网络做"沿时间维度的差分"算 per-token advantage(GAE),信号精细但工程沉重(显存翻倍 + value 估不准会全崩)。  
> **GRPO** 砍掉 value 网络,改用"沿 batch 维度的多组采样对比"做近似,工程轻盈但丢了 token 级 credit assignment。  
> 两条路本质都在解决 reward 太稀疏 —— 一个用模型预测填空,一个用多次采样填空。
