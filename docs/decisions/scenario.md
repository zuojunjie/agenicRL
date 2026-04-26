# 场景选择 · Search-R1

## 决策

**场景**：Search-R1 路径——训一个 LLM 交错进行推理与搜索引擎工具调用，奖励 = QA outcome 正确性（EM/F1）。

## 备选与权衡

| 类别 | 代表论文 | 优点 | 痛点 | 评价 |
|---|---|---|---|---|
| 工具调用（搜索/计算器/code） | ToRL, ARPO, GiGPO | 奖励信号干净、单 GPU 可跑 | "不够 agentic" | ✅ 选中 |
| Web/GUI Agent | WebArena 系 | 最像真 agent | env 工程量巨大 | ❌ env 成本太高 |
| 长程推理 | DeepSeek-R1, DAPO | 基础设施最成熟 | 已被刷满 | ❌ 与本目标错位 |

## 为什么是 Search-R1（而非 ToRL/ToolRL）

Search-R1 处在全景图的精确交叉点：

- **算法骨架** = DeepSeek-R1 的 GRPO（推理路径）
- **工具范式** = ToRL/ToolRL（工具使用路径）
- **奖励信号** = QA outcome（最干净的一类）

更关键的是：

- 官方代码基于 **verl**，verl 又是 DAPO/GSPO 的官方实现框架
- 后续读 DAPO/GSPO 论文时，改进可以直接 patch 进同一份代码
- A/B 对比成本极低

## 核心技术栈默认值

| 决策 | 默认 | 备选 |
|---|---|---|
| 基座模型 | Qwen2.5-3B-Instruct | Qwen2.5-7B / Llama-3.2-3B |
| 检索后端 | 本地 Wikipedia + E5 dense + FAISS | SerpAPI / BM25 |
| 数据集 | NQ + HotpotQA + 2Wiki + Musique + Bamboogle | 单一数据集 |
| 奖励 | 纯 outcome (EM 主) | + format / + step PRM |
| 算法 | GRPO（vanilla） → DAPO/GSPO → GiGPO → ARPO | PPO |
| 训练框架 | verl | OpenRLHF / TRL |
| 生成引擎 | vLLM | SGLang |
