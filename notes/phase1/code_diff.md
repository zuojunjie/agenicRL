# Phase 1 代码差异 — main_ppo vs main_ppo_format

来源：`diff verl/trainer/main_ppo.py main_ppo_format.py`

## 改动一览（全部在 main_ppo_format.py）

```diff
- from verl.utils.reward_score import qa_em
+ from verl.utils.reward_score import qa_em, qa_em_format

# RewardManager 构造
- def __init__(self, tokenizer, num_examine, format_score=0.):
+ def __init__(self, tokenizer, num_examine,
+              structure_format_score=0., final_format_score=0.,
+              retrieval_score=0., format_score=0.):

# data_source → 评分函数 路由
- 'nq' → qa_em.compute_score_em
+ 'nq' → qa_em_format.compute_score_em（含 5 档分级）
+ 新增数据源支持：'web_questions', 'strategyqa'

# 训练入口配置读取（main()）
+ structure_format_score = config.reward_model.structure_format_score
+ final_format_score     = config.reward_model.final_format_score
+ retrieval_score        = config.reward_model.retrieval_score
```

## 训练命令的额外 CLI override

```bash
python -m verl.trainer.main_ppo_format \
    ... 原所有参数 ... \
    +reward_model.structure_format_score=0.2 \
    +reward_model.final_format_score=0.1 \
    +reward_model.retrieval_score=0.1
```

（`+` 前缀是 Hydra 语法：在原 config 里**新增** key，因为 baseline 配置没有这几个字段）

## qa_em_format.is_valid_sequence 的 DFA（核心 ~70 行）

状态机 `start → in_think → after_think → (in_search → after_search → in_information → information → ...) | in_answer → end`

合规要求：
1. 必含 `<|im_start|>assistant` 标记
2. 4 标签 (think/search/information/answer) 数量平衡
3. 标签序列严格遵循上面 DFA
4. 标签外只允许空白

任何违反 → `is_valid_format=False` → 跌入低 reward 档位。

## 学习要点

1. **DFA 比简单 regex 严谨**：原以为 R1 风格只是检查 `<think>...</think><answer>...</answer>` 配对（我父 agent 任务书里就是这么描述的，**简化错了**）。Search-R1 实现强迫 search/information 也走规范流程。
2. **5 档而非 0/1**：稀疏的 0/1 reward → 模型学不动；分级让 partial credit 有 gradient 可走。
3. **零代码改动可逆性**：upstream 提供 `_format` 后缀的并行入口/工具是好实践，可借鉴到 Phase 2-5（不要改原 main_ppo，新建 main_ppo_dapo 等）。
