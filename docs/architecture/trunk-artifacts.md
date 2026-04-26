# Trunk 产物

一旦 Phase 0 建成，所有后续 phase 共享。**绝对不要**每个 phase 重做。

| 产物 | 建立位置 | 后续策略 |
|---|---|---|
| Retrieval FAISS index | P0 | 只读，所有 phase 共享 |
| Eval pipeline (EM/F1 + dev sets) | P0 | 每 phase 跑同一套，结果直接可比 |
| wandb project | P0 | 每 phase 一个 group，run name 带 phase 标签 |
| Git 分支策略 | P0 | `phase0-baseline` → `phase2-dapo` / `phase2-gspo` 各自分支 |
| 数据集 split | P0 | **绝对不要**中途换 split，否则横跨 phase 不可比 |

## wandb 命名规范（候选）

```
project: agentic-rl-search
group:   phase{0|2|3|4|5}-{algorithm}
name:    phase{N}-{algo}-{seed}-{timestamp}
tags:    [model:qwen2.5-3b, dataset:hotpotqa, ...]
```

例：`phase2-dapo-seed42-20260512` / `phase2-gspo-seed42-20260513`

## Git 分支策略（候选）

```
main                    # 始终是"已验证可跑"的状态
├── phase0-baseline     # baseline 复现
├── phase2-dapo         # 从 phase0 fork，加 DAPO 改进
├── phase2-gspo         # 从 phase0 fork，加 GSPO
├── phase3-torl         # 从 phase2 最佳分支 fork
├── phase4-gigpo        # 从 phase3 fork
└── phase5-arpo         # 从 phase4 fork
```

每个 phase 一个分支，便于 A/B、便于回退。

## "金标准" dev set

| 数据集 | split | 用途 | 大小 |
|---|---|---|---|
| NQ | dev | 单跳基准 | ~3.6K |
| HotpotQA | dev | 多跳基准 | ~7.4K |
| 2WikiMultiHop | dev | 多跳泛化 | ~12K |
| Bamboogle | test | 难题压力测试 | ~125 |

**铁律**：这些 split 一旦定下来，整个项目周期不换。
