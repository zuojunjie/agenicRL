# Phase 11-15：Tool Use 算法路线全程

> **优先级**：模型/算法优化为主，参数调优为辅
> **基线**：Phase 10 v22（1.5B-Math, 5h/run, train reward 0.62, val pass@1 58.6%）
> **守夜规则**：每晚 23-11 不启动新模型

---

## 排序原则

按"算法 ROI ÷ 工程成本"排：

| Phase | 论文 | 改动核心 | 工程量 | 预期 ROI |
|---|---|---|---|---|
| **P11** | **ARTIST** | outcome-based 多轮信用分配 | ~200 行 | ⭐⭐⭐ 最高 |
| **P12** | **ToolRL** | multi-tier reward（process + outcome）| ~100 行 | ⭐⭐⭐ |
| **P13** | **GiGPO** | 双层 advantage（inner + outer）| ~150 行 | ⭐⭐ |
| **P14** | **ARPO** | entropy-adaptive rollout 分支 | ~500 行 | ⭐⭐ |
| ~~P15~~ | ~~ToRL retry~~ | per-turn shaping | — | ❌ Phase 3 已证伪，可跳 |

旁路：每 Phase 穿插单变量参数调优（关 KL / max_turns / temperature / n_samples）。

---

## 时间表

| 日期 | 11:00 - 16:00 | 16:05 - 21:00 | 21:00 - 11:00 |
|---|---|---|---|
| **5/7（今）**| Phase 10 v22 跑 + V0 | — | ⛔ 守夜 |
| **5/8 周四** | v23 = 1.5B-NonMath baseline | **P11 ARTIST** | ⛔ 守夜 |
| **5/9 周五** | **P12 ToolRL** | **P13 GiGPO** | ⛔ 守夜 |
| **5/10 周六** | **P14 ARPO**（8h，越界）| | — |

总：**~30 h 训练 + ~10 h 实施 = 3 天完成**

---

## 各 Phase 实施要点

### P11 ARTIST（最大 ROI）
- 当前痛点：reward 0/1 sparse，trajectory 内每个 turn 都拿一样 advantage
- ARTIST 解：错答时回溯惩罚错的 turn，对答时加权对的 turn → **turn-level credit**
- 改 SkyRL 的 `compute_advantages_and_returns`
- 旁路：顺手关 KL loss

### P12 ToolRL（与 P11 互补）
- 当前痛点：只有 binary outcome reward，过程无监督
- ToolRL 解：加 process reward
  - syntax 正确（`<python>...</python>` 闭合）+0.1
  - sandbox 执行成功（无 Error）+0.1
  - 答案格式正确（含 `\boxed{}`）+0.1
- 改 env utils.py 的 `compute_score`
- 旁路：max_turns 4→8

### P13 GiGPO 重试
- Phase 4 失败因 NQ 单跳，Math 4-turn 才有发挥空间
- 改 `advantage_estimator`：
  - inner: trajectory 内 turn 间相对优势
  - outer: 同 prompt 8 sample 间相对优势（GRPO baseline）
- 旁路：temperature 1.0→0.7

### P14 ARPO 完整复现
- entropy-adaptive：高熵节点动态分多支（ARPO 论文 Algo 1）
- 与 ARTIST 配合：ARPO 决定 explore 哪里，ARTIST 给信用分
- 大改 SkyRL rollout loop
- 旁路：n_samples 8→16（如显存允许）

### ~~P15 ToRL retry~~（可跳）
- per-turn shaping reward
- Phase 3 已证伪 reward hacking
- 与 ARTIST 实现重叠，前 4 个跑通后不必再做

---

## 守夜规则

- 23:00-11:00 不启动任何新训练
- cron watchdog 在此时段检测进程死亡只 PushNotification 不重启
- GPU 闲置接受（夜间不烧钱也不冒险）

## 输出物

每个 Phase 完成后：
1. wandb 同 project (`agenic-rl-A6000New`) 不同 run
2. 飞书 wiki 单 Phase 节点
3. README 更新（指向飞书）
4. HTML 累积进步图加点

---

## 应急

- 任何 Phase OOM / 算法挂 → ROLLBACK 回 v22 配置（已 frozen 备份）
- Phase 11-14 中任一 reward < 0.55（v22 baseline 0.62）→ 标 fail，跳下一个
- 5/10 完成后 → 选最优组合 commit phase11_v01_组合最优
