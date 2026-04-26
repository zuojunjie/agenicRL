# agenicRL — 端到端 Agentic RL 学习项目

> 通过一个真实场景的训练（**Search-R1**），把 Agentic RL 的技术栈学透。
> 过程式记录、决策可追溯、每一次实验对照一篇论文。

## 项目快览

- **场景**：Search-R1 风格——LLM 交错进行推理与搜索引擎工具调用，奖励 = QA 正确率（EM/F1）
- **算力**：4×A100（AutoDL 按量租用，晚间常规型）
- **基座模型**：Qwen2.5-3B-Instruct（学为主，全程不升 7B）
- **算法路径**：DeepSeek-R1 → DAPO + GSPO → ToolRL→ToRL → GiGPO → ARPO
- **训练框架**：verl

## 文档导航

```
docs/
├── decisions/          # 关键决策与理由
│   ├── scenario.md     # 为什么选 Search-R1
│   ├── compute.md      # 算力/模型/节奏的权衡
│   └── curriculum.md   # 5 篇主干论文 vs 17 篇全景图
├── architecture/       # 项目架构
│   ├── dependencies.md # Phase 0–5 依赖图（硬/软）
│   └── trunk-artifacts.md  # 跨阶段共享产物
├── phase-minus-1/      # 当前阶段：本地 CPU 预热
│   └── deliverables.md
└── journal/            # 时间线日志
    └── 2026-04-26-kickoff.md
```

## 当前进度

- [x] 场景选定（Search-R1）
- [x] 算力方案敲定（AutoDL + 4×A100 + 3B）
- [x] 学习路径确定（5+2 篇论文）
- [x] 阶段依赖梳理
- [ ] **Phase -1：本地 CPU 预热**（进行中）
- [ ] Phase 0：Search-R1 baseline 跑通
- [ ] Phase 1：读 DeepSeek-R1
- [ ] Phase 2：DAPO + GSPO 双读双比
- [ ] Phase 3：ToRL（奖励设计）
- [ ] Phase 4：GiGPO（步级信用分配）
- [ ] Phase 5：ARPO（前沿）

## 本地预览文档站（可选）

```bash
pip install mkdocs-material
mkdocs serve   # 浏览器打开 http://127.0.0.1:8000
```

如果不跑 mkdocs，所有 markdown 在 GitHub/Gitee 网页上也能直接渲染。
