# AgenicRL · 项目知识中枢

> **这是一份 master doc**：只放导航 + 全局状态 + 术语表。所有细节都在 5 个子 doc 里，互不重叠。
> **维护原则**：每个子 doc 单一职责；master 只更新进度和子 doc 索引；**禁止把内容写在 master 里**。

---

## 一句话项目定位

通过一个**真实训练场景（Search-R1）**端到端学 Agentic RL。
不追 SOTA，把每一篇论文 **变成一次具体的代码改动 + A/B 实验**。

> Search-R1 = R1（推理算法）+ ToRL（工具范式）+ QA（最干净的 outcome reward）的交集场景。

---

## 当前阶段（活的，每周更新）

```
[Phase -1] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.x / 8 项交付物
   ↓
[Phase 0]  ⬜ baseline 跑通                       ← 下一站
   ↓
[Phase 1]  ⬜ 读 DeepSeek-R1
   ↓
[Phase 2]  ⬜ DAPO + GSPO（双读双比）
   ↓
[Phase 3]  ⬜ ToRL（code interpreter 工具）
   ↓
[Phase 4]  ⬜ GiGPO（步级 advantage）
   ↓
[Phase 5]  ⬜ ARPO（熵驱动分支）
```

**最近一次重大事件**（2026-04-26）：parallel_dl.py 把 80GB Wikipedia 索引下载从 11h 压到 ~55min（详见《云端运维大全》§9）。

---

## 5 个子 doc 索引

| # | 子 doc | 一句话职责 | 维护人 |
|---|---|---|---|
| 01 | [决策档案](https://my.feishu.cn/docx/TGYRdxgagolGv9xzj0Tc0oUsnfc) | 为什么是 Search-R1 / 5+2 论文路径 / 4×A100 + 3B 算力配置 | master agent |
| 02 | [架构与阶段依赖](https://my.feishu.cn/docx/J82yd1pjNo8yRBxMBf0c3sNGnvb) | Phase 0–5 DAG · trunk 共享产物 · git/wandb 命名规范 | master agent |
| 03 | [Search-R1 心智模型](https://my.feishu.cn/docx/OFjzdpdPkoWPRgxFj3ScOcWpnIy) | 60 行 agent loop · 两进程架构 · 与 GRPO 的连接 · 后续 phase 在哪改 | master agent |
| 04 | [云端运维大全](https://my.feishu.cn/docx/SnS0dQXACo3iOKxSKJEcqJqyntd) | AutoDL / HF / SSH / SCP 全部踩坑（§1–§9 master，§10 subAgent 1） | master + subAgent 1 |
| 05 | [Phase -1 进度档](https://my.feishu.cn/docx/PKV2dOZnTod9A6xp3ohce0IfnMd) | 8 项交付物 · kickoff 摘要 · 镜像保存策略 | master agent |

---

## 阅读顺序建议

| 你是谁 / 你要干什么 | 推荐读哪几个，按顺序 |
|---|---|
| **新接手这个项目的人** | 01 → 02 → 03 → 05 → 04（理论先行，实操垫底） |
| **要做 Phase 0 baseline** | 05（看清单）→ 03（理解 loop）→ 04（避坑） |
| **debug 云端环境** | 04 直接搜（已分 10 个症状-根因-解法节） |
| **决定下一步算法方向** | 02（DAG）→ 01（决策记录）→ 各 phase 论文 |
| **算预算** | 01 §3.8 |

---

## 关键术语 Glossary（一句话定义）

| 术语 | 一句话 | 详见 |
|---|---|---|
| **Search-R1** | 把搜索作为工具的 RL 训练范式，policy 输出 `<search>...</search>` 由外部 server 注入 `<information>...</information>` | 子 doc 03 |
| **agent loop** | `<think> → <search> → <information> → <think> → <answer>` 的多轮交互循环 | 子 doc 03 §2 |
| **GRPO** | DeepSeek-R1 的核心：对同 prompt 采样 G 条轨迹，按 reward 归一化得 advantage，token-level 更新 | 子 doc 03 §5 |
| **DAPO** | GRPO 的工程化改进 4 件套：动态采样、token-level loss、双 clip、长度衰减 | 子 doc 01 §2.3 |
| **GSPO** | Qwen 团队推荐的 IS 粒度改进：token → sequence level | 子 doc 01 §2.4 |
| **GiGPO** | 步级 advantage：每个 `<search>...<information>` 段单独算 advantage | 子 doc 01 §2.3 |
| **ARPO** | 熵驱动的 rollout 分支策略，前沿算法 | 子 doc 01 §2.3 |
| **verl** | volcengine 出的 RL 训练框架，Search-R1 / DAPO / GSPO 共用 | 子 doc 01 §1.4 |
| **trunk artifacts** | Phase 0 一旦建成、所有后续 phase 共享的产物（FAISS index / eval pipeline / wandb / git 分支） | 子 doc 02 §6 |
| **无卡模式** | AutoDL ¥0.5/h 的省钱状态，不分配 GPU 但实例和盘还在 | 子 doc 01 §3.5 |
| **学术加速** | AutoDL 内置的 PyPI/GitHub/HF 代理 `source /etc/network_turbo` | 子 doc 01 §3.4 / 04 §1 |

---

## Agent 协作约定（subAgent 留给后来者的接力规则）

> 这个项目同时跑了多个 Claude Code agent / worktree。为了知识不分裂、文档不冗余：

1. **Master agent**（主 `main` 分支）拥有所有 docs 的最终所有权
2. **subAgent**（worktree 分支）负责具体子任务，产出新知识时：
   - 优先**追加**到现有子 doc（如新踩坑追加到 §4 云端运维），不新建文件
   - 必须在追加段落前标注**贡献来源**（agent 名 / worktree 名）
   - commit 后由 master agent 决定何时合并回 main
3. **Master doc**（本文）由 master agent 唯一维护：
   - 里面只放**索引和导航**，不放细节内容
   - 子 doc 列表更新时，本文同步更新链接
4. **冗余检查**：新增内容前先全文搜（飞书内置搜索），避免和已有章节重复
5. **死链清理**：master agent 定期校验 5 个子 doc URL 是否还活着

---

## 维护元信息

- 最后一次重写：2026-04-26（subAgent 1 `sweet-archimedes-7376d4` 在飞书首次落盘）
- 源码版本：`docs/feishu-export/` @ git commit
- Feishu doc tokens 留底：`docs/feishu-export/UPLOAD_MANIFEST.json`

### 后续怎么改这套 doc

**唯一推荐路径 — `lark-cli` + 内置 `lark-doc` skill**（user 身份，已 `auth login`，全功能）：

```bash
# 整篇覆盖（最常用 — 改完本地 .md 后同步到飞书）
lark-cli docs +update --api-version v2 \
    --doc "https://my.feishu.cn/docx/<doc_token>" \
    --command overwrite --doc-format markdown \
    --content "@docs/feishu-export/04-cloud-ops.md"

# 局部替换
lark-cli docs +update --api-version v2 --doc "..." \
    --command str_replace --content '<旧文本>...</旧文本><新文本>...</新文本>'

# 追加
lark-cli docs +update --api-version v2 --doc "..." \
    --command append --content '<p>新章节</p>'

# 删除
lark-cli drive +delete --file-token <doc_token> --type docx --yes

# 创建新 doc（如果你写了第 7 个子文档）
lark-cli docs +create --api-version v2 --doc-format markdown \
    --content "@docs/feishu-export/06-foo.md"
```

详细见 `~/.claude/skills/lark-doc/SKILL.md` 及其 references。

### 关于 `scripts/feishu_uploader.py`（已弃用，保留作历史参考）

最初是 subAgent 1 在不知道有 `lark-cli` 的情况下用 stdlib 实现的 OpenAPI 上传脚本。功能被 `lark-cli` 全面覆盖，**不再用于生产**，但保留：

- 作为飞书 OpenAPI（`auth/upload_all/import_tasks`）的最小化参考实现，60 行 Python stdlib
- 当 lark-cli 不可用时的应急 fallback（如纯 CI 环境）

它依赖的 `~/.config/feishu/credentials.json`（app `cli_a964a2e703381bb4`）现已无用，**可以从飞书开发者后台删除该 app**，并 `rm ~/.config/feishu/credentials.json`。
