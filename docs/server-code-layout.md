# 服务器端代码放置规范（必读）

> 服务器是真源，本地是 git 中转。<br/>
> 所有 code 必须放在 `/root/autodl-tmp/agenicRL/` 下，否则会被 sync 漏掉，永远进不了 git。

## ✅ 正确路径（会被 sync）

| 类型 | 路径 |
|---|---|
| 训练 launcher | `/root/autodl-tmp/agenicRL/scripts/skyrl_phaseN_*.sh` |
| Phase 10+ 代码包 | `/root/autodl-tmp/agenicRL/skyrl_patches_phaseN/` |
| 数据 prep / 工具脚本 | `/root/autodl-tmp/agenicRL/scripts/*.py` |
| 论文/笔记 | `/root/autodl-tmp/agenicRL/notes/phaseN/*.md` |
| verl 时代 patch | `/root/autodl-tmp/agenicRL/patches/*.patch` |
| 一次性 ops 脚本（启动 retrieval / setup venv 等） | `/root/autodl-tmp/agenicRL/scripts/*.sh` |

## ❌ 错误路径（sync 看不到，永远丢失）

| ❌ | 为什么不行 |
|---|---|
| `/tmp/launch_retrieval.sh` | 重启即丢，sync 不覆盖 |
| `/root/autodl-tmp/retrieval/launch.sh` | 不在 agenicRL/ 下，rsync 不扫 |
| `/root/foo.py` | 同上 |
| `/root/autodl-tmp/external/SkyRL/skyrl_gym/envs/math_python/env.py` | 这是 SkyRL repo 内，不是 agenicRL |

**对最后一项的处理**：把 patch 源放在 `agenicRL/skyrl_patches_phaseN/` 下，然后用 `deploy.sh` 把它 scp/cp 到 SkyRL 下作为 sym-link 或 copy。这样源代码在 git，部署是 idempotent 的。

## ✅ 不需要 sync 的（已 .gitignore）

```
runs/  ckpts/  wandb/  *.log
.cache/  .skyrl-venv/  venvs/
data/  models/  external/
*.parquet  *.safetensors  *.faiss
nohup.out  *.bak
```

## 工作流

### Claude / 人在服务器上写新代码时

```bash
# ✅ 正确
cd /root/autodl-tmp/agenicRL
mkdir -p scripts/  # 或 skyrl_patches_phaseN/, notes/phaseN/
vim scripts/new_thing.sh

# ❌ 错误
vim /tmp/new_thing.sh
vim /root/scratch/new_thing.sh
```

### git 上传流程（本地操作）

```bash
# 1. 拉服务器最新（不删 local）
bash scripts/sync_with_server.sh pull connect.westd.seetacloud.com 12233

# 2. review
git status
git diff

# 3. commit + push
git add -A
git commit -m "..."
git push
```

### bootstrap：本地比服务器新时（如初次启用此流程）

```bash
# 把本地 git 内容覆盖到服务器
bash scripts/sync_with_server.sh push connect.westd.seetacloud.com 12233 --mirror
# 之后服务器与本地对齐，恢复 server-canonical 单向流
```

### 严格镜像 pull（清理 local 幽灵文件）

```bash
# ⚠️ 会删 local 上服务器没有的文件，慎用，先 git status 确认
bash scripts/sync_with_server.sh pull <host> <port> --mirror
```

## 检查清单（服务器上做完任何工作前自查）

- [ ] 新建的 `.sh` / `.py` / `.md` 是不是在 `/root/autodl-tmp/agenicRL/<某子目录>/` 下？
- [ ] 是否引用了 `/tmp/`、`/root/scratch/` 这种易丢路径？
- [ ] `runs/` `ckpts/` 等大产物是不是在该目录下而不是混进 git 区？

## 例外：训练产物

训练生成的 `runs/`、`ckpts/`、`wandb/` 即使在 `agenicRL/` 下，也通过 `.gitignore` 排除（rsync 同样 exclude），保持 git 干净。这些只活在 `autodl-tmp` 持久盘。
