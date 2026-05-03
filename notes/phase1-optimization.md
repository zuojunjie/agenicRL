# Phase 1+ 性能优化方案：retrieval 独占 GPU 3

## 背景

Phase 0d 跑的是 **4 卡共享布局**：
- 4 卡每卡 `faiss shard 9.7GB + vllm rollout 16-22GB` 共占 ~30GB
- FSDP 全 offload 到 CPU（actor update 慢，~155s）
- OOM 边界紧，已踩过一次（vllm util 0.6 → 0.4 修）

3B 模型 + 48GB×4 卡有大量未利用显存。Phase 1 起切到**dedicated retrieval** 布局以稳定 + 提速。

## 新布局

```
┌─ GPU 3 (retrieval 独占) ────────────────────┐
│  faiss IndexFlat fp16 (32GB) 整张卡          │
│  e5 encoder + KV cache + overhead ~3GB       │
│  total ~35GB / 48GB ✓                         │
└──────────────────────────────────────────────┘

┌─ GPU 0,1,2 (训练独占) ───────────────────────┐
│  vllm gpu_memory_utilization=0.4 = 19.2GB    │
│  FSDP shard (param+grad on GPU, optim CPU):   │
│    params 6.17/3 = 2.06GB                     │
│    grads  6.17/3 = 2.06GB                     │
│  activations + buffers ~5GB                   │
│  total ~28GB / 48GB ✓ (~20GB free buffer)    │
└──────────────────────────────────────────────┘
```

## 启动顺序

```bash
# 1. 杀当前 4 卡 shard 的 retrieval
ssh autodl-agenicrl 'kill $(cat /tmp/retrieval.pid); sleep 2'

# 2. 起单卡独占 retrieval（在 GPU 3）
ssh autodl-agenicrl 'bash /root/autodl-tmp/agenicRL/scripts/launch_retrieval_dedicated.sh'
# 等 ~5-10 min faiss 加载 + 自带 ready 检测

# 3. 起训练（用 GPU 0/1/2 + minimal offload + wandb）
ssh autodl-agenicrl '
cd /root/autodl-tmp/external/Search-R1
nohup env \
    MAX_STEPS=50 TEST_FREQ=20 SAVE_FREQ=25 \
    EXPERIMENT_NAME=phase1-grpo-dapo \
    LOGGER=wandb \
    TRAIN_GPUS=0,1,2 \
    FSDP_MODE=minimal \
    bash /root/autodl-tmp/agenicRL/scripts/phase0_train_grpo.sh \
    > /tmp/training.log 2>&1 &
'
```

## 预期收益

| 指标 | 4 卡共享 (Phase 0d) | 3+1 dedicated (Phase 1+) |
|---|---|---|
| 每步耗时 | 6.3 min | ~5.5 min (-13%) |
| update_actor | 155s | ~80s (param/grad on GPU, faster fwd/bwd) |
| OOM 风险 | 边界 (3GB free) | 充裕 (20GB free) |
| 50 步总时长 | 5.8h | 5.0h |
| Phase 2-5 累计省 | 0 | ~5h × ¥6 = ¥30 |

## 风险

- **首次启动 5-10 min faiss 单卡加载未实测**——3GB 余量需小心
- **若 minimal offload 仍 OOM** → fallback `FSDP_MODE=full` 撑过去（损失 75s/step）

## 决策原则

- **Phase 0 baseline 完成后**才切（不打断 50 步进度）
- Phase 1 第一次启动训练时切，**实测真效益**
- 如果 Phase 1 顺利 → Phase 2-5 全用此布局
