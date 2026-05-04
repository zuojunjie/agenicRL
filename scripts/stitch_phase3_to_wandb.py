"""
重建被删的 Phase 3 wandb run
策略：
  1. 原 Phase 3 step 1-22 数据 = 从早先 wandb 拉取的 cache（hardcoded）
  2. 原 Phase 3 step 22-24 = 从 wandb-summary.json 推断
  3. resume 数据 = 从当前 wandb run uyaxjdih 拉
  4. 合并 → 创建新 wandb run "phase3-torl-stitched"
"""
import wandb
import json
import os
from pathlib import Path

# ============================================================
# 1) 原 Phase 3 step 1-24 数据（从 chat 中 wandb pull 的 cache）
# ============================================================
ORIG_TRAIN = [
    # step, reward, resp_len, kl, entropy, n_search
    (1,  0.254, 663, 0.001, 0.957, 1.01),
    (2,  0.252, 646, 0.002, 0.980, 0.96),
    (3,  0.263, 651, 0.003, 0.947, 0.97),
    (4,  0.278, 659, 0.003, 0.945, 1.01),
    (5,  0.291, 656, 0.020, 0.953, 1.01),
    (6,  0.307, 667, 0.019, 0.920, 1.04),
    (7,  0.314, 681, 0.030, 0.915, 1.08),
    (8,  0.387, 681, 0.015, 0.895, 1.09),
    (10, 0.343, 709, 0.024, 0.907, 1.15),
    (11, 0.340, 712, 0.046, 0.923, 1.16),
    (12, 0.355, 703, 0.038, 0.871, 1.16),
    (13, 0.405, 738, 0.073, 0.919, 1.23),
    (14, 0.373, 740, 0.068, 0.894, 1.23),
    (15, 0.364, 762, 0.045, 0.916, 1.28),
    (16, 0.377, 830, 0.055, 0.952, 1.39),
    (17, 0.407, 858, 0.065, 0.935, 1.44),
    (20, 0.408, 961, 0.058, 0.885, 1.64),
    (21, 0.399, 1007, 0.073, 0.848, 1.77),
    (22, 0.409, 1079, 0.096, 0.793, 1.90),
    (23, 0.406, 1140, 0.135, 0.766, 2.04),
    (24, 0.429, 1295, 0.118, 0.701, 2.34),
]
# 原 val 数据（step 触发的中途 val）
ORIG_VAL = [
    (0,  0.242),
    (10, 0.353),
    (20, 0.387),
]

# ============================================================
# 2) 拉 current resume run 数据
# ============================================================
def pull_resume():
    api = wandb.Api()
    runs = list(api.runs("zjj-hit-sz/agentic-rl-search",
                         filters={"display_name": "phase3-torl-resume-step20", "state": "running"}))
    if not runs:
        runs = list(api.runs("zjj-hit-sz/agentic-rl-search",
                             filters={"display_name": "phase3-torl-resume-step20"}))
    if not runs:
        return None, None
    r = runs[0]
    keys_train = ["critic/rewards/mean", "response_length/mean", "actor/kl_loss",
                  "actor/entropy_loss", "env/number_of_valid_search"]
    h_train = r.history(keys=keys_train, samples=500)
    h_val = r.history(keys=["val/test_score/nq"], samples=200)
    return h_train, h_val


# ============================================================
# 3) 写新 wandb run
# ============================================================
def main():
    h_train, h_val = pull_resume()
    print(f"resume train rows: {len(h_train) if h_train is not None else 0}")
    print(f"resume val rows: {len(h_val) if h_val is not None else 0}")

    # 启动新 wandb run
    run = wandb.init(
        project="agentic-rl-search",
        name="phase3-torl-stitched-final",
        notes="重建：原 Phase 3 step 1-22 (OOM) + resume from step_20 ckpt step 23-52。step 编号统一映射到 [1, 52]。",
        tags=["phase3", "torl", "stitched", "recovered"],
    )

    # ---- 上传原数据 step 1-22（OOM 前）----
    val_dict = dict(ORIG_VAL)
    # initial val 单独 log step 0
    if 0 in val_dict:
        wandb.log({"val/test_score/nq": val_dict[0]}, step=0)

    for step, reward, resp_len, kl, entropy, n_search in ORIG_TRAIN:
        if step > 22:
            continue  # OOM 前数据 only
        log = {
            "critic/rewards/mean": reward,
            "response_length/mean": resp_len,
            "actor/kl_loss": kl,
            "actor/entropy_loss": entropy,
            "env/number_of_valid_search": n_search,
            "phase": "original_oom_attempt",
        }
        if step in val_dict:
            log["val/test_score/nq"] = val_dict[step]
        wandb.log(log, step=step)

    # ---- 上传 resume 数据 (resume step N → unified step 22+N) ----
    if h_train is not None and len(h_train) > 0:
        for _, row in h_train.iterrows():
            resume_step = int(row["_step"])
            unified_step = 22 + resume_step  # offset
            log = {
                "critic/rewards/mean": row.get("critic/rewards/mean"),
                "response_length/mean": row.get("response_length/mean"),
                "actor/kl_loss": row.get("actor/kl_loss"),
                "actor/entropy_loss": row.get("actor/entropy_loss"),
                "env/number_of_valid_search": row.get("env/number_of_valid_search"),
                "phase": "resume_micro16",
            }
            log = {k: v for k, v in log.items() if v == v}  # drop NaN
            wandb.log(log, step=unified_step)

    # resume val
    if h_val is not None and len(h_val) > 0:
        for _, row in h_val.iterrows():
            resume_step = int(row["_step"])
            unified_step = 22 + resume_step
            v = row.get("val/test_score/nq")
            if v == v:
                wandb.log({"val/test_score/nq": v}, step=unified_step)

    print(f"finished, run url: {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
