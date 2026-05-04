"""
Phase 3 ToRL 完整 stitch: recovrp3a (原 OOM 前) + uyaxjdih (resume) → 新 unified run
v2: 用恢复后的完整 50+ metrics，不再 hardcoded 5 metrics

Step 映射:
  recovrp3a step 0-20 → unified 0-20
  uyaxjdih step 1+   → unified 21+
"""
import wandb
import time

PROJECT = "zjj-hit-sz/agentic-rl-search"
RECOV_ID = "recovrp3a"
RESUME_NAME = "phase3-torl-resume-step20"
NEW_NAME = "phase3-torl-stitched-v2"


def get_running_run_by_name(api, name):
    runs = list(api.runs(PROJECT, filters={"display_name": name}))
    runs_running = [r for r in runs if r.state == "running"]
    if runs_running:
        return runs_running[0]
    return runs[0] if runs else None


def main():
    api = wandb.Api()
    print("--- pulling source runs ---")
    recov = api.run(f"{PROJECT}/{RECOV_ID}")
    print(f"  recovrp3a: state={recov.state}")
    resume = get_running_run_by_name(api, RESUME_NAME)
    print(f"  uyaxjdih:  state={resume.state}")

    # 拉所有 history（不指定 keys 避免稀疏 key 过滤）
    h_recov = recov.history(samples=500, pandas=True)
    h_resume = resume.history(samples=500, pandas=True)
    print(f"  recovrp3a rows: {len(h_recov)}, columns: {len(h_recov.columns)}")
    print(f"  uyaxjdih rows:  {len(h_resume)}, columns: {len(h_resume.columns)}")

    # 启新 run
    run = wandb.init(
        project="agentic-rl-search",
        name=NEW_NAME,
        notes="Phase 3 full timeline: original step 1-20 (recovered) + resume step 21+ (ppo_micro=16). Same reward function (ToRL DFA + per-turn shaping).",
        tags=["phase3", "torl", "stitched-v2", "complete"],
    )
    print(f"new run: {run.url}")

    # 1. 上传原始 recovrp3a step 0-20
    print("--- uploading original (step 0-20) ---")
    n_orig = 0
    for _, row in h_recov.iterrows():
        s = row.get("_step")
        if s is None or s != s:  # NaN check
            continue
        s = int(s)
        if s > 20:  # 跳过 OOM 区 21-24
            continue
        # 把 row 里所有非空字段拷过来
        log = {}
        for col, val in row.items():
            if col == "_step":
                continue
            if val is None:
                continue
            try:
                if val != val:  # NaN
                    continue
            except:
                pass
            log[col] = val
        log["phase"] = "original"
        wandb.log(log, step=s)
        n_orig += 1
    print(f"  logged {n_orig} rows from original")

    # 2. 上传 resume，offset +20
    print("--- uploading resume (step 21+) ---")
    n_resume = 0
    for _, row in h_resume.iterrows():
        s = row.get("_step")
        if s is None or s != s:
            continue
        s = int(s)
        if s == 0:  # initial val 与 original step 20 重复
            continue
        unified_step = 20 + s
        log = {}
        for col, val in row.items():
            if col == "_step":
                continue
            if val is None:
                continue
            try:
                if val != val:
                    continue
            except:
                pass
            log[col] = val
        log["phase"] = "resume_micro16"
        wandb.log(log, step=unified_step)
        n_resume += 1
    print(f"  logged {n_resume} rows from resume")

    print(f"\nTotal: {n_orig + n_resume} rows in unified run")
    print(f"URL: {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
