"""
拉 wandb 指标 + ASCII 趋势图 + Phase 0d/0e 对照（云端运行）

用法：
    # 默认拉当前 Phase 0e run
    python scripts/pull_wandb_metrics.py

    # 指定 run 名
    python scripts/pull_wandb_metrics.py --run phase2-dapo-cliphigh-50

    # 只看关键指标
    python scripts/pull_wandb_metrics.py --keys "val/test_score/nq,critic/rewards/mean"

    # 输出 JSON 给程序消费
    python scripts/pull_wandb_metrics.py --json

设计：纯只读，不写 wandb。父 agent 5min 心跳里调用此脚本。
"""
import argparse
import json
import sys
import os

DEFAULT_RUN = "phase1-deepseek-format-reward-50-coldstart"
DEFAULT_PROJECT = "agentic-rl-search"
DEFAULT_KEYS = [
    "val/test_score/nq",
    "critic/rewards/mean",
    "response_length/mean",
    "actor/grad_norm",
    "actor/kl_loss",
    "actor/pg_loss",
    "actor/entropy_loss",
    "env/finish_ratio",
    "env/number_of_valid_search",
    "timing_s/step",
]


def ascii_sparkline(values, width=40, height=6):
    """ASCII 趋势图，纯 stdlib"""
    if not values:
        return "(no data)"
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return f"|{'─' * width}| (flat={vmin:.3f})"
    # 取最近 width 个点
    pts = values[-width:]
    span = vmax - vmin
    rows = [[" "] * len(pts) for _ in range(height)]
    for col, v in enumerate(pts):
        norm = (v - vmin) / span
        row = height - 1 - int(round(norm * (height - 1)))
        rows[row][col] = "●"
    out = []
    for i, row in enumerate(rows):
        if i == 0:
            mark = f"  {vmax:.3f} ─┤"
        elif i == height - 1:
            mark = f"  {vmin:.3f} ─┤"
        else:
            mark = "         │"
        out.append(mark + "".join(row))
    out.append(f"           └─{'─' * len(pts)}→ step {pts and 'last:' + format(pts[-1], '.3f') or ''}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=DEFAULT_RUN, help="wandb run display_name (regex match)")
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--keys", default=",".join(DEFAULT_KEYS))
    p.add_argument("--json", action="store_true", help="output JSON for downstream")
    p.add_argument("--last", type=int, default=10, help="show last N rows in table")
    p.add_argument("--no-sparkline", action="store_true")
    p.add_argument("--compare", default=None, help="another run name to compare against")
    args = p.parse_args()

    try:
        import wandb
    except ImportError:
        print("❌ wandb 未安装，pip install wandb", file=sys.stderr)
        sys.exit(1)

    api = wandb.Api()
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]

    runs = api.runs(args.project, filters={"display_name": {"$regex": args.run}})
    runs = list(runs)
    if not runs:
        print(f"❌ 没找到 run 匹配 {args.run!r}", file=sys.stderr)
        sys.exit(2)

    # 优先级：running > finished > crashed/failed (crashed 通常是被 kill 的旧 run，要避开)
    state_priority = {"running": 0, "finished": 1, "crashed": 2, "failed": 2}
    runs_sorted = sorted(runs, key=lambda r: (state_priority.get(r.state, 3), -int(r.created_at.replace("-","").replace(":","").replace("T","").replace("Z","")[:14] or "0")))
    primary = runs_sorted[0]
    # 关键修复：history(keys=...) 在某些 wandb 版本下要求所有 keys 都有值才返回行；
    # val_*/* 是稀疏 key（只在 test_freq 步出现），会把训练步全过滤掉。
    # 解决方案：分开拉训练 keys 和 val keys，再 outer-merge
    val_keys = [k for k in keys if k.startswith("val/")]
    train_keys = [k for k in keys if not k.startswith("val/")]
    h_train = primary.history(keys=train_keys, samples=500) if train_keys else None
    h_val = primary.history(keys=val_keys, samples=200) if val_keys else None
    if h_train is None:
        hist = h_val
    elif h_val is None or len(h_val) == 0:
        hist = h_train
    else:
        import pandas as pd
        hist = pd.merge(h_train, h_val, on="_step", how="outer").sort_values("_step").reset_index(drop=True)

    if args.json:
        out = {
            "run_name": primary.name,
            "run_id": primary.id,
            "state": primary.state,
            "history": hist.fillna("").to_dict(orient="records") if len(hist) else [],
        }
        print(json.dumps(out, indent=2, default=str))
        return

    # human format
    print(f"# Run: {primary.name} (id={primary.id})  state={primary.state}")
    print(f"  rows: {len(hist)}, latest _step: {hist['_step'].max() if len(hist) else 'n/a'}")
    print()

    if len(hist) == 0:
        print("(暂无 history，可能还在 init / val_before_train 阶段)")
        return

    print(f"## Last {min(args.last, len(hist))} rows")
    cols = [c for c in hist.columns if c != "_step"]
    # 行表格（数字截断）
    print("| step | " + " | ".join(c.split("/")[-1][:14] for c in cols) + " |")
    print("|" + "---|" * (1 + len(cols)))
    for _, row in hist.tail(args.last).iterrows():
        cells = [str(int(row["_step"]))]
        for c in cols:
            v = row[c]
            cells.append(f"{v:.3f}" if isinstance(v, (int, float)) and v == v else "-")
        print("| " + " | ".join(cells) + " |")
    print()

    if not args.no_sparkline:
        print("## Trends (recent steps)")
        for c in cols:
            series = [v for v in hist[c].tolist() if v == v]
            if len(series) >= 2:
                print(f"\n### {c}")
                print(ascii_sparkline(series))
        print()

    if args.compare:
        comp_runs = list(api.runs(args.project, filters={"display_name": {"$regex": args.compare}}))
        if comp_runs:
            comp = comp_runs[0]
            comp_hist = comp.history(keys=keys, samples=500)
            print(f"## A/B compare vs {comp.name}")
            print("| metric | this final | compare final | Δ |")
            print("|---|---|---|---|")
            for c in cols:
                a = hist[c].dropna().iloc[-1] if len(hist[c].dropna()) else None
                b = comp_hist[c].dropna().iloc[-1] if c in comp_hist.columns and len(comp_hist[c].dropna()) else None
                if a is not None and b is not None:
                    delta = a - b
                    sign = "+" if delta >= 0 else ""
                    print(f"| {c} | {a:.3f} | {b:.3f} | {sign}{delta:.3f} |")
                else:
                    print(f"| {c} | {a if a is not None else '-'} | {b if b is not None else '-'} | - |")


if __name__ == "__main__":
    main()
