"""
事后把 verl/Search-R1 console logger 的 step 指标 replay 到 wandb。

用途：训练时 LOGGER=console，结束后用这个脚本把 metrics 推到 wandb dashboard。
所有数值（reward / KL / val 分数 / timing / grad_norm 等）都保留；
只是 wall-clock 时间戳全部贴在 replay 那一秒（不影响曲线分析）。

用法（在云端 conda env searchr1 下）：
    python scripts/replay_to_wandb.py \\
        --log /tmp/training.log \\
        --project agentic-rl-search \\
        --name phase0-baseline-50

前置：~/.netrc 已通过 wandb login 配置过。
"""
import argparse
import re
import sys
from pathlib import Path

import wandb


# 解析 verl console 输出的 step metrics 行：
#   step:1 - actor/kl_loss:0.001 - actor/pg_loss:-0.017 - ... - timing_s/step:1047.749
STEP_RE = re.compile(r"step:(\d+)\s*-\s*(.+)")
KV_RE = re.compile(r"([A-Za-z_][\w/]*?):([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)")
VAL_ONLY_RE = re.compile(r"step:(\d+)\s*-\s*val/test_score/(\S+):([\d.eE+-]+)\s*$")


def parse_log(log_path: Path):
    """返回 [(step, {metric: float, ...}), ...]"""
    rows = []
    with log_path.open() as f:
        for raw in f:
            # 剥掉 ANSI 颜色码 + ray 前缀  e.g. "[36m(main_task pid=...)[0m "
            line = re.sub(r"\x1b\[\d+m", "", raw)
            line = re.sub(r"\(main_task pid=\d+\)\s*", "", line)
            line = line.strip()

            # 优先匹配只有 val 的简洁行
            m = VAL_ONLY_RE.match(line)
            if m:
                step = int(m.group(1))
                key = f"val/test_score/{m.group(2)}"
                rows.append((step, {key: float(m.group(3))}))
                continue

            # 否则匹配完整 step 指标行
            m = STEP_RE.match(line)
            if not m:
                continue
            step = int(m.group(1))
            kvs = dict()
            for km in KV_RE.finditer(m.group(2)):
                k, v = km.group(1), km.group(2)
                try:
                    kvs[k] = float(v)
                except ValueError:
                    pass
            if kvs:
                rows.append((step, kvs))

    # 同一 step 多行（先 val 后 metrics 或反之）合并
    merged = {}
    for step, kvs in rows:
        merged.setdefault(step, {}).update(kvs)
    return sorted(merged.items())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, help="path to training console log")
    p.add_argument("--project", default="agentic-rl-search")
    p.add_argument("--name", required=True, help="wandb run name, e.g. phase0-baseline-50")
    p.add_argument("--config", default=None, help="optional wandb run config JSON")
    args = p.parse_args()

    log = Path(args.log)
    if not log.exists():
        print(f"❌ log not found: {log}", file=sys.stderr)
        sys.exit(1)

    rows = parse_log(log)
    if not rows:
        print(f"❌ no step metrics parsed from {log}", file=sys.stderr)
        sys.exit(1)

    print(f"parsed {len(rows)} step rows from {log}")
    first_step, first_kvs = rows[0]
    print(f"first step={first_step}, sample keys: {list(first_kvs.keys())[:6]}")

    cfg = {"replayed_from": str(log), "n_steps": len(rows)}
    if args.config:
        import json
        cfg.update(json.loads(Path(args.config).read_text()))

    run = wandb.init(project=args.project, name=args.name, config=cfg, reinit=True)
    print(f"wandb run url: {run.url}")

    for step, kvs in rows:
        wandb.log(kvs, step=step)
    wandb.finish()
    print("✅ replay done")


if __name__ == "__main__":
    main()
