"""
解析 verl console training log → 生成 metrics.csv + summary.md

用法（云端）:
    python scripts/generate_summary.py \\
        --log /tmp/training.log \\
        --output-dir /root/autodl-tmp/runs/phase0d-baseline-50 \\
        --run-name "Phase 0d Baseline 50 steps" \\
        --notes "GRPO + Qwen2.5-3B-Instruct + 4 卡共享布局"

输出:
    {output-dir}/metrics.csv
    {output-dir}/summary.md
"""
import argparse
import csv
import re
import sys
from pathlib import Path
from datetime import datetime


# 重要指标白名单（其他可计算但不展示）
KEY_METRICS = [
    ("critic/rewards/mean",         "reward"),
    ("val/test_score/nq",            "val_nq"),
    ("actor/grad_norm",              "grad_norm"),
    ("actor/kl_loss",                "kl_loss"),
    ("actor/pg_loss",                "pg_loss"),
    ("actor/entropy_loss",           "entropy"),
    ("env/finish_ratio",             "finish_ratio"),
    ("env/number_of_valid_search",   "n_search"),
    ("env/number_of_actions/mean",   "n_actions"),
    ("response_length/mean",         "resp_len"),
    ("timing_s/step",                "step_time_s"),
    ("timing_s/gen",                 "gen_s"),
    ("timing_s/update_actor",        "update_s"),
]


def parse_log(log_path: Path):
    """每个 step 的 metrics dict"""
    rows = {}
    with log_path.open() as f:
        for raw in f:
            # 剥 ANSI
            line = re.sub(r"\x1b\[\d+m", "", raw)
            line = re.sub(r"\(main_task pid=\d+\)\s*", "", line)
            line = line.strip()

            m = re.search(r"step:(\d+)\s*-\s*(.+)", line)
            if not m:
                # val 单行格式: "step:N - val/test_score/nq:VALUE"
                vm = re.match(r"step:(\d+)\s*-\s*val/test_score/(\S+):([\d.eE+-]+)\s*$", line)
                if vm:
                    step = int(vm.group(1))
                    rows.setdefault(step, {})[f"val/test_score/{vm.group(2)}"] = float(vm.group(3))
                continue

            step = int(m.group(1))
            kvs = rows.setdefault(step, {})
            for km in re.finditer(r"([A-Za-z_][\w/]*?):([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)", m.group(2)):
                k, v = km.group(1), km.group(2)
                try:
                    kvs[k] = float(v)
                except ValueError:
                    pass

    return sorted(rows.items())


def write_csv(rows, csv_path: Path):
    """写所有 metric 的完整 CSV"""
    if not rows:
        return
    all_keys = set()
    for _, kvs in rows:
        all_keys.update(kvs.keys())
    cols = ["step"] + sorted(all_keys)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for step, kvs in rows:
            w.writerow([step] + [kvs.get(k, "") for k in cols[1:]])


def fmt(v, prec=3):
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def write_summary(rows, output: Path, run_name: str, notes: str):
    """生成飞书友好的 markdown"""
    lines = []
    lines.append(f"# {run_name}")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")

    if notes:
        lines.append("## 说明")
        lines.append("")
        lines.append(notes)
        lines.append("")

    if not rows:
        lines.append("⚠️ 没解析到任何 step 数据。")
        output.write_text("\n".join(lines))
        return

    # 关键指标趋势
    n_steps = max(s for s, _ in rows) if rows else 0
    init_val = rows[0][1].get("val/test_score/nq")
    last_train_reward = None
    last_val = None
    for step, kvs in reversed(rows):
        if "critic/rewards/mean" in kvs and last_train_reward is None:
            last_train_reward = (step, kvs["critic/rewards/mean"])
        if "val/test_score/nq" in kvs and last_val is None:
            last_val = (step, kvs["val/test_score/nq"])
        if last_train_reward and last_val:
            break

    lines.append("## 关键结果")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 总训练 steps | {n_steps} |")
    if init_val is not None:
        lines.append(f"| Initial val/nq | {fmt(init_val)} |")
    if last_val:
        s, v = last_val
        delta = v - init_val if init_val else None
        delta_pct = (delta / init_val * 100) if init_val else None
        lines.append(f"| Final val/nq (step {s}) | {fmt(v)} |")
        if delta is not None:
            sign = "+" if delta >= 0 else ""
            lines.append(f"| val 提升 | {sign}{fmt(delta)} ({sign}{fmt(delta_pct, 1)}%) |")
    if last_train_reward:
        s, v = last_train_reward
        lines.append(f"| Final train reward (step {s}) | {fmt(v)} |")
    lines.append("")

    # val 曲线
    val_points = [(s, kvs.get("val/test_score/nq")) for s, kvs in rows if kvs.get("val/test_score/nq") is not None]
    if val_points:
        lines.append("## Validation 曲线")
        lines.append("")
        lines.append("| step | val/test_score/nq |")
        lines.append("|---|---|")
        for s, v in val_points:
            lines.append(f"| {s} | {fmt(v)} |")
        lines.append("")

    # 完整 step 表
    lines.append("## 全 step 指标表")
    lines.append("")
    headers = ["step"] + [label for _, label in KEY_METRICS]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for step, kvs in rows:
        cells = [str(step)]
        for k, _ in KEY_METRICS:
            cells.append(fmt(kvs.get(k)))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # 时间统计
    step_times = [kvs.get("timing_s/step") for _, kvs in rows if kvs.get("timing_s/step") is not None]
    if step_times:
        lines.append("## 时间分析")
        lines.append("")
        avg_t = sum(step_times) / len(step_times)
        lines.append(f"- 平均每步耗时: {avg_t:.1f}s ({avg_t/60:.1f} min)")
        lines.append(f"- 总训练 wall-clock: {sum(step_times):.0f}s ({sum(step_times)/3600:.2f}h)")
        lines.append("")

    output.write_text("\n".join(lines))
    print(f"✅ summary.md → {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--notes", default="")
    args = p.parse_args()

    log = Path(args.log)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not log.exists():
        print(f"❌ log not found: {log}", file=sys.stderr)
        sys.exit(1)

    rows = parse_log(log)
    print(f"parsed {len(rows)} steps from {log}")

    write_csv(rows, out / "metrics.csv")
    write_summary(rows, out / "summary.md", args.run_name, args.notes)
    print(f"✅ metrics.csv → {out / 'metrics.csv'}")


if __name__ == "__main__":
    main()
