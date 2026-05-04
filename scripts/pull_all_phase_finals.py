"""拉所有 phase 最终 val/nq + key metrics，给 5 phase summary 用"""
import wandb

api = wandb.Api()
runs = list(api.runs("agentic-rl-search"))
print("=== ALL phase final val/nq ===")
for r in sorted(runs, key=lambda x: x.created_at):
    if r.history(samples=1).empty:
        continue
    hv = list(r.scan_history(keys=["val/test_score/nq", "_step"]))
    if not hv:
        continue
    final = max(hv, key=lambda v: v["_step"])
    init = next((v for v in hv if v["_step"] == 0), None)
    init_v = init["val/test_score/nq"] if init else 0.0
    fin_v = final["val/test_score/nq"]
    fin_s = final["_step"]
    print(f"  {r.name[:48]:<48} {r.id} | init={init_v:.3f} final@step{fin_s}={fin_v:.3f}")
