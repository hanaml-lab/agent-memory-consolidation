"""Run the toy stream for every policy, budget and seed in a config and print a summary table."""
import argparse, sys, os, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import yaml
from tide import run_toy

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/toy.yaml"); a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
print(f"{'policy':8s} {'budget':>6s} {'acc':>6s} {'spent':>6s}")
for pol in cfg["policies"]:
    for b in (cfg["budgets"] if pol != "eager" else [1.0]):
        rs = [run_toy(b, pol, seed=s, **cfg["schedule"]) for s in cfg["seeds"]]
        acc = [r["accuracy"] for r in rs]
        print(f"{pol:8s} {b:6.2f} {statistics.mean(acc):6.3f} {statistics.mean(r['spent_frac'] for r in rs):6.2f}")
