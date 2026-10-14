"""TIDE on the CPU toy stream: consolidation policies at a range of budgets.

    python -m tide.demo [--config configs/toy.yaml]

For each budget (a fraction of the tokens eager consolidation spends) and each policy, the toy stream is consolidated
cycle by cycle and the store answers current-value queries. TIDE ranks units by the hazard model's probability of
access; the baselines rank them by recency or at random; eager consolidates everything into the graph.
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import yaml

from .sim import run_toy


def run(cfg: dict) -> dict:
    out = {}
    for policy in cfg["policies"]:
        for budget in ([1.0] if policy == "eager" else cfg["budgets"]):
            runs = [run_toy(budget, policy, seed=s, stream=cfg.get("stream"), **cfg["schedule"]) for s in cfg["seeds"]]
            out[(policy, budget)] = {k: statistics.mean(r[k] for r in runs) for k in ("accuracy", "spent_frac", "auroc")}
    return out


def config_path(name: str) -> Path:
    """``name`` from the working directory, else from the checkout this package is installed from."""
    path = Path(name)
    if not path.exists() and (Path(__file__).resolve().parents[2] / name).exists():
        return Path(__file__).resolve().parents[2] / name
    return path


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/toy.yaml")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(config_path(args.config).read_text())
    t0 = time.time()
    res = run(cfg)
    ranked = [p for p in cfg["policies"] if p != "eager"]
    seeds = cfg["seeds"]
    print(f"current-value accuracy, mean over seeds {seeds[0]}-{seeds[-1]}; budget = share of eager consolidation tokens")
    print(f"{'budget':>6}" + "".join(f"{p:>10}" for p in ranked) + f"{'max spent':>11}")
    for b in cfg["budgets"]:
        spent = max(res[(p, b)]["spent_frac"] for p in ranked)
        print(f"{b:>6.2f}" + "".join(f"{res[(p, b)]['accuracy']:>10.3f}" for p in ranked) + f"{spent:>11.2f}")
    if "eager" in cfg["policies"]:
        print(f"eager consolidation (everything to the graph): accuracy {res[('eager', 1.0)]['accuracy']:.3f}")
    auroc = statistics.mean(res[(ranked[0], b)]["auroc"] for b in cfg["budgets"])
    print(f"hazard model AUROC for access within {cfg['schedule']['delta']} sessions: {auroc:.3f}")
    print(f"done in {time.time() - t0:.1f}s")
    return res


if __name__ == "__main__":
    main()
