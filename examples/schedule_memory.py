"""Schedule consolidation for your own memory units with TIDE.

    python examples/schedule_memory.py examples/data/memory_units.jsonl --budget 0.4 --cycle 4

Each line of the input is one memory unit: ``id``, ``session`` (when it was written), ``text``, ``entities``,
``tokens``, the ``slot`` it asserts (``[subject, relation]``, or null) with its ``value``, and ``p_access``, the
probability that the unit is read within the scheduling horizon. Any model can supply ``p_access``; TIDE's own is
``tide.HazardModel``, a discrete-time survival model fitted on censored access logs.

Every ``cycle`` sessions the pending units are scheduled under the budget, a share of the tokens that eager
consolidation (every unit into the graph) would spend. The value of an action combines the probability of access,
the unit's links to other units (shared entities) and its conflict with the belief already consolidated for its
slot, minus a storage cost. The script prints each unit's action and the tokens spent against eager.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from tide import MemoryUnit, TieredStore, schedule_cycle
from tide.types import Action


def load(path: str) -> list[MemoryUnit]:
    units = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        u = MemoryUnit(d["id"], d["text"], tuple(d["entities"]), d["session"], np.zeros(1), d["tokens"],
                       slot=tuple(d["slot"]) if d.get("slot") else None, value=d.get("value"))
        u.p_access = float(d["p_access"])
        units.append(u)
    return units


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("units", help="JSON Lines file of memory units")
    ap.add_argument("--budget", type=float, default=0.4, help="share of eager consolidation tokens per cycle")
    ap.add_argument("--cycle", type=int, default=4, help="sessions between scheduling cycles")
    args = ap.parse_args(argv)
    units = load(args.units)
    store, actions, spent_total, eager_total = TieredStore(), {}, 0.0, 0.0
    last = max(u.created for u in units)
    for end in range(args.cycle - 1, last + args.cycle, args.cycle):
        pending = [u for u in units if end - args.cycle < u.created <= end]
        if not pending:
            continue
        for u in pending:
            store.add_raw(u)
        p = {u.uid: u.p_access for u in pending}
        kappa = {u.uid: store.kappa(u) for u in pending}
        chi = {u.uid: store.chi(u) for u in pending}
        act, spent, eager = schedule_cycle(pending, p, kappa, chi, args.budget, now=end)
        print(f"cycle ending at session {end}: spent {spent:.3f} of eager {eager:.3f} (k tokens)")
        for u in sorted(pending, key=lambda u: u.uid):
            note = f"conflicts with '{store.graph[u.slot][0]}'" if chi[u.uid] else ""
            store.apply(u, act[u.uid])
            actions[u.uid] = act[u.uid]
            print(f"  {u.uid:>3} {act[u.uid].value:<8} p={u.p_access:.2f} links={kappa[u.uid]:.0f}  {u.text}" + (f"  [{note}]" if note else ""))
        spent_total += spent
        eager_total += eager
    counts = Counter(a.value for a in actions.values())
    print(f"\nspent {spent_total / eager_total:.0%} of eager consolidation; " + ", ".join(f"{n} {a}" for a, n in counts.items()))
    beliefs = {f"{s[0]}.{s[1]}": v for s, (v, _, _) in store.graph.items()}
    print("consolidated beliefs:", ", ".join(f"{k} = {v}" for k, v in sorted(beliefs.items())))
    return {"actions": {k: v.value for k, v in actions.items()}, "spent_frac": spent_total / eager_total, "beliefs": beliefs,
            "forgot": [k for k, v in actions.items() if v is Action.FORGET]}


if __name__ == "__main__":
    main()
