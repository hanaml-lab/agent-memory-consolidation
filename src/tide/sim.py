"""Toy stream for an end-to-end CPU run: users state facts about slots, some facts change later,
and queries ask for the current value. Access propensity depends on unit features."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .types import MemoryUnit, Action
from .hazard import HazardModel, auroc
from .scheduler import schedule_cycle
from .store import TieredStore


@dataclass
class ToyStream:
    n_sessions: int = 60
    units_per_session: int = 12
    n_slots: int = 40
    d: int = 8
    seed: int = 0

    def generate(self):
        r = np.random.default_rng(self.seed)
        slot_names = [("user", f"attr{i}") for i in range(self.n_slots)]
        popularity = r.dirichlet(np.ones(self.n_slots) * 0.4)
        units, uid = [], 0
        for s in range(self.n_sessions):
            for _ in range(self.units_per_session):
                is_fact = r.random() < 0.45
                slot = slot_names[r.choice(self.n_slots, p=popularity)] if is_fact else None
                user = r.random() < 0.6
                x = r.normal(0, 1, self.d)
                x[0] = float(user)
                x[1] = float(is_fact)
                x[2] = np.log(popularity[int(slot[1][4:])] * self.n_slots) if slot else -2.0
                units.append(MemoryUnit(uid, f"u{uid}", (slot[1],) if slot else (f"misc{uid % 7}",), s, x,
                                        n_tokens=int(r.integers(80, 400)), slot=slot,
                                        value=f"v{s}_{int(r.integers(3))}" if slot else None))
                uid += 1
        return units, slot_names, popularity

    def access_logits(self, X):
        return -3.0 + 1.2 * X[:, 0] + 1.0 * X[:, 1] + 0.9 * X[:, 2] + 0.3 * X[:, 3]


def _simulate_access(stream, units, horizon, rng):
    X = np.stack([u.features for u in units])
    lam = 1 / (1 + np.exp(-stream.access_logits(X)))
    k = rng.geometric(np.clip(lam, 1e-3, 1))
    d = (k <= horizon).astype(int)
    return X, np.minimum(k, horizon), d


def run_toy(budget: float = 0.4, policy: str = "tide", seed: int = 0, cycle: int = 2, delta: int = 16, verbose: bool = False,
            stream: dict | None = None):
    """Run a full stream; returns dict with accuracy on current-value queries, spent fraction, AUROC.
    ``stream`` overrides ToyStream's sizes (n_sessions, units_per_session, n_slots, d)."""
    shape = dict(stream or {})
    stream = ToyStream(seed=seed, **shape)
    units, slots, pop = stream.generate()
    rng = np.random.default_rng(seed + 1)
    # train hazard on a separate replay (probe-access simulation), never on eval queries
    tr_units, _, _ = ToyStream(seed=seed + 100, **shape).generate()
    Xtr, ktr, dtr = _simulate_access(stream, tr_units, 32, rng)
    hz = HazardModel(stream.d, max_k=32, seed=seed)
    hz.fit(Xtr, ktr, dtr, steps=250)
    Xte, kte, dte = _simulate_access(stream, units, 32, rng)
    auc = auroc(hz.p_within(Xte, delta), (kte <= delta) & (dte == 1))

    store, pending, spent_tot, eager_tot = TieredStore(), [], 0.0, 0.0
    for s in range(stream.n_sessions):
        new = [u for u in units if u.created == s]
        for u in new:
            store.add_raw(u)
        pending += new
        if (s + 1) % cycle:
            continue
        X = np.stack([u.features for u in pending])
        if policy == "tide":
            p = dict(zip([u.uid for u in pending], hz.p_within(X, delta)))
        elif policy == "recency":
            p = {u.uid: 1.0 / (1 + s - u.created) for u in pending}
        elif policy == "random":
            p = {u.uid: float(rng.random()) for u in pending}
        elif policy == "eager":
            p = {u.uid: 1.0 for u in pending}
        else:
            raise ValueError(policy)
        kap = {u.uid: store.kappa(u) for u in pending}
        chi = {u.uid: store.chi(u) for u in pending}
        b = 1e9 if policy == "eager" else budget
        act, spent, eager = schedule_cycle(pending, p, kap, chi, b, now=s)
        if policy == "eager":
            act = {u.uid: Action.GRAPH for u in pending}
            spent = eager
        for u in sorted(pending, key=lambda u: u.created):
            store.apply(u, act[u.uid])
        spent_tot += spent
        eager_tot += eager
        pending = []
    # queries: current value of popular slots
    truth = {}
    for u in units:
        if u.slot is not None:
            truth[u.slot] = u.value
    qs = rng.choice(len(slots), 200, p=pop)
    correct = [store.answer(slots[i]) == truth.get(slots[i]) for i in qs if slots[i] in truth]
    out = dict(policy=policy, budget=budget, accuracy=float(np.mean(correct)),
               spent_frac=spent_tot / max(eager_tot, 1e-9), auroc=auc,
               n_graph_edges=len(store.graph), n_forgotten=len(store.forgotten))
    if verbose:
        print(out)
    return out
