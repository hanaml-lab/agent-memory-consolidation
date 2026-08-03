import numpy as np
from tide import MemoryUnit, Action, schedule_cycle


def _units(n=20):
    return [MemoryUnit(i, "t", ("e",), 0, np.zeros(2), n_tokens=200) for i in range(n)]


def test_budget_respected():
    us = _units()
    p = {u.uid: (u.uid + 1) / 20 for u in us}; z = {u.uid: 0.0 for u in us}
    for b in [0.0, 0.1, 0.4, 1.0]:
        act, spent, eager = schedule_cycle(us, p, z, z, b)
        assert spent <= b * eager + 1e-9


def test_high_access_units_upgraded_first():
    us = _units()
    p = {u.uid: (u.uid + 1) / 20 for u in us}; z = {u.uid: 0.0 for u in us}
    act, _, _ = schedule_cycle(us, p, z, z, 0.3)
    up = [u for u, a in act.items() if a is not Action.KEEP]
    kept = [u for u, a in act.items() if a is Action.KEEP]
    assert up and kept and min(up) > max(kept)


def test_conflict_term_buys_graph():
    us = _units(2)
    p = {0: 0.01, 1: 0.01}; k = {0: 0.0, 1: 0.0}; chi = {0: 1.0, 1: 0.0}
    act, _, _ = schedule_cycle(us, p, k, chi, 0.5)
    assert act[0] is Action.GRAPH and act[1] is not Action.GRAPH


def test_forget_old_cold_units():
    u = MemoryUnit(0, "t", ("e",), 0, np.zeros(2))
    act, _, _ = schedule_cycle([u], {0: 0.0}, {0: 0.0}, {0: 0.0}, 0.0, now=10)
    assert act[0] is Action.FORGET
