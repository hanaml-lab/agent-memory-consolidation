"""Greedy multiple-choice knapsack over consolidation actions (Algorithm 1)."""
from __future__ import annotations
import heapq
from .types import Action, MemoryUnit
from .voc import VoCWeights, value_of_consolidation

UPGRADES = [Action.SUMMARY, Action.GRAPH, Action.LORA]


def schedule_cycle(units: list[MemoryUnit], p: dict[int, float], kappa: dict[int, float], chi: dict[int, float],
                   budget: float, w: VoCWeights | None = None, now: int = 0, forget_eps=(0.04, 0.10, 6),
                   allow_lora: bool = False) -> tuple[dict[int, Action], float, float]:
    """Assign an action to each pending unit.

    budget is relative to eager consolidation (every unit -> GRAPH). Returns (actions, spent, eager_cost).
    """
    w = w or VoCWeights()
    eager = sum(u.cost(Action.GRAPH) for u in units)
    cap = budget * eager
    act: dict[int, Action] = {}
    for u in units:
        ef, ec, amin = forget_eps
        act[u.uid] = Action.FORGET if (p[u.uid] < ef and chi[u.uid] < ec and now - u.created >= amin) else Action.KEEP

    def val(u, a):
        return value_of_consolidation(p[u.uid], kappa[u.uid], chi[u.uid], a, w, allow_lora)

    heap = []
    by_id = {u.uid: u for u in units}

    def push(u):
        cur = act[u.uid]
        for a in UPGRADES:
            dv = val(u, a) - val(u, cur)
            dc = u.cost(a) - u.cost(cur)
            if dv > 0 and dc > 0:
                heapq.heappush(heap, (-dv / dc, u.uid, a.value, cur.value))
    for u in units:
        push(u)
    spent = 0.0
    while heap:
        _, uid, a, base = heapq.heappop(heap)
        u, a, base = by_id[uid], Action(a), Action(base)
        if act[uid] is not base:          # stale entry: unit already upgraded
            continue
        dc = u.cost(a) - u.cost(base)
        if spent + dc <= cap + 1e-12:
            spent += dc
            act[uid] = a
            push(u)
    return act, spent, eager
