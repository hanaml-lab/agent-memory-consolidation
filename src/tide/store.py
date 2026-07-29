"""A tiny tiered store with a temporal slot graph (stand-in for Graphiti/Zep-style graphs)."""
from __future__ import annotations
from .types import Action, MemoryUnit


class TieredStore:
    def __init__(self):
        self.raw: dict[int, MemoryUnit] = {}
        self.graph: dict[tuple[str, str], tuple[str, int, int]] = {}   # slot -> (value, valid_from, uid)
        self.summaries: dict[int, MemoryUnit] = {}
        self.forgotten: dict[int, MemoryUnit] = {}
        self.entity_index: dict[str, set[int]] = {}

    def add_raw(self, u: MemoryUnit):
        self.raw[u.uid] = u
        for e in u.entities:
            self.entity_index.setdefault(e, set()).add(u.uid)

    def kappa(self, u: MemoryUnit) -> float:
        return float(sum(len(self.entity_index.get(e, ())) - 1 for e in u.entities))

    def chi(self, u: MemoryUnit) -> float:
        """Contradiction against the current consolidated belief for the unit's slot."""
        if u.slot is None or u.slot not in self.graph:
            return 0.0
        return 1.0 if self.graph[u.slot][0] != u.value else 0.0

    def apply(self, u: MemoryUnit, a: Action):
        u.action = a
        if a is Action.GRAPH and u.slot is not None:
            cur = self.graph.get(u.slot)
            if cur is None or u.created >= cur[1]:
                self.graph[u.slot] = (u.value, u.created, u.uid)   # closes the superseded edge
        elif a is Action.SUMMARY:
            self.summaries[u.uid] = u
        elif a is Action.FORGET:
            self.forgotten[u.uid] = self.raw.pop(u.uid, u)

    def answer(self, slot: tuple[str, str]) -> str | None:
        """Reader: prefer the consolidated belief; otherwise fall back to the earliest raw mention found
        (a stand-in for a reader that is misled by unreconciled raw evidence)."""
        if slot in self.graph:
            return self.graph[slot][0]
        cands = sorted((u.created, u.value) for u in self.raw.values() if u.slot == slot)
        return cands[0][1] if cands else None
