from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class Action(str, Enum):
    KEEP = "keep"
    GRAPH = "graph"
    SUMMARY = "summary"
    LORA = "lora"
    FORGET = "forget"


# Relative LLM-token cost of each action per 1k tokens of unit text (GRAPH = eager reference).
COSTS = {Action.KEEP: 0.0, Action.FORGET: 0.0, Action.SUMMARY: 0.45, Action.GRAPH: 1.0, Action.LORA: 1.6}


@dataclass
class MemoryUnit:
    uid: int
    text: str
    entities: tuple[str, ...]
    created: int                      # session index
    features: np.ndarray              # dense feature vector x_u
    n_tokens: int = 200
    accesses: list[int] = field(default_factory=list)
    action: Action = Action.KEEP
    slot: tuple[str, str] | None = None   # (subject, relation) this unit asserts, if any
    value: str | None = None

    def cost(self, a: Action) -> float:
        return COSTS[a] * self.n_tokens / 1000.0
