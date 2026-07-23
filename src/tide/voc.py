"""Value of consolidation (Eq. 2)."""
from __future__ import annotations
from dataclasses import dataclass, field
from .types import Action

DEFAULT_DACC = {Action.KEEP: 0.0, Action.GRAPH: 0.11, Action.SUMMARY: 0.06, Action.LORA: 0.09, Action.FORGET: -0.25}
DEFAULT_DREL = {Action.KEEP: 0.0, Action.GRAPH: 1.0, Action.SUMMARY: 0.3, Action.LORA: 0.2, Action.FORGET: 0.0}
DEFAULT_DCONF = {Action.KEEP: 0.0, Action.GRAPH: 0.38, Action.SUMMARY: 0.05, Action.LORA: 0.02, Action.FORGET: 0.0}
DEFAULT_STORE = {Action.KEEP: 1.0, Action.GRAPH: 0.35, Action.SUMMARY: 0.2, Action.LORA: 0.0, Action.FORGET: 0.0}


@dataclass
class VoCWeights:
    lam_g: float = 0.15
    lam_c: float = 0.60
    lam_s: float = 0.02
    d_acc: dict = field(default_factory=lambda: dict(DEFAULT_DACC))
    d_rel: dict = field(default_factory=lambda: dict(DEFAULT_DREL))
    d_conf: dict = field(default_factory=lambda: dict(DEFAULT_DCONF))
    store: dict = field(default_factory=lambda: dict(DEFAULT_STORE))


def value_of_consolidation(p_access: float, kappa: float, chi: float, a: Action, w: VoCWeights, allow_lora: bool = False) -> float:
    if a is Action.LORA and not allow_lora:
        return float("-inf")
    return (p_access * w.d_acc[a] + w.lam_g * kappa * 0.01 * w.d_rel[a]
            + w.lam_c * chi * w.d_conf[a] - w.lam_s * w.store[a])
