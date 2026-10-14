import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _example(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_schedule_memory_on_the_bundled_units():
    out = _example("schedule_memory").main([str(ROOT / "examples" / "data" / "memory_units.jsonl"), "--budget", "0.4"])
    assert out["spent_frac"] <= 0.4 + 1e-9
    # the move to Lisbon contradicts the consolidated belief, so it is consolidated and replaces it
    assert out["actions"][7] == "graph" and out["beliefs"]["user.lives_in"] == "Lisbon"
    # small talk with a low probability of access is never consolidated
    assert out["actions"][6] == "keep" and out["actions"][10] == "keep"
