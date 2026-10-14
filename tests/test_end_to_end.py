import time
from tide import run_toy


def test_toy_runs_fast_and_orders_policies():
    t = time.time()
    tide = run_toy(0.4, "tide", seed=0)
    rnd = run_toy(0.4, "random", seed=0)
    eager = run_toy(1.0, "eager", seed=0)
    assert time.time() - t < 60
    assert tide["spent_frac"] <= 0.401
    assert tide["accuracy"] > rnd["accuracy"]
    assert eager["accuracy"] >= tide["accuracy"]
    assert tide["auroc"] > 0.75
