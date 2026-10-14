from tide.demo import run

CFG = {"stream": {"n_sessions": 30, "units_per_session": 8, "n_slots": 20, "d": 8}, "schedule": {"cycle": 2, "delta": 16},
       "budgets": [0.4, 1.0], "policies": ["eager", "tide", "random"], "seeds": [0]}


def test_demo_runs_and_respects_the_budget():
    out = run(CFG)
    assert out[("tide", 0.4)]["spent_frac"] <= 0.4 + 1e-9
    assert out[("eager", 1.0)]["accuracy"] >= out[("tide", 0.4)]["accuracy"]
    assert 0.5 < out[("tide", 0.4)]["auroc"] <= 1.0
