import numpy as np
from tide.hazard import HazardModel, auroc


def test_p_within_monotone_in_delta():
    m = HazardModel(4, max_k=16)
    X = np.random.default_rng(0).normal(size=(5, 4))
    p4, p16 = m.p_within(X, 4), m.p_within(X, 16)
    assert np.all(p16 >= p4) and np.all((p4 >= 0) & (p16 <= 1))


def test_fit_reduces_nll_and_ranks():
    r = np.random.default_rng(0)
    X = r.normal(size=(2000, 4))
    lam = 1 / (1 + np.exp(-(-2 + 2 * X[:, 0])))
    k = np.minimum(r.geometric(lam), 16)
    d = (r.geometric(lam) <= 16).astype(int)
    k = np.where(d == 1, k, 16)
    m = HazardModel(4, max_k=16)
    before = m.nll(X, k, d)
    m.fit(X, k, d, steps=300)
    after = m.nll(X, k, d)
    assert after < before
    assert auroc(m.p_within(X, 8), (k <= 8) & (d == 1)) > 0.7


def test_auroc_perfect():
    assert auroc(np.array([0.1, 0.2, 0.9]), np.array([0, 0, 1])) == 1.0
