import numpy as np
from volrisk.diagnostics import ess, rhat


def test_ess_and_rhat():
    """ESS of iid draws ~ N; R-hat ~ 1 for converged chains and > 2 for separated chains."""
    rng = np.random.default_rng(5)
    iid_draws = rng.standard_normal(3000)
    assert ess(iid_draws) > 2200

    converged = rng.standard_normal((4, 1500))
    diverged = converged + np.array([[0.0], [4.0], [-4.0], [8.0]])
    assert rhat(converged) < 1.02
    assert rhat(diverged) > 2.0