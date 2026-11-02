"""Tests for GARCH(1,1) simulation, variance recursion, and MLE estimation."""

import numpy as np
import pytest
from arch import arch_model

from volrisk.garch import (
    simulate_garch11,
    garch_variance,
    fit_garch,
    forecast_next_variance,
)


def test_variance_recursion_by_hand():
    """Verify 3-step variance recursion matches manual arithmetic."""
    eps = np.array([1.0, -2.0, 0.5])
    omega, alpha, beta = 0.1, 0.2, 0.7
    s2_0 = 1.0

    # Manual step-by-step arithmetic
    s2_1 = omega + alpha * (eps[0] ** 2) + beta * s2_0
    s2_2 = omega + alpha * (eps[1] ** 2) + beta * s2_1

    s2 = garch_variance(omega, alpha, beta, eps, gamma=0.0, s2_0=s2_0)
    np.testing.assert_allclose(s2, [s2_0, s2_1, s2_2])


def test_gjr_asymmetry_by_hand():
    """Verify GJR-GARCH adds gamma penalty only on negative shocks."""
    omega, alpha, beta, gamma = 0.1, 0.1, 0.7, 0.2
    # 3 elements needed so s2 has length 3: [s2_0, s2_1, s2_2]
    eps = np.array([1.0, -1.0, 0.5])
    s2_0 = 1.0

    # Day 1: eps[0] = 1.0 >= 0 -> shock multiplier is alpha only
    s2_1 = omega + alpha * (1.0**2) + beta * s2_0

    # Day 2: eps[1] = -1.0 < 0 -> shock multiplier is (alpha + gamma)
    s2_2 = omega + (alpha + gamma) * ((-1.0)**2) + beta * s2_1

    s2 = garch_variance(omega, alpha, beta, eps, gamma=gamma, s2_0=s2_0)
    np.testing.assert_allclose(s2, [s2_0, s2_1, s2_2])


def test_mle_recovers_true_parameters():
    """MLE recovers known (mu, omega, alpha, beta) on simulated series."""
    true_mu, true_omega, true_alpha, true_beta = 0.05, 0.05, 0.10, 0.85
    r = simulate_garch11(
        n=3000,
        mu=true_mu,
        omega=true_omega,
        alpha=true_alpha,
        beta=true_beta,
        seed=12,
    )

    fit = fit_garch(r, model="garch", dist="normal", compute_se=False)

    assert fit["success"] is True
    np.testing.assert_allclose(
        fit["params"],
        [true_mu, true_omega, true_alpha, true_beta],
        atol=0.05,
    )


def test_fit_matches_arch_library():
    """Custom MLE matches reference arch package parameters and log-likelihood."""
    r = simulate_garch11(n=2000, mu=0.03, omega=0.05, alpha=0.09, beta=0.88, seed=42)

    ours = fit_garch(r, model="garch", dist="normal", compute_se=False)
    ref = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")

    np.testing.assert_allclose(ours["params"], ref.params.to_numpy(), atol=0.03)
    assert abs(ours["loglik"] - ref.loglikelihood) < 2.5


def test_stationarity():
    """Fitted persistence alpha + beta < 1 and conditional variance path is strictly positive."""
    r = simulate_garch11(n=1000, mu=0.0, omega=0.05, alpha=0.08, beta=0.88, seed=7)
    res = fit_garch(r, compute_se=False)

    mu, omega, alpha, beta = res["params"][:4]
    assert alpha + beta < 1.0
    assert np.all(garch_variance(omega, alpha, beta, r - mu) > 0.0)


def test_forecast_next_variance():
    """Forecast produces a positive one-step standard deviation and unit-scaled residuals."""
    r = simulate_garch11(n=500, mu=0.0, omega=0.05, alpha=0.1, beta=0.85, seed=99)
    fit = fit_garch(r, compute_se=False)

    mu, sig_next, z = forecast_next_variance(fit["params"], r)

    assert sig_next > 0.0
    assert len(z) == len(r)
    # Standardized residuals z = eps / sigma should be centered near 0 with variance near 1
    assert -0.2 < np.mean(z) < 0.2
    assert 0.7 < np.var(z) < 1.3