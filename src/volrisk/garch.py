"""GARCH(1,1) and GJR-GARCH estimation by maximum likelihood."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln


def simulate_garch11(
    n: int,
    mu: float,
    omega: float,
    alpha: float,
    beta: float,
    seed: int = 0,
    burn: int = 500,
) -> np.ndarray:
    """Simulate r_t = mu + sigma_t * z_t with GARCH(1,1) variance and N(0,1) shocks."""
    rng = np.random.default_rng(seed)
    total_len = n + burn
    z = rng.standard_normal(total_len)
    s2 = np.empty(total_len)
    eps = np.empty(total_len)

    # Initial variance at long-run stationary expectation
    s2[0] = omega / max(1.0 - alpha - beta, 1e-4)
    eps[0] = np.sqrt(s2[0]) * z[0]

    for t in range(1, total_len):
        s2[t] = omega + alpha * (eps[t - 1] ** 2) + beta * s2[t - 1]
        eps[t] = np.sqrt(s2[t]) * z[t]

    return mu + eps[burn:]


def garch_variance(
    omega: float,
    alpha: float,
    beta: float,
    eps: np.ndarray,
    gamma: float = 0.0,
    s2_0: float | None = None,
) -> np.ndarray:
    """Compute conditional variance recursion:
    sigma^2_t = omega + (alpha + gamma * 1[eps_{t-1} < 0]) * eps_{t-1}^2 + beta * sigma^2_{t-1}
    """
    n = len(eps)
    s2 = np.empty(n)
    s2[0] = np.var(eps) if s2_0 is None else s2_0

    for t in range(1, n):
        indicator = gamma if eps[t - 1] < 0.0 else 0.0
        s2[t] = omega + (alpha + indicator) * (eps[t - 1] ** 2) + beta * s2[t - 1]

    return s2


def neg_loglik(
    theta: np.ndarray | list[float],
    r: np.ndarray,
    model: str = "garch",
    dist: str = "normal",
) -> float:
    """Negative log-likelihood function for GARCH/GJR models."""
    mu, omega, alpha, beta = theta[:4]
    k = 4
    gamma = 0.0

    if model == "gjr":
        gamma = theta[k]
        k += 1

    # Stationarity and non-negativity checks
    persistence = alpha + 0.5 * gamma + beta
    if omega <= 1e-8 or alpha < 0.0 or beta < 0.0 or gamma < 0.0 or persistence >= 0.9999:
        return 1e10

    eps = r - mu
    s2 = garch_variance(omega, alpha, beta, eps, gamma)

    if np.any(s2 <= 0.0) or not np.all(np.isfinite(s2)):
        return 1e10

    if dist == "normal":
        ll = -0.5 * (np.log(2.0 * np.pi) + np.log(s2) + (eps ** 2) / s2)
    else:  # Student-t standardized to unit variance
        nu = theta[k]
        if nu <= 2.05:
            return 1e10
        ll = (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(np.pi * (nu - 2.0))
            - 0.5 * np.log(s2)
            - ((nu + 1.0) / 2.0) * np.log1p((eps ** 2) / (s2 * (nu - 2.0)))
        )

    return float(-np.sum(ll))


def num_hessian(f, x: np.ndarray, rel: float = 1e-4) -> np.ndarray:
    """Compute central finite-difference Hessian matrix."""
    x = np.asarray(x, float)
    n = len(x)
    h = rel * np.maximum(np.abs(x), 1.0)
    H = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = h[i]
            ej[j] = h[j]
            H[i, j] = H[j, i] = (
                f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)
            ) / (4.0 * h[i] * h[j])

    return H


def fit_garch(
    r: np.ndarray | list[float],
    model: str = "garch",
    dist: str = "normal",
    compute_se: bool = True,
) -> dict:
    """Fit GARCH/GJR-GARCH by MLE."""
    r = np.asarray(r, float)
    var_uncond = float(np.var(r))

    x0 = [float(np.mean(r)), 0.05 * var_uncond, 0.05, 0.90]
    bounds = [(-np.inf, np.inf), (1e-6, None), (0.0, 1.0), (0.0, 1.0)]

    if model == "gjr":
        x0.append(0.05)
        bounds.append((0.0, 1.0))

    if dist == "t":
        x0.append(8.0)
        bounds.append((2.05, 100.0))

    f = lambda th: neg_loglik(th, r, model, dist)
    res = minimize(f, x0, method="L-BFGS-B", bounds=bounds)

    se = np.full(len(res.x), np.nan)
    if compute_se and res.success:
        try:
            H = num_hessian(f, res.x)
            inv_h = np.linalg.pinv(H)
            se = np.sqrt(np.maximum(np.diag(inv_h), 0.0))
        except Exception:
            pass

    k = len(res.x)
    ll = float(-res.fun)
    n_obs = len(r)

    return {
        "params": res.x,
        "se": se,
        "loglik": ll,
        "aic": 2.0 * k - 2.0 * ll,
        "bic": k * np.log(n_obs) - 2.0 * ll,
        "success": bool(res.success),
    }


def forecast_next_variance(
    params: np.ndarray,
    r: np.ndarray,
    model: str = "garch",
) -> tuple[float, float, np.ndarray]:
    """Calculate 1-step-ahead conditional standard deviation and standardized residuals."""
    mu, omega, alpha, beta = params[:4]
    gamma = params[4] if model == "gjr" else 0.0

    eps = r - mu
    s2 = garch_variance(omega, alpha, beta, eps, gamma)

    last_indicator = gamma if eps[-1] < 0.0 else 0.0
    s2_next = omega + (alpha + last_indicator) * (eps[-1] ** 2) + beta * s2[-1]

    std_resid = eps / np.sqrt(s2)
    return float(mu), float(np.sqrt(s2_next)), std_resid