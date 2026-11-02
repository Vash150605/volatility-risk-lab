"""Value at Risk (VaR) and Expected Shortfall (ES) models."""

from __future__ import annotations

import numpy as np
from scipy import stats
from volrisk.garch import fit_garch, forecast_next_variance


def var_es_historical(w: np.ndarray | list[float], a: float = 0.99) -> tuple[float, float]:
    """Historical simulation VaR and ES from empirical quantile."""
    w_arr = np.asarray(w, dtype=float)
    q = np.quantile(w_arr, 1.0 - a)
    tail = w_arr[w_arr <= q]
    es = tail.mean() if len(tail) > 0 else q
    return float(-q), float(-es)


def var_es_normal(w: np.ndarray | list[float], a: float = 0.99) -> tuple[float, float]:
    """Parametric Gaussian VaR and ES."""
    w_arr = np.asarray(w, dtype=float)
    mu, s = float(np.mean(w_arr)), float(np.std(w_arr, ddof=1))
    z = stats.norm.ppf(a)
    var = -mu + s * z
    es = -mu + s * (stats.norm.pdf(z) / (1.0 - a))
    return float(var), float(es)


def var_es_student_t(w: np.ndarray | list[float], a: float = 0.99) -> tuple[float, float]:
    """Parametric Student-t VaR and ES standardized to sample volatility."""
    w_arr = np.asarray(w, dtype=float)
    nu, _, _ = stats.t.fit(w_arr)
    nu = max(float(nu), 2.05)
    s = float(np.std(w_arr, ddof=1))
    k = np.sqrt((nu - 2.0) / nu)
    ta = stats.t.ppf(a, nu)

    var = -float(np.mean(w_arr)) + s * k * ta
    pdf_ta = stats.t.pdf(ta, nu)
    es = (
        -float(np.mean(w_arr))
        + s * k * (pdf_ta / (1.0 - a)) * ((nu + ta**2) / (nu - 1.0))
    )
    return float(var), float(es)


def var_es_fhs(
    w: np.ndarray | list[float],
    a: float = 0.99,
    params: np.ndarray | list[float] | None = None,
) -> tuple[float, float]:
    """GARCH-Filtered Historical Simulation (FHS) VaR and ES."""
    w_arr = np.asarray(w, dtype=float)
    if params is None:
        fit = fit_garch(w_arr, model="garch", dist="normal", compute_se=False)
        params = fit["params"]

    mu, sig_next, z = forecast_next_variance(params, w_arr, model="garch")
    q = np.quantile(z, 1.0 - a)
    tail = z[z <= q]
    tail_mean = tail.mean() if len(tail) > 0 else q

    var = -(mu + sig_next * q)
    es = -(mu + sig_next * tail_mean)
    return float(var), float(es)