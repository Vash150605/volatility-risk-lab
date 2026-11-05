import numpy as np
import pandas as pd
from scipy.special import xlogy, xlog1py
from scipy.stats import chi2
from volrisk.garch import fit_garch
from volrisk.var_models import (
    var_es_historical,
    var_es_normal,
    var_es_student_t,
    var_es_fhs,
)


def _ll(x, n, p):
    """Binomial log-likelihood without the constant; safe when x = 0 or x = n."""
    if p <= 0 or p >= 1:
        return 0.0
    return xlogy(x, p) + xlog1py(n - x, -p)


def kupiec_pof(hits, p):
    """Kupiec proportion-of-failures (unconditional coverage) test."""
    hits = np.asarray(hits, int)
    N, x = len(hits), int(hits.sum())
    if x == 0 or x == N:
        lr = -2 * _ll(x, N, p)
        return float(lr), float(chi2.sf(lr, 1))
    lr = -2 * (_ll(x, N, p) - _ll(x, N, x / N))
    return float(lr), float(chi2.sf(lr, 1))


def christoffersen_ind(hits):
    """Christoffersen independence test for clustered breaches."""
    h = np.asarray(hits, int)
    a, b = h[:-1], h[1:]
    n00 = int(np.sum((a == 0) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n11 = int(np.sum((a == 1) & (b == 1)))
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = _ll(n01 + n11, n00 + n01 + n10 + n11, pi)
    ll1 = _ll(n01, n00 + n01, pi01) + _ll(n11, n10 + n11, pi11)
    lr = max(0.0, -2 * (ll0 - ll1))
    return float(lr), float(chi2.sf(lr, 1))


def christoffersen_cc(hits, p):
    """Christoffersen conditional coverage test (POF + Independence)."""
    lr = kupiec_pof(hits, p)[0] + christoffersen_ind(hits)[0]
    return float(lr), float(chi2.sf(lr, 2))


def traffic_light(n_breaches_250):
    """Basel traffic-light zone over 250 trading days at 99% VaR."""
    if n_breaches_250 <= 4:
        return "green"
    if n_breaches_250 <= 9:
        return "yellow"
    return "red"


def rolling_forecasts(r_series, window=1000, levels=(0.99, 0.975), refit_every=25):
    """Rolling 1-day VaR/ES forecasts with zero look-ahead (window = r[t-W : t])."""
    r = r_series.to_numpy(dtype=float)
    dates = r_series.index
    rows = []
    garch_params = None

    for t in range(window, len(r)):
        w = r[t - window : t]  # Strictly excludes day t!
        if garch_params is None or (t - window) % refit_every == 0:
            garch_params = fit_garch(w, model="garch", dist="normal", compute_se=False)["params"]

        row = {"date": dates[t], "realized": r[t]}
        for a in levels:
            lbl = "99" if a == 0.99 else "975"
            res = {
                "hist": var_es_historical(w, a),
                "norm": var_es_normal(w, a),
                "t": var_es_student_t(w, a),
                "fhs": var_es_fhs(w, a, params=garch_params),
            }
            for m, (var_v, es_v) in res.items():
                row[f"var_{m}_{lbl}"] = var_v
                row[f"es_{m}_{lbl}"] = es_v
                row[f"hit_{m}_{lbl}"] = int(r[t] < -var_v)
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def backtest_table(df, levels=(0.99, 0.975)):
    """Build the 8-row evaluation table across all 4 methods and 2 confidence levels."""
    out = []
    for m in ["hist", "norm", "t", "fhs"]:
        for a in levels:
            lbl = "99" if a == 0.99 else "975"
            hits = df[f"hit_{m}_{lbl}"].to_numpy()
            es_pred = df[f"es_{m}_{lbl}"].to_numpy()
            realized = df["realized"].to_numpy()
            p = 1 - a
            n_b = int(hits.sum())
            _, kp = kupiec_pof(hits, p)
            _, ip = christoffersen_ind(hits)
            _, ccp = christoffersen_cc(hits, p)
            w250 = int(pd.Series(hits).rolling(250).sum().max()) if len(hits) >= 250 else n_b
            es_ratio = float((-realized[hits == 1]).mean() / es_pred[hits == 1].mean()) if n_b > 0 else np.nan
            out.append({
                "Method": m.upper(),
                "Level": f"{a*100:.1f}%",
                "Days": len(hits),
                "Breaches": n_b,
                "Expected": round(len(hits) * p, 1),
                "Kupiec_p": round(kp, 4),
                "Indep_p": round(ip, 4),
                "CC_p": round(ccp, 4),
                "ES_Ratio": round(es_ratio, 2),
                "Worst_Traffic_Light": f"{traffic_light(w250)} ({w250})" if a == 0.99 else "-",
            })
    return pd.DataFrame(out)