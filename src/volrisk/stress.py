import numpy as np


def max_drawdown(r_pct):
    """Maximum drawdown (reported as positive %) from daily percentage simple returns."""
    wealth = np.cumprod(1.0 + np.asarray(r_pct, float) / 100.0)
    peak = np.maximum.accumulate(wealth)
    return float(100.0 * np.max((peak - wealth) / peak))


def scenario_replay(port_ret, start, end, var_df=None):
    """Replay a historical crisis window on the 25-stock equal-weight portfolio."""
    w = port_ret.loc[start:end]
    cum_loss = 100.0 * (np.prod(1.0 + w.to_numpy() / 100.0) - 1.0)
    res = {
        "Window": f"{start} to {end}",
        "Days": len(w),
        "Cum_Return_%": round(float(cum_loss), 2),
        "Worst_Day_%": round(float(w.min()), 2),
        "Max_Drawdown_%": round(max_drawdown(w), 2),
    }
    if var_df is not None:
        sub = var_df.loc[start:end]
        for m in ["hist", "norm", "t", "fhs"]:
            res[f"Breaches_{m.upper()}_99"] = int(sub[f"hit_{m}_99"].sum())
    return res