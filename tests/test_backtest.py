import numpy as np
import pandas as pd
from volrisk.garch import simulate_garch11
from volrisk.backtest import (
    kupiec_pof,
    christoffersen_ind,
    traffic_light,
    rolling_forecasts,
)


def test_kupiec_exact_breach_rate():
    """Hit sequence with exact expected breach rate (10/1000 at 99%) gives LR = 0."""
    hits = np.zeros(1000, dtype=int)
    hits[:10] = 1
    lr, p_val = kupiec_pof(hits, 0.01)
    assert abs(lr) < 1e-10 and abs(p_val - 1.0) < 1e-10


def test_christoffersen_detects_clustering():
    """Clustered hits give small p-value; evenly spaced hits give large p-value."""
    clustered = np.zeros(1000, dtype=int)
    clustered[:10] = 1
    spaced = np.zeros(1000, dtype=int)
    spaced[::100] = 1
    assert christoffersen_ind(clustered)[1] < 0.001
    assert christoffersen_ind(spaced)[1] > 0.50


def test_traffic_light_boundaries():
    """4 -> green, 5 -> yellow, 9 -> yellow, 10 -> red."""
    assert [traffic_light(x) for x in (4, 5, 9, 10)] == ["green", "yellow", "yellow", "red"]


def test_rolling_forecasts_no_look_ahead():
    """Changing return on day t must not change the forecast made for day t."""
    r1 = pd.Series(simulate_garch11(120, 0.0, 0.05, 0.08, 0.85, seed=9))
    r2 = r1.copy()
    r2.iloc[105] = -50.0  # Massive shock on day 105
    df1 = rolling_forecasts(r1, window=100, levels=(0.99,), refit_every=50)
    df2 = rolling_forecasts(r2, window=100, levels=(0.99,), refit_every=50)
    for m in ["hist", "norm", "t", "fhs"]:
        assert df1.iloc[5][f"var_{m}_99"] == df2.iloc[5][f"var_{m}_99"]