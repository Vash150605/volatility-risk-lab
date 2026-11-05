import pandas as pd
from volrisk.stress import max_drawdown, scenario_replay


def test_scenario_replay_known_drawdown():
    """Wealth: 100 -> 110 (+10%) -> 88 (-20%) -> 96.8 (+10%) => Max Drawdown = 20%."""
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    r = pd.Series([0.0, 10.0, -20.0, 10.0], index=dates)
    assert abs(max_drawdown(r) - 20.0) < 1e-9
    rep = scenario_replay(r, "2020-01-01", "2020-01-04")
    assert rep["Max_Drawdown_%"] == 20.0
    assert rep["Worst_Day_%"] == -20.0