import pandas as pd
import pytest
from volrisk.data_loader import validate_prices, log_returns_pct


def test_validate_prices_rejects_bad_data():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    bad_prices = pd.DataFrame({"AAPL": [100.0, -5.0, 102.0]}, index=dates)
    with pytest.raises(ValueError, match="non-positive"):
        validate_prices(bad_prices)


def test_log_returns_pct_scaling():
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    p = pd.Series([100.0, 101.0], index=dates)
    r = log_returns_pct(p)
    assert len(r) == 1
    assert 0.99 < r.iloc[0] < 1.00  # ~0.995% return, not 0.00995