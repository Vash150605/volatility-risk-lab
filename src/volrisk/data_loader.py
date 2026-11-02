from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from volrisk.universe import ALL, START, END

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _cache_path(tickers, start, end) -> Path:
    """Same request -> same filename. Ticker order does not matter."""
    key = "|".join(sorted(tickers)) + f"|{start}|{end}"
    return CACHE_DIR / f"prices_{hashlib.md5(key.encode()).hexdigest()[:10]}.parquet"


def validate_prices(prices: pd.DataFrame, max_missing: float = 0.05) -> pd.DataFrame:
    """Raise if the price table is unusable; otherwise return a cleaned copy."""
    if prices.empty:
        raise ValueError("no data returned")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("index must be sorted with unique dates")
    missing = prices.isna().mean()
    too_many = missing[missing > max_missing]
    if not too_many.empty:
        raise ValueError(f"too many missing values: {too_many.round(3).to_dict()}")
    if (prices.dropna() <= 0).any().any():
        raise ValueError("non-positive prices found")
    return prices.ffill(limit=3).dropna()


def load_prices(
    tickers=ALL, start: str = START, end: str = END, refresh: bool = False
) -> pd.DataFrame:
    """Adjusted close prices (rows = dates, columns = tickers), cached as Parquet."""
    path = _cache_path(tickers, start, end)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(RAW_DIR / path.name.replace("prices_", "raw_"))
    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    cleaned = validate_prices(prices)
    if len(cleaned) < len(prices):
        log.warning("validation dropped %d of %d rows", len(prices) - len(cleaned), len(prices))
    prices = validate_prices(cleaned)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)
    log.info("cached %d rows x %d tickers -> %s", *prices.shape, path)
    return prices


def log_returns_pct(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Log returns scaled by 100: r_t = 100 * ln(P_t / P_{t-1})."""
    return 100.0 * np.log(prices / prices.shift(1)).dropna()