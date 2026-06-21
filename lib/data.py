from __future__ import annotations

import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent.parent / "cache"
DAILY_START = "2013-01-01"


def _cache_key(ticker: str, freq: str) -> str:
    return ticker.replace("^", "VIX").replace("=", "_") + f"_{freq}"


def fetch(
    ticker: str,
    freq: Literal["5m", "1d"],
    force_refresh: bool = False,
    prepost: bool = True,
) -> pd.DataFrame:
    """Download OHLCV data from yfinance, caching to parquet on first run.

    For 5m: downloads ~730 days with prepost=True (includes pre/after-market bars).
    For 1d: downloads from DAILY_START to today.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(ticker, freq)}.parquet"

    if path.exists() and not force_refresh:
        return pd.read_parquet(path)

    if freq == "5m":
        # Yahoo Finance caps 5m history at 60 days
        df = yf.download(
            ticker,
            period="60d",
            interval="5m",
            prepost=prepost,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
    else:
        df = yf.download(
            ticker,
            start=DAILY_START,
            interval="1d",
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )

    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.to_parquet(path)
    return df


def split_session(df_5m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split 5m bars into pre-market and regular session.

    Returns (pre_market, session) where:
        pre_market: 04:00–09:29 ET
        session:    09:30–16:00 ET
    """
    if df_5m.empty:
        return df_5m.copy(), df_5m.copy()

    if df_5m.index.tz is None:
        df = df_5m.tz_localize("UTC").tz_convert("America/New_York")
    else:
        df = df_5m.tz_convert("America/New_York")

    t = df.index.time
    pre = df[
        (t >= datetime.time(4, 0)) & (t < datetime.time(9, 30))
    ]
    session = df[
        (t >= datetime.time(9, 30)) & (t <= datetime.time(16, 0))
    ]
    return pre, session
