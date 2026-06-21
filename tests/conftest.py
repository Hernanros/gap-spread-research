from __future__ import annotations

import pandas as pd
import numpy as np
import pytest


def make_daily(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    opens: list[float] | None = None,
    start: str = "2024-01-02",
) -> pd.DataFrame:
    n = len(closes)
    if highs is None:
        highs = [c * 1.01 for c in closes]
    if lows is None:
        lows = [c * 0.99 for c in closes]
    if opens is None:
        opens = closes[:]
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=dates,
    )


def make_session_bars(
    date: str,
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    n = len(closes)
    if highs is None:
        highs = [c * 1.002 for c in closes]
    if lows is None:
        lows = [c * 0.998 for c in closes]
    times = pd.date_range(
        f"{date} 09:30", periods=n, freq="5min", tz="America/New_York"
    )
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100_000] * n,
        },
        index=times,
    )
