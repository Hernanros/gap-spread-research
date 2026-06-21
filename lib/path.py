from __future__ import annotations

import datetime
from typing import Sequence

import numpy as np
import pandas as pd

ENTRY_TIMES: list[str] = ["09:30", "09:45", "10:00", "10:30"]
VIX_REGIMES: dict[str, tuple[float, float]] = {
    "low": (0, 15),
    "mid": (15, 25),
    "high": (25, 999),
}


def analyze_paths(
    events: pd.DataFrame,
    session: pd.DataFrame,
    entry_times: Sequence[str] = ENTRY_TIMES,
) -> pd.DataFrame:
    """Compute gap fill, MAE, and fill timing for each event × entry_time.

    Args:
        events: Output of detect_events(). Indexed by date (tz-naive).
        session: 5m session bars (tz-aware, America/New_York).
        entry_times: List of entry time strings in "HH:MM" format.

    Returns:
        DataFrame with columns:
        date, ticker, direction, entry_time, entry_price, prev_close,
        gap_ratio, atr, premarket_fill_pct, vix_close,
        gap_filled, fill_time, mae, mae_pct_atr, eod_close
    """
    if session.empty:
        return pd.DataFrame()

    sess = session.copy()
    if sess.index.tz is None:
        sess = sess.tz_localize("America/New_York")
    else:
        sess = sess.tz_convert("America/New_York")

    records: list[dict] = []

    for date, row in events.iterrows():
        event_date = pd.Timestamp(date).date()
        day_bars = sess[sess.index.date == event_date]
        if day_bars.empty:
            continue

        prev_close = float(row["prev_close"])
        direction = row["direction"]
        eod_close = float(day_bars.iloc[-1]["close"])

        for entry_str in entry_times:
            h, m = map(int, entry_str.split(":"))
            entry_time_obj = datetime.time(h, m)
            bars = day_bars[day_bars.index.time >= entry_time_obj]
            if bars.empty:
                continue

            entry_price = float(bars.iloc[0]["close"])

            # Gap fill detection
            if direction == "gap_up":
                fill_bars = bars[bars["low"] <= prev_close]
            else:
                fill_bars = bars[bars["high"] >= prev_close]

            gap_filled = not fill_bars.empty
            fill_time: float | None = None
            if gap_filled:
                fb = fill_bars.iloc[0]
                open_minutes = 9 * 60 + 30
                bar_minutes = fb.name.hour * 60 + fb.name.minute
                fill_time = float(bar_minutes - open_minutes)

            # MAE: worst adverse excursion from entry_price
            if direction == "gap_up":
                mae = float((bars["high"] - entry_price).clip(lower=0).max())
            else:
                mae = float((entry_price - bars["low"]).clip(lower=0).max())

            atr = float(row["atr"])

            records.append(
                {
                    "date": date,
                    "ticker": row.get("ticker", ""),
                    "direction": direction,
                    "entry_time": entry_str,
                    "entry_price": entry_price,
                    "prev_close": prev_close,
                    "gap_ratio": float(row["gap_ratio"]),
                    "atr": atr,
                    "premarket_fill_pct": float(row.get("premarket_fill_pct", np.nan)),
                    "vix_close": float(row.get("vix_close", np.nan)),
                    "gap_filled": gap_filled,
                    "fill_time": fill_time,
                    "mae": mae,
                    "mae_pct_atr": mae / atr if atr > 0 else np.nan,
                    "eod_close": eod_close,
                }
            )

    return pd.DataFrame(records)


def label_vix_regime(
    vix: pd.Series,
    regimes: dict[str, tuple[float, float]] = VIX_REGIMES,
) -> pd.Series:
    """Map VIX values to regime labels: 'low', 'mid', 'high'."""
    result = pd.Series("unknown", index=vix.index, dtype=str)
    for label, (lo, hi) in regimes.items():
        result[(vix >= lo) & (vix < hi)] = label
    return result
