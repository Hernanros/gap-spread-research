from __future__ import annotations

import pandas as pd
import numpy as np

ATR_PERIOD = 14
GAP_THRESHOLD = 1.0


def compute_atr(daily: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Wilder EWM ATR. First value is NaN (no previous close for TR)."""
    high = daily["high"]
    low = daily["low"]
    prev_close = daily["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1, skipna=False)  # NaN when prev_close is missing (first row)
    return tr.ewm(span=period, adjust=False).mean()


def detect_events(
    daily: pd.DataFrame,
    premarket: pd.DataFrame,
    vix_daily: pd.DataFrame | None = None,
    threshold: float = GAP_THRESHOLD,
    atr_period: int = ATR_PERIOD,
) -> pd.DataFrame:
    """Return one row per gap event (|gap_ratio| >= threshold).

    Args:
        daily: Daily OHLCV with columns open/high/low/close.
        premarket: 5m pre-market bars (tz-aware, America/New_York).
                   Pass empty DataFrame if unavailable.
        vix_daily: Daily VIX close. Merged on date if provided.
        threshold: Minimum |gap| in ATR units to qualify as event.
        atr_period: Period for ATR computation.

    Returns:
        DataFrame indexed by date with columns:
        direction, gap, gap_ratio, atr, prev_close, open,
        [premarket_high, premarket_low, premarket_fill_pct if premarket provided],
        [vix_close if vix_daily provided]
    """
    d = daily.copy()
    d["atr"] = compute_atr(d, atr_period)
    d["prev_close"] = d["close"].shift(1)
    d["gap"] = d["open"] - d["prev_close"]
    d["gap_ratio"] = d["gap"] / d["atr"]
    d["direction"] = np.where(
        d["gap_ratio"] >= threshold,
        "gap_up",
        np.where(d["gap_ratio"] <= -threshold, "gap_down", "none"),
    )

    events = d[d["direction"] != "none"].copy()
    events = events.dropna(subset=["atr", "prev_close"])

    if not premarket.empty:
        pm_stats = _premarket_stats(premarket)
        events = events.join(pm_stats, how="left")
        events["premarket_fill_pct"] = _fill_pct(events)

    if vix_daily is not None and not vix_daily.empty:
        vix_col = vix_daily["close"] if "close" in vix_daily.columns else vix_daily.iloc[:, 0]
        vix_col.index = pd.to_datetime(vix_col.index).normalize()
        events.index = pd.to_datetime(events.index).normalize()
        events = events.join(vix_col.rename("vix_close"), how="left")

    keep = ["direction", "gap", "gap_ratio", "atr", "prev_close", "open"]
    for col in ["premarket_high", "premarket_low", "premarket_fill_pct", "vix_close"]:
        if col in events.columns:
            keep.append(col)

    return events[keep]


def _premarket_stats(pre: pd.DataFrame) -> pd.DataFrame:
    """Compute daily pre-market high and low from 5m bars."""
    df = pre.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York")
    df.index = df.index.normalize().tz_localize(None)
    return df.groupby(df.index).agg(
        premarket_high=("high", "max"),
        premarket_low=("low", "min"),
    )


def _fill_pct(events: pd.DataFrame) -> pd.Series:
    """Fraction of gap filled by pre-market action. Clipped to [0, 1].

    Gap up: measures how far premarket_low dipped back toward prev_close.
        fill_pct = max(0, open - premarket_low) / gap
    Gap down: measures how far premarket_high rallied back toward prev_close.
        fill_pct = max(0, premarket_high - open) / |gap|
    """
    result = pd.Series(0.0, index=events.index)

    up = events["direction"] == "gap_up"
    result[up] = (events.loc[up, "open"] - events.loc[up, "premarket_low"]) / events.loc[up, "gap"]

    down = ~up
    result[down] = (events.loc[down, "premarket_high"] - events.loc[down, "open"]) / events.loc[down, "gap"].abs()

    return result.clip(0.0, 1.0)
