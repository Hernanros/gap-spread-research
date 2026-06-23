"""Quick check: events when trend_30d = bullish only, no VIX filter, at 0.75×ATR.

trend_30d is bullish when close > close 30 days ago.
Bullish-trend events = (concurrent gap_up) ∪ (incongruent gap_down).
"""
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

import numpy as np
import pandas as pd

df = pd.read_csv("cache/phase2_events_075.csv")
YEARS = 7

# Derive trend_30d from existing columns
def derive_trend(row):
    if (row["direction"] == "gap_up"   and row["alignment"] == "concurrent") or \
       (row["direction"] == "gap_down" and row["alignment"] == "incongruent"):
        return "bullish"
    return "bearish"
df["trend_30d"] = df.apply(derive_trend, axis=1)

def stats(sub, label):
    if len(sub) == 0:
        print(f"{label}: empty")
        return
    n = len(sub)
    print(f"\n{label}")
    print(f"  n = {n}  ({n/YEARS:.1f}/yr)")
    print(f"  EV/trade   = {sub['eod_pnl'].mean():+.2f}")
    print(f"  Annual EV  = {sub['eod_pnl'].sum()/YEARS:+.1f}")
    print(f"  Win rate   = {sub['eod_safe'].mean()*100:.1f}%")
    print(f"  Median pnl = {sub['eod_pnl'].median():+.2f}")
    print(f"  p10        = {sub['eod_pnl'].quantile(0.10):+.2f}")
    print(f"  p25        = {sub['eod_pnl'].quantile(0.25):+.2f}")
    print(f"  Worst      = {sub['eod_pnl'].min():+.1f}")
    print(f"  Avg credit = {sub['credit'].mean():.2f}")
    print(f"  Avg width  = {sub['width'].mean():.2f}")

print("=" * 60)
print("Trend filter only (no VIX restriction) at 0.75×ATR")
print("=" * 60)

stats(df,                                          "ALL events (baseline reference)")
stats(df[df["trend_30d"] == "bullish"],            "BULLISH trend only (all VIX, any direction)")
stats(df[df["trend_30d"] == "bearish"],            "BEARISH trend only (all VIX, any direction)")

print("\n" + "=" * 60)
print("Split bullish-trend events by direction")
print("=" * 60)
b = df[df["trend_30d"] == "bullish"]
stats(b[b["direction"] == "gap_up"],   "Bullish trend + gap_up   (concurrent)")
stats(b[b["direction"] == "gap_down"], "Bullish trend + gap_down (incongruent — fear-spike)")

print("\n" + "=" * 60)
print("VIX breakdown WITHIN bullish-trend events")
print("=" * 60)
for regime in ["low", "mid", "high"]:
    stats(b[b["vix_regime"] == regime], f"Bullish trend × {regime} VIX")
