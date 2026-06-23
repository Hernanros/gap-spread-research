"""
Task 10 — gap-size tier interaction within candidate rules.

Tiers (in |gap_ratio| = |gap| / ATR units):
    T1: 0.75 ≤ |gap_ratio| < 1.00
    T2: 1.00 ≤ |gap_ratio| < 1.50
    T3: 1.50 ≤ |gap_ratio|

Reuses cache/phase2_events_075.csv (NS·EOD pnl already computed per event).

Goal: see whether within R1 / bullish / bullish×mid, deeper gaps give a more
favorable EV/tail ratio, or whether the bulk of the edge lives in the smallest
tier T1 (where there is structurally more time value to collect).
"""
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

import numpy as np
import pandas as pd

YEARS = 7

df = pd.read_csv("cache/phase2_events_075.csv")

# Derive trend label from existing alignment+direction
def trend(r):
    if (r["direction"] == "gap_up"   and r["alignment"] == "concurrent") or \
       (r["direction"] == "gap_down" and r["alignment"] == "incongruent"):
        return "bullish"
    return "bearish"
df["trend"] = df.apply(trend, axis=1)

# Gap-size tier
def tier(g):
    a = abs(g)
    if a < 1.00:  return "T1 [0.75-1.0)"
    if a < 1.50:  return "T2 [1.0-1.5)"
    return "T3 [1.5+)"
df["tier"] = df["gap_ratio"].apply(tier)

def stratify(sub, group_cols):
    if len(sub) == 0:
        return pd.DataFrame()
    s = sub.copy()
    s["c_over_w"] = s["credit"] / s["width"]
    g = s.groupby(group_cols, observed=True)
    return pd.DataFrame({
        "n":            g.size(),
        "n_yr":         (g.size() / YEARS).round(1),
        "ev_per_trade": g["eod_pnl"].mean().round(2),
        "annual_ev":    (g["eod_pnl"].sum() / YEARS).round(1),
        "win_%":        (g["eod_safe"].mean() * 100).round(1),
        "p10":          g["eod_pnl"].quantile(0.10).round(2),
        "worst":        g["eod_pnl"].min().round(1),
        "avg_credit":   g["credit"].mean().round(2),
        "avg_width":    g["width"].mean().round(1),
        "c/w_%":        (g["c_over_w"].mean() * 100).round(1),
    }).reset_index()


def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ── Rule definitions ──────────────────────────────────────────────────────────
rules = {
    "ALL (unfiltered)":          df,
    "Bullish trend only":        df[df["trend"] == "bullish"],
    "Bullish × mid VIX":         df[(df["trend"] == "bullish") & (df["vix_regime"] == "mid")],
    "R1: mid VIX × gap_down":    df[(df["vix_regime"] == "mid") & (df["direction"] == "gap_down")],
    "R2: bullish × mid × down":  df[(df["trend"] == "bullish") & (df["vix_regime"] == "mid") & (df["direction"] == "gap_down")],
}

for name, sub in rules.items():
    header(f"{name}   ({len(sub)} events, {len(sub)/YEARS:.1f}/yr)")
    print(stratify(sub, ["tier"]).to_string(index=False))


# ── Direction split within each tier of each rule ─────────────────────────────
header("DIRECTION × TIER within Bullish × mid VIX (the sweet-spot rule)")
print(stratify(df[(df["trend"] == "bullish") & (df["vix_regime"] == "mid")],
               ["tier", "direction"]).to_string(index=False))

header("DIRECTION × TIER within R1 (mid VIX × gap_down)")
print(stratify(df[(df["vix_regime"] == "mid") & (df["direction"] == "gap_down")],
               ["tier"]).to_string(index=False))


# ── Where does extra credit come from? ───────────────────────────────────────
header("Credit and width by tier — diagnostic")
print(stratify(df, ["tier"]).to_string(index=False))


# ── Candidate "deep-gap" sub-rule sweep ───────────────────────────────────────
header("Candidate sub-rules — combinations with stricter |gap_ratio| floor")
candidates = [
    ("Bullish × mid × |g|≥1.0",  df[(df["trend"]=="bullish") & (df["vix_regime"]=="mid") & (df["gap_ratio"].abs()>=1.0)]),
    ("Bullish × mid × |g|≥1.5",  df[(df["trend"]=="bullish") & (df["vix_regime"]=="mid") & (df["gap_ratio"].abs()>=1.5)]),
    ("R1 × |g|≥1.0",             df[(df["vix_regime"]=="mid") & (df["direction"]=="gap_down") & (df["gap_ratio"].abs()>=1.0)]),
    ("R1 × |g|≥1.5",             df[(df["vix_regime"]=="mid") & (df["direction"]=="gap_down") & (df["gap_ratio"].abs()>=1.5)]),
    ("R2 × |g|≥1.0",             df[(df["trend"]=="bullish") & (df["vix_regime"]=="mid") & (df["direction"]=="gap_down") & (df["gap_ratio"].abs()>=1.0)]),
]
rows = []
for name, s in candidates:
    if len(s) == 0:
        rows.append({"name": name, "n": 0})
        continue
    rows.append({
        "name": name,
        "n": len(s),
        "n_yr": round(len(s)/YEARS, 1),
        "ev_per_trade": round(s["eod_pnl"].mean(), 2),
        "annual_ev":   round(s["eod_pnl"].sum()/YEARS, 1),
        "win_%": round(s["eod_safe"].mean()*100, 1),
        "p10":   round(s["eod_pnl"].quantile(0.10), 2),
        "worst": round(s["eod_pnl"].min(), 1),
        "avg_credit": round(s["credit"].mean(), 2),
        "avg_width":  round(s["width"].mean(), 1),
    })
print(pd.DataFrame(rows).to_string(index=False))
