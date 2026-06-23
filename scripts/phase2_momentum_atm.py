"""
Task 10b — Momentum bet on deep gaps with strikes anchored at today's open.

Structure differs from the mean-reversion version:
    K_short = open          (ATM at entry, not prev_close)
    K_long  = open ± 0.5×ATR

  gap_up   → bull put spread anchored at open: profit if close > open
  gap_down → bear call spread anchored at open: profit if close < open

The bet is that the gap HOLDS / extends — price stays on the gap-side of open.

Reuses cache/phase2_events_075.csv for events. Re-prices credit + EOD pnl
under the new strike anchoring. Compares to the prev_close-anchored version
across all three tiers, with special focus on T3 (|gap| ≥ 1.5×ATR).
"""
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

from __future__ import annotations
import datetime, os
import numpy as np
import pandas as pd
from scipy.stats import norm as N

YEARS      = 7
WIDTH_ATR  = 0.5
T_FULL     = 1 / 252


def bs_call(S, K, T, r, sigma):
    if T <= 1e-12 or sigma <= 0: return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N.cdf(d1) - K * np.exp(-r * T) * N.cdf(d2)
def bs_put(S, K, T, r, sigma):
    return bs_call(S, K, T, r, sigma) - S + K * np.exp(-r * T)
def spread_value(S, K_short, K_long, direction, T, sigma, r=0.05):
    if direction == "gap_up":
        return bs_put(S, K_short, T, r, sigma) - bs_put(S, K_long, T, r, sigma)
    return bs_call(S, K_short, T, r, sigma) - bs_call(S, K_long, T, r, sigma)


# ── Reload raw daily data so we have today's CLOSE per event ──────────────────
SYMBOLS = ["ES", "NQ", "RTY", "YM"]
PATHS   = {s: f"data/{s}_1m_2019_2026.csv" for s in SYMBOLS}

def load_daily(path):
    df = pd.read_csv(path, parse_dates=["ts_event"]).set_index("ts_event")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    t = df.index.time
    sess = df[(t >= datetime.time(9, 30)) & (t <= datetime.time(16, 0))]
    daily = sess.resample("1D").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),     close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()
    return daily[daily["volume"] > 0]

print("Loading daily bars to attach today's close per event...")
daily_all = {s: load_daily(PATHS[s]) for s in SYMBOLS}

# Event table from task 9 stratify (has open, prev_close, atr, vix, eod_pnl old, etc.)
# But we need TODAY's CLOSE explicitly; it's already there as `eod_pnl` derives from close.
# Cache file rows: date is the event date (today). We need that day's close.
df = pd.read_csv("cache/phase2_events_075.csv", parse_dates=["date"])

# Attach today's close per row from the raw daily bars
def lookup_close(row):
    sym_daily = daily_all[row["symbol"]]
    d = pd.Timestamp(row["date"])
    if d.tzinfo is None:
        d = d.tz_localize("America/New_York") if sym_daily.index.tz else d
    # daily resample('1D') yields midnight-of-day index in tz; match by date
    mask = sym_daily.index.normalize() == d.normalize()
    if mask.any():
        return float(sym_daily.loc[mask, "close"].iloc[0]), float(sym_daily.loc[mask, "open"].iloc[0])
    return np.nan, np.nan

print("Attaching today's open and close...")
closes_opens = df.apply(lookup_close, axis=1, result_type="expand")
df["today_open"]  = closes_opens[1]
df["today_close"] = closes_opens[0]
df = df.dropna(subset=["today_open", "today_close"]).reset_index(drop=True)

print(f"{len(df)} events with today's open + close attached.")


# ── Re-price under both structures ────────────────────────────────────────────
def momentum_pnl(row):
    """Bet that gap HOLDS — strikes anchored at today's open.
    gap_up   → bull put spread at K_short=open
    gap_down → bear call spread at K_short=open
    """
    direction = row["direction"]
    atr       = row["atr"]
    sigma     = max(row["vix"] / 100.0, 0.05)
    S_open    = row["today_open"]
    close_px  = row["today_close"]
    width     = WIDTH_ATR * atr

    K_short = S_open
    K_long  = S_open - width if direction == "gap_up" else S_open + width

    credit = spread_value(S_open, K_short, K_long, direction, T_FULL, sigma)
    if credit <= 0:
        return pd.Series({"mom_credit": 0.0, "mom_eod_pnl": 0.0, "mom_win": False})

    if direction == "gap_up":
        eod_intrinsic = min(max(K_short - close_px, 0.0), width)
        win = close_px > S_open
    else:
        eod_intrinsic = min(max(close_px - K_short, 0.0), width)
        win = close_px < S_open
    return pd.Series({"mom_credit": credit, "mom_eod_pnl": credit - eod_intrinsic, "mom_win": win})

print("Re-pricing under momentum (K_short=open) structure...")
mom = df.apply(momentum_pnl, axis=1)
df = pd.concat([df, mom], axis=1)


# ── Tier label ───────────────────────────────────────────────────────────────
def tier(g):
    a = abs(g)
    if a < 1.00: return "T1 [0.75-1.0)"
    if a < 1.50: return "T2 [1.0-1.5)"
    return "T3 [1.5+)"
df["tier"] = df["gap_ratio"].apply(tier)


# ── Stratifier supporting either pnl column ──────────────────────────────────
def stratify(sub, group_cols, pnl_col, win_col, credit_col):
    if len(sub) == 0: return pd.DataFrame()
    s = sub.copy()
    g = s.groupby(group_cols, observed=True)
    return pd.DataFrame({
        "n":            g.size(),
        "n_yr":         (g.size() / YEARS).round(1),
        "ev_per_trade": g[pnl_col].mean().round(2),
        "annual_ev":    (g[pnl_col].sum() / YEARS).round(1),
        "win_%":        (g[win_col].mean() * 100).round(1),
        "p10":          g[pnl_col].quantile(0.10).round(2),
        "worst":        g[pnl_col].min().round(1),
        "avg_credit":   g[credit_col].mean().round(2),
    }).reset_index()


def header(s):
    print("\n" + "=" * 78); print(s); print("=" * 78)


# ── Headline comparison ──────────────────────────────────────────────────────
header("Mean-reversion (K_short = prev_close)  by tier")
print(stratify(df, ["tier"], "eod_pnl", "eod_safe", "credit").to_string(index=False))

header("Momentum (K_short = today's open)  by tier")
print(stratify(df, ["tier"], "mom_eod_pnl", "mom_win", "mom_credit").to_string(index=False))

header("Momentum  by tier × direction")
print(stratify(df, ["tier", "direction"], "mom_eod_pnl", "mom_win", "mom_credit").to_string(index=False))

header("Momentum on T3 only — tier × VIX regime × direction")
t3 = df[df["tier"] == "T3 [1.5+)"]
print(f"({len(t3)} T3 events, {len(t3)/YEARS:.1f}/yr)")
print(stratify(t3, ["vix_regime", "direction"], "mom_eod_pnl", "mom_win", "mom_credit").to_string(index=False))

header("Momentum on T3 only — VIX × alignment × direction")
print(stratify(t3, ["vix_regime", "alignment", "direction"], "mom_eod_pnl", "mom_win", "mom_credit").to_string(index=False))


# ── How often does the gap HOLD vs reverse? ──────────────────────────────────
header("Diagnostic: did the gap hold (close on the gap-side of open)?")
diag = []
for t_label in ["T1 [0.75-1.0)", "T2 [1.0-1.5)", "T3 [1.5+)"]:
    for direction in ["gap_up", "gap_down"]:
        sub = df[(df["tier"] == t_label) & (df["direction"] == direction)]
        if len(sub) == 0: continue
        hold_rate = sub["mom_win"].mean() * 100
        diag.append({
            "tier": t_label, "direction": direction,
            "n_yr": round(len(sub)/YEARS, 1),
            "gap_holds_%": round(hold_rate, 1),
            "avg_credit_mom": round(sub["mom_credit"].mean(), 2),
            "avg_pnl_mom": round(sub["mom_eod_pnl"].mean(), 2),
        })
print(pd.DataFrame(diag).to_string(index=False))


# ── Side-by-side: same trades, two structures ────────────────────────────────
header("Side-by-side: same events, mean-reversion (MR) vs momentum (MOM)")
rows = []
for t_label in ["T1 [0.75-1.0)", "T2 [1.0-1.5)", "T3 [1.5+)"]:
    sub = df[df["tier"] == t_label]
    if len(sub) == 0: continue
    rows.append({
        "tier":             t_label,
        "n_yr":             round(len(sub)/YEARS, 1),
        "MR_ev":            round(sub["eod_pnl"].mean(), 2),
        "MR_annual":        round(sub["eod_pnl"].sum()/YEARS, 1),
        "MR_win_%":         round(sub["eod_safe"].mean()*100, 1),
        "MR_worst":         round(sub["eod_pnl"].min(), 1),
        "MR_avg_credit":    round(sub["credit"].mean(), 2),
        "MOM_ev":           round(sub["mom_eod_pnl"].mean(), 2),
        "MOM_annual":       round(sub["mom_eod_pnl"].sum()/YEARS, 1),
        "MOM_win_%":        round(sub["mom_win"].mean()*100, 1),
        "MOM_worst":        round(sub["mom_eod_pnl"].min(), 1),
        "MOM_avg_credit":   round(sub["mom_credit"].mean(), 2),
    })
print(pd.DataFrame(rows).to_string(index=False))
