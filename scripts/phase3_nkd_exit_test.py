"""
Phase 3 — Exit-policy validation on NKD.

Mirrors Tasks 11 + 11b but on NKD only:
  - R1 cell (mid VIX × gap_down) — does EOD still dominate intraday exits?
  - M1 cell (T2 × gap_up)        — does SS75 still cap the tail?

Same PT/SS/Time grid as the US futures tests.
"""
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

from __future__ import annotations
import datetime, os
import numpy as np
import pandas as pd
from scipy.stats import norm as N

YEARS = 7
WIDTH_ATR = 0.5
T_FULL = 1/252


def bs_call_vec(S, K, T, r, sigma):
    S = np.asarray(S, dtype=float)
    safe = (T > 1e-12) & (sigma > 0)
    out = np.where(safe, 0.0, np.maximum(S - K, 0.0))
    if not safe.any(): return out
    d1 = (np.log(np.maximum(S,1e-12)/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(np.maximum(T,1e-12)))
    d2 = d1 - sigma*np.sqrt(np.maximum(T,1e-12))
    return np.where(safe, S*N.cdf(d1) - K*np.exp(-r*T)*N.cdf(d2), np.maximum(S - K, 0.0))
def bs_put_vec(S, K, T, r, sigma):
    return bs_call_vec(S, K, T, r, sigma) - S + K*np.exp(-r*T)
def spread_value_vec(S, K_short, K_long, direction, T, sigma, r=0.05):
    if direction == "gap_up":
        return bs_put_vec(S, K_short, T, r, sigma) - bs_put_vec(S, K_long, T, r, sigma)
    return bs_call_vec(S, K_short, T, r, sigma) - bs_call_vec(S, K_long, T, r, sigma)


def load_1min(p):
    df = pd.read_csv(p, parse_dates=["ts_event"]).set_index("ts_event")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    t = df.index.time
    return df[(t >= datetime.time(9,30)) & (t <= datetime.time(16,0))]


print("Loading NKD 1-min...")
min1 = load_1min("data/NKD_1m_2019_2026.csv")

# Use the cached event table from phase3_nkd_validate.py
events = pd.read_csv("cache/phase3_nkd_validate.csv", parse_dates=["date"])
nkd = events[events["symbol"] == "NKD"].copy()
print(f"NKD events: {len(nkd)}")


def simulate_event(row, structure):
    """structure ∈ {'r1','m1'}. Returns dict of pnl per exit policy."""
    direction = row["direction"]
    atr     = float(row["atr"])
    sigma   = max(float(row["vix"])/100.0, 0.05)
    pc      = float(row["prev_close"])
    today_open = float(row["open"])
    today_close = float(row["close"])
    width   = WIDTH_ATR * atr

    target_date = pd.Timestamp(row["date"]).date()
    day_bars = min1[min1.index.date == target_date]
    if day_bars.empty: return None
    closes = day_bars["close"].values.astype(float)
    highs  = day_bars["high"].values.astype(float)
    lows   = day_bars["low"].values.astype(float)
    n_bars = len(closes)
    if n_bars < 10: return None
    bar_idx = np.arange(n_bars)
    T_rem = np.maximum((1.0 - bar_idx/390.0) * T_FULL, 0.0)

    # Choose strike anchor
    if structure == "r1":
        if direction != "gap_down": return None
        K_short = pc
        K_long  = pc + width
        S_open = today_open
    else:  # m1
        if direction != "gap_up": return None
        K_short = today_open
        K_long  = today_open - width
        S_open = today_open

    val = spread_value_vec(closes, K_short, K_long, direction, T_rem, sigma)
    credit = float(val[0])
    if credit <= 0 or width <= 0: return None

    # EOD intrinsic
    if direction == "gap_up":
        eod_intrinsic = min(max(K_short - today_close, 0.0), width)
    else:
        eod_intrinsic = min(max(today_close - K_short, 0.0), width)
    eod_pnl = credit - eod_intrinsic

    out = {"credit": credit, "EOD": eod_pnl}

    for pt in [0.25, 0.50, 0.60, 0.75]:
        target = (1 - pt) * credit
        hit = np.where(val <= target)[0]
        out[f"PT{int(pt*100)}"] = (credit - float(val[hit[0]])) if len(hit) else eod_pnl

    for n_frac in [0.25, 0.50, 0.75]:
        if direction == "gap_up":
            trig = S_open - n_frac*width
            hit = np.where(lows <= trig)[0]
        else:
            trig = S_open + n_frac*width
            hit = np.where(highs >= trig)[0]
        if len(hit):
            t = int(hit[0])
            stop_val = float(spread_value_vec(
                np.array([trig]), K_short, K_long, direction,
                np.array([T_rem[t]]), sigma)[0])
            out[f"SS{int(n_frac*100)}"] = credit - stop_val
        else:
            out[f"SS{int(n_frac*100)}"] = eod_pnl

    for label, m in {"11:00":90, "12:00":150, "13:00":210, "14:00":270, "14:30":300, "15:00":330, "15:30":360}.items():
        idx = min(m, n_bars - 1)
        out[f"T{label}"] = credit - float(val[idx])

    return out


def report(sim, title):
    print("\n" + "="*78); print(title); print("="*78)
    if sim.empty:
        print("(no events)")
        return
    n_yr = len(sim) / YEARS
    print(f"n_events = {len(sim)} ({n_yr:.1f}/yr)")
    cols = ["EOD","PT25","PT50","PT60","PT75",
            "SS25","SS50","SS75",
            "T11:00","T12:00","T13:00","T14:00","T14:30","T15:00","T15:30"]
    rows = []
    for c in cols:
        p = sim[c].values
        rows.append({
            "policy": c,
            "ev_per_trade": round(p.mean(), 2),
            "annual_ev": round(p.sum()/YEARS, 1),
            "win_%": round((p > 0).mean()*100, 1),
            "p10": round(np.quantile(p, 0.10), 2),
            "worst": round(p.min(), 1),
            "delta_vs_EOD": round((p.sum() - sim["EOD"].sum())/YEARS, 1),
        })
    print(pd.DataFrame(rows).to_string(index=False))


# R1 cell on NKD
r1_rows = []
for _, ev in nkd[(nkd["vix_regime"]=="mid") & (nkd["direction"]=="gap_down")].iterrows():
    r = simulate_event(ev, "r1")
    if r is not None: r1_rows.append(r)
r1_sim = pd.DataFrame(r1_rows)
report(r1_sim, "R1 (mid-VIX × gap_down) — NKD intraday exit-policy sweep")

# M1 cell on NKD
m1_rows = []
for _, ev in nkd.iterrows():
    abs_gap = abs(float(ev["gap_ratio"]))
    if ev["direction"] != "gap_up" or not (1.0 <= abs_gap < 1.5): continue
    r = simulate_event(ev, "m1")
    if r is not None: m1_rows.append(r)
m1_sim = pd.DataFrame(m1_rows)
report(m1_sim, "M1 (T2 × gap_up) — NKD intraday exit-policy sweep")

# Save
os.makedirs("cache", exist_ok=True)
r1_sim.to_csv("cache/phase3_nkd_r1_sim.csv", index=False)
m1_sim.to_csv("cache/phase3_nkd_m1_sim.csv", index=False)
print("\nSaved cache/phase3_nkd_r1_sim.csv and phase3_nkd_m1_sim.csv")
