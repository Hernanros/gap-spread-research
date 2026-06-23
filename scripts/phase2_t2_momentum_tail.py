"""
Task 11b — Tail-risk control for T2 × gap_up momentum.

Two-pronged investigation:
  A. Diagnose: stratify the worst trades by VIX / alignment / symbol.
     Find a sub-filter that bounds the -229 pt worst case.
  B. Exit policies: rerun intraday PT/SS/Time sweep on momentum structure.
     With 60% win rate (vs 96% for R1), exit rules may help here.
"""
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

from __future__ import annotations
import datetime, os
import numpy as np
import pandas as pd
from scipy.stats import norm as N

SYMBOLS = ["ES", "NQ", "RTY", "YM"]
PATHS   = {s: f"data/{s}_1m_2019_2026.csv" for s in SYMBOLS}
YEARS   = 7
WIDTH_ATR = 0.5
T_FULL  = 1/252


# ── BS ────────────────────────────────────────────────────────────────────────
def bs_call_vec(S, K, T, r, sigma):
    S = np.asarray(S, dtype=float)
    safe = (T > 1e-12) & (sigma > 0)
    out = np.where(safe, 0.0, np.maximum(S - K, 0.0))
    if not safe.any(): return out
    d1 = (np.log(np.maximum(S,1e-12)/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(np.maximum(T,1e-12)))
    d2 = d1 - sigma*np.sqrt(np.maximum(T,1e-12))
    bs = S*N.cdf(d1) - K*np.exp(-r*T)*N.cdf(d2)
    return np.where(safe, bs, np.maximum(S - K, 0.0))

def bs_put_vec(S, K, T, r, sigma):
    return bs_call_vec(S, K, T, r, sigma) - S + K*np.exp(-r*T)

def spread_value_vec(S, K_short, K_long, direction, T, sigma, r=0.05):
    if direction == "gap_up":
        return bs_put_vec(S, K_short, T, r, sigma) - bs_put_vec(S, K_long, T, r, sigma)
    return bs_call_vec(S, K_short, T, r, sigma) - bs_call_vec(S, K_long, T, r, sigma)


# ── Data ──────────────────────────────────────────────────────────────────────
def load_1min(p):
    df = pd.read_csv(p, parse_dates=["ts_event"]).set_index("ts_event")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    t = df.index.time
    return df[(t >= datetime.time(9,30)) & (t <= datetime.time(16,0))]
def load_daily(p):
    df = pd.read_csv(p, parse_dates=["ts_event"]).set_index("ts_event")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    t = df.index.time
    sess = df[(t >= datetime.time(9,30)) & (t <= datetime.time(16,0))]
    daily = sess.resample("1D").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"),
        volume=("volume","sum"),
    ).dropna()
    return daily[daily["volume"] > 0]

print("Loading data...")
min1_all  = {s: load_1min(PATHS[s]) for s in SYMBOLS}
daily_all = {s: load_daily(PATHS[s]) for s in SYMBOLS}
for s in SYMBOLS:
    daily_all[s] = daily_all[s].copy()
    daily_all[s]["__date"] = daily_all[s].index.date

events = pd.read_csv("cache/phase2_events_075.csv", parse_dates=["date"])

# Attach today_open + today_close
def attach_oc(row):
    sd = daily_all[row["symbol"]]
    mask = sd["__date"].values == pd.Timestamp(row["date"]).date()
    if not mask.any(): return pd.Series({"today_open": np.nan, "today_close": np.nan})
    today = sd.iloc[int(np.where(mask)[0][0])]
    return pd.Series({"today_open": float(today["open"]), "today_close": float(today["close"])})

events = pd.concat([events, events.apply(attach_oc, axis=1)], axis=1)
events = events.dropna(subset=["today_open", "today_close"]).reset_index(drop=True)

# Tier
def tier(g):
    a = abs(g)
    if a < 1.00: return "T1"
    if a < 1.50: return "T2"
    return "T3"
events["tier"] = events["gap_ratio"].apply(tier)


# ── Momentum pnl (vectorized per row) ────────────────────────────────────────
def mom_credit_pnl(row):
    direction = row["direction"]
    atr     = float(row["atr"])
    sigma   = max(float(row["vix"])/100.0, 0.05)
    S_open  = float(row["today_open"])
    close_px = float(row["today_close"])
    width   = WIDTH_ATR * atr
    K_short = S_open
    K_long  = S_open - width if direction == "gap_up" else S_open + width
    credit  = float(spread_value_vec(np.array([S_open]), K_short, K_long, direction, np.array([T_FULL]), sigma)[0])
    if credit <= 0:
        return pd.Series({"mom_credit": 0.0, "mom_eod_pnl": 0.0, "mom_win": False, "K_short": np.nan, "K_long": np.nan})
    if direction == "gap_up":
        intrinsic = min(max(K_short - close_px, 0.0), width)
        win = close_px > S_open
    else:
        intrinsic = min(max(close_px - K_short, 0.0), width)
        win = close_px < S_open
    return pd.Series({"mom_credit": credit, "mom_eod_pnl": credit - intrinsic, "mom_win": win,
                      "K_short": K_short, "K_long": K_long})

mom = events.apply(mom_credit_pnl, axis=1)
events = pd.concat([events, mom], axis=1)

# Focus subset
t2u = events[(events["tier"] == "T2") & (events["direction"] == "gap_up")].copy().reset_index(drop=True)
print(f"\nT2 × gap_up events: {len(t2u)} ({len(t2u)/YEARS:.1f}/yr)")


# ── A. Diagnose worst trades ─────────────────────────────────────────────────
print("\n" + "="*78)
print("A. Worst 10 T2 × gap_up momentum trades")
print("="*78)
worst10 = t2u.nsmallest(10, "mom_eod_pnl")
print(worst10[["date", "symbol", "vix", "vix_regime", "alignment",
               "gap_ratio", "atr", "mom_credit", "mom_eod_pnl"]].to_string(index=False))


def stratify(sub, group_cols):
    if len(sub) == 0: return pd.DataFrame()
    g = sub.groupby(group_cols, observed=True)
    return pd.DataFrame({
        "n":            g.size(),
        "n_yr":         (g.size()/YEARS).round(1),
        "ev":           g["mom_eod_pnl"].mean().round(2),
        "annual_ev":    (g["mom_eod_pnl"].sum()/YEARS).round(1),
        "win_%":        (g["mom_win"].mean()*100).round(1),
        "p10":          g["mom_eod_pnl"].quantile(0.10).round(2),
        "worst":        g["mom_eod_pnl"].min().round(1),
        "avg_credit":   g["mom_credit"].mean().round(2),
    }).reset_index()


print("\n" + "="*78)
print("T2 × gap_up — stratified")
print("="*78)
print("\nBy VIX regime:")
print(stratify(t2u, ["vix_regime"]).to_string(index=False))
print("\nBy alignment:")
print(stratify(t2u, ["alignment"]).to_string(index=False))
print("\nBy symbol:")
print(stratify(t2u, ["symbol"]).to_string(index=False))
print("\nBy VIX × alignment:")
print(stratify(t2u, ["vix_regime", "alignment"]).to_string(index=False))


# ── B. Test exit policies on T2 × gap_up momentum ───────────────────────────
print("\n" + "="*78)
print("B. Intraday exit-policy sweep on T2 × gap_up momentum")
print("="*78)

def simulate_event(row):
    sym       = row["symbol"]
    direction = row["direction"]
    atr       = float(row["atr"])
    sigma     = max(float(row["vix"])/100.0, 0.05)
    S_open    = float(row["today_open"])
    close_px  = float(row["today_close"])
    width     = WIDTH_ATR * atr
    K_short   = S_open
    K_long    = S_open - width if direction == "gap_up" else S_open + width

    day_bars = min1_all[sym][min1_all[sym].index.date == pd.Timestamp(row["date"]).date()]
    if day_bars.empty: return None

    closes = day_bars["close"].values.astype(float)
    highs  = day_bars["high"].values.astype(float)
    lows   = day_bars["low"].values.astype(float)
    n_bars = len(closes)
    bar_idx = np.arange(n_bars)
    T_rem = np.maximum((1.0 - bar_idx/390.0) * T_FULL, 0.0)

    val = spread_value_vec(closes, K_short, K_long, direction, T_rem, sigma)
    credit = float(val[0])
    if credit <= 0: return None

    if direction == "gap_up":
        eod_intrinsic = min(max(K_short - close_px, 0.0), width)
    else:
        eod_intrinsic = min(max(close_px - K_short, 0.0), width)
    eod_pnl = credit - eod_intrinsic
    out = {"credit": credit, "EOD": eod_pnl}

    for pt in [0.25, 0.50, 0.60, 0.75]:
        target = (1 - pt) * credit
        hit = np.where(val <= target)[0]
        out[f"PT{int(pt*100)}"] = (credit - float(val[hit[0]])) if len(hit) else eod_pnl

    for n_frac in [0.25, 0.50, 0.75]:
        if direction == "gap_up":
            trig_lvl = S_open - n_frac*width
            hit = np.where(lows <= trig_lvl)[0]
        else:
            trig_lvl = S_open + n_frac*width
            hit = np.where(highs >= trig_lvl)[0]
        if len(hit):
            t = int(hit[0])
            stop_val = float(spread_value_vec(
                np.array([trig_lvl]), K_short, K_long, direction,
                np.array([T_rem[t]]), sigma)[0])
            out[f"SS{int(n_frac*100)}"] = credit - stop_val
        else:
            out[f"SS{int(n_frac*100)}"] = eod_pnl

    for label, m in {"11:00":90, "12:00":150, "13:00":210, "14:00":270, "14:30":300, "15:00":330, "15:30":360}.items():
        idx = min(m, n_bars - 1)
        out[f"T{label}"] = credit - float(val[idx])

    return out

sim_rows = []
for _, ev in t2u.iterrows():
    r = simulate_event(ev)
    if r is None: continue
    sim_rows.append(r)
sim = pd.DataFrame(sim_rows)
print(f"\nSimulated {len(sim)} of {len(t2u)} events.")

policy_cols = ["EOD",
               "PT25", "PT50", "PT60", "PT75",
               "SS25", "SS50", "SS75",
               "T11:00", "T12:00", "T13:00", "T14:00", "T14:30", "T15:00", "T15:30"]
n_yr = len(sim) / YEARS
rows = []
for col in policy_cols:
    p = sim[col].values
    rows.append({
        "policy": col,
        "ev_per_trade": round(p.mean(), 2),
        "annual_ev": round(p.sum()/YEARS, 1),
        "win_%": round((p > 0).mean()*100, 1),
        "p10": round(np.quantile(p, 0.10), 2),
        "worst": round(p.min(), 1),
        "delta_vs_EOD_annual": round((p.sum() - sim["EOD"].sum())/YEARS, 1),
    })
print(pd.DataFrame(rows).to_string(index=False))

# Save outputs
os.makedirs("cache", exist_ok=True)
sim.to_csv("cache/phase2_t2u_exit_sim.csv", index=False)
t2u.to_csv("cache/phase2_t2u_events.csv", index=False)
print("\nSaved cache/phase2_t2u_exit_sim.csv and cache/phase2_t2u_events.csv")
