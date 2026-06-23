"""
Task 11 — Exit-rule optimization within R1 (mid-VIX × gap_down).

Tests:
    PT exits:    PT25 / PT50 / PT60 / PT75   (close when spread worth (1-pt)*credit)
    Soft stops:  SS25 / SS50 / SS75          (close when price moves N% of width adverse)
    Time exits:  12:00 / 14:00 / 14:30 / 15:00 / 15:30
    Combined:    PT60 + time-15:00 (whichever first)
    Baseline:    EOD hold

For each event, walks 1-min bars from 09:30. Spread value at minute t:
    BS_value(S(t), K_short, K_long, direction, T_remaining(t), sigma)
    T_remaining(t) = (1 - t/390) * 1/252

Uses event subset: R1 = vix_regime == 'mid' AND direction == 'gap_down' at 0.75×ATR.
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
T_FULL  = 1 / 252


# ── BS ───────────────────────────────────────────────────────────────────────
def bs_call_vec(S, K, T, r, sigma):
    """Vectorized BS call. S can be array, K T sigma r scalar (or all arrays)."""
    S = np.asarray(S, dtype=float)
    safe = (T > 1e-12) & (sigma > 0)
    out = np.where(safe, 0.0, np.maximum(S - K, 0.0))
    if not safe.any():
        return out
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bs = S * N.cdf(d1) - K * np.exp(-r * T) * N.cdf(d2)
    return np.where(safe, bs, np.maximum(S - K, 0.0))


def bs_put_vec(S, K, T, r, sigma):
    return bs_call_vec(S, K, T, r, sigma) - S + K * np.exp(-r * T)


def spread_value_vec(S, K_short, K_long, direction, T, sigma, r=0.05):
    if direction == "gap_up":
        return bs_put_vec(S, K_short, T, r, sigma) - bs_put_vec(S, K_long, T, r, sigma)
    return bs_call_vec(S, K_short, T, r, sigma) - bs_call_vec(S, K_long, T, r, sigma)


# ── Data ─────────────────────────────────────────────────────────────────────
def load_1min(path):
    df = pd.read_csv(path, parse_dates=["ts_event"]).set_index("ts_event")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    t = df.index.time
    return df[(t >= datetime.time(9, 30)) & (t <= datetime.time(16, 0))]


def load_daily(path):
    df = pd.read_csv(path, parse_dates=["ts_event"]).set_index("ts_event")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    t = df.index.time
    sess = df[(t >= datetime.time(9, 30)) & (t <= datetime.time(16, 0))]
    daily = sess.resample("1D").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()
    return daily[daily["volume"] > 0]


print("Loading 1-min bars for all 4 symbols (~700K rows each)...")
min1_all = {s: load_1min(PATHS[s]) for s in SYMBOLS}
for s, d in min1_all.items():
    print(f"  {s}: {len(d):,} 1-min rows")

print("Loading daily bars to recover prev_close per event...")
daily_all = {s: load_daily(PATHS[s]) for s in SYMBOLS}

events = pd.read_csv("cache/phase2_events_075.csv", parse_dates=["date"])

# Attach prev_close, today_open, today_close — compare by date (not tz-aware ts)
for s in SYMBOLS:
    daily_all[s] = daily_all[s].copy()
    daily_all[s]["__date"] = daily_all[s].index.date

def attach_pc(row):
    sd = daily_all[row["symbol"]]
    target_date = pd.Timestamp(row["date"]).date()
    mask = sd["__date"].values == target_date
    if not mask.any():
        return pd.Series({"prev_close": np.nan, "today_open": np.nan, "today_close": np.nan})
    pos = int(np.where(mask)[0][0])
    today = sd.iloc[pos]
    prev  = sd.iloc[pos - 1] if pos > 0 else None
    return pd.Series({
        "prev_close":  float(prev["close"]) if prev is not None else np.nan,
        "today_open":  float(today["open"]),
        "today_close": float(today["close"]),
    })

attached = events.apply(attach_pc, axis=1)
events = pd.concat([events, attached], axis=1).dropna(subset=["prev_close"]).reset_index(drop=True)

# R1 subset
r1 = events[(events["vix_regime"] == "mid") & (events["direction"] == "gap_down")].copy().reset_index(drop=True)
print(f"\nR1 events: {len(r1)} ({len(r1)/YEARS:.1f}/yr)")


# ── Per-event intraday simulation ────────────────────────────────────────────
def simulate_event(row):
    """Returns pnl under each policy for one event.

    Returns dict mapping policy_name -> pnl (in points).
    """
    sym       = row["symbol"]
    direction = row["direction"]
    atr       = float(row["atr"])
    sigma     = max(float(row["vix"]) / 100.0, 0.05)
    pc        = float(row["prev_close"])
    width     = WIDTH_ATR * atr
    K_short   = pc
    K_long    = pc - width if direction == "gap_up" else pc + width

    # Get this event's 1-min session bars
    d_date  = pd.Timestamp(row["date"]).date()
    day_bars = min1_all[sym][min1_all[sym].index.date == d_date]
    if day_bars.empty:
        return None

    closes = day_bars["close"].values.astype(float)
    highs  = day_bars["high"].values.astype(float)
    lows   = day_bars["low"].values.astype(float)
    n_bars = len(closes)

    # Time-to-expiry per bar (linear decay across 390 min)
    bar_idx = np.arange(n_bars)
    T_remaining = (1.0 - bar_idx / 390.0) * T_FULL
    T_remaining = np.maximum(T_remaining, 0.0)

    # Spread value at each minute using close price
    val = spread_value_vec(closes, K_short, K_long, direction, T_remaining, sigma)

    # Credit at entry
    S_open = float(closes[0])
    credit = float(val[0])
    if credit <= 0 or width <= 0:
        return None

    # EOD intrinsic + EOD pnl (baseline)
    close_px = float(row["today_close"]) if "today_close" in row.index else float(closes[-1])
    if direction == "gap_up":
        eod_intrinsic = min(max(K_short - close_px, 0.0), width)
    else:
        eod_intrinsic = min(max(close_px - K_short, 0.0), width)
    eod_pnl = credit - eod_intrinsic

    results = {"credit": credit, "EOD": eod_pnl, "n_bars": n_bars}

    # PT exits: close when val[t] <= (1-pt) * credit
    for pt in [0.25, 0.50, 0.60, 0.75]:
        target = (1 - pt) * credit
        hit = np.where(val <= target)[0]
        if len(hit) > 0:
            t = hit[0]
            results[f"PT{int(pt*100)}"] = credit - float(val[t])
        else:
            results[f"PT{int(pt*100)}"] = eod_pnl

    # Soft stops: close when underlying moves N% of width adverse from S_open
    # gap_up: adverse = price falls; trigger when low <= S_open - N*width
    # gap_down: adverse = price rises; trigger when high >= S_open + N*width
    for n_frac in [0.25, 0.50, 0.75]:
        if direction == "gap_up":
            trigger_level = S_open - n_frac * width
            hit = np.where(lows <= trigger_level)[0]
        else:
            trigger_level = S_open + n_frac * width
            hit = np.where(highs >= trigger_level)[0]
        if len(hit) > 0:
            t = hit[0]
            # Use the trigger price as the underlying price at stop, conservative
            S_stop = trigger_level
            T_stop = T_remaining[t]
            stop_val = float(spread_value_vec(np.array([S_stop]), K_short, K_long, direction, np.array([T_stop]), sigma)[0])
            results[f"SS{int(n_frac*100)}"] = credit - stop_val
        else:
            results[f"SS{int(n_frac*100)}"] = eod_pnl

    # Time-based exits at specific minutes-since-open
    time_minutes = {"12:00": 150, "14:00": 270, "14:30": 300, "15:00": 330, "15:30": 360}
    for label, m in time_minutes.items():
        idx = min(m, n_bars - 1)
        results[f"T{label}"] = credit - float(val[idx])

    # Combined PT60 + time exit at 15:00 (whichever fires first)
    target60 = 0.40 * credit  # 60% of credit captured when val drops to 40%
    hit_pt = np.where(val <= target60)[0]
    pt_time = int(hit_pt[0]) if len(hit_pt) > 0 else 10**9
    time_15 = min(330, n_bars - 1)
    exit_t = min(pt_time, time_15)
    results["PT60+T15:00"] = credit - float(val[exit_t])

    return results


print("\nSimulating intraday policies on R1 events...")
sim_rows = []
for _, ev in r1.iterrows():
    r = simulate_event(ev)
    if r is None:
        continue
    r["date"] = ev["date"]; r["symbol"] = ev["symbol"]; r["gap_ratio"] = ev["gap_ratio"]
    r["mr_eod_pnl"] = ev["eod_pnl"]  # for sanity check vs daily-based EOD
    sim_rows.append(r)

sim = pd.DataFrame(sim_rows)
print(f"Simulated {len(sim)} of {len(r1)} events.")

# ── Sanity check: my new EOD must match cached daily EOD ─────────────────────
diff = (sim["EOD"] - sim["mr_eod_pnl"]).abs().mean()
print(f"\nEOD pnl sanity diff (intraday minute-based vs cached daily): mean |diff| = {diff:.3f} pts")

# ── Summary per policy ───────────────────────────────────────────────────────
policy_cols = ["EOD",
               "PT25", "PT50", "PT60", "PT75",
               "SS25", "SS50", "SS75",
               "T12:00", "T14:00", "T14:30", "T15:00", "T15:30",
               "PT60+T15:00"]

rows = []
n = len(sim)
n_yr = n / YEARS
for col in policy_cols:
    pnl = sim[col].values
    rows.append({
        "policy": col,
        "n_yr": round(n_yr, 1),
        "ev_per_trade": round(pnl.mean(), 2),
        "annual_ev": round(pnl.sum() / YEARS, 1),
        "win_%": round((pnl > 0).mean() * 100, 1),
        "median": round(np.median(pnl), 2),
        "p10": round(np.quantile(pnl, 0.10), 2),
        "p25": round(np.quantile(pnl, 0.25), 2),
        "worst": round(pnl.min(), 1),
        "delta_vs_EOD_annual": round((pnl.sum() - sim["EOD"].sum())/YEARS, 1),
    })

summary = pd.DataFrame(rows)
print("\n" + "=" * 78)
print("Exit-policy comparison on R1 events  (NS·EOD baseline + alternatives)")
print("=" * 78)
print(summary.to_string(index=False))

# Save
os.makedirs("cache", exist_ok=True)
sim.to_csv("cache/phase2_r1_exit_sim.csv", index=False)
summary.to_csv("cache/phase2_r1_exit_summary.csv", index=False)
print("\nSaved sim + summary CSVs.")
