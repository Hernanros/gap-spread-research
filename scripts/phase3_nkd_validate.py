"""
Phase 3 — Validate R1 + M1 on NKD using proper 1-min Databento data.

Mirrors the methodology used for ES/NQ/RTY/YM:
  - Same 09:30-16:00 ET session window
  - Same daily resample for ATR / gap detection
  - Same intraday minute simulation for M1's SS75 trigger
  - VIX as σ proxy (acknowledged caveat: JNIV would be more accurate)

Compares NKD per-symbol stats vs the 4 US futures, plus computes
event-day overlap to quantify diversification.
"""
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

from __future__ import annotations
import datetime, os
import numpy as np
import pandas as pd
from scipy.stats import norm as N

SYMBOLS = ["ES", "NQ", "RTY", "YM", "NKD"]
PATHS   = {s: f"data/{s}_1m_2019_2026.csv" for s in SYMBOLS}
YEARS   = 7
WIDTH_ATR = 0.5
T_FULL  = 1/252
T_STOP  = 0.5/252


# ── BS ────────────────────────────────────────────────────────────────────────
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

def load_daily_from_1m(min1):
    t = min1.index.time
    sess = min1[(t >= datetime.time(9,30)) & (t <= datetime.time(16,0))]
    daily = sess.resample("1D").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()
    return daily[daily["volume"] > 0]


print("Loading 1-min bars for all 5 symbols (incl NKD)...")
min1_all = {s: load_1min(PATHS[s]) for s in SYMBOLS}
daily_all = {s: load_daily_from_1m(min1_all[s]) for s in SYMBOLS}
for s in SYMBOLS:
    d = daily_all[s]
    print(f"  {s}: {len(min1_all[s]):,} 1-min bars, {len(d):,} daily bars  ({d.index[0].date()} → {d.index[-1].date()})")

# VIX
vix = pd.read_csv("data/VIX_daily_2019_2026.csv", index_col=0, parse_dates=True)
vix.index = pd.to_datetime(vix.index)
vix_series = vix["close"]


# ── Event detection ──────────────────────────────────────────────────────────
def detect_events(daily, sym, threshold=0.75, lookback=30):
    d = daily.copy()
    pc = d["close"].shift(1)
    tr = pd.concat([(d["high"]-d["low"]),
                    (d["high"]-pc).abs(),
                    (d["low"]-pc).abs()], axis=1).max(axis=1)
    d["atr"]        = tr.ewm(span=14, adjust=False).mean()
    d["prev_close"] = pc
    d["gap_ratio"]  = (d["open"] - pc) / d["atr"]
    d["direction"]  = np.where(d["gap_ratio"] >=  threshold, "gap_up",
                      np.where(d["gap_ratio"] <= -threshold, "gap_down", "none"))
    d["close_Nd"]   = d["close"].shift(lookback)
    d["trend_30d"]  = np.where(d["close"] > d["close_Nd"], "bullish", "bearish")
    d["alignment"]  = np.where(
        ((d["direction"]=="gap_up")   & (d["trend_30d"]=="bullish")) |
        ((d["direction"]=="gap_down") & (d["trend_30d"]=="bearish")),
        "concurrent",
        np.where(
            ((d["direction"]=="gap_up")   & (d["trend_30d"]=="bearish")) |
            ((d["direction"]=="gap_down") & (d["trend_30d"]=="bullish")),
            "incongruent", "none"))
    ev = d[d["direction"] != "none"].dropna(subset=["atr", "prev_close", "close_Nd"]).copy()
    ev["symbol"] = sym
    return ev

def vix_regime(v):
    if v < 15: return "low"
    if v < 25: return "mid"
    return "high"


# ── Per-event simulation (R1 + M1) ───────────────────────────────────────────
def simulate_event(row, sym, day_bars):
    direction = row["direction"]
    atr     = float(row["atr"])
    pc      = float(row["prev_close"])
    today_open  = float(row["open"])
    today_close = float(row["close"])
    width   = WIDTH_ATR * atr

    # VIX at prev business day
    d = pd.Timestamp(row.name).normalize()
    prev_d = d - pd.tseries.offsets.BDay(1)
    # vix_series index is tz-naive; row.name might be tz-aware. Normalize.
    prev_d = prev_d.tz_localize(None) if prev_d.tz is not None else prev_d
    try:
        vix_val = float(vix_series.loc[prev_d])
    except KeyError:
        # ffill nearest
        idx = vix_series.index.get_indexer([prev_d], method="ffill")[0]
        vix_val = float(vix_series.iloc[idx]) if idx >= 0 else 20.0
    if np.isnan(vix_val): vix_val = 20.0
    sigma = max(vix_val/100.0, 0.05)
    regime = vix_regime(vix_val)

    # Get 1-min bars for that day from session-window data
    if day_bars.empty:
        return None
    closes = day_bars["close"].values.astype(float)
    highs  = day_bars["high"].values.astype(float)
    lows   = day_bars["low"].values.astype(float)
    n_bars = len(closes)
    if n_bars < 10:  # need at least some bars
        return None
    bar_idx = np.arange(n_bars)
    T_rem = np.maximum((1.0 - bar_idx/390.0) * T_FULL, 0.0)

    result = {
        "date": d.date(), "symbol": sym, "direction": direction,
        "gap_ratio": float(row["gap_ratio"]), "atr": atr,
        "alignment": row["alignment"], "vix": vix_val, "vix_regime": regime,
        "prev_close": pc, "open": today_open, "close": today_close,
        "r1_pnl": np.nan, "m1_pnl": np.nan,
    }

    # R1: mid VIX × gap_down
    if regime == "mid" and direction == "gap_down":
        K_short = pc
        K_long  = pc + width
        val0 = float(spread_value_vec(
            np.array([today_open]), K_short, K_long, direction,
            np.array([T_FULL]), sigma)[0])
        credit = max(val0, 0.0)
        if credit > 0 and width > 0:
            eod_intrinsic = min(max(today_close - K_short, 0.0), width)
            result["r1_pnl"] = credit - eod_intrinsic
            result["r1_credit"] = credit

    # M1: T2 × gap_up (1.0 ≤ |gap_ratio| < 1.5)
    abs_gap = abs(float(row["gap_ratio"]))
    if direction == "gap_up" and 1.0 <= abs_gap < 1.5:
        S_open = today_open
        K_short = S_open
        K_long  = S_open - width
        credit = float(spread_value_vec(
            np.array([S_open]), K_short, K_long, direction,
            np.array([T_FULL]), sigma)[0])
        if credit > 0 and width > 0:
            # Intraday SS75 trigger
            trigger_level = S_open - 0.75 * width
            val = spread_value_vec(closes, K_short, K_long, direction, T_rem, sigma)
            hit = np.where(lows <= trigger_level)[0]
            if len(hit):
                t = int(hit[0])
                stop_val = float(spread_value_vec(
                    np.array([trigger_level]), K_short, K_long, direction,
                    np.array([T_rem[t]]), sigma)[0])
                result["m1_pnl"] = credit - stop_val
            else:
                eod_intrinsic = min(max(K_short - today_close, 0.0), width)
                result["m1_pnl"] = credit - eod_intrinsic
            result["m1_credit"] = credit

    return result


# ── Run on all symbols ────────────────────────────────────────────────────────
print("\nDetecting events + simulating per symbol...")
all_rows = []
for sym in SYMBOLS:
    ev = detect_events(daily_all[sym], sym)
    print(f"  {sym}: {len(ev)} events at 0.75x")
    for dt, row in ev.iterrows():
        target_date = pd.Timestamp(dt).date()
        day_bars = min1_all[sym][min1_all[sym].index.date == target_date]
        res = simulate_event(row, sym, day_bars)
        if res is not None:
            all_rows.append(res)

results = pd.DataFrame(all_rows)
print(f"\nTotal simulated events: {len(results)}")


# ── Summary per symbol ────────────────────────────────────────────────────────
def summarise(df, pnl_col, label):
    sub = df[df[pnl_col].notna()].copy()
    if sub.empty:
        return {"label": label, "n": 0}
    p = sub[pnl_col].values
    return {
        "label": label,
        "n": len(sub),
        "n_yr": round(len(sub)/YEARS, 1),
        "ev_per_trade": round(p.mean(), 2),
        "annual_ev": round(p.sum()/YEARS, 1),
        "win_%": round((p > 0).mean()*100, 1),
        "p10": round(np.quantile(p, 0.10), 2),
        "worst": round(p.min(), 1),
        "avg_credit": round(sub.get(pnl_col.replace("pnl", "credit"), pd.Series(dtype=float)).mean(), 2) if pnl_col.replace("pnl", "credit") in sub.columns else np.nan,
    }

def header(s):
    print("\n" + "="*78); print(s); print("="*78)

header("R1 — mid-VIX × gap_down × bear-call spread (hold to EOD), per symbol")
print(pd.DataFrame([summarise(results[results["symbol"]==s], "r1_pnl", s) for s in SYMBOLS]).to_string(index=False))

header("M1 — T2 × gap_up × bull-put @ open (intraday SS75), per symbol")
print(pd.DataFrame([summarise(results[results["symbol"]==s], "m1_pnl", s) for s in SYMBOLS]).to_string(index=False))


# ── Gap-day overlap ───────────────────────────────────────────────────────────
header("Gap-day overlap matrix (count of shared 0.75x events)")
def gap_days(df):
    return set(df["date"].unique())
ov = pd.DataFrame(0, index=SYMBOLS, columns=SYMBOLS)
for a in SYMBOLS:
    da = gap_days(results[results["symbol"]==a])
    for b in SYMBOLS:
        db = gap_days(results[results["symbol"]==b])
        ov.loc[a, b] = len(da & db)
print(ov)

print("\nFraction of A's days also in B (rows=A):")
frac = ov.copy().astype(float)
for a in SYMBOLS:
    da_n = len(gap_days(results[results["symbol"]==a]))
    for b in SYMBOLS:
        frac.loc[a, b] = round(ov.loc[a, b] / da_n, 2) if da_n > 0 else 0
print(frac)


# Save
os.makedirs("cache", exist_ok=True)
results.to_csv("cache/phase3_nkd_validate.csv", index=False)
print(f"\nSaved cache/phase3_nkd_validate.csv ({len(results)} rows)")
