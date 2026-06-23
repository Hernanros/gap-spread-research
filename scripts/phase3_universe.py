"""
Phase 3 / Tasks 7+8 — Universe expansion validation.

Fetches yfinance daily OHLC for:
  - SPY, QQQ, IWM, DIA   (US ETF analogues to ES, NQ, RTY, YM)
  - ^N225 (Nikkei cash)  (Asian session diversifier)
  - ^GDAXI (DAX cash)    (European session diversifier)
  - ^VIX                 (regime filter; reuse for non-US too as proxy)

Applies R1 and M1 with daily-only data:
  R1: 0.75x ATR gap_down, mid-VIX(prev), bear call @ prev_close,
      width = 0.5 x ATR, hold to close, EOD intrinsic from daily close.
  M1: 1.0-1.5x ATR gap_up, bull put @ today_open,
      width = 0.5 x ATR, SS75 trigger approximated via daily LOW
      (low <= open - 0.75 * width => stop at trigger price),
      else hold to close.

All EV in instrument-points (not dollars).

Caveats:
  - Daily OHLC only: SS75 trigger timing is approximate (uses daily low,
    assumes mid-day stop with T_stop = 0.5/252).
  - VIX is the only sigma proxy used. For DAX/Nikkei, this is a known
    miscalibration (V2X for DAX, JNIV for Nikkei would be more accurate).
    Treat the international results as directional, not calibrated.
"""
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm as N

try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    os.system(f"{sys.executable} -m pip install yfinance >/dev/null")
    import yfinance as yf

START = "2019-01-01"
END   = "2026-06-22"
YEARS = 7.0
WIDTH_ATR = 0.5
T_FULL    = 1/252
T_STOP    = 0.5/252


# ── BS ────────────────────────────────────────────────────────────────────────
def bs_call(S, K, T, r, sigma):
    if T <= 1e-12 or sigma <= 0: return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*N.cdf(d1) - K*np.exp(-r*T)*N.cdf(d2)
def bs_put(S, K, T, r, sigma):
    return bs_call(S, K, T, r, sigma) - S + K*np.exp(-r*T)
def spread_value(S, K_short, K_long, direction, T, sigma, r=0.05):
    if direction == "gap_up":
        return bs_put(S, K_short, T, r, sigma) - bs_put(S, K_long, T, r, sigma)
    return bs_call(S, K_short, T, r, sigma) - bs_call(S, K_long, T, r, sigma)


# ── Data fetch ────────────────────────────────────────────────────────────────
def fetch(ticker, name=None):
    name = name or ticker
    cache = f"cache/yf_{name}.parquet"
    os.makedirs("cache", exist_ok=True)
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    print(f"Fetching {name} ({ticker})...")
    df = yf.download(ticker, start=START, end=END, interval="1d", auto_adjust=True, progress=False, multi_level_index=False)
    if df.empty:
        print(f"  WARN: empty result for {ticker}")
        return df
    df.columns = [str(c).lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.to_parquet(cache)
    return df


# ── Event detection + backtest ────────────────────────────────────────────────
def compute_atr(daily, period=14):
    high = daily["high"]; low = daily["low"]
    pc = daily["close"].shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1, skipna=False)
    return tr.ewm(span=period, adjust=False).mean()


def detect_events(daily, threshold=0.75):
    d = daily.copy()
    d["atr"]        = compute_atr(d)
    d["prev_close"] = d["close"].shift(1)
    d["gap"]        = d["open"] - d["prev_close"]
    d["gap_ratio"]  = d["gap"] / d["atr"]
    d["direction"]  = np.where(d["gap_ratio"] >=  threshold, "gap_up",
                      np.where(d["gap_ratio"] <= -threshold, "gap_down", "none"))
    return d[d["direction"] != "none"].dropna(subset=["atr", "prev_close"]).copy()


def vix_regime(v):
    if v < 15: return "low"
    if v < 25: return "mid"
    return "high"


def backtest(daily, vix_series, symbol):
    """Apply R1 + M1 to one symbol. Returns per-event row + summary."""
    ev = detect_events(daily, threshold=0.75)
    if ev.empty:
        return pd.DataFrame()

    rows = []
    for dt, row in ev.iterrows():
        # VIX at previous business day
        prev_d = pd.Timestamp(dt).normalize() - pd.tseries.offsets.BDay(1)
        vix_val = float(vix_series.reindex([prev_d]).iloc[0]) if prev_d in vix_series.index else float(vix_series.iloc[(vix_series.index.get_indexer([prev_d], method="ffill")[0])])
        if np.isnan(vix_val): vix_val = 20.0
        sigma = max(vix_val/100.0, 0.05)

        atr     = float(row["atr"])
        width   = WIDTH_ATR * atr
        pc      = float(row["prev_close"])
        S_open  = float(row["open"])
        close_px = float(row["close"])
        low_d   = float(row["low"])
        high_d  = float(row["high"])
        direction = row["direction"]
        gap_ratio = float(row["gap_ratio"])
        abs_gap = abs(gap_ratio)
        regime = vix_regime(vix_val)

        # R1 candidate: mid VIX, gap_down, 0.75 <= |gap_ratio|
        r1_pnl = np.nan
        if regime == "mid" and direction == "gap_down":
            K_short = pc
            K_long  = pc + width
            credit  = spread_value(S_open, K_short, K_long, direction, T_FULL, sigma)
            if credit > 0 and width > 0:
                eod_intrinsic = min(max(close_px - K_short, 0.0), width)
                r1_pnl = credit - eod_intrinsic

        # M1 candidate: 1.0 <= |gap_ratio| < 1.5, gap_up, NO VIX filter
        m1_pnl = np.nan
        if direction == "gap_up" and 1.0 <= abs_gap < 1.5:
            K_short = S_open
            K_long  = S_open - width
            credit  = spread_value(S_open, K_short, K_long, direction, T_FULL, sigma)
            if credit > 0 and width > 0:
                # Detect SS75 trigger using daily low
                trigger_level = S_open - 0.75 * width
                if low_d <= trigger_level:
                    stop_val = spread_value(trigger_level, K_short, K_long, direction, T_STOP, sigma)
                    m1_pnl = credit - stop_val
                else:
                    eod_intrinsic = min(max(K_short - close_px, 0.0), width)
                    m1_pnl = credit - eod_intrinsic

        rows.append({
            "date": dt, "symbol": symbol, "direction": direction,
            "gap_ratio": gap_ratio, "atr": atr, "vix": vix_val, "vix_regime": regime,
            "open": S_open, "prev_close": pc, "high": high_d, "low": low_d, "close": close_px,
            "r1_pnl": r1_pnl, "m1_pnl": m1_pnl,
        })

    return pd.DataFrame(rows)


def summarise(df, rule_col, label, years=YEARS):
    sub = df[df[rule_col].notna()].copy()
    if sub.empty:
        return {"label": label, "n": 0}
    p = sub[rule_col].values
    return {
        "label": label,
        "n": len(sub),
        "n_yr": round(len(sub)/years, 1),
        "ev_per_trade": round(p.mean(), 3),
        "annual_ev": round(p.sum()/years, 2),
        "win_%": round((p > 0).mean()*100, 1),
        "median": round(np.median(p), 3),
        "p10": round(np.quantile(p, 0.10), 3),
        "worst": round(p.min(), 2),
    }


# ── Run ────────────────────────────────────────────────────────────────────────
TICKERS = {
    "SPY":    "SPY",
    "QQQ":    "QQQ",
    "IWM":    "IWM",
    "DIA":    "DIA",
    "Nikkei": "^N225",
    "DAX":    "^GDAXI",
}

print("Fetching VIX...")
vix = fetch("^VIX", "VIX")
if "close" not in vix.columns:
    vix.columns = [c.lower() for c in vix.columns]
vix_series = vix["close"]

print("\nFetching universe...")
data = {}
for label, ticker in TICKERS.items():
    df = fetch(ticker, label)
    if df.empty:
        continue
    data[label] = df
    print(f"  {label}: {len(df):,} daily bars  ({df.index[0].date()} → {df.index[-1].date()})")

# Truncate to common date range (2019-2026)
START_DT = pd.Timestamp(START)
for k in data:
    data[k] = data[k][data[k].index >= START_DT]

# Backtest each
all_events = []
for label, df in data.items():
    res = backtest(df, vix_series, label)
    res["instrument"] = label
    all_events.append(res)
combined = pd.concat(all_events, ignore_index=True)

# Per-instrument summary
print("\n" + "="*78)
print("R1 — mid-VIX × gap_down × bear-call spread (hold to EOD)")
print("="*78)
r1_rows = [summarise(combined[combined["instrument"]==k], "r1_pnl", k) for k in data.keys()]
print(pd.DataFrame(r1_rows).to_string(index=False))

print("\n" + "="*78)
print("M1 — T2 × gap_up × bull-put @ open (SS75 from daily low, else EOD)")
print("="*78)
m1_rows = [summarise(combined[combined["instrument"]==k], "m1_pnl", k) for k in data.keys()]
print(pd.DataFrame(m1_rows).to_string(index=False))


# ── Correlation: do these gap on the same days? ──────────────────────────────
def gap_days(df):
    return set(pd.Timestamp(d).date() for d in df["date"])

print("\n" + "="*78)
print("Gap-day overlap matrix (count of shared 0.75x events)")
print("="*78)
ginsts = list(data.keys())
overlap = pd.DataFrame(0, index=ginsts, columns=ginsts)
for a in ginsts:
    da = gap_days(combined[combined["instrument"]==a])
    for b in ginsts:
        db = gap_days(combined[combined["instrument"]==b])
        overlap.loc[a, b] = len(da & db)
print(overlap)

print("\nFraction of A's gap days also in B (rows=A, cols=B):")
frac = overlap.copy().astype(float)
for a in ginsts:
    da_n = len(gap_days(combined[combined["instrument"]==a]))
    for b in ginsts:
        frac.loc[a, b] = round(overlap.loc[a, b] / da_n, 2) if da_n > 0 else 0
print(frac)


# ── Combined portfolio: union of signals ──────────────────────────────────────
print("\n" + "="*78)
print("Combined portfolio — union of R1 + M1 signals across all instruments")
print("="*78)
r1_sum = summarise(combined, "r1_pnl", "R1 across all instruments")
m1_sum = summarise(combined, "m1_pnl", "M1 across all instruments")
print(pd.DataFrame([r1_sum, m1_sum]).to_string(index=False))

# Save event table
os.makedirs("cache", exist_ok=True)
combined.to_csv("cache/phase3_universe_events.csv", index=False)
print(f"\nSaved cache/phase3_universe_events.csv ({len(combined)} rows)")
