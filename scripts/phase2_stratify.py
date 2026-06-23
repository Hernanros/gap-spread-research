"""
Phase 2 / Task 9 — VIX regime × trend alignment × direction stratification

Goal: find a sub-population of 0.75×ATR gap events where NS·EOD policy gives:
  - positive EV per trade
  - worst-case trade better than -150 pts
  - at least ~20 events/yr

Variables:
  VIX regime:  low (<15), mid (15-25), high (>25) — measured at prev close
  Alignment:   concurrent  = gap direction matches 30-day trend
               incongruent = gap direction against 30-day trend
  Direction:   gap_up / gap_down
"""

import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

from __future__ import annotations
import datetime, os
import numpy as np
import pandas as pd
from scipy.stats import norm as N

SYMBOLS    = ["ES", "NQ", "RTY", "YM"]
PATHS      = {s: f"data/{s}_1m_2019_2026.csv" for s in SYMBOLS}
THRESHOLD  = 0.75
YEARS      = 7
WIDTH_ATR  = 0.5
T_FULL     = 1 / 252
TREND_LOOKBACK = 30
VIX_REGIMES = {"low": (0, 15), "mid": (15, 25), "high": (25, 999)}


# ── Load + detect + price ─────────────────────────────────────────────────────
def load_daily(path: str) -> pd.DataFrame:
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


def detect_gaps_with_alignment(daily, symbol, threshold, lookback):
    d = daily.copy()
    pc = d["close"].shift(1)
    tr = pd.concat([(d["high"] - d["low"]),
                    (d["high"] - pc).abs(),
                    (d["low"]  - pc).abs()], axis=1).max(axis=1)
    d["atr"]        = tr.ewm(span=14, adjust=False).mean()
    d["prev_close"] = pc
    d["gap_ratio"]  = (d["open"] - pc) / d["atr"]
    d["direction"]  = np.where(d["gap_ratio"] >=  threshold, "gap_up",
                      np.where(d["gap_ratio"] <= -threshold, "gap_down", "none"))
    d["close_Nd"]   = d["close"].shift(lookback)
    d["trend_30d"]  = np.where(d["close"] > d["close_Nd"], "bullish", "bearish")
    d["alignment"]  = np.where(
        ((d["direction"] == "gap_up")   & (d["trend_30d"] == "bullish")) |
        ((d["direction"] == "gap_down") & (d["trend_30d"] == "bearish")),
        "concurrent",
        np.where(
            ((d["direction"] == "gap_up")   & (d["trend_30d"] == "bearish")) |
            ((d["direction"] == "gap_down") & (d["trend_30d"] == "bullish")),
            "incongruent", "none"))
    ev = d[d["direction"] != "none"].dropna(subset=["atr", "prev_close", "close_Nd"]).copy()
    ev["symbol"] = symbol
    return ev


def bs_call(S, K, T, r, sigma):
    if T <= 1e-12 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N.cdf(d1) - K * np.exp(-r * T) * N.cdf(d2)


def bs_put(S, K, T, r, sigma):
    return bs_call(S, K, T, r, sigma) - S + K * np.exp(-r * T)


def spread_value(S, K_short, K_long, direction, T, sigma, r=0.05):
    if direction == "gap_up":
        return bs_put(S, K_short, T, r, sigma) - bs_put(S, K_long, T, r, sigma)
    return bs_call(S, K_short, T, r, sigma) - bs_call(S, K_long, T, r, sigma)


def vix_regime(v: float) -> str:
    for label, (lo, hi) in VIX_REGIMES.items():
        if lo <= v < hi:
            return label
    return "unknown"


# ── Build event table ─────────────────────────────────────────────────────────
def build_events(daily_all, vix_series):
    rows = []
    for sym in SYMBOLS:
        ev = detect_gaps_with_alignment(daily_all[sym], sym, THRESHOLD, TREND_LOOKBACK)
        for dt, row in ev.iterrows():
            prev_d  = pd.Timestamp(dt).normalize().tz_localize(None) - pd.tseries.offsets.BDay(1)
            vix_val = float(vix_series.get(prev_d, 20.0))
            sigma   = max(vix_val / 100.0, 0.05)
            atr     = float(row["atr"])
            width   = WIDTH_ATR * atr
            pc      = float(row["prev_close"])
            direction = row["direction"]
            S_open  = float(row["open"])

            K_short = pc
            K_long  = pc - width if direction == "gap_up" else pc + width

            credit = spread_value(S_open, K_short, K_long, direction, T_FULL, sigma)
            if credit <= 0 or width <= 0:
                continue

            close_px = float(row["close"])
            if direction == "gap_up":
                eod_intrinsic = min(max(K_short - close_px, 0.0), width)
            else:
                eod_intrinsic = min(max(close_px - K_short, 0.0), width)
            eod_pnl = credit - eod_intrinsic
            eod_safe = (direction == "gap_up"   and close_px > pc) or \
                       (direction == "gap_down" and close_px < pc)

            rows.append({
                "date": pd.Timestamp(dt).normalize(),
                "symbol": sym,
                "direction": direction,
                "alignment": row["alignment"],
                "vix": vix_val,
                "vix_regime": vix_regime(vix_val),
                "gap_ratio": float(row["gap_ratio"]),
                "atr": atr,
                "credit": credit,
                "width": width,
                "eod_pnl": eod_pnl,
                "eod_safe": eod_safe,
            })
    return pd.DataFrame(rows)


def stratify(df, group_cols):
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(group_cols, observed=True)
    out = pd.DataFrame({
        "n":            g.size(),
        "n_yr":         (g.size() / YEARS).round(1),
        "ev_per_trade": g["eod_pnl"].mean().round(2),
        "annual_ev":    (g["eod_pnl"].sum() / YEARS).round(1),
        "win_%":        (g["eod_safe"].mean() * 100).round(1),
        "median":       g["eod_pnl"].median().round(2),
        "p10":          g["eod_pnl"].quantile(0.10).round(2),
        "p25":          g["eod_pnl"].quantile(0.25).round(2),
        "worst":        g["eod_pnl"].min().round(1),
        "avg_credit":   g["credit"].mean().round(2),
    })
    return out.reset_index()


def main():
    print("Loading data...")
    daily_all = {s: load_daily(PATHS[s]) for s in SYMBOLS}
    vix = pd.read_csv("data/VIX_daily_2019_2026.csv", index_col=0, parse_dates=True)
    vix.index = pd.to_datetime(vix.index)
    vix_series = vix["close"]

    print(f"\nBuilding event table at {THRESHOLD}×ATR threshold...")
    df = build_events(daily_all, vix_series)
    print(f"  {len(df)} events / {len(df)/YEARS:.1f} per year across {len(SYMBOLS)} symbols")
    print(f"  VIX regime breakdown:   {dict(df['vix_regime'].value_counts())}")
    print(f"  Direction breakdown:    {dict(df['direction'].value_counts())}")
    print(f"  Alignment breakdown:    {dict(df['alignment'].value_counts())}")

    print("\n=== A · By direction (baseline) ===")
    print(stratify(df, ["direction"]).to_string(index=False))

    print("\n=== B · By VIX regime ===")
    print(stratify(df, ["vix_regime"]).to_string(index=False))

    print("\n=== C · By trend alignment ===")
    print(stratify(df, ["alignment"]).to_string(index=False))

    print("\n=== D · VIX × direction ===")
    print(stratify(df, ["vix_regime", "direction"]).to_string(index=False))

    print("\n=== E · Alignment × direction ===")
    print(stratify(df, ["alignment", "direction"]).to_string(index=False))

    print("\n=== F · VIX × alignment ===")
    print(stratify(df, ["vix_regime", "alignment"]).to_string(index=False))

    print("\n=== G · VIX × alignment × direction (full 3D) ===")
    print(stratify(df, ["vix_regime", "alignment", "direction"]).to_string(index=False))

    # Save for notebook reuse
    out_csv = "cache/phase2_events_075.csv"
    os.makedirs("cache", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nEvent table saved: {out_csv}  ({len(df)} rows)")

    # Find subsets meeting criteria
    print("\n=== H · Subsets meeting goal (ev>0, worst > -150, n_yr >= 5) ===")
    candidates = []
    for grouping in (["vix_regime", "direction"],
                     ["alignment", "direction"],
                     ["vix_regime", "alignment"],
                     ["vix_regime", "alignment", "direction"]):
        s = stratify(df, grouping)
        ok = s[(s["ev_per_trade"] > 0) & (s["worst"] > -150) & (s["n_yr"] >= 5)].copy()
        if len(ok) > 0:
            ok["grouping"] = "×".join(grouping)
            candidates.append(ok)
    if candidates:
        cand_df = pd.concat(candidates, ignore_index=True)
        print(cand_df.to_string(index=False))
    else:
        print("  No subset meets all three criteria. Relax thresholds or look at union of cells.")


if __name__ == "__main__":
    main()
