"""
Phase 2 / Task 8 — Stop loss at prev_close

Models Strategy A (OTM credit spread, K_short = prev_close, width = 0.5×ATR)
with a stop triggered when intraday price touches the short strike.

Compares four cells across ATR thresholds:
    (no stop / stop) × (PT60 / EOD)

Stop fill price approximated by Black-Scholes value of the spread at the
moment S = K_short, with configurable T_remaining (default 0.5/252 = mid-day).
"""

import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

from __future__ import annotations
import datetime, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm as N

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOLS    = ["ES", "NQ", "RTY", "YM"]
PATHS      = {s: f"data/{s}_1m_2019_2026.csv" for s in SYMBOLS}
THRESHOLDS = [0.50, 0.75, 1.00, 1.50, 2.00]
YEARS      = 7
WIDTH_ATR  = 0.5
PT_PCT     = 0.60
T_FULL     = 1 / 252        # full 1-DTE
T_STOP_DEFAULT = 0.5 / 252  # mid-day average stop


# ── Load ──────────────────────────────────────────────────────────────────────
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


def detect_gaps(daily: pd.DataFrame, symbol: str, threshold: float, atr_period: int = 14) -> pd.DataFrame:
    d = daily.copy()
    pc = d["close"].shift(1)
    tr = pd.concat([(d["high"] - d["low"]),
                    (d["high"] - pc).abs(),
                    (d["low"]  - pc).abs()], axis=1).max(axis=1)
    d["atr"]        = tr.ewm(span=atr_period, adjust=False).mean()
    d["prev_close"] = pc
    d["gap"]        = d["open"] - d["prev_close"]
    d["gap_ratio"]  = d["gap"] / d["atr"]
    d["direction"]  = np.where(d["gap_ratio"] >=  threshold, "gap_up",
                      np.where(d["gap_ratio"] <= -threshold, "gap_down", "none"))
    ev = d[d["direction"] != "none"].dropna(subset=["atr", "prev_close"]).copy()
    ev["symbol"] = symbol
    return ev


# ── BS pricing ────────────────────────────────────────────────────────────────
def bs_call(S, K, T, r, sigma):
    if T <= 1e-12 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N.cdf(d1) - K * np.exp(-r * T) * N.cdf(d2)


def bs_put(S, K, T, r, sigma):
    return bs_call(S, K, T, r, sigma) - S + K * np.exp(-r * T)


def spread_value(S, K_short, K_long, direction, T, sigma, r=0.05):
    """Net value (debit to close) of the credit spread at underlying price S."""
    if direction == "gap_up":
        return bs_put(S, K_short, T, r, sigma) - bs_put(S, K_long, T, r, sigma)
    return bs_call(S, K_short, T, r, sigma) - bs_call(S, K_long, T, r, sigma)


# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(daily_all, vix_series, threshold: float, t_stop: float) -> dict:
    rows = []
    for sym in SYMBOLS:
        ev = detect_gaps(daily_all[sym], sym, threshold=threshold)
        for dt, row in ev.iterrows():
            prev_d  = pd.Timestamp(dt).normalize().tz_localize(None) - pd.tseries.offsets.BDay(1)
            vix_val = float(vix_series.get(prev_d, 20.0))
            sigma   = max(vix_val / 100.0, 0.05)
            atr     = float(row["atr"])
            width   = WIDTH_ATR * atr
            pc      = float(row["prev_close"])
            direction = row["direction"]
            S_open  = float(row["open"])

            # Strikes
            K_short = pc
            K_long  = pc - width if direction == "gap_up" else pc + width

            # Credit collected at open
            credit_open = spread_value(S_open, K_short, K_long, direction, T_FULL, sigma)
            if credit_open <= 0 or width <= 0:
                continue

            # Intraday breach detection (daily low/high vs short strike)
            if direction == "gap_up":
                breached = float(row["low"]) <= K_short
            else:
                breached = float(row["high"]) >= K_short

            # Cost to close at K_short with T_remaining = t_stop
            stop_close_value = spread_value(K_short, K_short, K_long, direction, t_stop, sigma)
            stop_pnl = credit_open - stop_close_value

            # EOD intrinsic
            close_px = float(row["close"])
            if direction == "gap_up":
                eod_intrinsic = min(max(K_short - close_px, 0.0), width)
            else:
                eod_intrinsic = min(max(close_px - K_short, 0.0), width)
            eod_pnl = credit_open - eod_intrinsic

            # EOD safe: close on safe side of short strike
            eod_safe = (direction == "gap_up"   and close_px > pc) or \
                       (direction == "gap_down" and close_px < pc)

            rows.append({
                "symbol": sym, "direction": direction,
                "credit": credit_open, "width": width,
                "breached": breached, "eod_safe": eod_safe,
                "stop_pnl": stop_pnl, "eod_pnl": eod_pnl,
                "pt60_pnl": PT_PCT * credit_open,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return {"threshold": threshold, "n": 0}

    n = len(df)
    n_yr = n / YEARS
    breach_rate = df["breached"].mean()

    # Diagnostic: of breached trades, how many recovered to EOD safe?
    breached = df[df["breached"]]
    if len(breached) > 0:
        breach_recover_pct = breached["eod_safe"].mean() * 100
        breach_recover_eod_pnl = breached.loc[breached["eod_safe"], "eod_pnl"].mean() if breached["eod_safe"].any() else 0.0
        breach_lose_eod_pnl    = breached.loc[~breached["eod_safe"], "eod_pnl"].mean() if (~breached["eod_safe"]).any() else 0.0
    else:
        breach_recover_pct = 0.0
        breach_recover_eod_pnl = 0.0
        breach_lose_eod_pnl = 0.0

    # Cell 1: No stop, PT60 — win if eod_safe (PT60 pnl); loss = full max loss
    no_stop_pt60 = np.where(df["eod_safe"], PT_PCT * df["credit"], -(df["width"] - df["credit"]))
    # Cell 2: No stop, EOD — pnl = credit - eod_intrinsic for every trade
    no_stop_eod  = df["eod_pnl"].values
    # Cell 3: Stop, PT60 — if breached: stop_pnl; else: PT60 pnl
    stop_pt60    = np.where(df["breached"], df["stop_pnl"], PT_PCT * df["credit"])
    # Cell 4: Stop, EOD — if breached: stop_pnl; else: eod_pnl
    stop_eod     = np.where(df["breached"], df["stop_pnl"], df["eod_pnl"])

    def stats(pnl):
        return {
            "ev_trade": float(pnl.mean()),
            "annual_ev": float(pnl.mean()) * n_yr,
            "win_rate": float((pnl > 0).mean()) * 100,
            "worst": float(pnl.min()),
        }

    return {
        "threshold": threshold,
        "n_trades_total": n,
        "n_trades_yr": round(n_yr, 1),
        "avg_credit": float(df["credit"].mean()),
        "avg_width":  float(df["width"].mean()),
        "credit_over_width_pct": float((df["credit"] / df["width"]).mean()) * 100,
        "breach_rate_pct": float(breach_rate) * 100,
        "breach_recover_pct": float(breach_recover_pct),
        "breach_recover_eod_pnl": float(breach_recover_eod_pnl),
        "breach_lose_eod_pnl": float(breach_lose_eod_pnl),
        "no_stop_pt60": stats(no_stop_pt60),
        "no_stop_eod":  stats(no_stop_eod),
        "stop_pt60":    stats(stop_pt60),
        "stop_eod":     stats(stop_eod),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    daily_all = {s: load_daily(PATHS[s]) for s in SYMBOLS}
    vix = pd.read_csv("data/VIX_daily_2019_2026.csv", index_col=0, parse_dates=True)
    vix.index = pd.to_datetime(vix.index)
    vix_series = vix["close"]

    print("\nRunning backtest across thresholds (T_stop = 0.5/252)...")
    results = [run_backtest(daily_all, vix_series, t, T_STOP_DEFAULT) for t in THRESHOLDS]

    # Build summary table
    rows = []
    for r in results:
        if r.get("n_trades_total", 0) == 0:
            continue
        rows.append({
            "threshold": r["threshold"],
            "n_yr": r["n_trades_yr"],
            "breach_%": round(r["breach_rate_pct"], 1),
            "c/w_%":   round(r["credit_over_width_pct"], 1),
            "NS_PT60_ev": round(r["no_stop_pt60"]["ev_trade"], 3),
            "NS_PT60_yr": round(r["no_stop_pt60"]["annual_ev"], 2),
            "NS_EOD_ev":  round(r["no_stop_eod"]["ev_trade"], 3),
            "NS_EOD_yr":  round(r["no_stop_eod"]["annual_ev"], 2),
            "ST_PT60_ev": round(r["stop_pt60"]["ev_trade"], 3),
            "ST_PT60_yr": round(r["stop_pt60"]["annual_ev"], 2),
            "ST_EOD_ev":  round(r["stop_eod"]["ev_trade"], 3),
            "ST_EOD_yr":  round(r["stop_eod"]["annual_ev"], 2),
        })
    summary = pd.DataFrame(rows)
    print("\n=== EV per trade and annual EV (points, 1 contract, 1 symbol) ===")
    print("NS = no stop, ST = stop at prev_close")
    print(summary.to_string(index=False))

    # Diagnostic: breached-trade recovery profile
    diag_rows = []
    for r in results:
        if r.get("n_trades_total", 0) == 0:
            continue
        diag_rows.append({
            "threshold": r["threshold"],
            "breach_%": round(r["breach_rate_pct"], 1),
            "of_breached_recover_to_EOD_safe_%": round(r["breach_recover_pct"], 1),
            "avg_EOD_pnl_of_recoverers": round(r["breach_recover_eod_pnl"], 2),
            "avg_EOD_pnl_of_non_recoverers": round(r["breach_lose_eod_pnl"], 2),
            "NS_EOD_worst": round(r["no_stop_eod"]["worst"], 1),
            "ST_EOD_worst": round(r["stop_eod"]["worst"], 1),
            "ST_EOD_win_%": round(r["stop_eod"]["win_rate"], 1),
            "NS_EOD_win_%": round(r["no_stop_eod"]["win_rate"], 1),
        })
    diag = pd.DataFrame(diag_rows)
    print("\n=== Diagnostic: of the breached trades, do they recover by EOD? ===")
    print(diag.to_string(index=False))

    # Sensitivity on T_stop at 0.75×ATR (the headline threshold)
    print("\n=== T_stop sensitivity at 0.75×ATR (stop + EOD exit) ===")
    for t_stop_frac in [0.25, 0.50, 0.75, 1.00]:
        r = run_backtest(daily_all, vix_series, 0.75, t_stop_frac / 252)
        s = r["stop_eod"]
        print(f"  T_stop = {t_stop_frac:.2f}/252 → "
              f"EV/trade = {s['ev_trade']:+.3f}, annual = {s['annual_ev']:+.2f} pts, "
              f"win = {s['win_rate']:.1f}%, worst = {s['worst']:+.3f}")

    # Chart: annual EV per threshold for the 4 cells
    os.makedirs("charts", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    thr = summary["threshold"].values
    width_b = 0.18
    x = np.arange(len(thr))
    ax.bar(x - 1.5 * width_b, summary["NS_PT60_yr"], width_b, label="No stop · PT60", color="#ef476f", alpha=0.85)
    ax.bar(x - 0.5 * width_b, summary["NS_EOD_yr"],  width_b, label="No stop · EOD",  color="#f78c6b", alpha=0.85)
    ax.bar(x + 0.5 * width_b, summary["ST_PT60_yr"], width_b, label="Stop · PT60",    color="#06d6a0", alpha=0.85)
    ax.bar(x + 1.5 * width_b, summary["ST_EOD_yr"],  width_b, label="Stop · EOD",     color="#118ab2", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}×" for t in thr])
    ax.set_xlabel("ATR Threshold")
    ax.set_ylabel("Annual EV (points · 1 contract · 1 symbol)")
    ax.set_title("Strategy A — Stop-Loss at prev_close vs No Stop\n(T_stop = 0.5/252)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = "charts/p2_01_stop_loss_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved: {out}")


if __name__ == "__main__":
    main()
