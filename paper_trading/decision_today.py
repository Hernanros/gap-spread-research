"""Live decision layer — Phase 4.

For a candidate trade, computes:
  - R1 / M1 candidate strikes
  - Max risk per contract
  - Asks for the IBKR-quoted market credit
  - Computes real R:R, breakeven win rate
  - Compares to backtest expectations
  - Applies TAKE / SKIP / TAKE_MODIFIED thresholds
  - Logs every candidate to pricing_log.csv (whether taken or not)

Usage (interactive):
    python paper_trading/decision_today.py --symbol ES --open 7436.25

Usage (non-interactive, with market credit known):
    python paper_trading/decision_today.py --symbol NQ --open 29887.5 \\
        --rule R1 --market_credit_usd 158 --market_pop_pct 94

Thresholds (configurable):
    --max_rr 8.0
    --min_pop 88.0
"""
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

import argparse, csv, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm

WIDTH_ATR = 0.5
T_FULL = 1 / 252
RISK_FREE = 0.05
MULT = {"ES": 50, "NQ": 20, "RTY": 50, "YM": 5, "NKD": 5}
PRICING_LOG = "paper_trading/pricing_log.csv"

PRICING_FIELDS = [
    "date", "symbol", "rule", "prev_close", "today_open", "gap_ratio", "tier",
    "vix", "regime", "k_short", "k_long", "width_pts", "width_usd",
    "model_credit_usd", "market_credit_usd", "credit_haircut_pct",
    "max_risk_usd", "real_rr_ratio", "breakeven_win_rate_pct",
    "market_implied_pop_pct", "backtest_pop_pct",
    "decision", "actual_pnl_usd", "notes",
]

# Backtest win rates (used as a sanity comparison vs market-implied PoP)
BACKTEST_POP = {
    ("ES",  "R1"): 94, ("NQ",  "R1"): 94, ("RTY", "R1"): 95,
    ("YM",  "R1"): 100, ("NKD", "R1"): 100,
    ("ES",  "M1"): 50, ("NQ",  "M1"): 71, ("RTY", "M1"): 50,
    ("YM",  "M1"): 54, ("NKD", "M1"): 67,
}


def bs_call(S, K, T, sigma, r=RISK_FREE):
    if T <= 1e-12 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, sigma, r=RISK_FREE):
    return bs_call(S, K, T, sigma, r) - S + K * np.exp(-r * T)


def spread_credit_model(S_open, K_short, K_long, direction, sigma):
    """BS-priced credit (the model number, which we know is optimistic)."""
    if direction == "gap_up":
        return bs_put(S_open, K_short, T_FULL, sigma) - bs_put(S_open, K_long, T_FULL, sigma)
    return bs_call(S_open, K_short, T_FULL, sigma) - bs_call(S_open, K_long, T_FULL, sigma)


def latest_atr(symbol):
    df = pd.read_csv(f"data/{symbol}_1m_2019_2026.csv", parse_dates=["ts_event"]).set_index("ts_event")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    t = df.index.time
    sess = df[(t >= dt.time(9, 30)) & (t <= dt.time(16, 0))]
    daily = sess.resample("1D").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),     close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()
    daily = daily[daily["volume"] > 0]
    h, l, pc = daily["high"], daily["low"], daily["close"].shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    daily["atr14"] = tr.ewm(span=14, adjust=False).mean()
    return float(daily["atr14"].iloc[-1]), float(daily["close"].iloc[-1])


def vix_regime(v):
    return "low" if v < 15 else ("mid" if v < 25 else "high")


def analyze(symbol, today_open, prev_close, atr, vix, rule, market_credit_usd=None):
    mult = MULT[symbol]
    sigma = max(vix / 100, 0.05)
    width = WIDTH_ATR * atr
    width_usd = width * mult
    gap_pts = today_open - prev_close
    gap_ratio = gap_pts / atr
    direction = "gap_up" if gap_ratio > 0 else "gap_down"
    abs_g = abs(gap_ratio)
    tier = "T1" if abs_g < 1 else ("T2" if abs_g < 1.5 else "T3")
    regime = vix_regime(vix)

    # Determine strike anchors
    if rule == "R1":
        if direction != "gap_down":
            raise SystemExit(f"R1 only applies to gap_down. This event is {direction}.")
        k_short = prev_close
        k_long  = prev_close + width
    elif rule == "M1":
        if direction != "gap_up":
            raise SystemExit(f"M1 only applies to gap_up. This event is {direction}.")
        k_short = today_open
        k_long  = today_open - width
    else:
        raise SystemExit(f"Unknown rule: {rule}")

    # Model credit (BS-priced, what backtest assumed)
    model_credit_pts = spread_credit_model(today_open, k_short, k_long, direction, sigma)
    model_credit_usd = max(model_credit_pts, 0) * mult

    out = {
        "date": dt.date.today().isoformat(), "symbol": symbol, "rule": rule,
        "prev_close": prev_close, "today_open": today_open,
        "gap_ratio": round(gap_ratio, 3), "tier": tier,
        "vix": vix, "regime": regime, "direction": direction,
        "k_short": round(k_short, 2), "k_long": round(k_long, 2),
        "width_pts": round(width, 2), "width_usd": round(width_usd, 0),
        "model_credit_usd": round(model_credit_usd, 0),
        "market_credit_usd": market_credit_usd,
        "mult": mult,
    }

    if market_credit_usd is not None:
        out["credit_haircut_pct"] = round(100 * market_credit_usd / model_credit_usd, 1) if model_credit_usd > 0 else None
        out["max_risk_usd"] = round(width_usd - market_credit_usd, 0)
        out["real_rr_ratio"] = round(out["max_risk_usd"] / market_credit_usd, 1)
        out["breakeven_win_rate_pct"] = round(100 * out["max_risk_usd"] / (out["max_risk_usd"] + market_credit_usd), 1)

    return out


def decision(a, market_pop_pct, max_rr, min_pop):
    """Apply TAKE/SKIP thresholds."""
    rr = a.get("real_rr_ratio")
    if rr is None:
        return "NEEDS_MARKET_QUOTE"
    if rr > max_rr:
        return f"SKIP (R:R {rr:.1f}:1 > {max_rr}:1 threshold)"
    if market_pop_pct is not None and market_pop_pct < min_pop:
        return f"SKIP (market PoP {market_pop_pct}% < {min_pop}%)"
    if a["breakeven_win_rate_pct"] > a.get("backtest_pop_pct", 100):
        return f"SKIP (breakeven WR {a['breakeven_win_rate_pct']}% > backtest WR {a['backtest_pop_pct']}%)"
    return "TAKE"


def log_to_csv(row, path=PRICING_LOG):
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PRICING_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True, choices=list(MULT.keys()))
    p.add_argument("--open", type=float, required=True, dest="today_open",
                   help="Today's open (or hypothetical open)")
    p.add_argument("--prev_close", type=float, default=None,
                   help="Override prev_close (otherwise pulled from latest data)")
    p.add_argument("--vix", type=float, default=None,
                   help="Override VIX (otherwise pulled from latest data)")
    p.add_argument("--atr", type=float, default=None,
                   help="Override ATR (otherwise computed)")
    p.add_argument("--rule", choices=["R1", "M1", "AUTO"], default="AUTO",
                   help="Rule to evaluate (AUTO picks by gap direction)")
    p.add_argument("--market_credit_usd", type=float, default=None,
                   help="Market-quoted net credit in USD (per contract). If omitted, prints model-only and prompts.")
    p.add_argument("--market_pop_pct", type=float, default=None,
                   help="Market-implied probability of profit % (from IBKR or similar)")
    p.add_argument("--max_rr", type=float, default=8.0)
    p.add_argument("--min_pop", type=float, default=88.0)
    p.add_argument("--notes", default="")
    args = p.parse_args()

    # Get state
    atr_auto, prev_close_auto = latest_atr(args.symbol)
    atr        = args.atr or atr_auto
    prev_close = args.prev_close or prev_close_auto
    if args.vix is not None:
        vix = args.vix
    else:
        v = pd.read_csv("data/VIX_daily_2019_2026.csv", index_col=0, parse_dates=True)
        vix = float(v["close"].iloc[-1])

    # Determine rule
    gap_ratio = (args.today_open - prev_close) / atr
    if args.rule == "AUTO":
        rule = "R1" if gap_ratio < 0 else "M1"
    else:
        rule = args.rule

    a = analyze(args.symbol, args.today_open, prev_close, atr, vix, rule, args.market_credit_usd)
    a["backtest_pop_pct"] = BACKTEST_POP.get((args.symbol, rule), None)
    a["market_implied_pop_pct"] = args.market_pop_pct

    print("\n" + "=" * 70)
    print(f"  {args.symbol}  ·  rule {rule}  ·  {a['direction']}  ·  tier {a['tier']}")
    print("=" * 70)
    print(f"  prev_close  : {prev_close:.2f}")
    print(f"  today_open  : {args.today_open:.2f}")
    print(f"  gap_ratio   : {a['gap_ratio']:+.3f} ATR")
    print(f"  VIX / regime: {vix:.2f} / {a['regime']}")
    print()
    print(f"  Structure   : {'bear call' if a['direction']=='gap_down' else 'bull put'} spread")
    print(f"  K_short     : {a['k_short']:.2f}")
    print(f"  K_long      : {a['k_long']:.2f}")
    print(f"  Width       : {a['width_pts']:.2f} pts = ${a['width_usd']:.0f}")
    print()
    print(f"  Model credit (BS): ${a['model_credit_usd']:.0f}")

    if args.market_credit_usd is not None:
        print(f"  Market credit    : ${args.market_credit_usd:.0f}")
        print(f"  Credit haircut   : {a['credit_haircut_pct']}% of model")
        print(f"  Max risk         : ${a['max_risk_usd']:.0f}")
        print(f"  Real R:R         : {a['real_rr_ratio']}:1")
        print(f"  Breakeven WR     : {a['breakeven_win_rate_pct']}%")
        if args.market_pop_pct is not None:
            print(f"  Market PoP       : {args.market_pop_pct}%")
        print(f"  Backtest WR      : {a['backtest_pop_pct']}%")
        verdict = decision(a, args.market_pop_pct, args.max_rr, args.min_pop)
        print(f"\n  DECISION         : {verdict}")
        a["decision"] = verdict.split()[0]  # TAKE or SKIP
        a["notes"] = args.notes
        log_to_csv(a)
        print(f"\n  Logged to {PRICING_LOG}")
    else:
        print(f"\n  >> Pull IBKR chain for {a['k_short']:.0f}/{a['k_long']:.0f} {args.symbol} expiring today.")
        print(f"  >> Re-run with --market_credit_usd <usd> --market_pop_pct <pct> to get a decision.")
        a["decision"] = "NOT_QUOTED"
        a["notes"] = args.notes or "Awaiting market quote"
        log_to_csv(a)
        print(f"\n  Logged as NOT_QUOTED to {PRICING_LOG}")


if __name__ == "__main__":
    main()
