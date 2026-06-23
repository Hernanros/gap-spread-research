"""Morning automated signal scan — runs end-to-end without paid data.

What it does:
  1. Pulls TODAY'S OPEN for ES, NQ, RTY, YM, NKD (yfinance — free).
  2. Pulls TODAY'S VIX (yfinance).
  3. Uses cached daily ATR + prev_close from local Databento data.
  4. Computes R1 / M1 candidates for each symbol.
  5. Logs every candidate to pricing_log.csv (status NOT_QUOTED).
  6. Prints a clean morning briefing: TAKE candidates + skip reasons.

Run this once per day at ~09:35 ET (5 min after US open).

Then manually pull IBKR quotes for any TAKE candidates and re-run
decision_today.py to get the final TAKE/SKIP after seeing real market credit.
"""
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

import argparse, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
import csv

# Map our internal symbol → yfinance ticker for live/recent quotes
YF_TICKER = {
    "ES":  "ES=F",   # E-mini S&P 500 futures
    "NQ":  "NQ=F",   # E-mini Nasdaq 100
    "RTY": "RTY=F",  # E-mini Russell 2000
    "YM":  "YM=F",   # E-mini Dow
    "NKD": "NKD=F",  # Nikkei 225 USD
}
WIDTH_ATR = 0.5
MULT = {"ES": 50, "NQ": 20, "RTY": 50, "YM": 5, "NKD": 5}
SYMBOLS = list(MULT.keys())
PRICING_LOG = "paper_trading/pricing_log.csv"

PRICING_FIELDS = [
    "date","symbol","rule","prev_close","today_open","gap_ratio","tier",
    "vix","regime","k_short","k_long","width_pts","width_usd",
    "model_credit_usd","market_credit_usd","credit_haircut_pct",
    "max_risk_usd","real_rr_ratio","breakeven_win_rate_pct",
    "market_implied_pop_pct","backtest_pop_pct",
    "decision","actual_pnl_usd","notes",
]


def latest_atr_prev_close(symbol):
    """Pull from yfinance (FREE — uses live daily data). Fallback to cached Databento."""
    try:
        import yfinance as yf
        tk = yf.Ticker(YF_TICKER[symbol])
        # 40 days of daily data — enough for ATR(14) with warmup
        d = tk.history(period="60d", interval="1d", auto_adjust=False)
        if d.empty or len(d) < 20:
            raise RuntimeError("yfinance returned insufficient data")
        d = d.rename(columns={c: c.lower() for c in d.columns})
        h, l, pc = d["high"], d["low"], d["close"].shift(1)
        tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        d["atr14"] = tr.ewm(span=14, adjust=False).mean()
        prev_close = float(d["close"].iloc[-1])
        atr        = float(d["atr14"].iloc[-1])
        prev_date  = d.index[-1].date()
        return atr, prev_close, prev_date
    except Exception as e:
        # Fallback: cached Databento data (may be stale)
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
        print(f"  WARNING: yfinance failed for {symbol} ({e}); using cached Databento (may be stale)")
        return float(daily["atr14"].iloc[-1]), float(daily["close"].iloc[-1]), daily.index[-1].date()


def fetch_today_open(yf_ticker):
    """Pull today's regular-hours open (or most recent quote) from yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        # Get today's 1m bar; first bar after 09:30 ET is the open
        tk = yf.Ticker(yf_ticker)
        intraday = tk.history(period="1d", interval="1m", prepost=False, auto_adjust=False)
        if intraday.empty:
            return None
        # yfinance returns tz-aware Eastern; first regular-hours bar = open
        return float(intraday["Open"].iloc[0]), float(intraday["Close"].iloc[-1])
    except Exception:
        return None


def fetch_today_vix():
    try:
        import yfinance as yf
        tk = yf.Ticker("^VIX")
        d = tk.history(period="2d", interval="1d")
        if d.empty: return None
        return float(d["Close"].iloc[-1])
    except Exception:
        return None


def vix_regime(v):
    return "low" if v < 15 else ("mid" if v < 25 else "high")


def evaluate_symbol(symbol, today_open, vix):
    atr, prev_close, prev_date = latest_atr_prev_close(symbol)
    mult = MULT[symbol]
    width = WIDTH_ATR * atr
    gap_pts = today_open - prev_close
    gap_ratio = gap_pts / atr
    direction = "gap_up" if gap_ratio > 0 else "gap_down"
    abs_g = abs(gap_ratio)
    tier = "T1" if abs_g < 1.0 else ("T2" if abs_g < 1.5 else "T3")
    regime = vix_regime(vix)

    rule = None; k_short = None; k_long = None
    reasons = []

    # R1 check
    if regime == "mid" and direction == "gap_down" and abs_g >= 0.75:
        rule = "R1"
        k_short = prev_close
        k_long  = prev_close + width
    # M1 check
    elif direction == "gap_up" and 1.0 <= abs_g < 1.5:
        rule = "M1"
        k_short = today_open
        k_long  = today_open - width
    else:
        if regime != "mid": reasons.append(f"VIX {regime}")
        if abs_g < 0.75:   reasons.append(f"|gap|={abs_g:.2f} below 0.75")
        if direction == "gap_up" and abs_g < 1.0: reasons.append(f"|gap|={abs_g:.2f} below M1 T2 (1.0)")
        if direction == "gap_up" and abs_g >= 1.5: reasons.append(f"|gap|={abs_g:.2f} above M1 T2 (1.5)")

    return {
        "symbol": symbol, "rule": rule, "fires": rule is not None,
        "prev_close": prev_close, "today_open": today_open,
        "atr": atr, "prev_date": prev_date,
        "vix": vix, "regime": regime,
        "gap_pts": gap_pts, "gap_ratio": gap_ratio, "tier": tier, "direction": direction,
        "k_short": k_short, "k_long": k_long, "width": width, "width_usd": width * mult,
        "max_risk_usd": width * mult,
        "reasons": reasons,
    }


def log_candidate(row):
    csv_path = Path(PRICING_LOG)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PRICING_FIELDS, extrasaction="ignore")
        if write_header: w.writeheader()
        w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=SYMBOLS,
                   help="Symbols to scan (default: ES NQ RTY YM NKD)")
    p.add_argument("--manual_open", default=None,
                   help="Override format SYMBOL=PRICE, comma-separated. e.g. ES=7436.25,NQ=29887.5")
    p.add_argument("--manual_prev_close", default=None,
                   help="Override RTH prev_close (yfinance daily uses Globex close which differs ~$10-100)")
    p.add_argument("--manual_atr", default=None,
                   help="Override ATR per symbol, comma-separated. e.g. ES=114.5,NQ=750")
    p.add_argument("--manual_vix", type=float, default=None)
    p.add_argument("--no_log", action="store_true", help="Print only, don't log to pricing_log.csv")
    args = p.parse_args()

    print(f"\n{'='*72}\n  MORNING SCAN  ·  {dt.date.today().isoformat()}\n{'='*72}")

    # Resolve manual overrides
    def parse_kv(s):
        out = {}
        if not s: return out
        for pair in s.split(","):
            k, v = pair.split("=")
            out[k.strip()] = float(v.strip())
        return out
    manual = parse_kv(args.manual_open)
    manual_pc  = parse_kv(args.manual_prev_close)
    manual_atr = parse_kv(args.manual_atr)

    vix = args.manual_vix if args.manual_vix is not None else fetch_today_vix()
    if vix is None:
        print("  WARNING: VIX could not be fetched. Use --manual_vix to override.")
        return
    print(f"  VIX (live):            {vix:.2f}  [{vix_regime(vix)}]\n")

    results = []
    for sym in args.symbols:
        # Get today's open
        if sym in manual:
            today_open = manual[sym]
            src = "manual"
        else:
            yfp = fetch_today_open(YF_TICKER[sym])
            if yfp is None:
                print(f"  {sym}: could not fetch today's open from yfinance; use --manual_open {sym}=<price>")
                continue
            today_open, _ = yfp
            src = "yfinance"

        r = evaluate_symbol(sym, today_open, vix)
        # Apply manual overrides for prev_close / atr if provided (recompute gap)
        overridden = False
        if sym in manual_pc:
            r["prev_close"] = manual_pc[sym]
            overridden = True
        if sym in manual_atr:
            r["atr"] = manual_atr[sym]
            overridden = True
        if overridden:
            r["width"]    = WIDTH_ATR * r["atr"]
            r["width_usd"] = r["width"] * MULT[sym]
            r["max_risk_usd"] = r["width_usd"]
            r["gap_pts"]  = r["today_open"] - r["prev_close"]
            r["gap_ratio"] = r["gap_pts"] / r["atr"]
            abs_g = abs(r["gap_ratio"])
            r["direction"] = "gap_up" if r["gap_ratio"] > 0 else "gap_down"
            r["tier"] = "T1" if abs_g < 1.0 else ("T2" if abs_g < 1.5 else "T3")
            # Re-check rule
            r["fires"] = False; r["rule"] = None; r["reasons"] = []
            if r["regime"] == "mid" and r["direction"] == "gap_down" and abs_g >= 0.75:
                r["rule"] = "R1"; r["fires"] = True
                r["k_short"] = r["prev_close"]
                r["k_long"]  = r["prev_close"] + r["width"]
            elif r["direction"] == "gap_up" and 1.0 <= abs_g < 1.5:
                r["rule"] = "M1"; r["fires"] = True
                r["k_short"] = r["today_open"]
                r["k_long"]  = r["today_open"] - r["width"]
        r["src"] = src
        results.append(r)

    # ──  Print table ──
    print(f"  {'SYM':<5} {'PrevClose':>10} {'Open':>10} {'Gap pts':>10} {'GapR':>7} {'Tier':>5} {'Rule':>5}   {'Decision':<40}")
    print(f"  {'-'*100}")
    for r in results:
        rule_str = r["rule"] or "—"
        if r["fires"]:
            verdict = f"FIRES → K_short={r['k_short']:.0f} K_long={r['k_long']:.0f}"
        else:
            verdict = "no signal (" + "; ".join(r["reasons"]) + ")"
        print(f"  {r['symbol']:<5} {r['prev_close']:>10.2f} {r['today_open']:>10.2f} "
              f"{r['gap_pts']:>+10.2f} {r['gap_ratio']:>+7.2f} {r['tier']:>5} {rule_str:>5}   {verdict:<40}")

    # ──  Detail for firing signals ──
    print()
    fires = [r for r in results if r["fires"]]
    if not fires:
        print("  No R1 or M1 signals fire today. Nothing to do.")
    else:
        print(f"\n{'─'*72}\n  {len(fires)} CANDIDATE TRADE(S) — needs IBKR quote before TAKE\n{'─'*72}")
        for r in fires:
            print(f"\n  ▸ {r['symbol']} {r['rule']}  ({r['tier']} {r['direction']})")
            print(f"    Sell call/put at: {r['k_short']:.0f}")
            print(f"    Buy  call/put at: {r['k_long']:.0f}")
            print(f"    Width: {r['width']:.0f} pts = ${r['width_usd']:.0f} max risk per contract")
            print(f"    NEXT: pull IBKR chain at these strikes (today's expiry) and run:")
            mult = MULT[r["symbol"]]
            print(f"        python paper_trading/decision_today.py \\")
            print(f"            --symbol {r['symbol']} --open {r['today_open']} \\")
            print(f"            --prev_close {r['prev_close']} --atr {r['atr']:.2f} --vix {vix} \\")
            print(f"            --rule {r['rule']} \\")
            print(f"            --market_credit_usd <USD>  --market_pop_pct <PCT>")

    # ──  Log all candidates (whether firing or not) ──
    if not args.no_log:
        for r in fires:
            row = {
                "date": dt.date.today().isoformat(),
                "symbol": r["symbol"], "rule": r["rule"],
                "prev_close": round(r["prev_close"], 2),
                "today_open": round(r["today_open"], 2),
                "gap_ratio": round(r["gap_ratio"], 3),
                "tier": r["tier"], "vix": round(r["vix"], 2), "regime": r["regime"],
                "k_short": round(r["k_short"], 2), "k_long": round(r["k_long"], 2),
                "width_pts": round(r["width"], 2), "width_usd": round(r["width_usd"], 0),
                "model_credit_usd": "", "market_credit_usd": "",
                "credit_haircut_pct": "",
                "max_risk_usd": round(r["max_risk_usd"], 0),
                "real_rr_ratio": "", "breakeven_win_rate_pct": "",
                "market_implied_pop_pct": "", "backtest_pop_pct": "",
                "decision": "NOT_QUOTED", "actual_pnl_usd": "",
                "notes": f"Morning scan auto-detected; awaiting IBKR quote",
            }
            log_candidate(row)
        if fires:
            print(f"\n  Logged {len(fires)} candidate(s) to {PRICING_LOG} as NOT_QUOTED.")


if __name__ == "__main__":
    main()
