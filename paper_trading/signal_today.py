"""Detect today's R1 / M1 signal based on the latest data and a user-provided open price.

Usage:
    python paper_trading/signal_today.py --symbol ES --open 7450.46
    python paper_trading/signal_today.py --symbol ES --gap_pct -1.4

Prints structured output for each rule (FIRES / NO) with strike levels + model EV.
"""
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

import argparse, datetime as dt
import numpy as np
import pandas as pd

WIDTH_ATR = 0.5
MULTIPLIERS = {"ES": 50, "NQ": 20, "RTY": 50, "YM": 5, "NKD": 5}


def load_symbol_state(symbol: str):
    """Return prev_close, atr14, vix_close from the most recent fully-closed session."""
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

    prev_close = float(daily["close"].iloc[-1])
    atr14      = float(daily["atr14"].iloc[-1])
    prev_date  = daily.index[-1].date()

    vix = pd.read_csv("data/VIX_daily_2019_2026.csv", index_col=0, parse_dates=True)
    vix_close  = float(vix["close"].iloc[-1])
    vix_date   = vix.index[-1].date()
    return {"prev_close": prev_close, "atr14": atr14, "prev_date": prev_date,
            "vix_close": vix_close, "vix_date": vix_date}


def vix_regime(v):
    return "low" if v < 15 else ("mid" if v < 25 else "high")


def evaluate(symbol: str, open_px: float | None, gap_pct: float | None):
    s = load_symbol_state(symbol)
    pc = s["prev_close"]
    atr = s["atr14"]
    vix = s["vix_close"]
    regime = vix_regime(vix)
    mult = MULTIPLIERS[symbol]

    if open_px is None:
        if gap_pct is None:
            raise SystemExit("Provide either --open or --gap_pct")
        open_px = pc * (1 + gap_pct / 100)

    gap_pts = open_px - pc
    gap_ratio = gap_pts / atr
    direction = "gap_up" if gap_ratio > 0 else "gap_down"
    abs_g = abs(gap_ratio)
    tier = "T1" if abs_g < 1.0 else ("T2" if abs_g < 1.5 else "T3")

    print(f"\n{'='*68}")
    print(f"  Signal check for {symbol}   |   {dt.date.today()}")
    print(f"{'='*68}")
    print(f"  Prev close (data ts {s['prev_date']}):  {pc:.2f}")
    print(f"  ATR(14):                                 {atr:.2f}")
    print(f"  VIX (close {s['vix_date']}):             {vix:.2f}   [{regime}]")
    print(f"  Today's open / hypothetical:            {open_px:.2f}")
    print(f"  Gap:                                     {gap_pts:+.2f} pts   ({gap_ratio:+.2f} ATR)")
    print(f"  Tier:                                    {tier}    Direction: {direction}")

    width = WIDTH_ATR * atr

    # R1 check
    print(f"\n  R1 (mid-VIX × gap_DOWN, bear-call spread):")
    if regime == "mid" and direction == "gap_down" and abs_g >= 0.75:
        k_short = pc
        k_long  = pc + width
        max_risk = (width * mult)
        print(f"    >>> R1 FIRES <<<")
        print(f"    Structure:     bear call spread, 1-DTE")
        print(f"    K_short (sell):  {k_short:.2f}")
        print(f"    K_long  (buy):   {k_long:.2f}")
        print(f"    Width:           {width:.2f} pts = ${max_risk:.0f} max risk / contract")
        print(f"    Win condition:   close BELOW {k_short:.2f} at 16:00 ET")
        print(f"    Hist EV (T1):    +$489/trade, 94% win, worst 7yr -$20")
        print(f"    Entry time:      09:30 ET (or symbol's local open)")
    else:
        reasons = []
        if regime != "mid": reasons.append(f"VIX {regime}, need mid")
        if direction != "gap_down": reasons.append(f"direction {direction}, need gap_down")
        if abs_g < 0.75: reasons.append(f"|gap|={abs_g:.2f}, need >= 0.75 ATR")
        print(f"    NO. Reasons: {'; '.join(reasons)}")

    # M1 check
    print(f"\n  M1 (T2 × gap_UP, bull-put @ open, SS75 stop):")
    if direction == "gap_up" and 1.0 <= abs_g < 1.5:
        k_short = open_px
        k_long  = open_px - width
        max_risk = (width * mult)
        print(f"    >>> M1 FIRES <<<")
        print(f"    Structure:     bull put spread anchored at today's open, 1-DTE")
        print(f"    K_short (sell):  {k_short:.2f}  (=today's open)")
        print(f"    K_long  (buy):   {k_long:.2f}")
        print(f"    Width:           {width:.2f} pts = ${max_risk:.0f} max risk / contract")
        print(f"    Stop (SS75):     close if price falls to {open_px - 0.75*width:.2f}")
        print(f"    Else exit:       16:00 ET")
        print(f"    Hist EV:         +$575/trade, 56% win rate after SS75")
    else:
        reasons = []
        if direction != "gap_up": reasons.append(f"direction {direction}, need gap_up")
        if not (1.0 <= abs_g < 1.5): reasons.append(f"|gap|={abs_g:.2f}, need 1.0 <= |g| < 1.5 (T2)")
        print(f"    NO. Reasons: {'; '.join(reasons)}")

    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="ES", choices=list(MULTIPLIERS.keys()))
    p.add_argument("--open", type=float, default=None, help="Today's open price")
    p.add_argument("--gap_pct", type=float, default=None, help="Gap %% (e.g., -1.4 for 1.4%% down)")
    args = p.parse_args()
    evaluate(args.symbol, args.open, args.gap_pct)


if __name__ == "__main__":
    main()
