"""Evening check — close out any open paper trades and reconcile vs model.

Run after the 16:00 ET close. For each NOT_QUOTED / open entry in
pricing_log.csv or trades.csv it:
  - Pulls today's close from yfinance
  - Computes the EOD outcome (win/loss for the spread)
  - Prompts for the actual market exit fill (or uses intrinsic if not provided)
  - Updates the log

Free data only.
"""
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

import argparse, datetime as dt
from pathlib import Path
import pandas as pd

YF_TICKER = {"ES": "ES=F", "NQ": "NQ=F", "RTY": "RTY=F", "YM": "YM=F", "NKD": "NKD=F"}
MULT = {"ES": 50, "NQ": 20, "RTY": 50, "YM": 5, "NKD": 5}


def fetch_today_close(sym):
    try:
        import yfinance as yf
        tk = yf.Ticker(YF_TICKER[sym])
        intraday = tk.history(period="1d", interval="1m", prepost=False)
        if intraday.empty: return None
        return float(intraday["Close"].iloc[-1])
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=dt.date.today().isoformat())
    args = p.parse_args()

    print(f"\n{'='*72}\n  EVENING CHECK  ·  {args.date}\n{'='*72}\n")

    # Check pricing_log for NOT_QUOTED rows from today
    df = pd.read_csv("paper_trading/pricing_log.csv")
    today = df[df["date"] == args.date]
    print(f"  Pricing log entries for {args.date}: {len(today)}")
    if today.empty:
        print("  No candidates logged today.")
    else:
        print(f"  {'sym':<5} {'rule':<5} {'k_short':>8} {'k_long':>8} {'decision':<25}")
        for _, r in today.iterrows():
            print(f"  {r['symbol']:<5} {r['rule']:<5} {r['k_short']:>8} {r['k_long']:>8} {str(r['decision']):<25}")

    # Check trades.csv for open positions
    trades = pd.read_csv("paper_trading/trades.csv")
    open_pos = trades[trades["status"] == "open"]
    if open_pos.empty:
        print("\n  No open positions in trades.csv.")
    else:
        print(f"\n  {len(open_pos)} OPEN POSITION(S) — fetch close + record exits:\n")
        for _, t in open_pos.iterrows():
            sym = t["symbol"]
            close_px = fetch_today_close(sym)
            print(f"  Trade #{t['trade_id']}  {sym} {t['rule']}  "
                  f"K_short={t['k_short']}  K_long={t['k_long']}")
            if close_px is not None:
                k_short = float(t["k_short"])
                if t["direction"] == "gap_down":
                    intrinsic = max(close_px - k_short, 0)
                else:
                    intrinsic = max(k_short - close_px, 0)
                intrinsic = min(intrinsic, float(t["width_pts"]))
                print(f"    Today's close ({sym}): {close_px:.2f}")
                print(f"    Spread intrinsic at close: {intrinsic:.2f} pts")
                print(f"    → record exit: python paper_trading/record_exit.py "
                      f"--trade_id {t['trade_id']} --exit_price {intrinsic:.2f}")
            else:
                print(f"    yfinance fetch failed — pull close manually")

    print()


if __name__ == "__main__":
    main()
