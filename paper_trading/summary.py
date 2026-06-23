"""Summarize paper-trading performance to date.

Prints overall and per-rule stats: realized P&L, win rate, actual vs model EV.
"""
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

import pandas as pd

df = pd.read_csv("paper_trading/trades.csv")
if df.empty:
    print("No trades logged yet.")
    raise SystemExit

print(f"\n=== Paper-trading summary  ({len(df)} trades) ===")
print(f"Date range: {df['date'].min()} → {df['date'].max()}")
print()

closed = df[df["status"] == "closed"].copy()
open_  = df[df["status"] == "open"]

if not closed.empty:
    for col in ["net_pnl_usd", "model_ev_usd", "deviation_vs_model"]:
        closed[col] = pd.to_numeric(closed[col], errors="coerce")

    print(f"Closed: {len(closed)} | Open: {len(open_)}")
    print()
    print(f"Total realized PnL:  ${closed['net_pnl_usd'].sum():+,.0f}")
    print(f"Model EV (cumulated):${closed['model_ev_usd'].sum():+,.0f}")
    print(f"Deviation:           ${closed['deviation_vs_model'].sum():+,.0f}")
    print(f"Win rate:            {(closed['net_pnl_usd'] > 0).mean()*100:.1f}%")
    print()

    print("--- By rule ---")
    for rule, sub in closed.groupby("rule"):
        wins = (sub["net_pnl_usd"] > 0).mean() * 100
        print(f"{rule:>15}: n={len(sub):>3}  realized=${sub['net_pnl_usd'].sum():+,.0f}"
              f"  model=${sub['model_ev_usd'].sum():+,.0f}"
              f"  win={wins:.0f}%  worst=${sub['net_pnl_usd'].min():+,.0f}")

    print()
    print("--- Recent trades ---")
    cols_short = ["trade_id","date","symbol","rule","contracts","entry_credit_total_usd",
                  "net_pnl_usd","model_ev_usd","deviation_vs_model","status"]
    print(closed[cols_short].tail(10).to_string(index=False))

if not open_.empty:
    print()
    print("--- Open positions ---")
    print(open_[["trade_id","date","symbol","rule","k_short","k_long","contracts",
                "entry_credit_total_usd","model_ev_usd"]].to_string(index=False))
