"""Record a paper-trade entry.

Usage (R1 example):
    python paper_trading/record_entry.py \\
        --symbol ES --rule R1 \\
        --vix 16.4 --atr 114.5 \\
        --prev_close 7556.25 --today_open 7450.46 \\
        --contracts 1 --credit 3.5 \\
        --notes "R1 fired, mid-VIX, gap-down 0.92 ATR"

The script computes the rest (k_short, k_long, gap_ratio, tier, direction, model_ev_usd)
and appends a row to paper_trading/trades.csv with status=open.
"""
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

import argparse, csv, datetime as dt
from pathlib import Path

WIDTH_ATR = 0.5
MULT = {"ES": 50, "NQ": 20, "RTY": 50, "YM": 5, "NKD": 5}

# Model EV per trade (from backtest, after slippage). Used to compare actual vs expected.
MODEL_EV_USD_PER_CONTRACT = {
    ("ES",  "R1"): 489,   # per the T1 R1 cell (mid-VIX × gap_down)
    ("NQ",  "R1"): 178,
    ("RTY", "R1"): 13,
    ("YM",  "R1"): 30,
    ("NKD", "R1"): 78,
    ("ES",  "M1"): 220,
    ("NQ",  "M1"): 606,
    ("RTY", "M1"): 41,
    ("YM",  "M1"): 167,
    ("NKD", "M1"): 203,
}
TRADES_CSV = "paper_trading/trades.csv"

FIELDS = ["trade_id","date","symbol","rule","vix","atr","prev_close","today_open",
          "gap_pts","gap_ratio","direction","tier","structure",
          "k_short","k_long","width_pts",
          "contracts","entry_credit_per_contract","entry_credit_total_usd","entry_time_et",
          "exit_price_per_contract","exit_time_et","gross_pnl_usd","slippage_usd","net_pnl_usd",
          "model_ev_usd","deviation_vs_model","status","notes"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True, choices=list(MULT.keys()))
    p.add_argument("--rule", required=True, choices=["R1", "M1", "discretionary"])
    p.add_argument("--vix", type=float, required=True)
    p.add_argument("--atr", type=float, required=True)
    p.add_argument("--prev_close", type=float, required=True)
    p.add_argument("--today_open", type=float, required=True)
    p.add_argument("--contracts", type=int, default=1)
    p.add_argument("--credit", type=float, required=True,
                   help="Credit collected per contract in instrument points")
    p.add_argument("--entry_time_et", default=None, help="HH:MM ET (default: now)")
    p.add_argument("--notes", default="")
    args = p.parse_args()

    mult = MULT[args.symbol]
    width = WIDTH_ATR * args.atr
    gap_pts = args.today_open - args.prev_close
    gap_ratio = gap_pts / args.atr
    direction = "gap_up" if gap_ratio > 0 else "gap_down"
    abs_g = abs(gap_ratio)
    tier = "T1" if abs_g < 1.0 else ("T2" if abs_g < 1.5 else "T3")

    if args.rule == "R1":
        structure = "bear call spread"
        k_short = args.prev_close
        k_long  = args.prev_close + width
    elif args.rule == "M1":
        structure = "bull put spread at open"
        k_short = args.today_open
        k_long  = args.today_open - width
    else:
        structure = "discretionary"
        k_short = None
        k_long  = None

    entry_credit_total = args.credit * args.contracts * mult
    model_ev = MODEL_EV_USD_PER_CONTRACT.get((args.symbol, args.rule), 0) * args.contracts
    entry_time_et = args.entry_time_et or dt.datetime.utcnow().strftime("%H:%M") + " UTC"

    # Read existing trades to get next id
    csv_path = Path(TRADES_CSV)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    next_id = 1
    rows_existing = []
    if csv_path.exists() and csv_path.stat().st_size > len(",".join(FIELDS)):
        with csv_path.open() as f:
            r = csv.DictReader(f)
            rows_existing = list(r)
            if rows_existing:
                next_id = max(int(row["trade_id"]) for row in rows_existing if row["trade_id"]) + 1

    row = {
        "trade_id": next_id,
        "date": dt.date.today().isoformat(),
        "symbol": args.symbol,
        "rule": args.rule,
        "vix": round(args.vix, 2),
        "atr": round(args.atr, 2),
        "prev_close": round(args.prev_close, 2),
        "today_open": round(args.today_open, 2),
        "gap_pts": round(gap_pts, 2),
        "gap_ratio": round(gap_ratio, 3),
        "direction": direction,
        "tier": tier,
        "structure": structure,
        "k_short": round(k_short, 2) if k_short else "",
        "k_long":  round(k_long, 2)  if k_long  else "",
        "width_pts": round(width, 2),
        "contracts": args.contracts,
        "entry_credit_per_contract": round(args.credit, 2),
        "entry_credit_total_usd": round(entry_credit_total, 2),
        "entry_time_et": entry_time_et,
        "exit_price_per_contract": "",
        "exit_time_et": "",
        "gross_pnl_usd": "",
        "slippage_usd": "",
        "net_pnl_usd": "",
        "model_ev_usd": round(model_ev, 2),
        "deviation_vs_model": "",
        "status": "open",
        "notes": args.notes,
    }

    # Write back full table
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows_existing: w.writerow(r)
        w.writerow(row)

    print(f"Trade #{next_id} recorded (status=open).")
    print(f"  {args.symbol} {args.rule}: sell {k_short:.2f} / buy {k_long:.2f}, "
          f"{args.contracts} contract(s), credit ${entry_credit_total:.0f}")
    print(f"  Model EV expectation: ${model_ev:.0f}")
    print(f"  Run record_exit.py --trade_id {next_id} --exit_price X.XX to close.")


if __name__ == "__main__":
    main()
