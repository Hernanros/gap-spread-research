"""Record an exit for an open paper trade.

Usage:
    python paper_trading/record_exit.py --trade_id 1 --exit_price 0.8
    # or close at full max-loss if stop hit:
    python paper_trading/record_exit.py --trade_id 1 --exit_price max_loss

Computes gross PnL = (credit - exit_debit) × contracts × multiplier.
Subtracts 1.5 pt slippage round-trip (per the backtest convention).
Updates trades.csv in place, sets status=closed.
"""
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

import argparse, csv, datetime as dt
from pathlib import Path

WIDTH_ATR = 0.5
MULT = {"ES": 50, "NQ": 20, "RTY": 50, "YM": 5, "NKD": 5}
SLIPPAGE_PTS = {"ES": 1.5, "NQ": 1.5, "RTY": 1.5, "YM": 1.5, "NKD": 2.0}
TRADES_CSV = "paper_trading/trades.csv"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trade_id", type=int, required=True)
    p.add_argument("--exit_price", required=True,
                   help="Spread debit-to-close per contract, OR 'max_loss' if stopped/expired ITM")
    p.add_argument("--exit_time_et", default=None)
    p.add_argument("--notes", default=None, help="Append to existing notes")
    args = p.parse_args()

    rows = []
    found = False
    with Path(TRADES_CSV).open() as f:
        r = csv.DictReader(f)
        fields = r.fieldnames
        for row in r:
            if int(row["trade_id"]) == args.trade_id:
                found = True
                sym = row["symbol"]
                mult = MULT[sym]
                slip_pts = SLIPPAGE_PTS[sym]
                contracts = int(row["contracts"])
                credit = float(row["entry_credit_per_contract"])
                width = float(row["width_pts"])

                if args.exit_price == "max_loss":
                    exit_price = width  # spread closes worst-case at full width
                else:
                    exit_price = float(args.exit_price)

                gross_pnl_per_contract = credit - exit_price
                gross_pnl_total = gross_pnl_per_contract * contracts * mult
                slippage_total = slip_pts * contracts * mult
                net_pnl = gross_pnl_total - slippage_total
                model_ev = float(row.get("model_ev_usd") or 0)
                deviation = net_pnl - model_ev

                row["exit_price_per_contract"] = round(exit_price, 2)
                row["exit_time_et"] = args.exit_time_et or (dt.datetime.utcnow().strftime("%H:%M") + " UTC")
                row["gross_pnl_usd"] = round(gross_pnl_total, 2)
                row["slippage_usd"] = round(slippage_total, 2)
                row["net_pnl_usd"] = round(net_pnl, 2)
                row["deviation_vs_model"] = round(deviation, 2)
                row["status"] = "closed"
                if args.notes:
                    row["notes"] = (row["notes"] + " | " + args.notes).strip(" |")

                print(f"Trade #{args.trade_id} closed.")
                print(f"  Gross PnL: ${gross_pnl_total:+.0f}")
                print(f"  Slippage:  ${slippage_total:.0f}")
                print(f"  Net PnL:   ${net_pnl:+.0f}")
                print(f"  Model EV:  ${model_ev:+.0f}")
                print(f"  Deviation: ${deviation:+.0f}")
            rows.append(row)

    if not found:
        raise SystemExit(f"Trade #{args.trade_id} not found in {TRADES_CSV}")

    with Path(TRADES_CSV).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows: w.writerow(r)


if __name__ == "__main__":
    main()
