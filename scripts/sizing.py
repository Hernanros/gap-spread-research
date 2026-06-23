"""Compute yield at $2000 risk-per-contract sizing across all R1/M1 cells."""
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

import numpy as np
import pandas as pd

YEARS = 7
MULT = {"ES": 50, "NQ": 20, "RTY": 50, "YM": 5, "NKD": 5}
WIDTH_ATR = 0.5
BUDGET_USD = 2000.0  # max loss willing to take per single contract

events = pd.read_csv("cache/phase3_nkd_validate.csv")

rows = []
for sym in ["ES", "NQ", "RTY", "YM", "NKD"]:
    mult = MULT[sym]

    # R1 cell
    r1 = events[(events["symbol"]==sym) & (events["r1_pnl"].notna())].copy()
    if not r1.empty:
        # Theoretical max loss per spread = (width - credit) * mult, per trade
        # We can derive width from atr: width = 0.5 * ATR
        # Max loss in pts = width - credit (per trade); take avg and also worst-case
        r1["width_pts"] = WIDTH_ATR * r1["atr"]
        r1["maxloss_pts"] = r1["width_pts"] - r1["r1_credit"]
        avg_maxloss_dollars = (r1["maxloss_pts"] * mult).mean()
        p99_maxloss_dollars = (r1["maxloss_pts"] * mult).quantile(0.99)
        worst_observed_dollars = (r1["r1_pnl"].min()) * mult  # could be 0 or small
        ann_ev_per_contract = (r1["r1_pnl"].sum() / YEARS) * mult

        # Sizing: contracts = budget / theoretical max loss per spread
        contracts_at_avg = BUDGET_USD / max(avg_maxloss_dollars, 1)
        contracts_at_p99 = BUDGET_USD / max(p99_maxloss_dollars, 1)

        rows.append({
            "rule": "R1", "symbol": sym, "mult": mult,
            "n_yr": round(len(r1)/YEARS, 1),
            "avg_maxloss_$": round(avg_maxloss_dollars, 0),
            "p99_maxloss_$": round(p99_maxloss_dollars, 0),
            "worst_obs_$": round(worst_observed_dollars, 0),
            "ann_ev_per_1ct_$": round(ann_ev_per_contract, 0),
            "contracts@avg_maxloss": round(contracts_at_avg, 1),
            "contracts@p99_maxloss": round(contracts_at_p99, 1),
            "yearly_$_@avg": round(ann_ev_per_contract * contracts_at_avg, 0),
            "yearly_$_@p99": round(ann_ev_per_contract * contracts_at_p99, 0),
        })

    # M1 cell
    m1 = events[(events["symbol"]==sym) & (events["m1_pnl"].notna())].copy()
    if not m1.empty:
        m1["width_pts"] = WIDTH_ATR * m1["atr"]
        m1["maxloss_pts"] = m1["width_pts"] - m1["m1_credit"]
        avg_maxloss_dollars = (m1["maxloss_pts"] * mult).mean()
        p99_maxloss_dollars = (m1["maxloss_pts"] * mult).quantile(0.99)
        worst_observed_dollars = m1["m1_pnl"].min() * mult
        ann_ev_per_contract = (m1["m1_pnl"].sum() / YEARS) * mult

        contracts_at_avg = BUDGET_USD / max(avg_maxloss_dollars, 1)
        contracts_at_p99 = BUDGET_USD / max(p99_maxloss_dollars, 1)

        rows.append({
            "rule": "M1", "symbol": sym, "mult": mult,
            "n_yr": round(len(m1)/YEARS, 1),
            "avg_maxloss_$": round(avg_maxloss_dollars, 0),
            "p99_maxloss_$": round(p99_maxloss_dollars, 0),
            "worst_obs_$": round(worst_observed_dollars, 0),
            "ann_ev_per_1ct_$": round(ann_ev_per_contract, 0),
            "contracts@avg_maxloss": round(contracts_at_avg, 1),
            "contracts@p99_maxloss": round(contracts_at_p99, 1),
            "yearly_$_@avg": round(ann_ev_per_contract * contracts_at_avg, 0),
            "yearly_$_@p99": round(ann_ev_per_contract * contracts_at_p99, 0),
        })

df = pd.DataFrame(rows)
print("="*100)
print(f"Sizing analysis: $2000 max-loss budget per contract slot")
print("="*100)
print(df.to_string(index=False))

print(f"\nTotal yearly yield (avg-maxloss sizing): ${df['yearly_$_@avg'].sum():,.0f}")
print(f"Total yearly yield (p99-maxloss sizing, conservative): ${df['yearly_$_@p99'].sum():,.0f}")

# 7-year cumulative
print(f"\n7-year cumulative (avg-maxloss): ${df['yearly_$_@avg'].sum() * 7:,.0f}")
print(f"7-year cumulative (p99-maxloss): ${df['yearly_$_@p99'].sum() * 7:,.0f}")
