"""
Phase 3 robustness — walk-forward + slippage haircut.

Addresses the two real holes flagged by external critique:
  1. No walk-forward / out-of-sample validation
  2. Worst-trade headline doesn't include realistic slippage

Walk-forward split:
  in-sample (IS):  2019-01-01 → 2023-12-31   (5.0 years)
  hold-out (OOS):  2024-01-01 → 2026-06-22   (2.5 years)

Slippage assumptions:
  ES/NQ/RTY/YM:  1.5 pts round-trip per spread (≈ $30–75 per trade)
  NKD:           2.0 pts round-trip per spread (≈ $10 per trade, less liquid)
"""
from __future__ import annotations
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

import numpy as np
import pandas as pd

YEARS_IS  = 5.0
YEARS_OOS = 2.5
SLIPPAGE_PTS = {"ES": 1.5, "NQ": 1.5, "RTY": 1.5, "YM": 1.5, "NKD": 2.0}
MULT = {"ES": 50, "NQ": 20, "RTY": 50, "YM": 5, "NKD": 5}
IS_END = pd.Timestamp("2024-01-01")


def header(s):
    print("\n" + "=" * 80); print(s); print("=" * 80)


# Load the cached event table from phase3 NKD validation
events = pd.read_csv("cache/phase3_nkd_validate.csv", parse_dates=["date"])
events["date"] = pd.to_datetime(events["date"])
events["slippage_pts"] = events["symbol"].map(SLIPPAGE_PTS)
events["mult"] = events["symbol"].map(MULT)

# Apply slippage haircut (subtract from positive pnl, add to negative — i.e. always worse)
# Slippage is a friction cost — subtract from gross pnl regardless of sign
events["r1_pnl_net"] = events["r1_pnl"] - events["slippage_pts"]
events["m1_pnl_net"] = events["m1_pnl"] - events["slippage_pts"]

# Sample period
period = pd.Series(
    np.where(events["date"] < IS_END, "in_sample", "out_of_sample"),
    index=events.index
)
events["period"] = period


def summarise(df, pnl_col, period_label, years):
    sub = df[df[pnl_col].notna()].copy()
    if sub.empty:
        return {"period": period_label, "n": 0}
    p = sub[pnl_col].values
    dollar_p = (sub[pnl_col] * sub["mult"]).values
    return {
        "period": period_label,
        "n": len(sub),
        "n_yr": round(len(sub) / years, 1),
        "ev_pts": round(p.mean(), 2),
        "annual_ev_pts": round(p.sum() / years, 1),
        "annual_ev_usd": round(dollar_p.sum() / years, 0),
        "win_%": round((p > 0).mean() * 100, 1),
        "p10_pts": round(np.quantile(p, 0.10), 2),
        "worst_pts": round(p.min(), 2),
        "worst_usd": round(dollar_p.min(), 0),
    }


# ── 1. R1 walk-forward (no slippage) ──────────────────────────────────────────
header("R1 — mid-VIX × gap_down — walk-forward (model PnL, no slippage)")
is_r1 = events[events["period"] == "in_sample"]
oos_r1 = events[events["period"] == "out_of_sample"]
rows = [summarise(is_r1, "r1_pnl", "IS 2019-2023 (5y)", YEARS_IS),
        summarise(oos_r1, "r1_pnl", "OOS 2024-2026 (2.5y)", YEARS_OOS)]
print(pd.DataFrame(rows).to_string(index=False))


# ── 2. R1 walk-forward WITH slippage ──────────────────────────────────────────
header("R1 — same split, AFTER 1.5–2.0 pt slippage per trade")
rows = [summarise(is_r1,  "r1_pnl_net", "IS  + slippage", YEARS_IS),
        summarise(oos_r1, "r1_pnl_net", "OOS + slippage", YEARS_OOS)]
print(pd.DataFrame(rows).to_string(index=False))


# ── 3. M1 walk-forward (no slippage) ──────────────────────────────────────────
header("M1 — T2 × gap_up — walk-forward (model PnL, no slippage)")
rows = [summarise(is_r1,  "m1_pnl", "IS 2019-2023 (5y)", YEARS_IS),
        summarise(oos_r1, "m1_pnl", "OOS 2024-2026 (2.5y)", YEARS_OOS)]
print(pd.DataFrame(rows).to_string(index=False))


# ── 4. M1 walk-forward WITH slippage ──────────────────────────────────────────
header("M1 — same split, AFTER slippage")
rows = [summarise(is_r1,  "m1_pnl_net", "IS  + slippage", YEARS_IS),
        summarise(oos_r1, "m1_pnl_net", "OOS + slippage", YEARS_OOS)]
print(pd.DataFrame(rows).to_string(index=False))


# ── 5. Per-symbol OOS sanity check ────────────────────────────────────────────
header("R1 — out-of-sample, per symbol (with slippage)")
rows = []
for sym in ["ES", "NQ", "RTY", "YM", "NKD"]:
    sub = oos_r1[(oos_r1["symbol"]==sym) & (oos_r1["r1_pnl_net"].notna())]
    if sub.empty: continue
    p = sub["r1_pnl_net"].values
    d = (sub["r1_pnl_net"] * sub["mult"]).values
    rows.append({
        "symbol": sym, "n": len(sub), "n_yr": round(len(sub)/YEARS_OOS, 1),
        "ev_usd": round(d.mean(), 0),
        "annual_usd": round(d.sum()/YEARS_OOS, 0),
        "win_%": round((p > 0).mean()*100, 1),
        "worst_usd": round(d.min(), 0),
    })
print(pd.DataFrame(rows).to_string(index=False))

header("M1 — out-of-sample, per symbol (with slippage)")
rows = []
for sym in ["ES", "NQ", "RTY", "YM", "NKD"]:
    sub = oos_r1[(oos_r1["symbol"]==sym) & (oos_r1["m1_pnl_net"].notna())]
    if sub.empty: continue
    p = sub["m1_pnl_net"].values
    d = (sub["m1_pnl_net"] * sub["mult"]).values
    rows.append({
        "symbol": sym, "n": len(sub), "n_yr": round(len(sub)/YEARS_OOS, 1),
        "ev_usd": round(d.mean(), 0),
        "annual_usd": round(d.sum()/YEARS_OOS, 0),
        "win_%": round((p > 0).mean()*100, 1),
        "worst_usd": round(d.min(), 0),
    })
print(pd.DataFrame(rows).to_string(index=False))


# ── 6. Headline summary for the brief ─────────────────────────────────────────
header("HEADLINE COMPARISON: in-sample vs out-of-sample (USD, with slippage)")
rule_rows = []
for rule_col, rule_name in [("r1_pnl_net", "R1 (gap_down)"), ("m1_pnl_net", "M1 (gap_up)")]:
    is_sub  = events[(events["period"]=="in_sample") & (events[rule_col].notna())]
    oos_sub = events[(events["period"]=="out_of_sample") & (events[rule_col].notna())]
    is_d  = (is_sub[rule_col] * is_sub["mult"]).values if len(is_sub) else np.array([0])
    oos_d = (oos_sub[rule_col] * oos_sub["mult"]).values if len(oos_sub) else np.array([0])
    rule_rows.append({
        "rule": rule_name,
        "IS_n_yr": round(len(is_sub) / YEARS_IS, 1),
        "IS_ann_usd": round(is_d.sum() / YEARS_IS, 0),
        "IS_win_%": round((is_d > 0).mean()*100, 1),
        "IS_worst_usd": round(is_d.min(), 0),
        "OOS_n_yr": round(len(oos_sub) / YEARS_OOS, 1),
        "OOS_ann_usd": round(oos_d.sum() / YEARS_OOS, 0),
        "OOS_win_%": round((oos_d > 0).mean()*100, 1),
        "OOS_worst_usd": round(oos_d.min(), 0),
    })
print(pd.DataFrame(rule_rows).to_string(index=False))

# Save for the notebook
out_df = pd.DataFrame(rule_rows)
out_df.to_csv("cache/phase3_walkforward.csv", index=False)
print("\nSaved cache/phase3_walkforward.csv")
