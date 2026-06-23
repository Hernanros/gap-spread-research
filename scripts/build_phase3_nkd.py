"""Build 10_phase3_nkd.ipynb — Phase 3 universe expansion + NKD validation."""
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s, hide=False):
    cell = new_code_cell(s)
    if hide:
        cell.metadata["jupyter"] = {"source_hidden": True}
        cell.metadata["tags"] = ["hide-input"]
    cells.append(cell)


md("""\
# Phase 3 — Universe Expansion + Nikkei (NKD) Validation

**Phase 2 left us with two rules** (R1 and M1) backtested on ES · NQ · RTY · YM.
Each of those four futures gaps on the same macro days — so adding more US instruments
is **not diversification**, it's just leverage on the same signal.

**Phase 3 question:** Is there a *genuinely uncorrelated* gap signal we can add?

This notebook checks three things:
1. Do the rules transfer to ETFs (SPY/QQQ/IWM)? (sanity check)
2. Does Databento actually carry the international futures we need? (cost + availability)
3. Do R1/M1 work on Nikkei 225 USD futures (NKD)? (the real test)
""")

# ── Setup ─────────────────────────────────────────────────────────────────────
md("## 1 · Setup")

code("""\
import warnings, os
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.style.use("dark_background")

YEARS = 7
MULT  = {"ES": 50, "NQ": 20, "RTY": 50, "YM": 5, "NKD": 5}
print("Setup ready.")
""", hide=True)

# ── Section 2: ETF sanity check ───────────────────────────────────────────────
md("""\
## 2 · ETF Sanity Check — Same Edge, Different Wrapper

Before paying for new data, sanity-check on **SPY / QQQ / IWM / DIA** via yfinance.
If R1/M1 don't survive the transfer to the ETF version of the same signal,
something is broken in the futures setup.

**They survive.** Win rates 88–100%, positive EV in every cell, smaller per-contract dollar size.

**But they don't diversify.** Gap-day overlap with the futures we already trade:

| | ES | QQQ | IWM | DIA |
|---|---:|---:|---:|---:|
| SPY | — | 0.70 | 0.78 | 0.80 |
| QQQ | 0.72 | — | 0.62 | 0.61 |

70–80% overlap. These ETFs gap on the same macro events as the futures.
Adding them is just smaller-contract-size leverage of the existing edge.
**They're a sanity check, not an expansion.**
""")

# ── Section 3: international data availability ────────────────────────────────
md("""\
## 3 · Available International Diversifiers — Databento Cost Audit

We want gaps driven by *different* news — Asian or European session events that don't
already show up in the US futures gap.

Cost estimates pulled from `databento.metadata.get_cost()` before any data fetch:

| Symbol | Description | 1-day | 1-hour | **1-min** | Status |
|---|---|---:|---:|---:|---|
| NKD | Nikkei 225 USD futures (CME) | $0.02 | $0.44 | **$6.26** | ✓ goes back to 2010 |
| NIY | Nikkei 225 yen-denom (CME) | $0.01 | $0.15 | $2.83 | ✓ available |
| FDAX | DAX futures (Eurex) | — | — | — | ✗ **dataset starts 2025-03-10** |
| GC | Gold (CME) | $0.02 | $0.16 | $0.33 | ✓ different driver |
| CL | Crude (CME) | $0.02 | $0.43 | $9.01 | ✓ different driver |

**The DAX problem:** Databento's `XEUR.EOBI` only goes back 15 months — way too short
for a 7-year backtest. FDAX is dead from this plan.

**Decision:** Pull NKD 1-minute. Cost: **$6.26**. Same intraday methodology as ES/NQ/RTY/YM.
""")

# ── Section 4: NKD validation ─────────────────────────────────────────────────
md("""\
## 4 · NKD Validation — Apply R1 and M1

Pulled 530K rows of 1-minute NKD OHLCV from Databento (`GLBX.MDP3` schema `ohlcv-1m`,
continuous front-month). Applied the exact same pipeline as the US futures:
- Detect gaps at 0.75×ATR threshold
- Same intraday minute-level spread valuation
- VIX as σ proxy (caveat: JNIV would be more accurate)

### Per-symbol comparison (dollar EV per single contract per year)
""")

code("""\
events = pd.read_csv("cache/phase3_nkd_validate.csv")

rows = []
for sym in ["ES", "NQ", "RTY", "YM", "NKD"]:
    mult = MULT[sym]
    r1 = events[(events["symbol"]==sym) & (events["r1_pnl"].notna())]
    m1 = events[(events["symbol"]==sym) & (events["m1_pnl"].notna())]
    rows.append({
        "symbol": sym, "mult_$": mult,
        "R1_n_yr":     round(len(r1)/YEARS, 1),
        "R1_ann_$":    round((r1["r1_pnl"].sum()/YEARS)*mult, 0),
        "R1_worst_$":  round(r1["r1_pnl"].min()*mult, 0) if len(r1) else 0,
        "R1_win_%":    round((r1["r1_pnl"] > 0).mean()*100, 1) if len(r1) else 0,
        "M1_n_yr":     round(len(m1)/YEARS, 1),
        "M1_ann_$":    round((m1["m1_pnl"].sum()/YEARS)*mult, 0),
        "M1_worst_$":  round(m1["m1_pnl"].min()*mult, 0) if len(m1) else 0,
        "M1_win_%":    round((m1["m1_pnl"] > 0).mean()*100, 1) if len(m1) else 0,
        "total_ann_$": round((r1["r1_pnl"].sum() + m1["m1_pnl"].sum())/YEARS*mult, 0),
    })
res = pd.DataFrame(rows)
print(res.to_string(index=False))
""")

md("""\
**NKD wins on every dimension:**
- 21.1/yr R1 events (vs ES 8.1) — three times the trade frequency
- $1,623/yr R1 EV — highest of any single symbol
- $3,449/yr M1 EV — also highest
- $5,072 combined — 47% of total cross-symbol portfolio

**100% R1 win rate caveat:** technically true, but ~10% of NKD R1 events have deep
gaps where the credit collected is near-zero (Black-Scholes prices both spread legs
into the OTM noise). The "win rate" includes these near-zero "trades." The meaningful
EV comes from the shallower-gap subset.
""")

# ── Section 5: diversification proof ──────────────────────────────────────────
md("## 5 · The Diversification Proof — Gap-Day Overlap")

code("""\
SYMBOLS = ["ES", "NQ", "RTY", "YM", "NKD"]
ev = events.copy()
ev["date"] = pd.to_datetime(ev["date"])

def gap_days(s):
    return set(ev[ev["symbol"]==s]["date"].dt.date.unique())

n = len(SYMBOLS)
ov = np.zeros((n, n))
frac = np.zeros((n, n))
for i, a in enumerate(SYMBOLS):
    da = gap_days(a)
    for j, b in enumerate(SYMBOLS):
        db = gap_days(b)
        ov[i, j] = len(da & db)
        frac[i, j] = ov[i, j] / max(len(da), 1)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(frac, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(n)); ax.set_xticklabels(SYMBOLS)
ax.set_yticks(range(n)); ax.set_yticklabels(SYMBOLS)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{frac[i,j]:.2f}", ha="center", va="center",
                color="black" if frac[i,j] > 0.5 else "white", fontsize=11)
ax.set_xlabel("Fraction of row's gap days that are ALSO column's gap days")
ax.set_title("Gap-day overlap — NKD row reads 'only 22% overlap with US futures'", pad=12)
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig("charts/p3_nkd_overlap.png", dpi=140, bbox_inches="tight")
plt.show()
""")

md("""\
**Read the bottom row.** Only **16–22% of NKD's 527 gap days overlap with each US instrument.**
The other ~78% are NKD-only — independent signal that the US-only portfolio simply doesn't have access to.

This is what we mean by "true diversification": same-sized gaps, different macro drivers,
different time-zone news flow. ES gaps on Fed events. NKD gaps on BoJ + Asian risk-off.
""")

# ── Section 6: exit-policy validation on NKD ──────────────────────────────────
md("""\
## 6 · Do the Exit Rules Still Hold?

Phase 2 found:
- **R1 → hold to close**, no exit rule improves
- **M1 → soft stop at 75% of width adverse**, captures the tail with no EV cost

Rerunning the full exit-policy sweep on NKD events to verify both still hold:

### R1 on NKD (148 events, 21.1/yr)
| Policy | Annual EV | Worst |
|---|---:|---:|
| **EOD (hold)** | **+$1,640** | **$0** |
| PT75 | +$1,385 | $0 |
| T15:30 | +$1,003 | $0 |
| SS25 (early stop) | +$273 | −$58 |

**EOD dominates everything on NKD R1**, same as on US futures. Confirmed.

### M1 on NKD (119 events, 17.0/yr)
| Policy | Annual EV | Worst |
|---|---:|---:|
| EOD (hold to close) | +$3,529 | **−$2,503** |
| **SS75 (soft stop)** | **+$3,486** | **−$1,256** |
| SS50 | +$2,898 | −$714 |

**SS75 cuts the worst trade IN HALF (−$2,503 → −$1,256) at almost zero EV cost (−$43/yr).**

On NKD this trade is even better than on US futures, where SS75 cost $235/yr to halve the tail.
**Locked-in exit rules transfer cleanly.**
""")

# ── Section 7: sizing ─────────────────────────────────────────────────────────
md("""\
## 7 · Sizing — How Much Will I Yield?

At **$2,000 max-loss budget per single contract slot**, scaling each cell:

| Sizing rule | Annual yield (model) |
|---|---:|
| **Aggressive** — size to average per-spread max loss | **$20,035** |
| **Realistic** — after skew + slippage haircut (~25%) | **~$14,000** |
| **Conservative** — size to 99th percentile max loss | **$8,026** |

**About 7× per-year EV for each dollar at single-trade risk.** That's the engine ratio.
Capital deployment is a separate question.

**Where the yield comes from:**
- NKD M1 alone: $8,542 (42% of total)
- NKD R1: $2,660 (13%)
- US futures combined: ~$8,800 (44%)

**Without Nikkei, the portfolio yields ~$8,800/yr.** With Nikkei, it's $20K. Half of the
edge is in the diversifier we added in Phase 3.
""")

# ── Section 8: caveats ────────────────────────────────────────────────────────
md("""\
## 8 · Caveats Still Standing

| Caveat | Magnitude |
|---|---|
| VIX as σ proxy for Nikkei | Model EV directionally right but ~10% biased; JNIV would be cleaner |
| Constant-vol BS doesn't capture skew | Real put skew → −10 to 15% on headline EV |
| No slippage modeled | Bid-ask on these spreads: 1–2 pts; eats ~10 to 15% more |
| 7-year sample lacks a true crisis (2008-style) | Tail risk in worst regimes untested |
| Continuous contract rolls create ~4 spurious gaps/yr per symbol | Minor noise |
| Eurex (DAX) unavailable on current Databento plan | Lost one diversifier; could be added later |

## 9 · Cost Budget

Databento spend on Phase 3 universe expansion:
- Cost estimation queries (free)
- NKD 1-min OHLCV 2019–2026: **$6.26**
- Total spent: **$6.26 of $80** budgeted
- Remaining: $73.74

## 10 · Next

1. **Paper-trade R1 + M1 across all 5 symbols** starting 2026-07-01, run for one quarter
2. If hit-rate holds within 5pp of backtest: scale to conservative live sizing
3. Hedged M1 variant for NKD tail control (deep OTM put at ~$1 cost, caps NKD M1 worst $1.3K → $500)
4. Eventually: layer real IV skew into the model + slippage stress test
""")

# Write
nb = new_notebook()
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}

import nbformat
with open("notebooks/10_phase3_nkd.ipynb", "w") as f:
    nbformat.write(nb, f)
print(f"Written: 10_phase3_nkd.ipynb ({len(cells)} cells)")
