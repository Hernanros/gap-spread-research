# Sizing Decision Log — $2,000 per-trade risk

**Date:** 2026-06-22
**Question:** At $2,000 max-loss budget per single contract slot, what's the annual yield?

## Answers

| Sizing rule | Annual yield | 7-year cumulative |
|---|---:|---:|
| **Aggressive** — size by AVG max-loss/spread | **$20,035** | $140,245 |
| **Conservative** — size by p99 max-loss/spread | **$8,026** | $56,182 |

Real-world expected outcome is somewhere between (a real trader dynamically sizes per trade).
After skew + slippage haircut (~25%), realistic deliverable ≈ **$13,000–$15,000/yr** at aggressive sizing, **$5,000–$6,000/yr** at conservative.

## Per-cell breakdown (aggressive sizing)

| Rule | Symbol | Contracts | Yearly $ | % of total |
|---|---|---:|---:|---:|
| M1 | NKD | 2.5 | $8,542 | **42.6%** |
| R1 | NKD | 1.6 | $2,660 | 13.3% |
| M1 | NQ | 1.0 | $1,761 | 8.8% |
| M1 | YM | 2.8 | $1,602 | 8.0% |
| M1 | ES | 2.2 | $1,523 | 7.6% |
| R1 | ES | 1.5 | $1,513 | 7.6% |
| R1 | YM | 2.0 | $1,512 | 7.6% |
| R1 | NQ | 0.7 | $596 | 3.0% |
| M1 | RTY | 2.6 | $213 | 1.1% |
| R1 | RTY | 1.9 | $113 | 0.6% |

## Key observations

- **NKD M1 is the headline:** 43% of total yield. Skip it and yield halves.
- **NQ R1 is capital-inefficient:** high contract price × high ATR → only 0.7 contracts at $2K budget → $596/yr.
- **RTY barely contributes:** ~$326/yr combined across both rules. Candidate to drop.
- **Per-trade risk capped at $2K by spread structure** (defined-risk vertical spreads).
- **Approximate annual-EV / max-single-trade-loss ratio at aggressive sizing: 10:1.**

## Caveats carried forward

1. All numbers from constant-vol BS — overstates EV by 10–20% vs real put skew.
2. No slippage modeled — subtract another 10–15%.
3. 7-year sample includes COVID 2020 (positive vol spike); doesn't include 2008-style crisis.
4. "Average max loss sizing" assumption: real trader would dynamically size per trade.

## Files

- `sizing.py` — analysis script
- `cache/phase3_nkd_validate.csv` — per-event data
- `.planning/HANDOFF.json` — full project state
