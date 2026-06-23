# Gap-Spread Research

> *Investigating one trader's intuition with 7 years of futures data.*

A research project: my mentor showed me a credit-spread trade he made on June 15.
I'm a data person, not a trader. I wanted to know if there was a systematic
pattern behind his move, or just calibrated experience.

7 years of 1-minute futures data later: **two viable rules**, one of which
is exactly what my mentor was doing intuitively.

---

## Start Here

| File | What it is |
|---|---|
| **[`notebooks/11_strategy_brief.html`](notebooks/11_strategy_brief.html)** | **The story. Open this first.** Plain English, lots of charts. |
| [`social/linkedin_post.md`](social/linkedin_post.md) | LinkedIn version of the story |
| [`social/twitter_thread.md`](social/twitter_thread.md) | Twitter/X thread version |

---

## The Research Trail

Everything in `notebooks/`, numbered chronologically:

| # | Notebook | What happened |
|---|---|---|
| 01 | `01_initial_research` | First exploration of the gap pattern |
| 02 | `02_eda` | Daily-resolution EDA on event frequency |
| 03 | `03_strategy_v2` | Three-strategy backtest sweep |
| 04 | `04_phase1_presentation` | Phase 1 summary |
| 05 | `05_phase2_stop_loss` | Tested hard stops — **rejected** |
| 06 | `06_phase2_stratify` | **R1 identified** (mid-VIX × gap-down) |
| 07 | `07_phase2_gap_tiers` | Deeper gaps give *less* credit (BS effect) |
| 07b | `07b_phase2_momentum_anchor` | **M1 identified** (open-anchored on gap-ups) |
| 08 | `08_phase2_exit_rules` | EOD hold is optimal for R1 |
| 08b | `08b_phase2_t2_momentum_tail` | SS75 stop is optimal for M1 |
| 09 | `09_phase2_presentation` | Phase 2 narrative |
| 10 | `10_phase3_nkd` | Nikkei expansion via Databento |
| **11** | **`11_strategy_brief`** | **Final brief (read this)** |

---

## Repo Structure

```
.
├── notebooks/              # research trail (01–11) + HTML exports
├── scripts/                # all Python build & analysis scripts
├── lib/                    # importable library code (data, events, path, spreads)
├── tests/                  # unit tests for lib/
├── data/                   # raw OHLCV data (Databento)
├── cache/                  # intermediate analysis artifacts
├── charts/                 # generated PNGs
├── social/                 # LinkedIn / Twitter drafts
├── docs/                   # design specs
├── .planning/              # project state + handoff
├── requirements.txt
└── venv/
```

## Running the Code

All scripts use absolute path handling — runnable from anywhere:

```bash
source venv/bin/activate
python scripts/build_strategy_brief.py          # rebuilds notebooks/11_strategy_brief.ipynb
python scripts/sizing.py                        # sizing analysis
python scripts/phase3_nkd_validate.py           # NKD R1/M1 validation
```

To execute notebooks:
```bash
jupyter nbconvert --to notebook --execute notebooks/11_strategy_brief.ipynb \
    --output 11_strategy_brief.ipynb
```

## Data Sources

- **CME futures (ES, NQ, RTY, YM, NKD):** Databento 1-minute OHLCV
- **VIX:** Databento daily close
- **Date range:** 2019-01-01 → 2026-06-22 (≈7 years)

Databento cost for this project: ~$6.27 USD.

## The Two Rules

**R1 — Mean reversion (gap-downs):**
- Trigger: |gap| / ATR(14) ≥ 0.75, direction = gap_down, VIX(prev) ∈ [15, 25)
- Trade: 1-DTE bear call spread at prev_close, hold to 16:00 ET
- Universe: ES, NQ, RTY, YM, NKD
- ~50 events/yr, ~$4,300 annual EV at single-contract sizing

**M1 — Momentum (moderate gap-ups):**
- Trigger: 1.0 ≤ |gap|/ATR < 1.5, direction = gap_up
- Trade: 1-DTE bull put spread at today's open
- Exit: SS75 stop (close if low ≤ open − 0.75 × width) or 16:00 ET
- Universe: ES, NQ, RTY, YM, NKD
- ~29 events/yr, ~$6,600 annual EV at single-contract sizing

## Caveats

- Constant-vol Black-Scholes ignores real put-side skew (−10–15% EV)
- No slippage modeled (another −10–15%)
- Sample includes 2020 COVID; doesn't include 2008-style crisis
- **Has not been paper-traded yet.** Backtests flatter, markets humble.

Paper-trading begins 2026-07-01.

---

Built with Python · Databento · pandas · numpy · scipy · matplotlib.
