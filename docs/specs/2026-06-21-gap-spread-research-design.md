# Gap-Based Credit Spread Research — Design Spec

**Date:** 2026-06-21  
**Author:** Hernan Rosenblum  
**Purpose:** Portfolio showcase + pre-cursor to Swing Trainer integration  
**Status:** Approved, ready for implementation

---

## 1. Hypothesis

After overnight gaps ≥ 1 ATR in equity index ETFs/futures, selling short-dated credit spreads
anchored at the previous day's close may yield positive expectancy due to mean reversion or
limited immediate trend continuation.

---

## 2. Goals

1. **Portfolio showcase** — publication-quality research notebook demonstrating quant research skills:
   clean narrative, documented assumptions, professional charts.
2. **Swing Trainer integration** — `lib/` modules are architected to port directly into
   `backend/services/` when the feature moves into the app.

---

## 3. File Structure

```
~/Documents/gap-spread-research/
├── gap_spread_research.ipynb       ← research report / narrative
├── lib/
│   ├── __init__.py
│   ├── data.py                     ← yfinance fetch (session + pre-market), parquet cache
│   ├── events.py                   ← ATR(14), gap detection, pre-market fill %
│   ├── path.py                     ← MAE, fill timing, entry timing grid
│   └── spreads.py                  ← Black-Scholes layer (labeled synthetic)
├── cache/
│   └── *.parquet                   ← cached downloads (avoid re-hitting yfinance)
├── charts/
│   └── *.png                       ← exported figures for portfolio use
├── docs/specs/
│   └── 2026-06-21-gap-spread-research-design.md
└── requirements.txt
```

---

## 4. Data Layer (`lib/data.py`)

### Sources

| Data | Ticker(s) | Frequency | Depth | Source |
|------|-----------|-----------|-------|--------|
| Session + pre-market OHLCV | SPY, QQQ, IWM, ES=F, NQ=F | 5m | ~2 years | yfinance (`prepost=True`) |
| Daily OHLCV | SPY, QQQ, IWM, ES=F, NQ=F | 1d | 10+ years | yfinance |
| Volatility index | ^VIX | 1d | 10+ years | yfinance |

### Asset Notes

- **SPY, QQQ, IWM** — primary assets; clean intraday data, no roll gaps.
- **ES=F, NQ=F** — secondary; continuous contract roll gaps complicate gap detection;
  treated as supplementary and clearly labeled in output.
- **^VIX** — IV proxy for Black-Scholes layer; regime label (low < 15, mid 15–25, high > 25).

### Cache Strategy

- First run: fetch → write `cache/{ticker}_5m.parquet` and `cache/{ticker}_daily.parquet`
- Subsequent runs: load from parquet
- `force_refresh=True` flag to re-fetch

### Session Split

Downloaded with `prepost=True`. Split by timestamp:
- `pre_market`: 04:00–09:29 ET
- `session`: 09:30–16:00 ET

---

## 5. Analysis Pipeline

### 5a. `lib/events.py` — Gap Detection

**Inputs:** daily OHLC, 5m pre-market bars  
**Outputs:** event DataFrame with one row per gap event

**Logic:**
1. ATR(14) computed on daily data using Wilder EWM method
2. `gap = open - prev_close`
3. `gap_ratio = gap / ATR(14)`
4. Event when `|gap_ratio| >= 1.0`
5. Label: `gap_up` (positive) / `gap_down` (negative)
6. `premarket_fill_pct`: fraction of gap covered by pre-market price action before 9:30
   - Gap up (open > prev_close): `max(0, open - premarket_low) / gap` — how far the pre-market low dipped back toward prev_close
   - Gap down (open < prev_close): `max(0, premarket_high - open) / |gap|` — how far the pre-market high rallied back toward prev_close
   - Clipped to [0, 1]

**Event columns:**

```
date, ticker, direction, gap, gap_ratio, atr,
prev_close, open, premarket_high, premarket_low,
premarket_fill_pct, vix_close
```

---

### 5b. `lib/path.py` — Price-Path Layer (Layer 1)

**No synthetic assumptions — pure price data.**

**Inputs:** event DataFrame, 5m session bars  
**Outputs:** path results DataFrame

**Per event:**

| Feature | Definition |
|---------|------------|
| `gap_filled` | bool — did session price touch `prev_close`? |
| `fill_time` | minutes from open to first touch of `prev_close`; NaN if not filled |
| `mae` | maximum adverse excursion: worst 5m close against spread direction from entry |
| `mae_pct_atr` | MAE / ATR |
| `eod_close` | closing price relative to `prev_close` |

**Entry timing grid:**

Simulate entering at open (09:30), 09:45, 10:00, 10:30. For each entry time:
- `entry_price`: 5m bar close at or after entry time
- Recompute `mae` and `gap_filled` from that entry point forward
- Output: 4 rows per event (one per entry time) with `entry_time` column

**Segmentation variables:**

- `premarket_fill_pct` quartiles (Q1–Q4) — predictor analysis
- VIX regime (low / mid / high) — regime breakdown

---

### 5c. `lib/spreads.py` — Options P&L Layer (Layer 2, Synthetic)

**All outputs labeled `[Synthetic — Black-Scholes + VIX/16 proxy]`.**

**Pricing:**

- `S` = entry price at selected entry time
- `K_short` = `prev_close` (short strike, ATM)
- `K_long` = `prev_close ± 0.5 × ATR` (long strike, wing)
- `σ` = `VIX_close / 100` (annualized IV proxy)
- `T` = `DTE / 252` where DTE = 1
- `r` = 0.05 (risk-free rate)
- Gap up → Bull Put Spread (sell put at K_short, buy put at K_long below)
- Gap down → Bear Call Spread (sell call at K_short, buy call at K_long above)

**Exit strategies simulated:**

| Exit | Logic |
|------|-------|
| EOD | P&L based on EOD intrinsic value vs. strikes |
| PT25 / PT50 / PT75 | Hit if MAE stays within `(1 - target) × spread_width`; proxy for intraday target touch |
| SL1× / SL2× / SL3× | Stop triggered if `mae > N × credit`; P&L = `-N × credit` |

**Output metrics per (exit_strategy × entry_time × ticker):**

- `n_trades`, `win_rate`, `avg_win`, `avg_loss`
- `expectancy`, `expectancy_per_dollar_risked`
- `profit_factor`
- MAE distribution: p50, p75, p90, p95, p99, worst

---

## 6. Notebook Narrative

```
0. Setup & Config
   — imports, constants, asset list

1. Data
   — load/fetch, event counts per ticker, data quality notes

2. Gap Event Characterization
   — gap size distribution (ATR units), gap_up vs. gap_down frequency,
     premarket_fill_pct distribution, VIX at event time

3. Price-Path Analysis  [Layer 1 — no synthetic assumptions]
   3a. Gap fill rate & time-to-fill (by ticker, direction)
   3b. MAE distribution (histogram + tail statistics)
   3c. Entry timing grid (heat map: fill rate × entry time × ticker)
   3d. Pre-market fill % as predictor (quartile segmentation)
   3e. VIX regime breakdown (low / mid / high)

4. Credit Spread Simulation  [Layer 2 — Synthetic: Black-Scholes + VIX proxy]
   4a. Spread pricing at each entry time
   4b. Exit strategy comparison (EOD / PT25/50/75 / SL1×/2×/3×)
   4c. Expectancy by exit × entry time (summary table)
   4d. Tail risk: MAE distribution and worst-case scenarios

5. Multi-Asset Comparison
   — fill rate, MAE, expectancy side-by-side: SPY / QQQ / IWM / ES=F / NQ=F

6. Key Findings & Limitations
   — bullet summary of what the data shows
   — explicit limitations: synthetic options, ~2-year 5m sample,
     no bid/ask spread, no transaction costs, no liquidity constraints
   — Swing Trainer integration map (lib/ → backend/services/)
```

---

## 7. Swing Trainer Integration Map

| Research module | Target Swing Trainer file |
|-----------------|--------------------------|
| `lib/data.py` | `backend/services/data.py` (extend existing) |
| `lib/events.py` | `backend/services/gap_events.py` (new) |
| `lib/path.py` | `backend/services/gap_path.py` (new) |
| `lib/spreads.py` | `backend/services/options.py` (extend existing `YFinanceOptionsProvider`) |

Integration phase: after research confirms edge, expose via `GET /api/gaps/events` and
`GET /api/gaps/analysis` endpoints, with a new "Gap Scanner" section in the Bull Assistant UI.

---

## 8. Key Constants (configurable in notebook Section 0)

```python
TICKERS_PRIMARY   = ["SPY", "QQQ", "IWM"]
TICKERS_SECONDARY = ["ES=F", "NQ=F"]
VIX_TICKER        = "^VIX"
DAILY_START       = "2013-01-01"
ATR_PERIOD        = 14
GAP_THRESHOLD     = 1.0      # in ATR units
SPREAD_WIDTH_ATR  = 0.5      # wing width as fraction of ATR
DTE               = 1        # days to expiration
RISK_FREE_RATE    = 0.05
PROFIT_TARGETS    = [0.25, 0.50, 0.75]
STOP_MULTIPLES    = [1, 2, 3]
ENTRY_TIMES       = ["09:30", "09:45", "10:00", "10:30"]
VIX_REGIMES       = {"low": (0, 15), "mid": (15, 25), "high": (25, 999)}
```

---

## 9. Requirements

```
yfinance>=0.2.40
pandas>=2.0
numpy>=1.24
scipy>=1.11
matplotlib>=3.7
pyarrow>=14.0      # parquet cache
jupyter>=1.0
```

---

## 10. Out of Scope

- Live trading / order execution
- Real historical options chain data (would require paid API)
- Intraday options pricing (assumes entry at open IV snapshot)
- Earnings event filtering (noted as future enhancement)
- Transaction costs and slippage modeling
