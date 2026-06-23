# Paper-Trading Workflow

Automated daily routine. All scripts use **free data only** (yfinance + cached Databento) — never pulls paid data without explicit permission.

## Daily routine

### 1. Morning — once at ~09:35 ET (5 min after US open)

```bash
python paper_trading/morning_scan.py
```

Auto-fetches today's open + VIX from yfinance, computes R1/M1 signals for all 5 symbols, logs candidates to `pricing_log.csv`, and prints the exact `decision_today.py` command for any firing signals.

**If yfinance's daily close differs from your broker's RTH close** (it does — yfinance uses Globex aggregate), override with manual values:

```bash
python paper_trading/morning_scan.py \
    --manual_prev_close ES=7547,NQ=30659 \
    --manual_atr ES=114.5,NQ=749.64 \
    --manual_vix 16.4
```

### 2. For each firing signal — pull IBKR quote, decide

```bash
python paper_trading/decision_today.py \
    --symbol NQ --open 29887.5 --prev_close 30659 --atr 749.64 \
    --vix 16.4 --rule R1 \
    --market_credit_usd 158 --market_pop_pct 94
```

Computes real R:R and breakeven WR vs market-implied PoP, applies TAKE/SKIP thresholds (default `R:R ≤ 8` AND `market_PoP ≥ 88%`), logs to `pricing_log.csv`.

### 3. If TAKE — log the entry

```bash
python paper_trading/record_entry.py \
    --symbol NQ --rule R1 --vix 16.4 --atr 749.64 \
    --prev_close 30659 --today_open 29887.5 \
    --contracts 1 --credit 7.9 \
    --notes "Pulled IBKR @ 09:42 ET"
```

(Credit is in instrument points, e.g. 7.9 = $158 NQ credit.)

### 4. Evening — at the close

```bash
python paper_trading/evening_check.py
```

Lists today's candidates + open trades, fetches today's close, computes spread intrinsic, prints the `record_exit.py` command to log final P&L.

### 5. Record the exit

```bash
python paper_trading/record_exit.py --trade_id 1 --exit_price 0.5
```

### 6. Periodic — see how it's going

```bash
python paper_trading/summary.py
```

Realized P&L by rule, win rate, deviation vs model EV.

---

## Files

| File | Purpose |
|---|---|
| `morning_scan.py` | Daily signal scan, free data, logs candidates |
| `decision_today.py` | Per-candidate TAKE/SKIP decision with real market quote |
| `record_entry.py` | Log a paper-trade entry |
| `record_exit.py` | Close a paper trade, compute net P&L vs model |
| `evening_check.py` | Post-close reconciliation reminder |
| `summary.py` | Cumulative performance vs backtest expectations |
| `trades.csv` | All paper trades (entries + exits) |
| `pricing_log.csv` | Every candidate signal — taken, skipped, or NOT_QUOTED |

## Decision thresholds (current defaults)

- Max R:R: **8:1**
- Min market-implied PoP: **88%**
- Auto-SKIP if breakeven WR > backtest WR

Tune in `decision_today.py` with `--max_rr` and `--min_pop`.

## Data sources

- **Live / current:** yfinance (free, may have ~15 min delay)
- **Historical 1-min:** local Databento CSVs (pulled once, do not refresh without permission)
- **VIX:** yfinance daily
- **NEVER:** auto-pull from Databento without asking
