# Gap-Based Credit Spread Research — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular Python research package and Jupyter notebook that backtests gap-based credit spread strategies on index ETFs using 5m intraday data, pre-market fill % as a predictor, and Black-Scholes spread pricing as an explicitly-labeled synthetic layer.

**Architecture:** `lib/` contains four focused modules (data, events, path, spreads); the notebook imports from `lib/` and serves as the research narrative. All logic is unit-tested before being wired into the notebook. The two-layer design keeps pure price-path analysis (Layer 1) separate from synthetic options P&L (Layer 2).

**Tech Stack:** Python 3.10+, yfinance, pandas, numpy, scipy, matplotlib, pyarrow (parquet cache), nbformat (notebook assembly), pytest

**Spec:** `docs/specs/2026-06-21-gap-spread-research-design.md`

---

## File Map

| File | Created/Modified | Responsibility |
|------|-----------------|----------------|
| `requirements.txt` | Create | Dependencies |
| `.gitignore` | Create | Ignore cache, charts, .ipynb_checkpoints |
| `lib/__init__.py` | Create | Empty package marker |
| `lib/data.py` | Create | yfinance fetch, parquet cache, session split |
| `lib/events.py` | Create | ATR(14), gap detection, premarket_fill_pct |
| `lib/path.py` | Create | MAE, gap fill, entry timing grid, VIX regime labels |
| `lib/spreads.py` | Create | Black-Scholes pricing, exit simulation, metrics |
| `tests/__init__.py` | Create | Empty |
| `tests/conftest.py` | Create | Shared fixtures |
| `tests/test_events.py` | Create | Unit tests for lib/events.py |
| `tests/test_path.py` | Create | Unit tests for lib/path.py |
| `tests/test_spreads.py` | Create | Unit tests for lib/spreads.py |
| `gap_spread_research.ipynb` | Create | Research report (assembled in Tasks 6-9) |

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `lib/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `cache/.gitkeep`
- Create: `charts/.gitkeep`

- [ ] **Step 1: Write requirements.txt**

```
yfinance>=0.2.40
pandas>=2.0
numpy>=1.24
scipy>=1.11
matplotlib>=3.7
pyarrow>=14.0
nbformat>=5.9
jupyter>=1.0
pytest>=7.0
```

- [ ] **Step 2: Write .gitignore**

```
cache/
charts/
.ipynb_checkpoints/
__pycache__/
*.pyc
.DS_Store
```

- [ ] **Step 3: Write tests/conftest.py**

```python
import pandas as pd
import numpy as np
import pytest


def make_daily(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    opens: list[float] | None = None,
    start: str = "2024-01-02",
) -> pd.DataFrame:
    n = len(closes)
    if highs is None:
        highs = [c * 1.01 for c in closes]
    if lows is None:
        lows = [c * 0.99 for c in closes]
    if opens is None:
        opens = closes[:]
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=dates,
    )


def make_session_bars(
    date: str,
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    n = len(closes)
    if highs is None:
        highs = [c * 1.002 for c in closes]
    if lows is None:
        lows = [c * 0.998 for c in closes]
    times = pd.date_range(
        f"{date} 09:30", periods=n, freq="5min", tz="America/New_York"
    )
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100_000] * n,
        },
        index=times,
    )
```

- [ ] **Step 4: Create empty package markers and cache/charts dirs**

```bash
mkdir -p lib tests cache charts
touch lib/__init__.py tests/__init__.py cache/.gitkeep charts/.gitkeep
```

- [ ] **Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore lib/__init__.py tests/__init__.py tests/conftest.py cache/.gitkeep charts/.gitkeep
git commit -m "chore: project scaffold"
```

---

## Task 2: lib/data.py — Data Fetching & Cache

**Files:**
- Create: `lib/data.py`

- [ ] **Step 1: Write lib/data.py**

```python
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent.parent / "cache"
DAILY_START = "2013-01-01"


def _cache_key(ticker: str, freq: str) -> str:
    return ticker.replace("^", "VIX").replace("=", "_") + f"_{freq}"


def fetch(
    ticker: str,
    freq: Literal["5m", "1d"],
    force_refresh: bool = False,
    prepost: bool = True,
) -> pd.DataFrame:
    """Download OHLCV data from yfinance, caching to parquet on first run.

    For 5m: downloads ~730 days with prepost=True (includes pre/after-market bars).
    For 1d: downloads from DAILY_START to today.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(ticker, freq)}.parquet"

    if path.exists() and not force_refresh:
        return pd.read_parquet(path)

    if freq == "5m":
        df = yf.download(
            ticker,
            period="730d",
            interval="5m",
            prepost=prepost,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
    else:
        df = yf.download(
            ticker,
            start=DAILY_START,
            interval="1d",
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )

    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.to_parquet(path)
    return df


def split_session(df_5m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split 5m bars into pre-market and regular session.

    Returns (pre_market, session) where:
        pre_market: 04:00–09:29 ET
        session:    09:30–16:00 ET
    """
    if df_5m.empty:
        return df_5m.copy(), df_5m.copy()

    if df_5m.index.tz is None:
        df = df_5m.tz_localize("UTC").tz_convert("America/New_York")
    else:
        df = df_5m.tz_convert("America/New_York")

    t = df.index.time
    pre = df[
        (t >= datetime.time(4, 0)) & (t < datetime.time(9, 30))
    ]
    session = df[
        (t >= datetime.time(9, 30)) & (t <= datetime.time(16, 0))
    ]
    return pre, session
```

- [ ] **Step 2: Smoke-test the cache (manual, no pytest)**

```bash
python - <<'EOF'
from lib.data import fetch, split_session
df = fetch("SPY", "1d")
print(f"SPY daily: {len(df)} rows, {df.index[0].date()} → {df.index[-1].date()}")
df5 = fetch("SPY", "5m")
pre, sess = split_session(df5)
print(f"SPY 5m: {len(df5)} total, {len(pre)} pre-market, {len(sess)} session bars")
EOF
```

Expected output (values will vary): `SPY daily: 2xxx rows ... SPY 5m: xxxxx total ...`

- [ ] **Step 3: Commit**

```bash
git add lib/data.py
git commit -m "feat: data fetch with parquet cache and session split"
```

---

## Task 3: lib/events.py + Tests

**Files:**
- Create: `lib/events.py`
- Create: `tests/test_events.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_events.py
import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_daily
from lib.events import compute_atr, detect_events, _fill_pct


# --- ATR ---

def test_atr_first_value_nan():
    df = make_daily([100.0] * 20)
    atr = compute_atr(df, period=14)
    assert pd.isna(atr.iloc[0])


def test_atr_all_positive_after_warmup():
    df = make_daily([100.0] * 20)
    atr = compute_atr(df, period=14)
    assert (atr.dropna() > 0).all()


def test_atr_larger_range_yields_larger_atr():
    df_narrow = make_daily([100.0] * 20, highs=[101.0] * 20, lows=[99.0] * 20)
    df_wide = make_daily([100.0] * 20, highs=[105.0] * 20, lows=[95.0] * 20)
    assert compute_atr(df_wide).iloc[-1] > compute_atr(df_narrow).iloc[-1]


# --- detect_events ---

def test_detects_gap_up():
    closes = [100.0] * 20
    opens = closes[:]
    opens[-1] = 108.0  # large gap up on last day
    df = make_daily(closes, opens=opens)
    events = detect_events(df, pd.DataFrame())
    assert "gap_up" in events["direction"].values


def test_detects_gap_down():
    closes = [100.0] * 20
    opens = closes[:]
    opens[-1] = 92.0  # large gap down on last day
    df = make_daily(closes, opens=opens)
    events = detect_events(df, pd.DataFrame())
    assert "gap_down" in events["direction"].values


def test_no_event_small_gap():
    closes = [100.0] * 20
    opens = closes[:]
    opens[-1] = 100.05  # tiny gap, well below 1 ATR
    df = make_daily(closes, opens=opens)
    events = detect_events(df, pd.DataFrame())
    assert len(events) == 0


def test_gap_ratio_correct():
    closes = [100.0] * 20
    opens = closes[:]
    opens[-1] = 108.0
    df = make_daily(closes, opens=opens)
    events = detect_events(df, pd.DataFrame())
    gap_up = events[events["direction"] == "gap_up"].iloc[0]
    assert gap_up["gap"] == pytest.approx(8.0)
    assert gap_up["gap_ratio"] == pytest.approx(gap_up["gap"] / gap_up["atr"])


# --- _fill_pct ---

def _make_events_row(direction, open_, prev_close, gap, pm_high, pm_low):
    return pd.DataFrame(
        {
            "direction": [direction],
            "open": [open_],
            "prev_close": [prev_close],
            "gap": [gap],
            "premarket_high": [pm_high],
            "premarket_low": [pm_low],
        },
        index=pd.DatetimeIndex(["2024-06-03"]),
    )


def test_fill_pct_gap_up_full_fill():
    ev = _make_events_row("gap_up", open_=105, prev_close=100, gap=5,
                          pm_high=106, pm_low=100)
    assert _fill_pct(ev).iloc[0] == pytest.approx(1.0)


def test_fill_pct_gap_up_half_fill():
    ev = _make_events_row("gap_up", open_=105, prev_close=100, gap=5,
                          pm_high=106, pm_low=102.5)
    assert _fill_pct(ev).iloc[0] == pytest.approx(0.5)


def test_fill_pct_gap_up_no_fill_clipped():
    # pre-market low above open → no retracement at all
    ev = _make_events_row("gap_up", open_=105, prev_close=100, gap=5,
                          pm_high=107, pm_low=106)
    assert _fill_pct(ev).iloc[0] == pytest.approx(0.0)


def test_fill_pct_gap_down_full_fill():
    # gap down: open=95, prev_close=100, gap=-5
    # pm high = 100 → fully filled
    ev = _make_events_row("gap_down", open_=95, prev_close=100, gap=-5,
                          pm_high=100, pm_low=94)
    assert _fill_pct(ev).iloc[0] == pytest.approx(1.0)


def test_fill_pct_gap_down_half_fill():
    ev = _make_events_row("gap_down", open_=95, prev_close=100, gap=-5,
                          pm_high=97.5, pm_low=94)
    assert _fill_pct(ev).iloc[0] == pytest.approx(0.5)


def test_fill_pct_capped_at_one():
    # pre-market low went below prev_close → clip to 1.0
    ev = _make_events_row("gap_up", open_=105, prev_close=100, gap=5,
                          pm_high=106, pm_low=98)  # below prev_close
    assert _fill_pct(ev).iloc[0] == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests — expect all to FAIL**

```bash
cd ~/Documents/gap-spread-research && pytest tests/test_events.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'compute_atr' from 'lib.events'`

- [ ] **Step 3: Write lib/events.py**

```python
from __future__ import annotations

import pandas as pd
import numpy as np

ATR_PERIOD = 14
GAP_THRESHOLD = 1.0


def compute_atr(daily: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Wilder EWM ATR. First value is NaN (no previous close for TR)."""
    high = daily["high"]
    low = daily["low"]
    prev_close = daily["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def detect_events(
    daily: pd.DataFrame,
    premarket: pd.DataFrame,
    vix_daily: pd.DataFrame | None = None,
    threshold: float = GAP_THRESHOLD,
    atr_period: int = ATR_PERIOD,
) -> pd.DataFrame:
    """Return one row per gap event (|gap_ratio| >= threshold).

    Args:
        daily: Daily OHLCV with columns open/high/low/close.
        premarket: 5m pre-market bars (tz-aware, America/New_York).
                   Pass empty DataFrame if unavailable.
        vix_daily: Daily VIX close. Merged on date if provided.
        threshold: Minimum |gap| in ATR units to qualify as event.
        atr_period: Period for ATR computation.

    Returns:
        DataFrame indexed by date with columns:
        direction, gap, gap_ratio, atr, prev_close, open,
        [premarket_high, premarket_low, premarket_fill_pct if premarket provided],
        [vix_close if vix_daily provided]
    """
    d = daily.copy()
    d["atr"] = compute_atr(d, atr_period)
    d["prev_close"] = d["close"].shift(1)
    d["gap"] = d["open"] - d["prev_close"]
    d["gap_ratio"] = d["gap"] / d["atr"]
    d["direction"] = np.where(
        d["gap_ratio"] >= threshold,
        "gap_up",
        np.where(d["gap_ratio"] <= -threshold, "gap_down", "none"),
    )

    events = d[d["direction"] != "none"].copy()
    events = events.dropna(subset=["atr", "prev_close"])

    if not premarket.empty:
        pm_stats = _premarket_stats(premarket)
        events = events.join(pm_stats, how="left")
        events["premarket_fill_pct"] = _fill_pct(events)

    if vix_daily is not None and not vix_daily.empty:
        vix_col = vix_daily["close"] if "close" in vix_daily.columns else vix_daily.iloc[:, 0]
        vix_col.index = pd.to_datetime(vix_col.index).normalize()
        events.index = pd.to_datetime(events.index).normalize()
        events = events.join(vix_col.rename("vix_close"), how="left")

    keep = ["direction", "gap", "gap_ratio", "atr", "prev_close", "open"]
    for col in ["premarket_high", "premarket_low", "premarket_fill_pct", "vix_close"]:
        if col in events.columns:
            keep.append(col)

    return events[keep]


def _premarket_stats(pre: pd.DataFrame) -> pd.DataFrame:
    """Compute daily pre-market high and low from 5m bars."""
    df = pre.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York")
    df.index = df.index.normalize().tz_localize(None)
    return df.groupby(df.index).agg(
        premarket_high=("high", "max"),
        premarket_low=("low", "min"),
    )


def _fill_pct(events: pd.DataFrame) -> pd.Series:
    """Fraction of gap filled by pre-market action. Clipped to [0, 1].

    Gap up: measures how far premarket_low dipped back toward prev_close.
        fill_pct = max(0, open - premarket_low) / gap
    Gap down: measures how far premarket_high rallied back toward prev_close.
        fill_pct = max(0, premarket_high - open) / |gap|
    """
    result = pd.Series(0.0, index=events.index)

    up = events["direction"] == "gap_up"
    result[up] = (events.loc[up, "open"] - events.loc[up, "premarket_low"]) / events.loc[up, "gap"]

    down = ~up
    result[down] = (events.loc[down, "premarket_high"] - events.loc[down, "open"]) / events.loc[down, "gap"].abs()

    return result.clip(0.0, 1.0)
```

- [ ] **Step 4: Run tests — expect all to PASS**

```bash
pytest tests/test_events.py -v
```

Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add lib/events.py tests/test_events.py
git commit -m "feat: gap event detection with ATR and premarket fill pct"
```

---

## Task 4: lib/path.py + Tests

**Files:**
- Create: `lib/path.py`
- Create: `tests/test_path.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_path.py
import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_daily, make_session_bars
from lib.path import analyze_paths, label_vix_regime


def _make_event(
    date: str = "2024-06-03",
    direction: str = "gap_up",
    open_: float = 105.0,
    prev_close: float = 100.0,
    atr: float = 2.5,
    premarket_fill_pct: float = 0.2,
    vix_close: float = 18.0,
) -> pd.DataFrame:
    gap = open_ - prev_close
    return pd.DataFrame(
        {
            "direction": [direction],
            "open": [open_],
            "prev_close": [prev_close],
            "gap": [gap],
            "gap_ratio": [gap / atr],
            "atr": [atr],
            "premarket_fill_pct": [premarket_fill_pct],
            "vix_close": [vix_close],
        },
        index=pd.DatetimeIndex([date]),
    )


def test_gap_filled_when_low_touches_prev_close():
    session = make_session_bars(
        "2024-06-03",
        closes=[105, 104, 103, 101, 100, 100, 101],
        lows=[104.5, 103.5, 102.5, 100.5, 99.5, 99.5, 100.5],
    )
    events = _make_event(direction="gap_up", open_=105.0, prev_close=100.0)
    result = analyze_paths(events, session, entry_times=["09:30"])
    assert result.iloc[0]["gap_filled"] is True or result.iloc[0]["gap_filled"] == True


def test_gap_not_filled_when_price_stays_above_prev_close():
    session = make_session_bars(
        "2024-06-03",
        closes=[105, 106, 107, 106, 105, 104, 103],
        lows=[104.5, 105.5, 106.5, 105.5, 104.5, 103.5, 102.5],
    )
    events = _make_event(direction="gap_up", open_=105.0, prev_close=100.0)
    result = analyze_paths(events, session, entry_times=["09:30"])
    assert result.iloc[0]["gap_filled"] is False or result.iloc[0]["gap_filled"] == False


def test_fill_time_is_minutes_from_open():
    # Price touches prev_close (100) at 10:00 = 30 mins from open
    closes = [105, 104, 103, 102, 101, 100] + [100] * 7
    lows = [c - 0.3 for c in closes]
    lows[5] = 99.5  # bar at 10:00 (index 6 from 09:30 = 6*5=30 mins)
    session = make_session_bars("2024-06-03", closes=closes, lows=lows)
    events = _make_event(direction="gap_up", open_=105.0, prev_close=100.0)
    result = analyze_paths(events, session, entry_times=["09:30"])
    fill_time = result.iloc[0]["fill_time"]
    assert fill_time is not None and fill_time >= 0


def test_mae_gap_up_is_adverse_upward_move():
    # Gap up bull put: adverse = price rising away from prev_close
    # Entry at 105, session goes to high of 111
    closes = [105, 107, 110, 108, 106]
    highs = [106, 108, 111, 109, 107]
    lows = [104, 106, 109, 107, 105]
    session = make_session_bars("2024-06-03", closes=closes, highs=highs, lows=lows)
    events = _make_event(direction="gap_up", open_=105.0)
    result = analyze_paths(events, session, entry_times=["09:30"])
    # MAE = max(high - entry_price) = 111 - 105 = 6
    assert result.iloc[0]["mae"] == pytest.approx(6.0)


def test_mae_gap_down_is_adverse_downward_move():
    # Gap down bear call: adverse = price falling away from prev_close
    # Entry at 95, session goes to low of 89
    closes = [95, 93, 90, 92, 94]
    highs = [96, 94, 91, 93, 95]
    lows = [94, 92, 89, 91, 93]
    session = make_session_bars("2024-06-03", closes=closes, highs=highs, lows=lows)
    events = _make_event(direction="gap_down", open_=95.0, prev_close=100.0)
    result = analyze_paths(events, session, entry_times=["09:30"])
    # MAE = max(entry_price - low) = 95 - 89 = 6
    assert result.iloc[0]["mae"] == pytest.approx(6.0)


def test_entry_timing_grid_returns_four_rows_per_event():
    closes = [105] * 20
    session = make_session_bars("2024-06-03", closes=closes)
    events = _make_event()
    result = analyze_paths(events, session, entry_times=["09:30", "09:45", "10:00", "10:30"])
    assert len(result) == 4
    assert set(result["entry_time"]) == {"09:30", "09:45", "10:00", "10:30"}


def test_later_entry_has_smaller_or_equal_mae():
    # As we wait longer, the worst adverse from entry should not increase in an uptrend
    closes = [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.1 for c in closes]
    session = make_session_bars("2024-06-03", closes=closes, highs=highs, lows=lows)
    events = _make_event(direction="gap_up")
    result = analyze_paths(events, session, entry_times=["09:30", "09:45"])
    mae_open = result[result["entry_time"] == "09:30"]["mae"].iloc[0]
    mae_945 = result[result["entry_time"] == "09:45"]["mae"].iloc[0]
    # 09:45 entry misses first 3 bars of the uptrend → smaller MAE from entry
    assert mae_945 <= mae_open


def test_mae_pct_atr_computed():
    closes = [105] * 13
    session = make_session_bars("2024-06-03", closes=closes)
    events = _make_event(atr=2.5)
    result = analyze_paths(events, session, entry_times=["09:30"])
    row = result.iloc[0]
    assert row["mae_pct_atr"] == pytest.approx(row["mae"] / row["atr"])


def test_label_vix_regime_low():
    vix = pd.Series([12.0])
    assert label_vix_regime(vix).iloc[0] == "low"


def test_label_vix_regime_mid():
    vix = pd.Series([20.0])
    assert label_vix_regime(vix).iloc[0] == "mid"


def test_label_vix_regime_high():
    vix = pd.Series([32.0])
    assert label_vix_regime(vix).iloc[0] == "high"
```

- [ ] **Step 2: Run tests — expect all to FAIL**

```bash
pytest tests/test_path.py -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'analyze_paths' from 'lib.path'`

- [ ] **Step 3: Write lib/path.py**

```python
from __future__ import annotations

import datetime
from typing import Sequence

import numpy as np
import pandas as pd

ENTRY_TIMES: list[str] = ["09:30", "09:45", "10:00", "10:30"]
VIX_REGIMES: dict[str, tuple[float, float]] = {
    "low": (0, 15),
    "mid": (15, 25),
    "high": (25, 999),
}


def analyze_paths(
    events: pd.DataFrame,
    session: pd.DataFrame,
    entry_times: Sequence[str] = ENTRY_TIMES,
) -> pd.DataFrame:
    """Compute gap fill, MAE, and fill timing for each event × entry_time.

    Args:
        events: Output of detect_events(). Indexed by date (tz-naive).
        session: 5m session bars (tz-aware, America/New_York).
        entry_times: List of entry time strings in "HH:MM" format.

    Returns:
        DataFrame with columns:
        date, ticker, direction, entry_time, entry_price, prev_close,
        gap_ratio, atr, premarket_fill_pct, vix_close,
        gap_filled, fill_time, mae, mae_pct_atr, eod_close
    """
    if session.empty:
        return pd.DataFrame()

    sess = session.copy()
    if sess.index.tz is None:
        sess = sess.tz_localize("America/New_York")
    else:
        sess = sess.tz_convert("America/New_York")

    records: list[dict] = []

    for date, row in events.iterrows():
        event_date = pd.Timestamp(date).date()
        day_bars = sess[sess.index.date == event_date]
        if day_bars.empty:
            continue

        prev_close = float(row["prev_close"])
        direction = row["direction"]
        eod_close = float(day_bars.iloc[-1]["close"])

        for entry_str in entry_times:
            h, m = map(int, entry_str.split(":"))
            entry_time_obj = datetime.time(h, m)
            bars = day_bars[day_bars.index.time >= entry_time_obj]
            if bars.empty:
                continue

            entry_price = float(bars.iloc[0]["close"])

            # Gap fill detection
            if direction == "gap_up":
                fill_bars = bars[bars["low"] <= prev_close]
            else:
                fill_bars = bars[bars["high"] >= prev_close]

            gap_filled = not fill_bars.empty
            fill_time: float | None = None
            if gap_filled:
                fb = fill_bars.iloc[0]
                open_minutes = 9 * 60 + 30
                bar_minutes = fb.name.hour * 60 + fb.name.minute
                fill_time = float(bar_minutes - open_minutes)

            # MAE: worst adverse excursion from entry_price
            if direction == "gap_up":
                mae = float((bars["high"] - entry_price).clip(lower=0).max())
            else:
                mae = float((entry_price - bars["low"]).clip(lower=0).max())

            atr = float(row["atr"])

            records.append(
                {
                    "date": date,
                    "ticker": row.get("ticker", ""),
                    "direction": direction,
                    "entry_time": entry_str,
                    "entry_price": entry_price,
                    "prev_close": prev_close,
                    "gap_ratio": float(row["gap_ratio"]),
                    "atr": atr,
                    "premarket_fill_pct": float(row.get("premarket_fill_pct", np.nan)),
                    "vix_close": float(row.get("vix_close", np.nan)),
                    "gap_filled": gap_filled,
                    "fill_time": fill_time,
                    "mae": mae,
                    "mae_pct_atr": mae / atr if atr > 0 else np.nan,
                    "eod_close": eod_close,
                }
            )

    return pd.DataFrame(records)


def label_vix_regime(
    vix: pd.Series,
    regimes: dict[str, tuple[float, float]] = VIX_REGIMES,
) -> pd.Series:
    """Map VIX values to regime labels: 'low', 'mid', 'high'."""
    result = pd.Series("unknown", index=vix.index, dtype=str)
    for label, (lo, hi) in regimes.items():
        result[(vix >= lo) & (vix < hi)] = label
    return result
```

- [ ] **Step 4: Run tests — expect all to PASS**

```bash
pytest tests/test_path.py -v
```

Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add lib/path.py tests/test_path.py
git commit -m "feat: price path analysis — MAE, gap fill, entry timing grid"
```

---

## Task 5: lib/spreads.py + Tests

**Files:**
- Create: `lib/spreads.py`
- Create: `tests/test_spreads.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_spreads.py
import pandas as pd
import numpy as np
import pytest
from lib.spreads import (
    _bs_put,
    _bs_call,
    price_spread,
    simulate_exits,
    compute_metrics,
)


# --- Black-Scholes ---

def test_bs_put_intrinsic_at_zero_time():
    assert _bs_put(S=95, K=100, T=0, r=0.05, sigma=0.20) == pytest.approx(5.0)


def test_bs_put_zero_when_otm_at_zero_time():
    assert _bs_put(S=105, K=100, T=0, r=0.05, sigma=0.20) == pytest.approx(0.0)


def test_bs_call_intrinsic_at_zero_time():
    assert _bs_call(S=105, K=100, T=0, r=0.05, sigma=0.20) == pytest.approx(5.0)


def test_bs_call_zero_when_otm_at_zero_time():
    assert _bs_call(S=95, K=100, T=0, r=0.05, sigma=0.20) == pytest.approx(0.0)


def test_bs_put_positive_for_positive_time():
    val = _bs_put(S=100, K=100, T=1 / 252, r=0.05, sigma=0.20)
    assert val > 0


def test_bs_put_atm_greater_than_otm():
    atm = _bs_put(S=100, K=100, T=1 / 252, r=0.05, sigma=0.20)
    otm = _bs_put(S=100, K=90, T=1 / 252, r=0.05, sigma=0.20)
    assert atm > otm


# --- price_spread ---

def test_price_spread_gap_up_positive_credit():
    result = price_spread(
        S=105, K_short=100, atr=5, direction="gap_up",
        vix=20, r=0.05, dte=1, width_atr=0.5,
    )
    assert result["credit"] >= 0
    assert result["K_long"] < result["K_short"]
    assert result["spread_width"] == pytest.approx(2.5)


def test_price_spread_gap_down_positive_credit():
    result = price_spread(
        S=95, K_short=100, atr=5, direction="gap_down",
        vix=20, r=0.05, dte=1, width_atr=0.5,
    )
    assert result["credit"] >= 0
    assert result["K_long"] > result["K_short"]


def test_price_spread_higher_vix_higher_credit():
    low_vix = price_spread(S=105, K_short=100, atr=5, direction="gap_up", vix=15)
    high_vix = price_spread(S=105, K_short=100, atr=5, direction="gap_up", vix=30)
    assert high_vix["credit"] > low_vix["credit"]


# --- simulate_exits ---

def _make_path_df(**kwargs) -> pd.DataFrame:
    defaults = dict(
        date=pd.Timestamp("2024-06-03"),
        ticker="SPY",
        direction="gap_up",
        entry_time="09:30",
        entry_price=105.0,
        prev_close=100.0,
        atr=5.0,
        mae=1.0,
        eod_close=103.0,
        vix_close=20.0,
        gap_filled=True,
        gap_ratio=2.0,
        premarket_fill_pct=0.3,
    )
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


def test_eod_win_when_price_stays_above_short_strike():
    # Gap up, EOD close above prev_close → spread expires worthless → full credit
    df = _make_path_df(direction="gap_up", prev_close=100, eod_close=103, mae=0.5)
    exits = simulate_exits(df, profit_targets=[], stop_multiples=[])
    eod = exits[exits["exit_strategy"] == "EOD"].iloc[0]
    assert eod["win"] is True or eod["win"] == True
    assert eod["pnl"] > 0


def test_eod_loss_when_price_breaches_short_strike():
    # Gap up, EOD close well below prev_close (deep in the spread)
    df = _make_path_df(direction="gap_up", prev_close=100, eod_close=95, mae=6.0, atr=5.0)
    exits = simulate_exits(df, profit_targets=[], stop_multiples=[])
    eod = exits[exits["exit_strategy"] == "EOD"].iloc[0]
    assert eod["pnl"] < eod["credit"]


def test_stop_loss_triggers_caps_loss():
    # MAE larger than 1× credit → stop triggers
    df = _make_path_df(mae=100.0)  # absurdly large MAE to guarantee trigger
    exits = simulate_exits(df, profit_targets=[], stop_multiples=[1])
    sl = exits[exits["exit_strategy"] == "SL1x"].iloc[0]
    credit = sl["credit"]
    assert sl["pnl"] == pytest.approx(-credit, abs=1e-6)


def test_profit_target_win_when_mae_small():
    # Small MAE → price barely moved → PT50 hit
    df = _make_path_df(mae=0.01)
    exits = simulate_exits(df, profit_targets=[0.50], stop_multiples=[])
    pt = exits[exits["exit_strategy"] == "PT50"].iloc[0]
    credit = pt["credit"]
    assert pt["pnl"] == pytest.approx(0.50 * credit, abs=1e-6)


def test_all_exit_strategies_present():
    df = _make_path_df()
    exits = simulate_exits(df, profit_targets=[0.25, 0.50, 0.75], stop_multiples=[1, 2, 3])
    strategies = set(exits["exit_strategy"])
    assert strategies == {"EOD", "PT25", "PT50", "PT75", "SL1x", "SL2x", "SL3x"}


# --- compute_metrics ---

def test_compute_metrics_win_rate():
    rows = [
        _make_path_df(eod_close=103).iloc[0].to_dict(),  # win
        _make_path_df(eod_close=103).iloc[0].to_dict(),  # win
        _make_path_df(eod_close=95, mae=10.0).iloc[0].to_dict(),  # likely loss
    ]
    path_df = pd.DataFrame(rows)
    exits = simulate_exits(path_df, profit_targets=[], stop_multiples=[])
    metrics = compute_metrics(exits)
    eod = metrics[
        (metrics["exit_strategy"] == "EOD") & (metrics["entry_time"] == "09:30")
    ].iloc[0]
    assert 0 < eod["win_rate"] <= 1.0


def test_compute_metrics_profit_factor_positive():
    rows = [_make_path_df(eod_close=103).iloc[0].to_dict() for _ in range(5)]
    rows += [_make_path_df(eod_close=95, mae=10.0).iloc[0].to_dict()]
    path_df = pd.DataFrame(rows)
    exits = simulate_exits(path_df, profit_targets=[], stop_multiples=[])
    metrics = compute_metrics(exits)
    eod = metrics[metrics["exit_strategy"] == "EOD"].iloc[0]
    assert eod["profit_factor"] > 0
```

- [ ] **Step 2: Run tests — expect all to FAIL**

```bash
pytest tests/test_spreads.py -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name '_bs_put' from 'lib.spreads'`

- [ ] **Step 3: Write lib/spreads.py**

```python
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

PROFIT_TARGETS: list[float] = [0.25, 0.50, 0.75]
STOP_MULTIPLES: list[int] = [1, 2, 3]
RISK_FREE_RATE: float = 0.05
SPREAD_WIDTH_ATR: float = 0.5
DTE: int = 1


def _bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def price_spread(
    S: float,
    K_short: float,
    atr: float,
    direction: str,
    vix: float,
    r: float = RISK_FREE_RATE,
    dte: int = DTE,
    width_atr: float = SPREAD_WIDTH_ATR,
) -> dict:
    """Price a credit spread using Black-Scholes.

    Args:
        S: Entry price (underlying at time of entry).
        K_short: Short strike = prev_close (ATM for the spread).
        atr: ATR value for computing spread width.
        direction: 'gap_up' → bull put spread; 'gap_down' → bear call spread.
        vix: VIX close, used as annualized IV proxy (vix / 100).
        r: Risk-free rate.
        dte: Days to expiration.
        width_atr: Spread width as multiple of ATR.

    Returns:
        dict with keys: credit, spread_width, K_short, K_long, sigma.

    Note: All P&L values from this function are SYNTHETIC (Black-Scholes + VIX proxy).
    """
    sigma = vix / 100.0
    T = dte / 252.0
    spread_width = width_atr * atr

    if direction == "gap_up":
        K_long = K_short - spread_width
        credit = _bs_put(S, K_short, T, r, sigma) - _bs_put(S, K_long, T, r, sigma)
    else:
        K_long = K_short + spread_width
        credit = _bs_call(S, K_short, T, r, sigma) - _bs_call(S, K_long, T, r, sigma)

    return {
        "credit": max(credit, 0.0),
        "spread_width": spread_width,
        "K_short": K_short,
        "K_long": K_long,
        "sigma": sigma,
    }


def simulate_exits(
    path_df: pd.DataFrame,
    profit_targets: Sequence[float] = PROFIT_TARGETS,
    stop_multiples: Sequence[int] = STOP_MULTIPLES,
    r: float = RISK_FREE_RATE,
    dte: int = DTE,
    width_atr: float = SPREAD_WIDTH_ATR,
) -> pd.DataFrame:
    """Simulate exit strategies for each row in path_df.

    Each event × entry_time row produces N exit rows
    (1 EOD + len(profit_targets) PTs + len(stop_multiples) SLs).

    Note: P&L values are SYNTHETIC — Black-Scholes + VIX/100 as IV proxy.
    """
    records: list[dict] = []

    for _, row in path_df.iterrows():
        vix = float(row.get("vix_close", 20.0))
        if np.isnan(vix):
            vix = 20.0

        sp = price_spread(
            S=float(row["entry_price"]),
            K_short=float(row["prev_close"]),
            atr=float(row["atr"]),
            direction=str(row["direction"]),
            vix=vix,
            r=r,
            dte=dte,
            width_atr=width_atr,
        )
        credit = sp["credit"]
        width = sp["spread_width"]
        mae = float(row["mae"])
        eod_close = float(row["eod_close"])
        prev_close = float(row["prev_close"])
        direction = str(row["direction"])

        # EOD P&L: intrinsic value of short option at close
        if direction == "gap_up":
            intrinsic = min(max(prev_close - eod_close, 0.0), width)
        else:
            intrinsic = min(max(eod_close - prev_close, 0.0), width)
        eod_pnl = credit - intrinsic

        def _record(exit_name: str, pnl: float) -> dict:
            return {
                "date": row["date"],
                "ticker": row.get("ticker", ""),
                "direction": direction,
                "entry_time": row["entry_time"],
                "exit_strategy": exit_name,
                "credit": credit,
                "spread_width": width,
                "sigma": sp["sigma"],
                "mae": mae,
                "pnl": pnl,
                "win": bool(pnl > 0),
                "gap_ratio": float(row.get("gap_ratio", np.nan)),
                "premarket_fill_pct": float(row.get("premarket_fill_pct", np.nan)),
                "vix_close": vix,
                "gap_filled": bool(row.get("gap_filled", False)),
            }

        records.append(_record("EOD", eod_pnl))

        for pt in profit_targets:
            target_hit = (mae < (1 - pt) * width) if width > 0 else False
            pnl = pt * credit if target_hit else eod_pnl
            records.append(_record(f"PT{int(pt * 100)}", pnl))

        for sl in stop_multiples:
            stop_level = sl * credit
            pnl = -stop_level if mae > stop_level else eod_pnl
            records.append(_record(f"SL{sl}x", pnl))

    return pd.DataFrame(records)


def compute_metrics(exits_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate win rate, expectancy, and profit factor per exit × entry_time × direction."""
    records: list[dict] = []
    group_cols = ["exit_strategy", "entry_time", "direction"]

    for keys, group in exits_df.groupby(group_cols, observed=True):
        exit_strat, entry_time, direction = keys
        wins = group[group["win"]]
        losses = group[~group["win"]]
        n = len(group)
        win_rate = len(wins) / n
        avg_win = float(wins["pnl"].mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses["pnl"].mean()) if len(losses) > 0 else 0.0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        total_loss = float(losses["pnl"].sum())
        profit_factor = float(wins["pnl"].sum()) / abs(total_loss) if total_loss != 0 else np.inf
        avg_credit = float(group["credit"].mean())

        records.append(
            {
                "exit_strategy": exit_strat,
                "entry_time": entry_time,
                "direction": direction,
                "n_trades": n,
                "win_rate": round(win_rate, 3),
                "avg_win": round(avg_win, 4),
                "avg_loss": round(avg_loss, 4),
                "expectancy": round(expectancy, 4),
                "expectancy_per_dollar": round(expectancy / avg_credit, 3) if avg_credit > 0 else 0.0,
                "profit_factor": round(profit_factor, 2),
                "total_pnl": round(float(group["pnl"].sum()), 4),
                "mae_p50": round(float(group["mae"].quantile(0.50)), 4),
                "mae_p90": round(float(group["mae"].quantile(0.90)), 4),
                "mae_p99": round(float(group["mae"].quantile(0.99)), 4),
                "mae_worst": round(float(group["mae"].max()), 4),
            }
        )

    return pd.DataFrame(records).sort_values(group_cols).reset_index(drop=True)
```

- [ ] **Step 4: Run all tests — expect full suite to PASS**

```bash
pytest tests/ -v
```

Expected: `all tests passed` (events + path + spreads)

- [ ] **Step 5: Commit**

```bash
git add lib/spreads.py tests/test_spreads.py
git commit -m "feat: Black-Scholes spread pricing, exit simulation, metrics"
```

---

## Task 6: Notebook — Sections 0, 1, 2 (Setup, Data, Gap Characterization)

**Files:**
- Create: `gap_spread_research.ipynb` (use nbformat)

- [ ] **Step 1: Write notebook builder script and run it**

Save this as `build_notebook.py` then run it. Each task appends cells to the notebook.

```python
# build_notebook.py
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
}


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(src):
    return nbf.v4.new_code_cell(src)


# ── Section 0: Setup & Config ──────────────────────────────────────────────
nb.cells.append(md("""# Gap-Based Credit Spread Research

**Hypothesis:** After overnight gaps ≥ 1 ATR in equity index ETFs, selling short-dated credit spreads
anchored at the previous day's close may yield positive expectancy due to mean reversion or limited
immediate trend continuation.

**Data:** yfinance 5m intraday (~2 years) + daily OHLCV (10+ years) + VIX.
**Assets:** SPY, QQQ, IWM (primary); ES=F, NQ=F (secondary — roll gap caveat applies).

**Two-layer design:**
- **Layer 1 — Price Path:** pure price analysis, no synthetic assumptions.
- **Layer 2 — Credit Spread [Synthetic]:** Black-Scholes + VIX/100 as IV proxy. Labeled throughout.

---"""))

nb.cells.append(code("""\
import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from lib.data import fetch, split_session
from lib.events import detect_events
from lib.path import analyze_paths, label_vix_regime
from lib.spreads import simulate_exits, compute_metrics

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
CHARTS_DIR = Path("charts")
CHARTS_DIR.mkdir(exist_ok=True)\
"""))

nb.cells.append(code("""\
# ── Configuration (edit here to re-run with different parameters) ──────────
TICKERS_PRIMARY   = ["SPY", "QQQ", "IWM"]
TICKERS_SECONDARY = ["ES=F", "NQ=F"]
ALL_TICKERS       = TICKERS_PRIMARY + TICKERS_SECONDARY
VIX_TICKER        = "^VIX"

ATR_PERIOD        = 14
GAP_THRESHOLD     = 1.0      # minimum |gap| in ATR units
SPREAD_WIDTH_ATR  = 0.5      # wing width as fraction of ATR
DTE               = 1        # days to expiration for spread pricing
RISK_FREE_RATE    = 0.05

PROFIT_TARGETS    = [0.25, 0.50, 0.75]
STOP_MULTIPLES    = [1, 2, 3]
ENTRY_TIMES       = ["09:30", "09:45", "10:00", "10:30"]
VIX_REGIMES       = {"low": (0, 15), "mid": (15, 25), "high": (25, 999)}

COLORS_DIR = {"gap_up": "#2ca02c", "gap_down": "#d62728"}
COLORS_REGIME = {"low": "#1f77b4", "mid": "#ff7f0e", "high": "#d62728"}\
"""))

# ── Section 1: Data ────────────────────────────────────────────────────────
nb.cells.append(md("""## 1. Data

Loading daily and 5m intraday data. First run fetches from yfinance and writes parquet cache;
subsequent runs load from cache instantly. Set `force_refresh=True` to re-fetch."""))

nb.cells.append(code("""\
print("Loading daily data...")
daily   = {t: fetch(t, "1d") for t in ALL_TICKERS}
vix_daily = fetch(VIX_TICKER, "1d")

print("Loading 5m intraday data (may take a minute on first run)...")
intraday = {t: fetch(t, "5m") for t in ALL_TICKERS}
print("Done.")

# Split into pre-market and session bars
premarkets = {}
sessions   = {}
for t in ALL_TICKERS:
    pm, sess = split_session(intraday[t])
    premarkets[t] = pm
    sessions[t]   = sess

print("\\nSample counts:")
for t in ALL_TICKERS:
    print(f"  {t}: {len(daily[t])} daily bars, "
          f"{len(premarkets[t])} pre-market 5m bars, "
          f"{len(sessions[t])} session 5m bars")\
"""))

nb.cells.append(code("""\
print("Detecting gap events...")
events_by_ticker = {}
for t in ALL_TICKERS:
    ev = detect_events(
        daily[t], premarkets[t], vix_daily,
        threshold=GAP_THRESHOLD, atr_period=ATR_PERIOD,
    )
    ev["ticker"] = t
    events_by_ticker[t] = ev

all_events = pd.concat(events_by_ticker.values())

summary = all_events.groupby(["ticker", "direction"]).size().unstack(fill_value=0)
print(f"\\nTotal gap events: {len(all_events)}")
display(summary)\
"""))

# ── Section 2: Gap Event Characterization ─────────────────────────────────
nb.cells.append(md("""## 2. Gap Event Characterization

Before simulating trades, we characterize the events themselves: how large are gaps,
how often do they occur in each direction, and how much of the gap is already retracing
in pre-market before the 9:30 open."""))

nb.cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Panel 1: Gap ratio distribution by direction
for direction, color in COLORS_DIR.items():
    subset = all_events[all_events["direction"] == direction]["gap_ratio"].abs()
    axes[0].hist(subset, bins=30, alpha=0.6, color=color, label=direction, density=True)
axes[0].set_xlabel("|Gap| / ATR(14)")
axes[0].set_ylabel("Density")
axes[0].set_title("Gap Size Distribution")
axes[0].legend()

# Panel 2: Direction frequency per ticker
counts = all_events.groupby(["ticker", "direction"]).size().unstack(fill_value=0)
counts.plot(kind="bar", ax=axes[1], color=[COLORS_DIR.get(c, "gray") for c in counts.columns],
            rot=0)
axes[1].set_title("Event Frequency by Ticker")
axes[1].set_xlabel("")
axes[1].set_ylabel("Count")

# Panel 3: Pre-market fill % distribution
all_events["premarket_fill_pct"].dropna().plot(
    kind="hist", bins=25, ax=axes[2], color="#7f7f7f", edgecolor="white"
)
axes[2].set_xlabel("Pre-market Fill %")
axes[2].set_title("Pre-market Gap Fill Distribution")

plt.tight_layout()
plt.savefig(CHARTS_DIR / "01_gap_characterization.png", dpi=150, bbox_inches="tight")
plt.show()\
"""))

nb.cells.append(code("""\
# VIX distribution at event time
fig, ax = plt.subplots(figsize=(10, 4))
for direction, color in COLORS_DIR.items():
    subset = all_events[all_events["direction"] == direction]["vix_close"].dropna()
    ax.hist(subset, bins=25, alpha=0.6, color=color, label=direction, density=True)
ax.set_xlabel("VIX at Event Date")
ax.set_ylabel("Density")
ax.set_title("VIX Distribution at Gap Events")
ax.axvline(15, color="blue", linestyle="--", alpha=0.5, label="Low/Mid boundary (15)")
ax.axvline(25, color="red", linestyle="--", alpha=0.5, label="Mid/High boundary (25)")
ax.legend()
plt.tight_layout()
plt.savefig(CHARTS_DIR / "02_vix_at_events.png", dpi=150, bbox_inches="tight")
plt.show()

vix_regime_counts = all_events.copy()
vix_regime_counts["regime"] = label_vix_regime(vix_regime_counts["vix_close"].fillna(20))
print("Events by VIX regime:")
display(vix_regime_counts.groupby(["regime", "direction"]).size().unstack(fill_value=0))\
"""))

with open("gap_spread_research.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written (Sections 0-2).")
```

- [ ] **Step 2: Run the builder**

```bash
cd ~/Documents/gap-spread-research && python build_notebook.py
```

Expected: `Notebook written (Sections 0-2).`

- [ ] **Step 3: Verify notebook opens**

```bash
jupyter nbconvert --to notebook --execute gap_spread_research.ipynb --output gap_spread_research_test.ipynb 2>&1 | tail -5
```

Expected: notebook executes without errors; `gap_spread_research_test.ipynb` created. Delete the test file after verifying:
```bash
rm -f gap_spread_research_test.ipynb
```

- [ ] **Step 4: Commit**

```bash
git add gap_spread_research.ipynb build_notebook.py
git commit -m "feat: notebook sections 0-2 (setup, data, gap characterization)"
```

---

## Task 7: Notebook — Section 3 (Price-Path Analysis, Layer 1)

**Files:**
- Modify: `build_notebook.py` (append cells before `nbf.write`)
- Rebuild: `gap_spread_research.ipynb`

- [ ] **Step 1: Append Section 3 cells to build_notebook.py**

Add this block immediately before the final `with open(...)` / `nbf.write(nb, f)` lines:

```python
# ── Section 3: Price-Path Analysis (Layer 1) ──────────────────────────────
nb.cells.append(md("""## 3. Price-Path Analysis — Layer 1

**No synthetic assumptions.** All results derived from 5m price bars only.

We compute gap fill rate, time-to-fill, MAE, and repeat across four entry times
(09:30, 09:45, 10:00, 10:30) to find whether waiting improves the edge."""))

nb.cells.append(code("""\
print("Computing price paths (this takes ~1-2 minutes for all tickers)...")
paths_by_ticker = {}
for t in ALL_TICKERS:
    print(f"  {t}...", end=" ", flush=True)
    paths_by_ticker[t] = analyze_paths(
        events_by_ticker[t], sessions[t], entry_times=ENTRY_TIMES
    )
    paths_by_ticker[t]["ticker"] = t
    print(f"{len(paths_by_ticker[t])} rows")

all_paths = pd.concat(paths_by_ticker.values(), ignore_index=True)
all_paths["vix_regime"] = label_vix_regime(all_paths["vix_close"].fillna(20))
print(f"\\nTotal path rows: {len(all_paths)}  ({len(all_paths) // len(ENTRY_TIMES)} events × {len(ENTRY_TIMES)} entry times)")\
"""))

nb.cells.append(md("""### 3a. Gap Fill Rate & Time-to-Fill"""))

nb.cells.append(code("""\
open_paths = all_paths[all_paths["entry_time"] == "09:30"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Fill rate by ticker and direction
fill_rate = open_paths.groupby(["ticker", "direction"])["gap_filled"].mean().unstack()
fill_rate.plot(kind="bar", ax=axes[0], color=[COLORS_DIR.get(c, "gray") for c in fill_rate.columns], rot=0)
axes[0].set_title("Gap Fill Rate by Ticker (Entry: Open)")
axes[0].set_ylabel("Fill Rate")
axes[0].set_ylim(0, 1)
axes[0].axhline(0.5, color="black", linestyle="--", alpha=0.4, label="50%")
axes[0].legend()

# Time-to-fill distribution (filled events only)
filled = open_paths[open_paths["gap_filled"]]["fill_time"].dropna()
axes[1].hist(filled, bins=30, color="#1f77b4", edgecolor="white")
axes[1].set_xlabel("Minutes from Open to Fill")
axes[1].set_title("Time-to-Fill Distribution (Filled Events Only)")
axes[1].axvline(filled.median(), color="red", linestyle="--", label=f"Median: {filled.median():.0f} min")
axes[1].legend()

plt.tight_layout()
plt.savefig(CHARTS_DIR / "03_fill_rate_time.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Overall fill rate (open entry): {open_paths['gap_filled'].mean():.1%}")
print(f"Median time to fill: {filled.median():.0f} minutes")\
"""))

nb.cells.append(md("""### 3b. MAE Distribution"""))

nb.cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for direction, color in COLORS_DIR.items():
    subset = open_paths[open_paths["direction"] == direction]["mae_pct_atr"].dropna()
    axes[0].hist(subset, bins=30, alpha=0.6, color=color, label=direction, density=True)
axes[0].set_xlabel("MAE / ATR(14)")
axes[0].set_title("MAE Distribution (in ATR units)")
axes[0].legend()

# Tail statistics table
tail_stats = open_paths.groupby("direction")["mae_pct_atr"].describe(
    percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
).round(3)
axes[1].axis("off")
tbl = axes[1].table(
    cellText=tail_stats.values,
    rowLabels=tail_stats.index,
    colLabels=tail_stats.columns,
    loc="center",
    cellLoc="center",
)
tbl.scale(1, 1.5)
axes[1].set_title("MAE / ATR Tail Statistics")

plt.tight_layout()
plt.savefig(CHARTS_DIR / "04_mae_distribution.png", dpi=150, bbox_inches="tight")
plt.show()\
"""))

nb.cells.append(md("""### 3c. Entry Timing Grid

Does waiting 15–60 minutes before entering reduce MAE or improve fill rate?"""))

nb.cells.append(code("""\
# Heat map: fill rate × entry_time × ticker
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, metric, label in [
    (axes[0], "gap_filled", "Gap Fill Rate"),
    (axes[1], "mae_pct_atr", "MAE / ATR (mean)"),
]:
    pivot = all_paths.groupby(["ticker", "entry_time"])[metric].mean().unstack()
    pivot = pivot[ENTRY_TIMES]  # ensure consistent column order
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn" if "fill" in metric else "RdYlGn_r")
    ax.set_xticks(range(len(ENTRY_TIMES)))
    ax.set_xticklabels(ENTRY_TIMES)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{label} by Entry Time")
    plt.colorbar(im, ax=ax)
    for i in range(len(pivot.index)):
        for j in range(len(ENTRY_TIMES)):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(CHARTS_DIR / "05_entry_timing_grid.png", dpi=150, bbox_inches="tight")
plt.show()\
"""))

nb.cells.append(md("""### 3d. Pre-market Fill % as Predictor

Does a gap that's already partially filling in pre-market revert faster/more reliably?"""))

nb.cells.append(code("""\
open_paths_pm = open_paths.dropna(subset=["premarket_fill_pct"])
open_paths_pm = open_paths_pm.copy()
open_paths_pm["pm_quartile"] = pd.qcut(
    open_paths_pm["premarket_fill_pct"], q=4,
    labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Fill rate by quartile
fill_by_q = open_paths_pm.groupby("pm_quartile", observed=True)["gap_filled"].mean()
fill_by_q.plot(kind="bar", ax=axes[0], color="#1f77b4", rot=30)
axes[0].set_title("Gap Fill Rate by Pre-market Fill % Quartile")
axes[0].set_ylabel("Fill Rate")
axes[0].set_ylim(0, 1)
axes[0].axhline(0.5, color="red", linestyle="--", alpha=0.4)

# MAE by quartile
mae_by_q = open_paths_pm.groupby("pm_quartile", observed=True)["mae_pct_atr"].mean()
mae_by_q.plot(kind="bar", ax=axes[1], color="#ff7f0e", rot=30)
axes[1].set_title("Mean MAE/ATR by Pre-market Fill % Quartile")
axes[1].set_ylabel("MAE / ATR (mean)")

plt.tight_layout()
plt.savefig(CHARTS_DIR / "06_premarket_predictor.png", dpi=150, bbox_inches="tight")
plt.show()\
"""))

nb.cells.append(md("""### 3e. VIX Regime Breakdown"""))

nb.cells.append(code("""\
regime_summary = open_paths.groupby(["vix_regime", "direction"]).agg(
    n_events=("gap_filled", "count"),
    fill_rate=("gap_filled", "mean"),
    mae_mean=("mae_pct_atr", "mean"),
    mae_p90=("mae_pct_atr", lambda x: x.quantile(0.90)),
).round(3)
print("Price-path metrics by VIX regime:")
display(regime_summary)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric, label in [
    (axes[0], "fill_rate", "Fill Rate"),
    (axes[1], "mae_mean", "Mean MAE / ATR"),
]:
    pivot = regime_summary[metric].unstack()
    pivot.plot(kind="bar", ax=ax,
               color=[COLORS_DIR.get(c, "gray") for c in pivot.columns], rot=0)
    ax.set_title(f"{label} by VIX Regime")
    ax.set_xlabel("VIX Regime")
    ax.set_ylabel(label)

plt.tight_layout()
plt.savefig(CHARTS_DIR / "07_vix_regime.png", dpi=150, bbox_inches="tight")
plt.show()\
"""))
```

- [ ] **Step 2: Rebuild and verify notebook**

```bash
python build_notebook.py
jupyter nbconvert --to notebook --execute gap_spread_research.ipynb --output gap_spread_research_test.ipynb 2>&1 | tail -5
rm -f gap_spread_research_test.ipynb
```

Expected: executes without errors.

- [ ] **Step 3: Commit**

```bash
git add gap_spread_research.ipynb build_notebook.py
git commit -m "feat: notebook section 3 — price-path analysis (Layer 1)"
```

---

## Task 8: Notebook — Section 4 (Credit Spread Simulation, Layer 2)

**Files:**
- Modify: `build_notebook.py` (append cells before `nbf.write`)
- Rebuild: `gap_spread_research.ipynb`

- [ ] **Step 1: Append Section 4 cells to build_notebook.py**

Add before the final `with open(...)` / `nbf.write(nb, f)`:

```python
# ── Section 4: Credit Spread Simulation (Layer 2 — Synthetic) ─────────────
nb.cells.append(md("""## 4. Credit Spread Simulation — Layer 2 [Synthetic]

> ⚠️ **All P&L values in this section are SYNTHETIC.** Spreads are priced using
> Black-Scholes with VIX/100 as annualized IV proxy. There is no real options
> chain data — actual market prices will differ due to IV smile, bid/ask spread,
> liquidity, and early assignment risk.

For each gap event and entry time we:
1. Price a 1-DTE credit spread anchored at prev_close (width = 0.5 × ATR).
2. Simulate 7 exit strategies: EOD, PT25/50/75 (profit targets), SL1×/2×/3× (stop-loss).
3. Compute win rate, expectancy, and profit factor per combination."""))

nb.cells.append(code("""\
print("Simulating credit spreads...")
exits = simulate_exits(
    all_paths,
    profit_targets=PROFIT_TARGETS,
    stop_multiples=STOP_MULTIPLES,
)
print(f"Exit rows: {len(exits):,}  ({len(all_paths):,} path rows × 7 exit strategies)")
print(f"Sample credit (SPY, first event): ${exits[exits['ticker']=='SPY']['credit'].iloc[0]:.4f}")\
"""))

nb.cells.append(md("""### 4a. Spread Pricing at Each Entry Time"""))

nb.cells.append(code("""\
eod_exits = exits[exits["exit_strategy"] == "EOD"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Credit distribution by entry time
eod_exits.boxplot(column="credit", by="entry_time", ax=axes[0])
axes[0].set_title("Spread Credit by Entry Time [Synthetic]")
axes[0].set_xlabel("Entry Time")
axes[0].set_ylabel("Credit Received ($)")
plt.sca(axes[0])
plt.title("Spread Credit by Entry Time [Synthetic]")

# Credit vs gap_ratio scatter
axes[1].scatter(
    eod_exits["gap_ratio"].abs(), eod_exits["credit"],
    c=[COLORS_DIR.get(d, "gray") for d in eod_exits["direction"]],
    alpha=0.4, s=15,
)
axes[1].set_xlabel("|Gap| / ATR")
axes[1].set_ylabel("Credit ($) [Synthetic]")
axes[1].set_title("Credit vs Gap Magnitude")
from matplotlib.patches import Patch
axes[1].legend(handles=[
    Patch(color=COLORS_DIR["gap_up"], label="gap_up"),
    Patch(color=COLORS_DIR["gap_down"], label="gap_down"),
])

plt.tight_layout()
plt.savefig(CHARTS_DIR / "08_spread_pricing.png", dpi=150, bbox_inches="tight")
plt.show()\
"""))

nb.cells.append(md("""### 4b. Exit Strategy Comparison"""))

nb.cells.append(code("""\
metrics = compute_metrics(exits)

# Focus on open entry for clarity
metrics_open = metrics[metrics["entry_time"] == "09:30"].copy()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

EXIT_ORDER = ["EOD", "PT25", "PT50", "PT75", "SL1x", "SL2x", "SL3x"]

for direction in ["gap_up", "gap_down"]:
    m = metrics_open[metrics_open["direction"] == direction].set_index("exit_strategy")
    m = m.reindex(EXIT_ORDER).dropna()

    for ax, col, label in [
        (axes[0], "win_rate", "Win Rate"),
        (axes[1], "expectancy_per_dollar", "Expectancy / $1 Risked [Synthetic]"),
        (axes[2], "profit_factor", "Profit Factor [Synthetic]"),
    ]:
        ax.plot(
            m.index, m[col],
            marker="o", label=direction, color=COLORS_DIR[direction]
        )

for ax, title in zip(axes, ["Win Rate", "Expectancy / $ Risked", "Profit Factor"]):
    ax.set_title(f"{title}\n(Entry: Open)")
    ax.set_xlabel("Exit Strategy")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.axhline(0 if "Expectancy" in title else 1, color="gray", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(CHARTS_DIR / "09_exit_comparison.png", dpi=150, bbox_inches="tight")
plt.show()\
"""))

nb.cells.append(md("""### 4c. Expectancy by Exit × Entry Time"""))

nb.cells.append(code("""\
# Summary table: expectancy_per_dollar for all exit × entry_time combinations
pivot_exp = metrics[metrics["direction"] == "gap_up"].pivot_table(
    index="exit_strategy", columns="entry_time", values="expectancy_per_dollar"
)
pivot_exp = pivot_exp.reindex(EXIT_ORDER).reindex(columns=ENTRY_TIMES)

print("[Synthetic] Gap Up — Expectancy per $1 Risked:")
display(pivot_exp.round(3).style.background_gradient(cmap="RdYlGn", axis=None))

pivot_exp_dn = metrics[metrics["direction"] == "gap_down"].pivot_table(
    index="exit_strategy", columns="entry_time", values="expectancy_per_dollar"
)
pivot_exp_dn = pivot_exp_dn.reindex(EXIT_ORDER).reindex(columns=ENTRY_TIMES)

print("\\n[Synthetic] Gap Down — Expectancy per $1 Risked:")
display(pivot_exp_dn.round(3).style.background_gradient(cmap="RdYlGn", axis=None))\
"""))

nb.cells.append(md("""### 4d. Tail Risk — MAE Distribution and Worst Cases"""))

nb.cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MAE distribution vs spread width
eod_exits_open = eod_exits[eod_exits["entry_time"] == "09:30"].copy()
eod_exits_open["mae_as_pct_width"] = eod_exits_open["mae"] / eod_exits_open["spread_width"]

for direction, color in COLORS_DIR.items():
    subset = eod_exits_open[eod_exits_open["direction"] == direction]["mae_as_pct_width"]
    axes[0].hist(subset, bins=30, alpha=0.6, color=color, label=direction, density=True)
axes[0].axvline(1.0, color="black", linestyle="--", label="Strike breach (MAE = width)")
axes[0].set_xlabel("MAE / Spread Width")
axes[0].set_title("MAE as Fraction of Spread Width [Synthetic]")
axes[0].legend()

# Tail loss CDF
for direction, color in COLORS_DIR.items():
    subset = eod_exits_open[eod_exits_open["direction"] == direction]["pnl"].sort_values()
    axes[1].plot(subset.values, np.linspace(0, 1, len(subset)), color=color, label=direction)
axes[1].axvline(0, color="black", linestyle="--", alpha=0.5)
axes[1].set_xlabel("P&L ($) [Synthetic]")
axes[1].set_ylabel("CDF")
axes[1].set_title("P&L CDF — EOD Exit [Synthetic]")
axes[1].legend()

plt.tight_layout()
plt.savefig(CHARTS_DIR / "10_tail_risk.png", dpi=150, bbox_inches="tight")
plt.show()

worst = eod_exits_open.nsmallest(5, "pnl")[["date", "ticker", "direction", "mae", "pnl", "gap_ratio"]]
print("\\nFive worst EOD outcomes [Synthetic]:")
display(worst.round(4))\
"""))
```

- [ ] **Step 2: Rebuild and verify**

```bash
python build_notebook.py
jupyter nbconvert --to notebook --execute gap_spread_research.ipynb --output gap_spread_research_test.ipynb 2>&1 | tail -5
rm -f gap_spread_research_test.ipynb
```

Expected: executes without errors.

- [ ] **Step 3: Commit**

```bash
git add gap_spread_research.ipynb build_notebook.py
git commit -m "feat: notebook section 4 — credit spread simulation (Layer 2, synthetic)"
```

---

## Task 9: Notebook — Sections 5 & 6 (Multi-Asset + Findings) + Final Polish

**Files:**
- Modify: `build_notebook.py` (append final cells)
- Rebuild: `gap_spread_research.ipynb`
- Delete: `build_notebook.py` (not needed after notebook is final)

- [ ] **Step 1: Append Sections 5 and 6 to build_notebook.py**

Add before the final `with open(...)` / `nbf.write(nb, f)`:

```python
# ── Section 5: Multi-Asset Comparison ─────────────────────────────────────
nb.cells.append(md("""## 5. Multi-Asset Comparison

SPY, QQQ, IWM are the primary assets. ES=F and NQ=F are included as secondary
(note: continuous contract roll gaps may inflate event counts for futures)."""))

nb.cells.append(code("""\
open_paths_all = all_paths[all_paths["entry_time"] == "09:30"]
metrics_open_all = metrics[metrics["entry_time"] == "09:30"]

# Fill rate and MAE by ticker
fill_mae = open_paths_all.groupby("ticker").agg(
    n_events=("gap_filled", "count"),
    fill_rate=("gap_filled", "mean"),
    mae_mean=("mae_pct_atr", "mean"),
    mae_p90=("mae_pct_atr", lambda x: x.quantile(0.90)),
).round(3)
print("Fill rate and MAE by ticker (entry: open):")
display(fill_mae)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

fill_mae["fill_rate"].plot(kind="bar", ax=axes[0], color="#1f77b4", rot=30)
axes[0].set_title("Gap Fill Rate by Ticker")
axes[0].set_ylabel("Fill Rate")
axes[0].set_ylim(0, 1)

fill_mae["mae_mean"].plot(kind="bar", ax=axes[1], color="#ff7f0e", rot=30)
axes[1].set_title("Mean MAE / ATR by Ticker")
axes[1].set_ylabel("MAE / ATR")

# Expectancy EOD by ticker [Synthetic]
eod_metrics_by_ticker = exits[
    (exits["exit_strategy"] == "EOD") & (exits["entry_time"] == "09:30")
].groupby("ticker").agg(
    expectancy=("pnl", "mean"),
    win_rate=("win", "mean"),
).round(3)
eod_metrics_by_ticker["expectancy"].plot(kind="bar", ax=axes[2], color="#2ca02c", rot=30)
axes[2].set_title("EOD Expectancy by Ticker [Synthetic]")
axes[2].set_ylabel("Mean P&L ($)")
axes[2].axhline(0, color="black", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(CHARTS_DIR / "11_multi_asset.png", dpi=150, bbox_inches="tight")
plt.show()\
"""))

# ── Section 6: Key Findings & Limitations ─────────────────────────────────
nb.cells.append(md("""## 6. Key Findings & Limitations

### Findings

_(Run the notebook and fill in with actual results — placeholder headings below.)_

**Price-Path (Layer 1 — no synthetic assumptions):**
- Overall gap fill rate and how it varies by ticker and direction
- Whether waiting (09:45/10:00/10:30) improves fill rate or reduces MAE
- Pre-market fill % quartile effect on intraday behavior
- VIX regime differences (high-VIX gaps: larger MAE, different fill rate?)

**Credit Spread [Layer 2 — Synthetic]:**
- Best exit strategy by expectancy/dollar risked
- Whether profit targets or stop-losses materially improve results
- Tail risk: worst-case scenarios and MAE distribution vs. spread width

---

### Limitations

1. **Synthetic options pricing.** All P&L values use Black-Scholes with VIX/100 as IV proxy.
   Real spreads have IV smile, bid/ask spread (~$0.05–0.15 per spread), liquidity constraints,
   and early assignment risk (American-style options). Actual P&L will differ.

2. **~2-year 5m sample.** yfinance provides ~730 days of 5m bars. Results may not generalize
   across different macro regimes. Daily-data analysis (Layer 1 summary) uses 10+ years.

3. **No transaction costs or slippage.** Commissions (~$0.65/contract) and slippage are excluded.

4. **No earnings filter.** Gaps on earnings days behave differently; filtering them may
   strengthen or weaken the hypothesis.

5. **ES=F / NQ=F roll gaps.** Continuous contract price series have artificial gaps at
   roll dates — futures event counts should be treated as indicative only.

---

### Swing Trainer Integration Map

When this research confirms edge worth trading, the `lib/` modules port directly:

| Research | Swing Trainer |
|----------|---------------|
| `lib/data.py` | `backend/services/data.py` (extend existing yfinance provider) |
| `lib/events.py` | `backend/services/gap_events.py` (new service) |
| `lib/path.py` | `backend/services/gap_path.py` (new service) |
| `lib/spreads.py` | `backend/services/options.py` (extend `YFinanceOptionsProvider`) |

**Suggested integration phase:** `GET /api/gaps/events` returns today's qualifying gaps,
`GET /api/gaps/analysis` returns metrics for the chosen ticker. New "Gap Scanner" tab
in the Bull Assistant UI."""))
```

- [ ] **Step 2: Rebuild notebook one final time**

```bash
python build_notebook.py
```

Expected: `Notebook written (Sections 0-2).` (message from original scaffold — final notebook has all 6 sections).

- [ ] **Step 3: Execute full notebook end-to-end**

```bash
jupyter nbconvert --to notebook --execute gap_spread_research.ipynb \
  --output gap_spread_research.ipynb \
  --ExecutePreprocessor.timeout=300 2>&1 | tail -10
```

Expected: notebook executes completely with all output cells populated.

- [ ] **Step 4: Verify charts directory has output files**

```bash
ls charts/
```

Expected: `01_gap_characterization.png 02_vix_at_events.png 03_fill_rate_time.png 04_mae_distribution.png 05_entry_timing_grid.png 06_premarket_predictor.png 07_vix_regime.png 08_spread_pricing.png 09_exit_comparison.png 10_tail_risk.png 11_multi_asset.png`

- [ ] **Step 5: Run full test suite one last time**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Final commit**

```bash
git add gap_spread_research.ipynb build_notebook.py charts/
git commit -m "feat: notebook complete — sections 5-6 multi-asset comparison and findings"
```

---

## Self-Review Notes

**Spec coverage check:**

| Spec requirement | Task |
|-----------------|------|
| Two-layer design (price-path + synthetic options) | Tasks 4, 5, 7, 8 |
| lib/data.py with parquet cache + prepost=True | Task 2 |
| lib/events.py ATR(14), gap detection, premarket_fill_pct | Task 3 |
| lib/path.py MAE, fill time, entry timing grid | Task 4 |
| lib/spreads.py Black-Scholes, exits, metrics | Task 5 |
| Gap event characterization (Section 2) | Task 6 |
| Fill rate, time-to-fill, MAE (Section 3a-b) | Task 7 |
| Entry timing grid heat map (Section 3c) | Task 7 |
| Pre-market fill % as predictor (Section 3d) | Task 7 |
| VIX regime breakdown (Section 3e) | Task 7 |
| Spread pricing at each entry time (Section 4a) | Task 8 |
| Exit strategy comparison (Section 4b) | Task 8 |
| Expectancy × exit × entry time table (Section 4c) | Task 8 |
| Tail risk analysis (Section 4d) | Task 8 |
| Multi-asset comparison (Section 5) | Task 9 |
| Findings + limitations + integration map (Section 6) | Task 9 |
| Swing Trainer integration map documented | Spec + Task 9 |
| charts/ directory with exported figures | Tasks 7, 8, 9 |
| requirements.txt | Task 1 |
| All tests pass | Tasks 3, 4, 5 |

All spec requirements covered. ✓
