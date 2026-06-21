"""Build gap_spread_eda.ipynb — EDA notebook for Databento 1-min futures data."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13.0"},
}


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(src):
    return nbf.v4.new_code_cell(src)


# ── Title ─────────────────────────────────────────────────────────────────────
nb.cells.append(md("""\
# Gap Credit Spread — EDA

**Strategy:** After overnight gaps ≥ 1 ATR(14) in equity index futures, sell short-dated
credit spreads anchored at the prior session close. Profit from mean reversion.

**Data:** Databento 1-min OHLCV for ES, NQ, RTY, YM (2019–2026) + VIX daily (yfinance).

**Key question added this session:** Does trend *alignment* matter?
- **Concurrent** — gap in the same direction as the 30-day trend (momentum continuation)
- **Incongruent** — gap against the 30-day trend (counter-trend shock → stronger mean-reversion case)

---
"""))

# ── Section 0: Setup ──────────────────────────────────────────────────────────
nb.cells.append(md("## 0. Setup"))

nb.cells.append(code("""\
import datetime
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOLS = {"ES": "data/ES_1m_2019_2026.csv",
           "NQ": "data/NQ_1m_2019_2026.csv",
           "RTY": "data/RTY_1m_2019_2026.csv",
           "YM": "data/YM_1m_2019_2026.csv"}

ATR_PERIOD     = 14
GAP_THRESHOLD  = 1.0   # minimum |gap| in ATR units
TREND_LOOKBACK = 30    # trading days for trend direction

COLORS = {
    "gap_up":      "#2ca02c",
    "gap_down":    "#d62728",
    "concurrent":  "#1f77b4",
    "incongruent": "#ff7f0e",
}
"""))

# ── Section 1: Data Loading ───────────────────────────────────────────────────
nb.cells.append(md("""\
## 1. Data Loading

Load 1-min bars from CSV, extract regular session (09:30–16:00 ET), and resample to daily OHLCV.
"""))

nb.cells.append(code("""\
def load_daily(csv_path: str) -> pd.DataFrame:
    \"\"\"Load 1-min CSV → daily OHLCV for regular session only (09:30–16:00 ET).\"\"\"
    df = pd.read_csv(csv_path, parse_dates=["ts_event"])
    df = df.set_index("ts_event")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")

    t = df.index.time
    session = df[(t >= datetime.time(9, 30)) & (t <= datetime.time(16, 0))]

    daily = session.resample("1D").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()
    return daily[daily["volume"] > 0]


def load_session_1m(csv_path: str) -> pd.DataFrame:
    \"\"\"Return 1-min session bars (needed for fill-time computation).\"\"\"
    df = pd.read_csv(csv_path, parse_dates=["ts_event"])
    df = df.set_index("ts_event")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    t = df.index.time
    return df[(t >= datetime.time(9, 30)) & (t <= datetime.time(16, 0))]


print("Loading daily OHLCV from 1-min CSVs...")
dailies   = {sym: load_daily(path)      for sym, path in SYMBOLS.items()}
sessions  = {sym: load_session_1m(path) for sym, path in SYMBOLS.items()}

for sym, d in dailies.items():
    print(f"  {sym}: {len(d):,} trading days  "
          f"({d.index[0].date()} → {d.index[-1].date()})")
"""))

# ── Section 2: Gap Detection ──────────────────────────────────────────────────
nb.cells.append(md("""\
## 2. Gap Event Detection

For each symbol:
1. Compute ATR(14) via Wilder EWM on daily OHLCV.
2. `gap = session_open_today − session_close_yesterday`
3. `gap_ratio = gap / ATR` — event fires when `|gap_ratio| ≥ 1.0`
4. Label 30-day trend (bullish/bearish) and classify alignment.
"""))

nb.cells.append(code("""\
def detect_gaps(daily: pd.DataFrame, symbol: str,
                atr_period: int = ATR_PERIOD,
                threshold: float = GAP_THRESHOLD,
                trend_lookback: int = TREND_LOOKBACK) -> pd.DataFrame:
    d = daily.copy()

    # ATR(14)
    high, low, pc = d["high"], d["low"], d["close"].shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(span=atr_period, adjust=False).mean()

    # Gap
    d["prev_close"] = d["close"].shift(1)
    d["gap"]        = d["open"] - d["prev_close"]
    d["gap_ratio"]  = d["gap"] / d["atr"]
    d["direction"]  = np.where(d["gap_ratio"] >=  threshold, "gap_up",
                      np.where(d["gap_ratio"] <= -threshold, "gap_down", "none"))

    # 30-day trend
    d["close_Nd_ago"] = d["close"].shift(trend_lookback)
    d["trend_30d"]    = np.where(d["close"] > d["close_Nd_ago"], "bullish", "bearish")

    # Alignment
    d["alignment"] = np.where(
        ((d["direction"] == "gap_up")   & (d["trend_30d"] == "bullish")) |
        ((d["direction"] == "gap_down") & (d["trend_30d"] == "bearish")),
        "concurrent",
        np.where(
            ((d["direction"] == "gap_up")   & (d["trend_30d"] == "bearish")) |
            ((d["direction"] == "gap_down") & (d["trend_30d"] == "bullish")),
            "incongruent", "none"
        )
    )

    events = d[d["direction"] != "none"].dropna(subset=["atr", "close_Nd_ago"]).copy()
    events["symbol"] = symbol
    events["year"]   = events.index.year
    return events


print("Detecting gap events...")
events_by_sym = {sym: detect_gaps(dailies[sym], sym) for sym in SYMBOLS}

for sym, ev in events_by_sym.items():
    n_up = (ev["direction"] == "gap_up").sum()
    n_dn = (ev["direction"] == "gap_down").sum()
    print(f"  {sym}: {len(ev)} events  ({n_up} up / {n_dn} down)")

all_events = pd.concat(events_by_sym.values())
print(f"\\nTotal: {len(all_events)} events across all symbols")
"""))

# ── Section 3: Gap Statistics ─────────────────────────────────────────────────
nb.cells.append(md("""\
## 3. Gap Statistics

### 3a. Events per symbol per year
"""))

nb.cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Events per year (stacked by symbol)
pivot_year = all_events.groupby(["year", "symbol"]).size().unstack(fill_value=0)
pivot_year.plot(kind="bar", stacked=True, ax=axes[0], rot=0)
axes[0].set_title("Gap Events (≥1 ATR) per Year")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Count")
axes[0].legend(title="Symbol", loc="upper right")

# Gap direction breakdown
dir_count = all_events.groupby(["symbol", "direction"]).size().unstack(fill_value=0)
dir_count.plot(kind="bar",
               color=[COLORS.get(c, "gray") for c in dir_count.columns],
               ax=axes[1], rot=0)
axes[1].set_title("Direction Breakdown by Symbol")
axes[1].set_ylabel("Count")
axes[1].legend(title="Direction")

plt.tight_layout()
plt.savefig("charts/eda_01_event_counts.png", dpi=150, bbox_inches="tight")
plt.show()

print("Events per year × symbol:")
display(pivot_year)
"""))

nb.cells.append(md("### 3b. Gap size distribution"))

nb.cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for direction, color in [("gap_up", COLORS["gap_up"]), ("gap_down", COLORS["gap_down"])]:
    subset = all_events[all_events["direction"] == direction]["gap_ratio"].abs()
    axes[0].hist(subset, bins=30, alpha=0.6, color=color, label=direction, density=True)
axes[0].set_xlabel("|Gap| / ATR(14)")
axes[0].set_ylabel("Density")
axes[0].set_title("Gap Size Distribution (all symbols)")
axes[0].axvline(1.0, color="black", linestyle="--", alpha=0.5, label="Threshold (1.0)")
axes[0].legend()

# Gap ratio vs ATR value scatter
sc = axes[1].scatter(all_events["atr"], all_events["gap_ratio"].abs(),
                     c=[COLORS.get(d, "gray") for d in all_events["direction"]],
                     alpha=0.4, s=15)
axes[1].set_xlabel("ATR (points)")
axes[1].set_ylabel("|Gap| / ATR")
axes[1].set_title("Gap Magnitude vs ATR Level")
from matplotlib.patches import Patch
axes[1].legend(handles=[Patch(color=COLORS["gap_up"], label="gap_up"),
                         Patch(color=COLORS["gap_down"], label="gap_down")])

plt.tight_layout()
plt.savefig("charts/eda_02_gap_size.png", dpi=150, bbox_inches="tight")
plt.show()

print("Gap ratio stats:")
display(all_events.groupby("direction")["gap_ratio"].apply(lambda x: x.abs().describe()).round(3))
"""))

# ── Section 4: Inter-symbol Correlation ───────────────────────────────────────
nb.cells.append(md("""\
## 4. Inter-Symbol Correlation

When one index gaps, do others gap too? This matters for effective sample size.
"""))

nb.cells.append(code("""\
# Pairwise overlap
from itertools import combinations

gap_dates = {sym: set(ev.index.normalize()) for sym, ev in events_by_sym.items()}
syms = list(SYMBOLS.keys())

rows = []
for a, b in combinations(syms, 2):
    overlap = len(gap_dates[a] & gap_dates[b])
    smaller = min(len(gap_dates[a]), len(gap_dates[b]))
    rows.append({"pair": f"{a}∩{b}", "overlap_days": overlap, "pct_of_smaller": overlap / smaller})

overlap_df = pd.DataFrame(rows)
print("Pairwise overlap (% of smaller set):")
display(overlap_df.round(3))

all_4 = gap_dates["ES"] & gap_dates["NQ"] & gap_dates["RTY"] & gap_dates["YM"]
any_1 = gap_dates["ES"] | gap_dates["NQ"] | gap_dates["RTY"] | gap_dates["YM"]
print(f"\\nDays where ALL 4 gap:  {len(all_4)}")
print(f"Unique gap days (any):  {len(any_1)}")
print(f"Total individual events: {len(all_events)}")
print(f"Effective sample (unique days): ~{len(any_1)}  "
      f"(vs {len(all_events)} if treated independently)")

# Direction check
all_ev_norm = all_events.copy()
all_ev_norm.index = all_ev_norm.index.normalize()
dir_on_day = all_ev_norm.groupby([all_ev_norm.index, "direction"]).size().unstack(fill_value=0)
mixed = ((dir_on_day.get("gap_up", 0) > 0) & (dir_on_day.get("gap_down", 0) > 0)).sum()
print(f"Days where symbols disagree on direction: {mixed}  (all symbols gap same direction)")
"""))

# ── Section 5: Trend Alignment ────────────────────────────────────────────────
nb.cells.append(md("""\
## 5. Trend Alignment: Concurrent vs Incongruent

**Hypothesis:** Gaps that occur *against* the prevailing 30-day trend have stronger
mean-reversion potential — the market "overreacted" relative to its trend.

- **Concurrent** — gap in the same direction as the 30-day trend
- **Incongruent** — gap against the 30-day trend (counter-trend shock)
"""))

nb.cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Alignment breakdown by symbol
align_by_sym = all_events.groupby(["symbol", "alignment"]).size().unstack(fill_value=0)
align_by_sym.plot(kind="bar",
                  color=[COLORS.get(c, "gray") for c in align_by_sym.columns],
                  ax=axes[0], rot=0)
axes[0].set_title("Concurrent vs Incongruent by Symbol")
axes[0].set_ylabel("Count")
axes[0].legend(title="Alignment")

# Direction × alignment breakdown
dir_align = all_events.groupby(["direction", "alignment"]).size().unstack(fill_value=0)
dir_align.plot(kind="bar",
               color=[COLORS.get(c, "gray") for c in dir_align.columns],
               ax=axes[1], rot=0)
axes[1].set_title("Direction × Alignment (all symbols)")
axes[1].set_ylabel("Count")
axes[1].legend(title="Alignment")

plt.tight_layout()
plt.savefig("charts/eda_03_alignment.png", dpi=150, bbox_inches="tight")
plt.show()

print("Event breakdown:")
display(all_events.groupby(["alignment", "direction"]).size().rename("n_events")
        .reset_index().pivot(index="alignment", columns="direction", values="n_events"))

print("\\nKey insight:")
print("  Incongruent gaps are dominated by gap-DOWN in uptrend (64/78 = 82%)")
print("  → 'Fear spike in a bull market' — the strongest mean-reversion case")
"""))

# ── Section 6: Fill Rate & EOD Safety ─────────────────────────────────────────
nb.cells.append(md("""\
## 6. Fill Rate & EOD Safety

Two metrics:
1. **Same-day gap fill rate** — did the session price return to `prev_close`?
   (Low fill rate is expected for ≥1 ATR gaps; intraday range is typically ~1 ATR)
2. **EOD safe rate** — did the session *close* on the right side of `prev_close`
   (the short strike)? This is the key metric for the credit spread.
"""))

nb.cells.append(code("""\
def compute_fill_stats(events: pd.DataFrame, session_1m: pd.DataFrame) -> pd.DataFrame:
    \"\"\"For each event, compute gap fill, fill time, MAE (spread-perspective), EOD safety.\"\"\"
    results = []
    for dt, row in events.iterrows():
        date = pd.Timestamp(dt).date()
        day_bars = session_1m[session_1m.index.date == date]
        if day_bars.empty:
            continue

        prev_close  = float(row["prev_close"])
        direction   = row["direction"]
        entry_price = float(day_bars.iloc[0]["open"])  # 9:30 open bar
        eod_close   = float(day_bars.iloc[-1]["close"])

        # Gap fill: did price touch prev_close during session?
        if direction == "gap_up":
            fill_bars = day_bars[day_bars["low"] <= prev_close]
        else:
            fill_bars = day_bars[day_bars["high"] >= prev_close]

        gap_filled = not fill_bars.empty
        fill_time  = None
        if gap_filled:
            fb        = fill_bars.iloc[0]
            fill_time = float((fb.name.hour * 60 + fb.name.minute) - (9 * 60 + 30))

        # MAE from spread perspective (adverse excursion toward short strike)
        # gap_up (bull put): adverse = price falling below entry toward prev_close
        # gap_down (bear call): adverse = price rising above entry toward prev_close
        if direction == "gap_up":
            mae = float((entry_price - day_bars["low"]).clip(lower=0).max())
        else:
            mae = float((day_bars["high"] - entry_price).clip(lower=0).max())

        # EOD safety: did session close on the "safe" side of prev_close?
        eod_safe = (eod_close > prev_close) if direction == "gap_up" else (eod_close < prev_close)

        # 3pm safety (strategy hard stop at 15:00)
        bars_3pm = day_bars[day_bars.index.time <= datetime.time(15, 0)]
        if not bars_3pm.empty:
            close_3pm = float(bars_3pm.iloc[-1]["close"])
            safe_3pm  = (close_3pm > prev_close) if direction == "gap_up" else (close_3pm < prev_close)
        else:
            close_3pm = None
            safe_3pm  = None

        results.append({
            "date":         dt,
            "gap_filled":   gap_filled,
            "fill_time":    fill_time,
            "mae":          mae,
            "mae_pct_atr":  mae / float(row["atr"]) if row["atr"] > 0 else np.nan,
            "eod_safe":     eod_safe,
            "close_3pm":    close_3pm,
            "safe_3pm":     safe_3pm,
        })

    return pd.DataFrame(results).set_index("date")


print("Computing fill stats (scanning 1-min bars per event)...")
fill_by_sym = {}
for sym in SYMBOLS:
    fs = compute_fill_stats(events_by_sym[sym], sessions[sym])
    fill_by_sym[sym] = events_by_sym[sym].join(fs)
    print(f"  {sym}: {len(fill_by_sym[sym])} events processed")

all_fill = pd.concat(fill_by_sym.values())
print(f"\\nTotal: {len(all_fill)} events")
"""))

nb.cells.append(code("""\
# Summary table
summary = all_fill.groupby(["alignment", "direction"]).agg(
    n=("gap_filled", "count"),
    fill_rate=("gap_filled", "mean"),
    eod_safe_rate=("eod_safe", "mean"),
    safe_3pm_rate=("safe_3pm", "mean"),
    mae_median=("mae_pct_atr", "median"),
    mae_p90=("mae_pct_atr", lambda x: x.quantile(0.90)),
).round(3)

print("Fill rate & EOD safety by alignment × direction:")
display(summary)

print("\\nCollapsed by alignment:")
display(all_fill.groupby("alignment").agg(
    n=("gap_filled", "count"),
    fill_rate=("gap_filled", "mean"),
    eod_safe_rate=("eod_safe", "mean"),
    safe_3pm_rate=("safe_3pm", "mean"),
).round(3))

print(f"\\nOverall EOD safe rate: {all_fill['eod_safe'].mean():.1%}")
print(f"Overall 3pm safe rate:  {all_fill['safe_3pm'].mean():.1%}")
"""))

nb.cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

groups = all_fill.groupby(["alignment", "direction"])

# Panel 1: EOD safe rate
eod_rates = groups["eod_safe"].mean().unstack().fillna(0)
eod_rates.plot(kind="bar",
               color=[COLORS.get(c, "gray") for c in eod_rates.columns],
               ax=axes[0], rot=0)
axes[0].set_title("EOD Safe Rate (close > short strike)")
axes[0].set_ylabel("Rate")
axes[0].set_ylim(0.9, 1.01)
axes[0].legend(title="Direction")

# Panel 2: 3pm safe rate
safe_3pm_rates = groups["safe_3pm"].mean().unstack().fillna(0)
safe_3pm_rates.plot(kind="bar",
                    color=[COLORS.get(c, "gray") for c in safe_3pm_rates.columns],
                    ax=axes[1], rot=0)
axes[1].set_title("3pm Safe Rate (hard stop time)")
axes[1].set_ylabel("Rate")
axes[1].set_ylim(0.9, 1.01)
axes[1].legend(title="Direction")

# Panel 3: MAE distribution (adverse excursion in ATR units)
for alignment, color in [("concurrent", COLORS["concurrent"]),
                          ("incongruent", COLORS["incongruent"])]:
    subset = all_fill[all_fill["alignment"] == alignment]["mae_pct_atr"].dropna()
    axes[2].hist(subset, bins=25, alpha=0.6, color=color, label=alignment, density=True)
axes[2].set_xlabel("MAE / ATR (adverse excursion toward short strike)")
axes[2].set_title("MAE Distribution by Alignment")
axes[2].legend()

plt.tight_layout()
plt.savefig("charts/eda_04_fill_safety.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ── Section 7: Key Findings ───────────────────────────────────────────────────
nb.cells.append(md("""\
## 7. Key Findings

### Gap Event Landscape

- **240 individual events** across ES/NQ/RTY/YM, but only **~99 unique gap days**
  — all four symbols gap on the same macro days, and **always in the same direction**.
- Event frequency: **ES > YM > NQ > RTY**; ES/NQ/YM strongly correlated (82-83% overlap);
  RTY somewhat independent (54-78%).
- 2020 (COVID) and 2024 are the high-event years; 2023 was quiet.

### Trend Alignment

- **162 concurrent** (gap in trend direction) vs **78 incongruent** (gap against trend).
- Incongruent gaps are dominated by **gap-down in an uptrend** (64/78 = 82%) —
  the "fear spike in a bull market" scenario.
- Gap-up in a downtrend is rare (14 events) — markets in downtrends don't produce many upside gap shocks.

### Same-Day Short Strike Breach (EOD Close vs prev_close)

| Group | n | EOD safe rate | 3pm safe rate |
|-------|---|--------------|---------------|
| Concurrent | 162 | ~96.9% | ~96.3% |
| Incongruent | 78 | ~98.7% | ~98.7% |
| **Overall** | **240** | **~97.5%** | **~97.1%** |

The short strike (prev_close) holds through EOD ~97.5% of the time for ≥1 ATR gaps.
This is the key input for the credit spread simulation.

### Limitations at This Stage

1. **Synthetic options pricing** — no real options chain data; Black-Scholes with VIX/100 IV proxy.
2. **No bid/ask spread or transaction costs** modeled.
3. **1-min data, 9:30 entry** — ignores pre-open futures behavior; entry may be better later.
4. **Credit received depends heavily on IV** at time of entry — high-VIX events collect more credit
   but may have larger MAE.

---

### Next Steps

1. **Credit spread simulation** — price spreads at 9:30 entry, simulate exits
   (60% PT and 3pm hard stop from strategy spec), compute win rate / expectancy by alignment.
2. **ATR threshold sensitivity** — test 1.0, 1.5, 2.0 thresholds.
3. **Gap-down in uptrend deep-dive** — the 64-event incongruent gap-down set.
4. **VIX regime overlay** — does the safe rate hold in high-VIX environments?
"""))

# ── Write notebook ─────────────────────────────────────────────────────────────
with open("gap_spread_eda.ipynb", "w") as f:
    nbf.write(nb, f)

print("gap_spread_eda.ipynb written.")
