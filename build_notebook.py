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


# ── Section 0: Setup & Config ──────────────────────────────────────────────
nb.cells.append(md("""# Gap-Based Credit Spread Research

**Hypothesis:** After overnight gaps ≥ 1 ATR in equity index ETFs, selling short-dated credit spreads
anchored at the previous day's close may yield positive expectancy due to mean reversion or limited
immediate trend continuation.

**Data:** yfinance 5m intraday (~60 days) + daily OHLCV (10+ years) + VIX.
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
subsequent runs load from cache instantly. Set `force_refresh=True` to re-fetch.

Note: Yahoo Finance limits 5m data to ~60 days. Daily data goes back to 2013."""))

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
    im = ax.imshow(pivot.values.astype(float), aspect="auto", cmap="RdYlGn" if "fill" in metric else "RdYlGn_r")
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
spy_exits = exits[exits["ticker"] == "SPY"]
if len(spy_exits) > 0:
    print(f"Sample credit (SPY, first event): ${spy_exits['credit'].iloc[0]:.4f}")\
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
    ax.set_title(f"{title}\\n(Entry: Open)")
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

2. **~60-day 5m sample.** yfinance provides ~60 days of 5m bars. Results may not generalize
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

with open("gap_spread_research.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written (all sections 0-6).")
