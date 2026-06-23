"""Build 09_phase2_presentation.ipynb — Task 12, final Phase 2 narrative + paper-trade spec."""
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
        cell.metadata['jupyter'] = {'source_hidden': True}
        cell.metadata['tags'] = ['hide-input']
    cells.append(cell)


md("""\
# Gap Credit Spread Strategy — Phase 2 Research Summary

**Author:** Hernan Rosenblum · June 2026
**Universe:** ES · NQ · RTY · YM (CME Globex, continuous futures)
**Data:** 7 years · 1-minute OHLCV · 2019–2026 · ~10M bars (Databento) + VIX daily

---

> *Phase 1 established that overnight gaps ≥ 1×ATR are 97.5 % EOD-safe.
> The simple credit spread barely captured the edge:* −334 *annual EV at*
> *0.75×ATR with PT60 exit.*

> **Phase 2 question:** can we extract a tradeable rule from this 97.5 %
> finding — bounded tail, after-cost positive EV, repeatable?

---
""")

# ── HIDDEN SETUP ────────────────────────────────────────────────────────────
code("""\
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BG = '#0d1117'
C_GREEN = '#3fb950'
C_RED = '#f78166'
C_BLUE = '#58a6ff'
C_GOLD = '#e3b341'
plt.style.use('dark_background')
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
                     'figure.dpi': 110, 'font.size': 10})
YEARS = 7
os.makedirs('charts', exist_ok=True)

# Load pre-computed event tables
events  = pd.read_csv('cache/phase2_events_075.csv', parse_dates=['date'])
t2u_ev  = pd.read_csv('cache/phase2_t2u_events.csv', parse_dates=['date'])
r1_sim  = pd.read_csv('cache/phase2_r1_exit_sim.csv')
t2u_sim = pd.read_csv('cache/phase2_t2u_exit_sim.csv')

# Tier
def tier(g):
    a = abs(g)
    if a < 1.00: return 'T1'
    if a < 1.50: return 'T2'
    return 'T3'
events['tier'] = events['gap_ratio'].apply(tier)
print('Setup complete.')
""", hide=True)

# ── 1 · The path ────────────────────────────────────────────────────────────
md("""\
## 1 · Where Phase 1 Left Off — and the Phase 2 Path

Phase 1 confirmed the EOD-safety pattern but the simple structure was
EV-negative. Five questions drove Phase 2, each one re-shaping the next:

1. **Does a stop at prev_close fix the losers?** ⟶ No, it cuts winners that recover.
2. **Is there a high-conviction sub-population?** ⟶ Yes — mid-VIX × gap_down.
3. **Do deeper gaps give more credit?** ⟶ No, they give *less* (BS effect).
4. **Should we bet on momentum instead?** ⟶ Yes, but on T2 × gap_up, not T3.
5. **Does any exit rule improve EOD hold?** ⟶ For R1 no; for momentum yes.

Each finding flipped a starting assumption. The two rules that survived
are presented in §3.
""")

# ── 2 · Stratification — the headline filter ────────────────────────────────
md("## 2 · The Filter That Made the Strategy Tradeable")

code("""\
# Bubble plot: every 3D cell at 0.75x threshold as (worst, annual_EV)
def stratify(df, cols):
    g = df.groupby(cols, observed=True)
    return pd.DataFrame({
        'n': g.size(),
        'n_yr': g.size() / YEARS,
        'ev': g['eod_pnl'].mean(),
        'annual_ev': g['eod_pnl'].sum() / YEARS,
        'worst': g['eod_pnl'].min(),
    }).reset_index()

s3d = stratify(events, ['vix_regime', 'alignment', 'direction'])

fig, ax = plt.subplots(figsize=(11, 6.5))
for direction, color in [('gap_down', C_GREEN), ('gap_up', C_BLUE)]:
    sub = s3d[s3d['direction'] == direction]
    ax.scatter(sub['worst'], sub['annual_ev'], s=sub['n_yr']*30, alpha=0.8,
               c=color, edgecolor='white', label=direction, linewidths=1.2)
    for _, row in sub.iterrows():
        ax.annotate(f"{row['vix_regime']}/{row['alignment'][:4]}",
                    (row['worst'], row['annual_ev']),
                    xytext=(8, 5), textcoords='offset points', fontsize=8.5, alpha=0.9)
ax.axhline(0, color='white', lw=0.5, alpha=0.5)
ax.axvline(-150, color=C_GOLD, lw=1, ls='--', label='tail goal: worst > -150')
# Highlight R1 region
ax.add_patch(mpatches.Rectangle((-25, 0), 30, 200, alpha=0.15, color=C_GREEN, label='R1 cells'))
ax.set_xlabel('Worst single trade (pts)')
ax.set_ylabel('Annual EV (pts · 1 contract · 1 symbol)')
ax.set_title('3D stratification of 0.75x events — VIX x alignment x direction')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('charts/p2_pres_01_stratify.png', dpi=140, bbox_inches='tight')
plt.show()
""", hide=True)

md("""\
**Reading the chart.** Each bubble is a (VIX × alignment × direction) cell;
size is trades per year, position is (worst trade, annual EV). The cells
**clustered top-right** are what we want — positive EV, bounded worst case.
Two stand out cleanly: `mid/concurrent/gap_down` and `mid/incongruent/gap_down`.
Together they form **Rule R1**.

| Unfiltered NS·EOD at 0.75×ATR | After R1 filter |
|---|---|
| 88 trades/yr | 28.7 trades/yr |
| +347 annual EV | +228 annual EV |
| Worst trade −338 pts | **Worst trade −19.6 pts** |

Lost 1/3 of EV — cut tail by **17×**. Risk-adjusted return improved
dramatically because losers per cell are bounded by `width − credit`, and
filtering on VIX caps the spread width via ATR.
""")

# ── 3 · The tier surprise ────────────────────────────────────────────────────
md("## 3 · Counterintuitive — Deeper Gaps Are Not Better")

code("""\
# Two-panel: credit collapse + EV by tier
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
colors = {'low': C_BLUE, 'mid': C_GREEN, 'high': C_RED}
for vr, c in colors.items():
    sub = events[events['vix_regime'] == vr]
    ax.scatter(sub['gap_ratio'].abs(), sub['credit'], alpha=0.4, c=c, s=14, label=f'{vr} VIX')
ax.set_xlabel('|gap_ratio| (gap / ATR)')
ax.set_ylabel('Credit collected (pts)')
ax.set_title('Credit collapses with gap depth — both legs OTM in 1-DTE')
ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0.7, 3.5)

ax = axes[1]
tier_ev = events.groupby('tier').agg(annual_ev=('eod_pnl', lambda x: x.sum()/YEARS),
                                      n_yr=('eod_pnl', lambda x: len(x)/YEARS),
                                      avg_credit=('credit', 'mean')).reset_index()
tier_ev = tier_ev.set_index('tier').reindex(['T1', 'T2', 'T3']).reset_index()
bars = ax.bar(tier_ev['tier'], tier_ev['annual_ev'], color=[C_GREEN, C_BLUE, C_RED], alpha=0.85)
for bar, v in zip(bars, tier_ev['annual_ev']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+5, f'{v:+.0f}',
            ha='center', fontsize=10)
ax.set_xlabel('Gap-size tier')
ax.set_ylabel('Annual EV (pts)')
ax.set_title('Annual EV concentrates in T1 — every rule shows the pattern')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('charts/p2_pres_02_tiers.png', dpi=140, bbox_inches='tight')
plt.show()
""", hide=True)

md("""\
| Tier | Avg credit | Annual EV (% of total) |
|---|---:|---:|
| **T1** [0.75 – 1.0 × ATR] | 10.7 | **236 (69%)** |
| T2 [1.0 – 1.5 × ATR]      | 5.7  | 95  (28%) |
| T3 [1.5 + × ATR]          | 1.6  | 13  (4 %) |

At deeper gaps, **both spread legs move OTM together** in a 1-DTE option.
Premium drops off as a Gaussian function of strike distance, so the
difference (= credit) shrinks faster than either leg.

In real markets put-side skew makes this worse: the OTM long leg trades at
elevated IV vs constant-vol BS, further compressing the credit. **The
T1-dominance is structural, not a model artifact.**
""")

# ── 4 · The second edge — momentum ───────────────────────────────────────────
md("## 4 · A Second Edge — Momentum on T2 × gap_up")

code("""\
# Anchor comparison
tiers = ['T1', 'T2', 'T3']
mr_vals  = [events[events['tier']==t]['eod_pnl'].sum()/YEARS for t in tiers]
# Use t2u_ev which has mom_eod_pnl. But t2u_ev is only T2 gap_up subset.
# Re-derive for all tiers needs all-events momentum. Skip - use scalar values.
mom_vals = [-24, 262, 17]  # from task 10b output

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(tiers)); w = 0.35
ax.bar(x - w/2, mr_vals,  w, label='Mean reversion (K_short=prev_close)', color=C_GREEN, alpha=0.85)
ax.bar(x + w/2, mom_vals, w, label='Momentum (K_short=today_open)', color=C_BLUE, alpha=0.85)
ax.axhline(0, color='white', lw=0.5)
ax.set_xticks(x); ax.set_xticklabels(tiers)
ax.set_ylabel('Annual EV (pts)')
ax.set_title('Strike anchor by tier — momentum dominates T2, mean reversion dominates T1')
ax.legend()
for i, v in enumerate(mr_vals):
    ax.text(i - w/2, v + (8 if v>=0 else -14), f'{v:+.0f}', ha='center', fontsize=10)
for i, v in enumerate(mom_vals):
    ax.text(i + w/2, v + (8 if v>=0 else -14), f'{v:+.0f}', ha='center', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('charts/p2_pres_03_anchor.png', dpi=140, bbox_inches='tight')
plt.show()
""", hide=True)

md("""\
**Mean-reversion structure** (K_short = yesterday's close) wins on T1.
**Momentum structure** (K_short = today's open) wins on T2.

The two edges are non-overlapping — different gap tier, different direction:
- **R1** trades T1 × gap_down (fear reversion)
- **M1** trades T2 × gap_up (greed extension)

T3 (deepest gaps) doesn't carry either edge — gap-hold rate falls to ~44 %
because the move is "exhausted" by the open. Marginal participants by 09:30
are the contrarians.
""")

# ── 5 · Exit rules — the asymmetry ───────────────────────────────────────────
md("## 5 · Exit Rules — Why EOD Wins for R1 but SS Wins for Momentum")

code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: R1 — every policy ranked by annual EV
r1_pol = ['EOD','PT75','T15:30','PT60','PT50','T15:00','PT25','SS75','SS50','SS25']
r1_ev = []
for p in r1_pol:
    r1_ev.append(r1_sim[p].sum()/YEARS)
ax = axes[0]
colors_r1 = [C_GREEN if p=='EOD' else (C_GOLD if p.startswith('PT') else C_RED if p.startswith('SS') else C_BLUE) for p in r1_pol]
bars = ax.barh(r1_pol, r1_ev, color=colors_r1, alpha=0.85)
ax.axvline(0, color='white', lw=0.5)
ax.set_xlabel('Annual EV (pts)')
ax.set_title('R1: EOD hold dominates — no exit rule improves')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
for bar, v in zip(bars, r1_ev):
    ax.text(v + (5 if v >= 0 else -5), bar.get_y() + bar.get_height()/2,
            f'{v:+.0f}', va='center', fontsize=9,
            ha='left' if v >= 0 else 'right')

# Right: T2 × gap_up momentum
t2_pol = ['EOD','T15:30','SS75','T15:00','T14:30','PT75','SS50','PT60','T14:00','PT50','T13:00']
t2_ev = []; t2_worst = []
for p in t2_pol:
    t2_ev.append(t2u_sim[p].sum()/YEARS)
    t2_worst.append(t2u_sim[p].min())
ax = axes[1]
colors_t2 = [C_GREEN if p=='EOD' else (C_GOLD if p.startswith('PT') else C_RED if p.startswith('SS') else C_BLUE) for p in t2_pol]
bars = ax.barh(t2_pol, t2_ev, color=colors_t2, alpha=0.85)
ax.axvline(0, color='white', lw=0.5)
ax.set_xlabel('Annual EV (pts)')
ax.set_title('T2 × gap_up momentum: SS75 doubles risk-adjusted return')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
for bar, v, w in zip(bars, t2_ev, t2_worst):
    ax.text(v + 5, bar.get_y() + bar.get_height()/2,
            f'{v:+.0f} (worst {w:+.0f})', va='center', fontsize=8.5)

plt.tight_layout()
plt.savefig('charts/p2_pres_04_exits.png', dpi=140, bbox_inches='tight')
plt.show()
""", hide=True)

md("""\
The asymmetry is structural:

| | R1 (mean reversion) | M1 (momentum) |
|---|---|---|
| Win rate | 96 % | 73 % |
| Short strike at entry | OTM | **ATM** |
| Gamma | low | high |
| Effect of stop | cuts mostly winners | cuts mostly **real losers** |

R1 doesn't need a stop — most "adverse" moves are noise that mean-reverts.
M1 does — ATM gamma escalates losers fast, and a SS75 exit (close on a
75 %-of-width adverse move) captures the worst trades *before* they hit
max loss. Risk-adjusted return on M1 jumps from 1.26 to 3.56 with SS75
alone.
""")

# ── 6 · The two final rules ──────────────────────────────────────────────────
md("""\
## 6 · The Two Final Rules

### R1 — Mean Reversion (primary)

```
Universe:    ES, NQ, RTY, YM
Trigger:     Overnight |gap| / ATR(14) ≥ 0.75
Direction:   gap_down only
Regime:      VIX(prev close) ∈ [15, 25)
Structure:   1-DTE bear call spread
             K_short = prev_close
             K_long  = prev_close + 0.5 × ATR(14)
Entry:       09:30 ET
Exit:        16:00 ET (HOLD TO CLOSE)
```

| Metric | Value |
|---|---:|
| Events / yr | **28.7** |
| Annual EV   | **+228 pts** |
| EV / trade  | +7.96 pts |
| Win rate    | 93.5 % |
| Worst trade | **−19.6 pts** |
| Avg credit  | 8.6 pts |

---

### M1 — Momentum (secondary)

```
Universe:    ES, NQ, RTY, YM
Trigger:     Overnight 1.0 ≤ |gap| / ATR(14) < 1.5
Direction:   gap_up only
Regime:      no VIX filter
Structure:   1-DTE bull put spread anchored at OPEN
             K_short = today_open
             K_long  = today_open − 0.5 × ATR(14)
Entry:       09:30 ET
Exit:        SS75 — close if low <= today_open − 0.75 × spread_width
             else 16:00 ET
```

| Metric | Value |
|---|---:|
| Events / yr | **11.6** |
| Annual EV   | **+226 pts** |
| EV / trade  | +19.5 pts |
| Win rate    | 56 % (after SS75) |
| Worst trade | **−63.5 pts** |
| Avg credit  | 44.2 pts |
""")

# ── 7 · Combined portfolio ───────────────────────────────────────────────────
md("## 7 · Combined Portfolio")

code("""\
fig, ax = plt.subplots(figsize=(10.5, 5.5))
labels = ['R1\\n(MR · gap_down)', 'M1\\n(MOM · gap_up SS75)', 'Combined']
annual = [228, 226, 454]
worst  = [-19.6, -63.5, -63.5]
freq   = [28.7, 11.6, 40.3]

x = np.arange(len(labels))
ax2 = ax.twinx()
bars = ax.bar(x, annual, color=[C_GREEN, C_BLUE, C_GOLD], alpha=0.85, width=0.55,
              label='Annual EV (pts)')
for bar, v, w, n in zip(bars, annual, worst, freq):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+15,
            f'+{v} pts\\n{n}/yr\\nworst {w:+.1f}', ha='center', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Annual EV (pts · 1 contract · 1 symbol)')
ax.set_title('Phase 2 Final — Two rules, one combined portfolio')
ax.set_ylim(0, 550)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('charts/p2_pres_05_portfolio.png', dpi=140, bbox_inches='tight')
plt.show()
""", hide=True)

md("""\
R1 and M1 are **non-overlapping**: R1 fires on gap_down days, M1 on gap_up.
They cluster on macro signal days but never on the same trade. Combined:

| Combined R1 + M1 | Value |
|---|---:|
| Signals / yr   | **40.3** |
| Annual EV      | **+454 pts** |
| Worst single trade | **−63.5 pts** (M1) |

For ES at $50/pt and 1-contract-per-symbol-per-rule sizing:
- Annual EV ≈ **$22,700 / year / symbol**
- Worst single-day loss ≈ **$3,175** (from M1)

Per-symbol position sizing should differ between rules because the
worst-trade ratio is ~3×.
""")

# ── 8 · Limitations ──────────────────────────────────────────────────────────
md("""\
## 8 · Limitations & Open Questions

### Modeling caveats
- All EV in **BS-model points** with constant-vol σ = VIX/100. Real put skew
  inflates the OTM long-leg cost — credit estimates are slightly optimistic.
- No slippage or commissions. Realistic mid-fill on a 5-point-wide ES spread
  is 0.5–1.0 pt round trip. At R1's 8.6-pt avg credit, slippage erodes
  ~10–15 % of EV. M1's 44-pt credit is more robust to slippage.
- 1-DTE assumption matches 0-DTE / 1-DTE ES, SPX-style options. Index-future
  options have specific expirations; in live trading, the nearest weekly is
  the appropriate proxy.

### Statistical caveats
- 7-year sample. Includes 2020 COVID volatility and 2022 inflation regime.
  A pre-2008 / 1998-crisis regime is **not** in this sample.
- Survivorship in 4-symbol universe — all are major liquid index futures.
- Cluster risk: 4 symbols all gap on the same macro days; treat as roughly
  one signal with leveraged contract sizing, not 4 independent signals.

### What Phase 3 should ask
1. **Skew-aware repricing.** Pull empirical IV skew (from CBOE or simulated)
   and recompute R1 / M1 credit. Likely reduces EV ~10–20 %.
2. **Slippage stress test.** Subtract 0.5 / 1.0 / 2.0 pt slippage per trade.
3. **Live forward test.** Paper trade R1 + M1 from 2026-07-01 for one
   quarter using the locked specs in §6. Verify hit-rate matches.
4. **Hedged variants.** Buy a deep OTM long option to cap M1 tail from
   −63 to ~−20 at a cost of ~1 pt credit. Worth modeling.
5. **Position sizing.** Risk-budget per rule based on the worst-trade
   ratio. R1 can size 3× larger than M1 for equal max-drawdown exposure.

### What was NOT a viable rule
- Hard stops at prev_close (Task 8) — cuts winners that recover
- Deep-gap rules (T3) — gaps are exhausted by open, no momentum
- PT60 exit on R1 (Task 11) — caps winners without saving losers

---

*Phase 2 complete. 7 years of data, ~11M intraday bars analyzed across
6 sub-tasks. Two rules identified, one combined portfolio. Ready for
Phase 3: skew-aware repricing, slippage stress test, and live paper trading.*
""")

# Write notebook
nb = new_notebook()
nb.cells = cells
nb.metadata = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
}

import nbformat
with open('notebooks/09_phase2_presentation.ipynb', 'w') as f:
    nbformat.write(nb, f)
print(f'Written: 09_phase2_presentation.ipynb ({len(cells)} cells)')
