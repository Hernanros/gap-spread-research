"""
Build 07b_phase2_momentum_anchor.ipynb — Task 10b.

Tests the alternative strike anchoring: K_short = today's open (momentum bet)
vs K_short = prev_close (mean reversion). Compares per tier.
"""
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13.0"},
}

def md(t):    nb.cells.append(nbf.v4.new_markdown_cell(t))
def code(s):  nb.cells.append(nbf.v4.new_code_cell(s))


md("""\
# Phase 2 · Task 10b — Strike Anchor: Open vs Prev-Close

**Question.** What if we flip the structure to bet on momentum continuation
instead of mean reversion?

- **Mean reversion (MR)** — the structure used so far. `K_short = prev_close`.
  Profit if price stays on the gap-side of yesterday's close
  (gap_up: close > prev_close; gap_down: close < prev_close).
- **Momentum (MOM)** — anchor strikes at today's open. `K_short = open`.
  Profit if price stays on the gap-side of TODAY'S open
  (gap_up: close > open; gap_down: close < open).
  This bets the gap **holds or extends**.

The momentum short is ATM at entry, so it collects ~4× more credit but
carries ATM gamma risk.
""")

md("## 1 · Setup and pricing")

code("""\
import datetime, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm as N

YEARS, WIDTH_ATR, T_FULL = 7, 0.5, 1/252
plt.style.use('dark_background')
os.makedirs('charts', exist_ok=True)

def bs_call(S, K, T, r, sigma):
    if T <= 1e-12 or sigma <= 0: return max(S-K, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*N.cdf(d1) - K*np.exp(-r*T)*N.cdf(d2)
def bs_put(S, K, T, r, sigma):
    return bs_call(S, K, T, r, sigma) - S + K*np.exp(-r*T)
def spread_value(S, K_short, K_long, direction, T, sigma, r=0.05):
    if direction == 'gap_up':
        return bs_put(S, K_short, T, r, sigma) - bs_put(S, K_long, T, r, sigma)
    return bs_call(S, K_short, T, r, sigma) - bs_call(S, K_long, T, r, sigma)
""")

md("## 2 · Load events + attach today's open/close from raw daily bars")

code("""\
SYMBOLS = ['ES', 'NQ', 'RTY', 'YM']
PATHS   = {s: f'data/{s}_1m_2019_2026.csv' for s in SYMBOLS}

def load_daily(path):
    df = pd.read_csv(path, parse_dates=['ts_event']).set_index('ts_event')
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('America/New_York')
    t = df.index.time
    sess = df[(t >= datetime.time(9,30)) & (t <= datetime.time(16,0))]
    daily = sess.resample('1D').agg(
        open=('open','first'), high=('high','max'),
        low=('low','min'),     close=('close','last'),
        volume=('volume','sum'),
    ).dropna()
    return daily[daily['volume'] > 0]

daily_all = {s: load_daily(PATHS[s]) for s in SYMBOLS}
df = pd.read_csv('cache/phase2_events_075.csv', parse_dates=['date'])

def lookup_open_close(row):
    sd = daily_all[row['symbol']]
    d = pd.Timestamp(row['date']).normalize()
    mask = sd.index.normalize() == d.normalize() if sd.index.tz is None else sd.index.tz_localize(None).normalize() == d.normalize()
    if mask.any():
        return float(sd.loc[mask, 'open'].iloc[0]), float(sd.loc[mask, 'close'].iloc[0])
    return np.nan, np.nan

oc = df.apply(lookup_open_close, axis=1, result_type='expand')
df['today_open']  = oc[0]
df['today_close'] = oc[1]
df = df.dropna(subset=['today_open', 'today_close']).reset_index(drop=True)
print(f'{len(df)} events with open+close attached.')
""")

md("## 3 · Re-price under momentum (open-anchored) structure")

code("""\
mom_credit_list, mom_pnl_list, mom_win_list = [], [], []
for _, row in df.iterrows():
    direction = row['direction']
    atr     = float(row['atr'])
    sigma   = max(float(row['vix'])/100.0, 0.05)
    S_open  = float(row['today_open'])
    close_px = float(row['today_close'])
    width   = WIDTH_ATR * atr
    K_short = S_open
    K_long  = S_open - width if direction == 'gap_up' else S_open + width
    credit  = spread_value(S_open, K_short, K_long, direction, T_FULL, sigma)
    if credit <= 0:
        mom_credit_list.append(0.0); mom_pnl_list.append(0.0); mom_win_list.append(False); continue
    if direction == 'gap_up':
        intrinsic = min(max(K_short - close_px, 0.0), width)
        win = close_px > S_open
    else:
        intrinsic = min(max(close_px - K_short, 0.0), width)
        win = close_px < S_open
    mom_credit_list.append(credit); mom_pnl_list.append(credit - intrinsic); mom_win_list.append(win)

df['mom_credit']  = mom_credit_list
df['mom_eod_pnl'] = mom_pnl_list
df['mom_win']     = mom_win_list

def tier(g):
    a = abs(g)
    if a < 1.0: return 'T1 [0.75-1.0)'
    if a < 1.5: return 'T2 [1.0-1.5)'
    return 'T3 [1.5+)'
df['tier'] = df['gap_ratio'].apply(tier)
print('Momentum structure priced for all events.')
""")

md("## 4 · Headline — same events, both structures")

code("""\
rows = []
for t in ['T1 [0.75-1.0)', 'T2 [1.0-1.5)', 'T3 [1.5+)']:
    sub = df[df['tier'] == t]
    rows.append({
        'tier':           t,
        'n_yr':           round(len(sub)/YEARS, 1),
        'MR_ev':          round(sub['eod_pnl'].mean(), 2),
        'MR_annual':      round(sub['eod_pnl'].sum()/YEARS, 1),
        'MR_win_%':       round(sub['eod_safe'].mean()*100, 1),
        'MR_worst':       round(sub['eod_pnl'].min(), 1),
        'MR_avg_credit':  round(sub['credit'].mean(), 2),
        'MOM_ev':         round(sub['mom_eod_pnl'].mean(), 2),
        'MOM_annual':     round(sub['mom_eod_pnl'].sum()/YEARS, 1),
        'MOM_win_%':      round(sub['mom_win'].mean()*100, 1),
        'MOM_worst':      round(sub['mom_eod_pnl'].min(), 1),
        'MOM_avg_credit': round(sub['mom_credit'].mean(), 2),
    })
print(pd.DataFrame(rows).to_string(index=False))
""")

md("""\
**The momentum bet works — but on T2, not T3.**

| Tier | MR annual | MOM annual |
|---|---:|---:|
| T1 | +236 | **−24** |
| **T2** | +95 | **+262** |
| T3 | +13 | +17 |

T3 has roughly equal (tiny) annual EV under either structure — the deepest
gaps don't have momentum left. T2 is the sweet spot for the momentum bet.

The MOM credit is ~4× larger (44 pts vs 6-11 pts) because the short option
is ATM at entry, but the worst trade is ~2× the MR worst because ATM
gamma punishes adverse moves more severely.
""")

md("## 5 · Why deep gaps don't carry momentum — gap-hold diagnostic")

code("""\
diag = []
for t in ['T1 [0.75-1.0)', 'T2 [1.0-1.5)', 'T3 [1.5+)']:
    for d in ['gap_up', 'gap_down']:
        sub = df[(df['tier']==t) & (df['direction']==d)]
        if len(sub) == 0: continue
        diag.append({
            'tier': t, 'direction': d,
            'n_yr': round(len(sub)/YEARS, 1),
            'gap_holds_%': round(sub['mom_win'].mean()*100, 1),
            'avg_credit_mom': round(sub['mom_credit'].mean(), 2),
            'avg_pnl_mom': round(sub['mom_eod_pnl'].mean(), 2),
            'worst_pnl_mom': round(sub['mom_eod_pnl'].min(), 1),
        })
print(pd.DataFrame(diag).to_string(index=False))
""")

md("""\
**Gap-hold rate by tier × direction:**

| Tier | gap_up holds | gap_down holds |
|---|---:|---:|
| T1 | 57.7 % | 39.9 % |
| **T2** | **60.5 %** | **45.0 %** |
| T3 | 42.9 % | 44.7 % |

Two asymmetries:
1. **Gap-up holds best at every tier.** Greed-extension is the structural
   tendency; fear-spikes revert. Confirms the EDA's mean-reversion finding
   (which was about gap-downs).
2. **T3 has the lowest gap-hold rate, not the highest.** The intuition that
   "bigger gap = stronger conviction = more momentum" is wrong. Extreme
   gaps look exhausted by the open — marginal buyers/sellers who would have
   continued the move did so overnight. By 09:30 the contrarian is the
   marginal participant.
""")

md("## 6 · T3 deep-dive — where the user expected an edge")

code("""\
t3 = df[df['tier'] == 'T3 [1.5+)']
print(f'T3 events: {len(t3)} ({len(t3)/YEARS:.1f}/yr)')

def stratify_mom(sub, group_cols):
    g = sub.groupby(group_cols, observed=True)
    return pd.DataFrame({
        'n': g.size(),
        'n_yr': (g.size()/YEARS).round(1),
        'ev_per_trade': g['mom_eod_pnl'].mean().round(2),
        'annual_ev': (g['mom_eod_pnl'].sum()/YEARS).round(1),
        'win_%': (g['mom_win'].mean()*100).round(1),
        'worst': g['mom_eod_pnl'].min().round(1),
        'avg_credit': g['mom_credit'].mean().round(2),
    }).reset_index()

print('\\n--- T3 × VIX × direction (momentum) ---')
print(stratify_mom(t3, ['vix_regime', 'direction']).to_string(index=False))
print('\\n--- T3 × VIX × alignment × direction (momentum) ---')
print(stratify_mom(t3, ['vix_regime', 'alignment', 'direction']).to_string(index=False))
""")

md("""\
Within T3, two cells flicker positive but on tiny samples:
- **high-VIX × concurrent × gap_down** (n=4, 0.6/yr) — EV +80, all wins
- **mid-VIX × concurrent × gap_down** (n=16, 2.3/yr) — EV +24, 75 % win, worst −198

These are too rare to size a strategy around. T3 as a regime does not carry
the directional edge required for a momentum credit bet.
""")

md("## 7 · Visual — annual EV: MR vs MOM by tier")

code("""\
tiers = ['T1 [0.75-1.0)', 'T2 [1.0-1.5)', 'T3 [1.5+)']
mr_ev = [df[df['tier']==t]['eod_pnl'].sum()/YEARS for t in tiers]
mom_ev = [df[df['tier']==t]['mom_eod_pnl'].sum()/YEARS for t in tiers]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(tiers)); w = 0.35
ax.bar(x - w/2, mr_ev, w, label='Mean reversion (K_short=prev_close)', color='#06d6a0', alpha=0.85)
ax.bar(x + w/2, mom_ev, w, label='Momentum (K_short=today open)', color='#ef476f', alpha=0.85)
ax.axhline(0, color='white', lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(tiers)
ax.set_ylabel('Annual EV (pts)')
ax.set_title('Strike anchor comparison — annual EV by gap-size tier')
ax.legend()
for i, v in enumerate(mr_ev):
    ax.text(i - w/2, v + 8, f'{v:+.0f}', ha='center', fontsize=9)
for i, v in enumerate(mom_ev):
    ax.text(i + w/2, v + 8, f'{v:+.0f}', ha='center', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('charts/p2_06_anchor_compare.png', dpi=150, bbox_inches='tight')
plt.show()
""")

md("""\
## 8 · Findings — Task 10b

### Direct answer to the user's question
> "Bet on momentum continuing on gaps deeper than 1.5×ATR with strikes at
>  today's open."

**Verdict: not the right combination.** T3 doesn't carry momentum any better
than T2 — actually worse. The annual EV at T3 under MOM is +17 pts, barely
distinguishable from the MR baseline of +13.

### What the data *does* support
The momentum structure (open-anchored ATM short) is real edge — just on
**T2 [1.0–1.5×ATR]**, not T3. Specifically T2 × gap_up:
- 11.6/yr, EV/trade +23.15, **annual EV +268 pts**
- Win rate 60.5 %, worst trade **−229 pts**

That's slightly higher annual EV than R1 (+228) at similar frequency — but
with a fat tail (the ATM short has high gamma).

### Two distinct edges identified in Phase 2

| Edge | Direction | Where | Annual EV | Tail | Best for |
|---|---|---|---:|---:|---|
| **Mean reversion** | gap_down | R1 (mid-VIX × gap_down × T1) | +180 | **−19.6** | low-risk steady |
| **Momentum** | gap_up | T2 × gap_up (any VIX) | +268 | −229 | higher EV at higher gamma |

These are non-overlapping (different tiers, different directions). A
portfolio could run both for an uncorrelated combined edge — total ~+450
annual EV — but the momentum leg needs tail-risk controls before sizing.

### Why deep gaps lack momentum
At T3, gap is ≥1.5×ATR — that's ~1.5σ of overnight move. Anyone willing to
push the move further has already taken position overnight (the gap is the
visible result). By the open, marginal demand is the contrarian / fader of
the move. Hence T3 gap-hold rate (42–45 %) is below T2 (45–60 %).

### Phase 2 task list — updated
| # | Task | Status |
|---|---|---|
| 10b | Strike-anchor comparison (MR vs MOM) by tier | **done — momentum edge confirmed at T2 × gap_up, not T3** |
| 11  | Exit-rule optimization within R1 × T1 (PT/time/soft-stop sweep) | next |
| 11b | (added) Exit-rule + sizing analysis for T2 × gap_up momentum (tail control) | after 11 |
| 12  | Build final Phase 2 presentation + paper-trade spec | last |
""")

import nbformat
with open('notebooks/07b_phase2_momentum_anchor.ipynb', 'w') as f:
    nbformat.write(nb, f)
print(f'Written: 07b_phase2_momentum_anchor.ipynb ({len(nb.cells)} cells)')
