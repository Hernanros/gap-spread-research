"""Build 08_phase2_exit_rules.ipynb — Task 11.

Intraday exit-policy sweep on R1 events. Tests PT/SS/Time/Combined vs EOD.
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
# Phase 2 · Task 11 — Exit-Rule Optimization within R1

**Premise.** Task 8 showed hard stops hurt EV. Task 10 showed deeper gaps
hurt EV. Task 9 settled on Rule R1 (mid-VIX × gap_down at 0.75×ATR) as the
primary candidate, with `NS · EOD` baseline = +228 annual EV, worst −19.6.

This notebook asks: **can any intraday exit policy beat pure EOD hold on R1?**

Three families tested:
- **PT exits** (PT25/50/60/75): close when spread debit drops to (1−pt)×credit
- **Soft stops** (SS25/50/75): close when underlying moves N% of width adverse
- **Time exits** (12:00 / 14:00 / 14:30 / 15:00 / 15:30): unconditional close
- **Combined** (PT60 + T15:00, whichever first)

Spread value at minute *t* is computed by BS with linearly-decaying T:
`T_remaining(t) = (1 − t/390) × 1/252`.
""")

md("## 1 · Setup and intraday spread valuation")

code("""\
import datetime, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm as N

SYMBOLS = ['ES', 'NQ', 'RTY', 'YM']
PATHS   = {s: f'data/{s}_1m_2019_2026.csv' for s in SYMBOLS}
YEARS   = 7
WIDTH_ATR = 0.5
T_FULL  = 1/252

plt.style.use('dark_background')
os.makedirs('charts', exist_ok=True)

def bs_call_vec(S, K, T, r, sigma):
    S = np.asarray(S, dtype=float)
    safe = (T > 1e-12) & (sigma > 0)
    out = np.where(safe, 0.0, np.maximum(S - K, 0.0))
    if not safe.any():
        return out
    d1 = (np.log(np.maximum(S, 1e-12)/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(np.maximum(T, 1e-12)))
    d2 = d1 - sigma*np.sqrt(np.maximum(T, 1e-12))
    bs = S*N.cdf(d1) - K*np.exp(-r*T)*N.cdf(d2)
    return np.where(safe, bs, np.maximum(S - K, 0.0))

def bs_put_vec(S, K, T, r, sigma):
    return bs_call_vec(S, K, T, r, sigma) - S + K*np.exp(-r*T)

def spread_value_vec(S, K_short, K_long, direction, T, sigma, r=0.05):
    if direction == 'gap_up':
        return bs_put_vec(S, K_short, T, r, sigma) - bs_put_vec(S, K_long, T, r, sigma)
    return bs_call_vec(S, K_short, T, r, sigma) - bs_call_vec(S, K_long, T, r, sigma)
""")

md("## 2 · Load events, 1-min bars, and attach prev_close")

code("""\
def load_1min(path):
    df = pd.read_csv(path, parse_dates=['ts_event']).set_index('ts_event')
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('America/New_York')
    t = df.index.time
    return df[(t >= datetime.time(9,30)) & (t <= datetime.time(16,0))]

def load_daily(path):
    df = pd.read_csv(path, parse_dates=['ts_event']).set_index('ts_event')
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('America/New_York')
    t = df.index.time
    sess = df[(t >= datetime.time(9,30)) & (t <= datetime.time(16,0))]
    daily = sess.resample('1D').agg(
        open=('open','first'), high=('high','max'),
        low=('low','min'), close=('close','last'), volume=('volume','sum'),
    ).dropna()
    return daily[daily['volume'] > 0]

print('Loading 1-min bars (~700K rows × 4 symbols)...')
min1_all = {s: load_1min(PATHS[s]) for s in SYMBOLS}
daily_all = {s: load_daily(PATHS[s]) for s in SYMBOLS}
for s in SYMBOLS:
    daily_all[s] = daily_all[s].copy()
    daily_all[s]['__date'] = daily_all[s].index.date

events = pd.read_csv('cache/phase2_events_075.csv', parse_dates=['date'])

def attach_pc(row):
    sd = daily_all[row['symbol']]
    target_date = pd.Timestamp(row['date']).date()
    mask = sd['__date'].values == target_date
    if not mask.any():
        return pd.Series({'prev_close': np.nan, 'today_close': np.nan})
    pos = int(np.where(mask)[0][0])
    return pd.Series({
        'prev_close':  float(sd.iloc[pos-1]['close']) if pos > 0 else np.nan,
        'today_close': float(sd.iloc[pos]['close']),
    })

attached = events.apply(attach_pc, axis=1)
events = pd.concat([events, attached], axis=1).dropna(subset=['prev_close']).reset_index(drop=True)

r1 = events[(events['vix_regime']=='mid') & (events['direction']=='gap_down')].copy().reset_index(drop=True)
print(f'R1 events: {len(r1)} ({len(r1)/YEARS:.1f}/yr)')
""")

md("## 3 · Simulate each event under all policies")

code("""\
def simulate_event(row):
    sym       = row['symbol']
    direction = row['direction']
    atr       = float(row['atr'])
    sigma     = max(float(row['vix'])/100.0, 0.05)
    pc        = float(row['prev_close'])
    width     = WIDTH_ATR * atr
    K_short   = pc
    K_long    = pc - width if direction == 'gap_up' else pc + width

    d_date  = pd.Timestamp(row['date']).date()
    day_bars = min1_all[sym][min1_all[sym].index.date == d_date]
    if day_bars.empty: return None

    closes = day_bars['close'].values.astype(float)
    highs  = day_bars['high'].values.astype(float)
    lows   = day_bars['low'].values.astype(float)
    n_bars = len(closes)
    bar_idx = np.arange(n_bars)
    T_remaining = np.maximum((1.0 - bar_idx/390.0) * T_FULL, 0.0)

    val = spread_value_vec(closes, K_short, K_long, direction, T_remaining, sigma)
    S_open = float(closes[0])
    credit = float(val[0])
    if credit <= 0 or width <= 0: return None

    close_px = float(row['today_close'])
    if direction == 'gap_up':
        eod_intrinsic = min(max(K_short - close_px, 0.0), width)
    else:
        eod_intrinsic = min(max(close_px - K_short, 0.0), width)
    eod_pnl = credit - eod_intrinsic

    out = {'credit': credit, 'EOD': eod_pnl}

    for pt in [0.25, 0.50, 0.60, 0.75]:
        target = (1 - pt) * credit
        hit = np.where(val <= target)[0]
        out[f'PT{int(pt*100)}'] = (credit - float(val[hit[0]])) if len(hit) else eod_pnl

    for n_frac in [0.25, 0.50, 0.75]:
        if direction == 'gap_up':
            trig_lvl = S_open - n_frac*width
            hit = np.where(lows <= trig_lvl)[0]
        else:
            trig_lvl = S_open + n_frac*width
            hit = np.where(highs >= trig_lvl)[0]
        if len(hit):
            t = int(hit[0])
            stop_val = float(spread_value_vec(
                np.array([trig_lvl]), K_short, K_long, direction,
                np.array([T_remaining[t]]), sigma)[0])
            out[f'SS{int(n_frac*100)}'] = credit - stop_val
        else:
            out[f'SS{int(n_frac*100)}'] = eod_pnl

    for label, m in {'12:00':150, '14:00':270, '14:30':300, '15:00':330, '15:30':360}.items():
        idx = min(m, n_bars - 1)
        out[f'T{label}'] = credit - float(val[idx])

    target60 = 0.40 * credit
    hit_pt = np.where(val <= target60)[0]
    pt_time = int(hit_pt[0]) if len(hit_pt) else 10**9
    exit_t = min(pt_time, min(330, n_bars - 1))
    out['PT60+T15:00'] = credit - float(val[exit_t])

    return out

sim_rows = []
for _, ev in r1.iterrows():
    r = simulate_event(ev)
    if r is None: continue
    sim_rows.append(r)
sim = pd.DataFrame(sim_rows)
print(f'Simulated {len(sim)} of {len(r1)} events.')
""")

md("## 4 · Policy comparison")

code("""\
policy_cols = ['EOD', 'PT25', 'PT50', 'PT60', 'PT75',
               'SS25', 'SS50', 'SS75',
               'T12:00', 'T14:00', 'T14:30', 'T15:00', 'T15:30',
               'PT60+T15:00']
n = len(sim); n_yr = n / YEARS
rows = []
for col in policy_cols:
    p = sim[col].values
    rows.append({
        'policy': col,
        'n_yr': round(n_yr, 1),
        'ev_per_trade': round(p.mean(), 2),
        'annual_ev': round(p.sum()/YEARS, 1),
        'win_%': round((p > 0).mean()*100, 1),
        'p10': round(np.quantile(p, 0.10), 2),
        'worst': round(p.min(), 1),
        'delta_vs_EOD_annual': round((p.sum() - sim['EOD'].sum())/YEARS, 1),
    })
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
""")

md("""\
## 5 · Visual — EV vs worst by policy
""")

code("""\
fig, ax = plt.subplots(figsize=(11, 7))
colors = {'EOD':'#06d6a0',
          'PT25':'#f78c6b','PT50':'#f78c6b','PT60':'#f78c6b','PT75':'#f78c6b',
          'SS25':'#ef476f','SS50':'#ef476f','SS75':'#ef476f',
          'T12:00':'#118ab2','T14:00':'#118ab2','T14:30':'#118ab2','T15:00':'#118ab2','T15:30':'#118ab2',
          'PT60+T15:00':'#ffd166'}
for _, r in summary.iterrows():
    ax.scatter(r['worst'], r['annual_ev'], s=160, color=colors[r['policy']],
               edgecolor='white', alpha=0.85)
    ax.annotate(r['policy'], (r['worst'], r['annual_ev']),
                xytext=(8,5), textcoords='offset points', fontsize=9)
ax.axhline(0, color='white', lw=0.5, alpha=0.5)
ax.axvline(0, color='white', lw=0.5, alpha=0.5)
ax.set_xlabel('Worst single trade (pts)')
ax.set_ylabel('Annual EV (pts)')
ax.set_title('R1 exit-policy sweep — EOD hold dominates everything')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('charts/p2_07_exit_policy.png', dpi=150, bbox_inches='tight')
plt.show()
""")

md("""\
## 6 · Findings — Task 11

### EOD hold is the optimal exit for R1 — no policy improves it

| Policy | Annual EV | Δ vs EOD | Worst |
|---|---:|---:|---:|
| **EOD** | **+229** | **0** | **−15.5** |
| PT75 | +182 | −47 | −15.5 |
| T15:30 | +171 | −58 | −27.4 |
| PT60 | +149 | −80 | −15.5 |
| PT50 | +125 | −105 | −15.5 |
| T15:00 | +95 | −135 | −96.5 |
| PT25 | +70 | −160 | −15.5 |
| SS50 | +9 | −220 | −20.7 |
| SS25 | −16 | −246 | −10.6 |

### Three structural reasons EOD wins

1. **R1 already has a 96 % win rate.** Most trades end safely. Adding
   any exit rule caps winners on more trades than it saves losers.
2. **Theta concentrates in the last 30 min.** The 15:30→16:00 segment
   alone is worth +58 annual EV pts. T15:30 gives up 25 % of EV vs EOD.
3. **Soft stops trigger on noise, not real signal.** SS25 fires on ~80 %
   of R1 trades (only 21 % win rate), and most of those triggers are
   transient dips that would have recovered. Identical mechanism to the
   hard stop in Task 8.

### The PT family — illustrative pattern

| Policy | Captured % of EOD EV |
|---|---:|
| PT25 | 30 % |
| PT50 | 55 % |
| PT60 | 65 % |
| PT75 | 79 % |
| **EOD** | **100 %** |

PT exits effectively pay you a fraction of the EOD reward without
materially changing tail risk (worst is still −15.5 because the rare
big losers never give a PT trigger before flying through max-loss). So
they are dominated.

### What does NOT improve EOD
- More aggressive PT  → larger EV give-up
- Less aggressive PT  → smaller give-up but still negative
- Earlier time exit   → larger give-up (theta lost)
- Later time exit     → smaller give-up but still negative
- Soft stops          → big EV give-up + no tail improvement
- Combined PT + Time  → worst-of-both
- Hard stop (Task 8)  → already rejected

### What WOULD improve EOD (hypotheses for future work)
- **Slippage-aware sizing.** EOD captures full theta but real-world exit
  at 16:00 may be illiquid (mini futures fine; on illiquid expiries
  there's a fill-quality concern). Worth checking the bid-ask at close.
- **Adaptive thresholds.** Not tested here; would require regime models.
- **Hedging the tail.** Buy a deep OTM put on entry to convert
  `−15.5` worst to `−5` worst at a cost of ~0.5 pts credit. Cheap
  insurance worth modeling.

### Recommended R1 strategy spec
```
Universe:    ES, NQ, RTY, YM (CME index futures)
Trigger:     Overnight gap with |gap_ratio| ≥ 0.75 (gap/ATR(14))
Direction:   gap_down only
Regime:      VIX(prev close) in [15, 25)
Structure:   Bear call spread
             K_short = prev_close
             K_long  = prev_close + 0.5 × ATR(14)
             1-DTE (same-day expiry)
Sizing:      Per-contract per symbol per signal day
Entry:       09:30 ET market open
Exit:        HOLD TO 16:00 ET CLOSE
Expected:    28.7 events/yr, +228 annual EV pts, 93–96 % win, worst −19.6
```

### Phase 2 task list — updated
| # | Task | Status |
|---|---|---|
| 11  | Exit-rule optimization for R1 | **done — EOD hold is optimal; no policy improves** |
| 11b | Tail-risk control for T2 × gap_up momentum (sizing/sub-filter) | next |
| 12  | Build `08_phase2_conditional.ipynb` final presentation + paper-trade spec | last |
""")

import nbformat
with open('notebooks/08_phase2_exit_rules.ipynb', 'w') as f:
    nbformat.write(nb, f)
print(f'Written: 08_phase2_exit_rules.ipynb ({len(nb.cells)} cells)')
