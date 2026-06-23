"""Quick combo: T2×gap_up with filters + best exit policies."""
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

import numpy as np, pandas as pd

sim = pd.read_csv("cache/phase2_t2u_exit_sim.csv")
ev  = pd.read_csv("cache/phase2_t2u_events.csv")

# Align indices
sim["__i"] = sim.index
ev["__i"]  = ev.index
m = ev[["__i", "symbol", "vix", "vix_regime", "alignment", "atr", "gap_ratio"]].merge(sim, on="__i")

YEARS = 7

def stats(df, pcol, label):
    n = len(df); n_yr = n/YEARS
    p = df[pcol].values
    return {
        "filter": label,
        "policy": pcol,
        "n": n, "n_yr": round(n_yr,1),
        "ev_per_trade": round(p.mean(),2),
        "annual_ev": round(p.sum()/YEARS,1),
        "win_%": round((p>0).mean()*100,1),
        "p10": round(np.quantile(p,0.10),2),
        "worst": round(p.min(),1),
    }

filters = [
    ("ALL T2×up",          m),
    ("drop high VIX",      m[m["vix_regime"]!="high"]),
    ("drop YM",            m[m["symbol"]!="YM"]),
    ("mid VIX only",       m[m["vix_regime"]=="mid"]),
    ("mid VIX × conc",     m[(m["vix_regime"]=="mid") & (m["alignment"]=="concurrent")]),
    ("mid VIX × inc",      m[(m["vix_regime"]=="mid") & (m["alignment"]=="incongruent")]),
    ("drop high VIX + YM", m[(m["vix_regime"]!="high") & (m["symbol"]!="YM")]),
]

# Best policies to try
policies = ["EOD", "SS50", "SS75", "T14:30", "T15:00"]

rows = []
for label, sub in filters:
    for pol in policies:
        rows.append(stats(sub, pol, label))

df = pd.DataFrame(rows)
print("\n=== Filter × Policy grid for T2 × gap_up momentum ===")
for label, sub in filters:
    block = df[df["filter"] == label]
    print(f"\n--- {label}  (n={len(sub)}, {len(sub)/YEARS:.1f}/yr) ---")
    print(block[["policy", "ev_per_trade", "annual_ev", "win_%", "worst"]].to_string(index=False))

# Best risk-adjusted: EV/|worst|
df["ev_per_worst"] = df.apply(lambda r: round(abs(r["annual_ev"]/r["worst"]), 2) if r["worst"]<0 else float("inf"), axis=1)
print("\n=== Top 10 combos sorted by annual_EV / |worst|  (Sharpe-ish) ===")
top = df[df["worst"] < 0].nlargest(10, "ev_per_worst")
print(top[["filter", "policy", "n_yr", "annual_ev", "worst", "ev_per_worst"]].to_string(index=False))
