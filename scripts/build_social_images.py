"""LinkedIn / Twitter — social-optimized visuals.

Square 1080x1080, large readable text, one key idea per image.
"""
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG    = "#0d1117"
GREEN = "#3fb950"
BLUE  = "#58a6ff"
GOLD  = "#e3b341"
RED   = "#f78166"
GREY  = "#6e7681"
GREY2 = "#8b949e"
WHITE = "#f0f6fc"
MUTED = "#30363d"

plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GREY, "axes.labelcolor": WHITE, "text.color": WHITE,
    "xtick.color": WHITE, "ytick.color": WHITE,
    "figure.dpi": 140, "font.family": "sans-serif",
})

import os
os.makedirs("charts", exist_ok=True)
events = pd.read_csv("cache/phase2_events_075.csv")


# ── Option A: "Only one regime survived" — bubble chart, social-optimized ────
def build_main():
    s = events.groupby(["vix_regime", "direction"]).agg(
        n=("eod_pnl", "size"),
        annual_ev=("eod_pnl", lambda x: x.sum() / 7),
        worst=("eod_pnl", "min"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10.8, 10.8))  # 1080x1080 at dpi=100
    fig.patch.set_facecolor(BG)

    # Title block
    fig.text(0.5, 0.94, "Only one regime survived.",
             ha="center", fontsize=30, fontweight="bold", color=WHITE)
    fig.text(0.5, 0.89, "Mid-VIX × gap-DOWN — across 7 years and 4 US futures.",
             ha="center", fontsize=15, color=GREY2)

    # Plot the 6 cells
    win_mask = (s["vix_regime"] == "mid") & (s["direction"] == "gap_down")
    for _, r in s.iterrows():
        is_winner = (r["vix_regime"] == "mid") and (r["direction"] == "gap_down")
        color = GOLD if is_winner else MUTED
        edge = WHITE if is_winner else GREY
        size = r["n"] * 8 if is_winner else r["n"] * 4
        ax.scatter(r["worst"], r["annual_ev"], s=size,
                   c=color, alpha=0.95 if is_winner else 0.5,
                   edgecolor=edge, linewidths=2.2 if is_winner else 1.0,
                   zorder=10 if is_winner else 5)

    # Label each cell
    for _, r in s.iterrows():
        is_winner = (r["vix_regime"] == "mid") and (r["direction"] == "gap_down")
        label = f"{r['vix_regime']}-VIX · {r['direction'].replace('_', ' ')}"
        offset_y = 14 if r["annual_ev"] < 100 else -22
        ax.annotate(label, (r["worst"], r["annual_ev"]),
                    xytext=(15, offset_y), textcoords="offset points",
                    fontsize=12 if not is_winner else 14,
                    color=WHITE if is_winner else GREY2,
                    fontweight="bold" if is_winner else "normal")

    # Highlight the winner with a callout
    winner = s[win_mask].iloc[0]
    ax.annotate(
        f"+{winner['annual_ev']:.0f} pts/yr\nworst trade: {winner['worst']:.0f} pts",
        xy=(winner["worst"], winner["annual_ev"]),
        xytext=(-340, 90), textcoords="offset points",
        fontsize=16, color=GOLD, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=2),
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#1c2026",
                  edgecolor=GOLD, linewidth=1.5),
    )

    ax.axhline(0, color=GREY, lw=0.8, alpha=0.5)
    ax.set_xlabel("Worst single trade (pts)", fontsize=13, color=GREY2)
    ax.set_ylabel("Annual EV (pts per single contract)", fontsize=13, color=GREY2)
    ax.grid(alpha=0.12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)

    # Bottom credit
    fig.text(0.5, 0.04,
             "6 cells tested · 5 failed · only one delivered positive EV with a bounded tail",
             ha="center", fontsize=12, color=GREY2, style="italic")
    fig.text(0.5, 0.01, "@hernan.rosenblum — gap-spread research",
             ha="center", fontsize=10, color=GREY)

    plt.subplots_adjust(left=0.10, right=0.95, top=0.86, bottom=0.10)
    out = "charts/social_main_only_one_regime.png"
    plt.savefig(out, dpi=100, facecolor=BG, bbox_inches=None)
    print(f"Written: {out}")
    plt.close()


# ── Option B: Walk-forward — IS vs OOS, square crop ────────────────────────────
def build_walkforward():
    fig, ax = plt.subplots(figsize=(10.8, 10.8))
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.94, "Trained on 5 years. Held out 2.5.",
             ha="center", fontsize=30, fontweight="bold", color=WHITE)
    fig.text(0.5, 0.89, "Out-of-sample, both rules held — and improved.",
             ha="center", fontsize=15, color=GREY2)

    rules = ["R1\n(gap-down)", "M1\n(gap-up)"]
    is_vals  = [2112, 4188]
    oos_vals = [3622, 8260]

    x = np.arange(len(rules))
    w = 0.36
    b1 = ax.bar(x - w/2, is_vals,  w, label="2019–2023 (in-sample)",
                color=BLUE, alpha=0.9, edgecolor=WHITE, linewidth=1.2)
    b2 = ax.bar(x + w/2, oos_vals, w, label="2024–2026 (held-out)",
                color=GOLD, alpha=0.95, edgecolor=WHITE, linewidth=1.2)

    for bars, vals in [(b1, is_vals), (b2, oos_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 200,
                    f"${v:,}", ha="center", fontsize=16, fontweight="bold",
                    color=WHITE)

    ax.set_xticks(x)
    ax.set_xticklabels(rules, fontsize=18, fontweight="bold")
    ax.set_ylabel("Annual EV (USD, after slippage)", fontsize=13, color=GREY2)
    ax.legend(loc="upper left", fontsize=13, frameon=False)
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 10500)
    ax.tick_params(labelsize=11)

    fig.text(0.5, 0.04,
             "If a backtested rule survives an untouched holdout, the edge is more credible.",
             ha="center", fontsize=12, color=GREY2, style="italic")
    fig.text(0.5, 0.01, "@hernan.rosenblum — gap-spread research",
             ha="center", fontsize=10, color=GREY)

    plt.subplots_adjust(left=0.13, right=0.95, top=0.86, bottom=0.10)
    out = "charts/social_walkforward.png"
    plt.savefig(out, dpi=100, facecolor=BG, bbox_inches=None)
    print(f"Written: {out}")
    plt.close()


# ── Option C: The trichotomy — three panels, low/mid/high VIX ────────────────
def build_trichotomy():
    fig = plt.figure(figsize=(10.8, 10.8))
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.95, "The signal only lives in the middle.",
             ha="center", fontsize=28, fontweight="bold", color=WHITE)
    fig.text(0.5, 0.91, "Why gap-trading on US equity index futures only works at mid VIX.",
             ha="center", fontsize=13, color=GREY2)

    panels = [
        ("Low VIX  (<15)",     "Premium too thin",
         "Credit collected is small;\nslippage eats the edge.",  RED),
        ("Mid VIX  (15–25)",   "The sweet spot",
         "Bounded losses + meaningful credit.\nMean-reversion structurally bounded.", GOLD),
        ("High VIX (>25)",     "Tails dominate",
         "One trade in the wrong direction\nwipes a year of gains.",   RED),
    ]

    for i, (title, headline, body, color) in enumerate(panels):
        ax = fig.add_axes([0.05 + i*0.30, 0.18, 0.27, 0.65])
        ax.set_facecolor("#161b22")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.2)
        # Symbol
        symbol = "✗" if color == RED else "✓"
        ax.text(0.5, 0.80, symbol, ha="center", va="center", fontsize=64,
                color=color, transform=ax.transAxes, fontweight="bold")
        ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=17,
                color=WHITE, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.46, headline, ha="center", va="center", fontsize=14,
                color=color, transform=ax.transAxes, fontweight="bold")
        ax.text(0.5, 0.22, body, ha="center", va="center", fontsize=11,
                color=GREY2, transform=ax.transAxes)

    fig.text(0.5, 0.10,
             "Filtering out the extremes was what turned a losing strategy into a winning one.",
             ha="center", fontsize=12, color=WHITE, style="italic")
    fig.text(0.5, 0.06,
             "5-of-6 stratified regimes failed. The one that worked had no positive analog.",
             ha="center", fontsize=11, color=GREY2)
    fig.text(0.5, 0.02, "@hernan.rosenblum — gap-spread research",
             ha="center", fontsize=10, color=GREY)

    out = "charts/social_trichotomy.png"
    plt.savefig(out, dpi=100, facecolor=BG, bbox_inches=None)
    print(f"Written: {out}")
    plt.close()


build_main()
build_walkforward()
build_trichotomy()
print("\nAll 3 social images built at 1080x1080 (square).")
