# LinkedIn Post — Gap Research

---

A trader I follow made a credit-spread trade on ES last Monday. Took him 5 minutes.

It looked routine for him. I stared at the chart for a week trying to understand whether it was a real pattern, or just experience that looks like a pattern after the fact.
Here’s what surfaced.

>>

I broke the market into volatility regimes, gap directions, and trend contexts.
Most of it was noise.
Only one mean-reversion regime survived every cut.
When the gap is DOWN and VIX sits in a mid-range band (roughly 15–25), price tends to drift back toward the previous day’s close into the session end. Across ES, NQ and YM over 7 years — the structurewas surprisingly consistent.
Outside that regime, the edge disappears. Low volatility → too weak. High volatility → dominated by tails. The signal only lives in the middle.

>>

The skeptic in me didn’t let it pass.
“Maybe this is just selection bias.”
So I split the data properly: 2019–2023 as in-sample, 2024–2026 held out, no peeking.
The pattern didn’t break out-of-sample — if anything, it held cleaner than expected.

>>

Next question: real trading costs.
Slippage and spread were applied to every trade. The edge compressed, but didn’t disappear. Sharper under friction, but still positive.
One market (RTY) failed the holdout entirely. It was removed.

>>

After all the filtering:

→ 4 instruments (ES, NQ, YM, NKD)→ 2 distinct setups (mean-reversion + momentum extension variants)→ ~50 trades per year after costs


The headline I started with was: “I built a strategy.”
The more accurate version is:
I found a market condition where a simple structure keeps working — and learned exactly where it stops.
That second version is the only one that actually matters.

Paper trading starts July 1. 

Full research trail is documented for anyone who wants to dig deeper. what you think?

