---
title: Gap Credit Spread Strategy — v1 Spec
date: 2026-06-21
context: Crystallized from Socratic exploration session
---

## The Strategy (v1)

**Universe:** ES, NQ, RTY, YM futures (primary) + SPY, QQQ, IWM, IWB ETFs (proxy)

**Trigger:** Overnight gap ≥ 1 ATR(14) in either direction

**Trade:**
- Gap UP → Bull Put Spread (short put at prior day's close)
- Gap DOWN → Bear Call Spread (short call at prior day's close)

**Entry:** Market open (or futures pre-market open)

**Spread width:** 0.5 ATR (to be optimized)

**Exit rules (first hit wins):**
1. 60% of max profit reached → close position
2. 3:00 PM ET hard stop → close regardless

**Sub-questions to investigate:**
- Does gap-DOWN outperform gap-UP? (hypothesis: fear selling overshoots more than euphoria)
- What is the optimal profit target? (test 25%, 50%, 75%, EOD)
- What is the actual same-day gap fill rate at ≥1 ATR?
- Does strategy hold across different VIX regimes?
- Is there a sweet spot ATR threshold (1.0, 1.5, 2.0)?

## Outputs Needed
1. Trading rule (backtested, statistically sound)
2. Interview presentation (rigorous methodology walkthrough)
3. Social media infographics (clean, visual)

## Data Decision
- **Provider:** Databento (pay-as-you-go, $125 free credits)
- **Resolution:** 15-min bars
- **History:** 5 years
- **Symbols:** ES, NQ, RTY, YM (CME Globex) + VIX intraday
- **Action needed:** User must sign up at databento.com and get API key

## Origin
Inspired by a friend's live trade: 2.5% gap-up on ES+NQ, sold bull put spread at prior close,
held until ~90 min before close, captured ~95% of max profit.
