# Cron Setup — Automated Daily Workflow

Hands-off daily routine. The cron jobs run morning_scan and evening_check on schedule and push a notification to your Mac + phone when a signal fires.

## One-time setup

### 1. Phone push notifications (optional but recommended — free)

1. Install **ntfy** on your phone (iOS App Store or Google Play, free, no signup)
2. Pick a unique topic name, e.g. `hr-gap-trader-7x2k`
3. In the ntfy app: tap `+` → enter topic name → subscribe
4. Add the topic to `.env`:
   ```bash
   echo 'NTFY_TOPIC=hr-gap-trader-7x2k' >> .env
   ```
5. Test it:
   ```bash
   source venv/bin/activate
   python paper_trading/notify.py "Test from gap-trader"
   ```
   Should appear on phone + Mac banner.

If you skip this, you'll still get Mac banner notifications (no phone push).

### 2. Install the cron jobs

Open crontab:
```bash
crontab -e
```

Add these two lines (times are **Israel local** — adjust if you travel):

```cron
# Morning scan — 9:35 ET = 16:35 IDT (Israel DST) Mon-Fri
35 16 * * 1-5  /Users/hernanrosenblum/Documents/gap-spread-research/scripts/cron_morning.sh

# Evening check — 16:10 ET = 23:10 IDT Mon-Fri
10 23 * * 1-5  /Users/hernanrosenblum/Documents/gap-spread-research/scripts/cron_evening.sh
```

Save and exit. Verify with `crontab -l`.

### 3. Allow cron to run scripts (macOS gotcha)

macOS Catalina+ blocks cron from accessing most folders by default. Grant access:

1. **System Settings → Privacy & Security → Full Disk Access**
2. Click `+` → Cmd-Shift-G → type `/usr/sbin/cron` → add
3. Toggle it ON

Without this, the cron will silently fail to read your Documents folder.

## What happens automatically

| Time | Job | If signal fires |
|---|---|---|
| 16:35 IDT (9:35 ET) | `morning_scan.py` | Mac banner + phone push: "N signal(s) fire today: ES NQ — pull IBKR" |
| 23:10 IDT (16:10 ET) | `evening_check.py` | Mac banner + phone push if there's an open trade to close |

Logs go to `logs/cron_morning_YYYY-MM-DD.log` and `logs/cron_evening_YYYY-MM-DD.log` so you can debug if a notification doesn't arrive.

## When a signal fires — your manual steps

1. Notification arrives on phone/Mac with the firing symbols
2. Open the latest log: `cat logs/cron_morning_$(date +%Y-%m-%d).log`
3. Pull IBKR quote at the printed strikes
4. Run:
   ```bash
   python paper_trading/decision_today.py --symbol ES --open ... \
     --prev_close ... --atr ... --vix ... --rule R1 \
     --market_credit_usd <USD> --market_pop_pct <PCT>
   ```
5. If TAKE: `record_entry.py`
6. At close (after evening notification): `record_exit.py`

## Important data-freshness note

yfinance daily closes use **Globex aggregate**, which differs from your broker's RTH 4pm close by ~$10–100. The cron-fired `morning_scan.py` uses yfinance defaults. If a signal flags borderline, override with the broker's actual numbers:

```bash
python paper_trading/morning_scan.py \
    --manual_prev_close ES=7547,NQ=30659 \
    --manual_atr ES=114.5,NQ=749.64 --manual_vix 16.4
```

Eventually we'd refresh local Databento data weekly (with your permission) so the script uses RTH data without manual overrides.

## Options-chain auto-fetch

**Currently:** `decision_today.py` asks you to paste the IBKR market credit manually. There is no reliable free real-time CME futures option chain feed.

**Path forward:**
- yfinance has SPY/QQQ option chains (proxies for ES/NQ at ~10× scale) — could be added as a rough fallback
- Databento has the real ES/NQ chain but costs ~$5 for a year of daily data; **will not pull without your explicit approval**
- IBKR API (`ib_insync`) can pull live chains if you have TWS running — engineering effort but doable

When you want me to add the SPY/QQQ proxy or wire IBKR auto-pull, just say so.

## Stopping the cron

```bash
crontab -e         # remove the two lines
# or
crontab -r         # remove all your cron jobs
```
