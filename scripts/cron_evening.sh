#!/bin/bash
# Cron wrapper — runs evening check after US close, notifies about open trades to close.
#
# Add to crontab (~16:10 ET = 23:10 IDT during Israel summer DST):
#   10 23 * * 1-5  /Users/hernanrosenblum/Documents/gap-spread-research/scripts/cron_evening.sh

set -euo pipefail
REPO="/Users/hernanrosenblum/Documents/gap-spread-research"
cd "$REPO"
mkdir -p logs
LOGFILE="logs/cron_evening_$(date +%Y-%m-%d).log"

source venv/bin/activate
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs) 2>/dev/null || true
fi

OUT="$(python paper_trading/evening_check.py 2>&1)"
echo "$OUT" > "$LOGFILE"

# Count OPEN positions
OPEN=$(echo "$OUT" | grep -c "OPEN POSITION" || true)
if [ "$OPEN" -gt 0 ]; then
  python -m paper_trading.notify "Open trades to close. Run record_exit.py." \
    "Gap Trader · close-of-day" 2>/dev/null || true
  echo "[NOTIFIED — open trades present]" >> "$LOGFILE"
fi
echo "---END $(date)---" >> "$LOGFILE"
