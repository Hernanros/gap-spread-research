#!/bin/bash
# Cron wrapper — runs morning_scan, parses output, sends notification if any signal fires.
#
# Add to crontab (your local time, ~9:35 ET = 16:35 IDT during Israel summer DST):
#   35 16 * * 1-5  /Users/hernanrosenblum/Documents/gap-spread-research/scripts/cron_morning.sh
#
# Logs to logs/cron_morning_YYYY-MM-DD.log

set -euo pipefail

REPO="/Users/hernanrosenblum/Documents/gap-spread-research"
cd "$REPO"

mkdir -p logs
LOGFILE="logs/cron_morning_$(date +%Y-%m-%d).log"

# Activate venv
source venv/bin/activate

# Load env vars (e.g. NTFY_TOPIC for push to phone)
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs) 2>/dev/null || true
fi

# Run scan; capture stdout for parsing
SCAN_OUTPUT="$(python paper_trading/morning_scan.py 2>&1)"
echo "$SCAN_OUTPUT" > "$LOGFILE"

# Count firing signals from the output table (lines containing "FIRES")
FIRES=$(echo "$SCAN_OUTPUT" | grep -c "FIRES" || true)

if [ "$FIRES" -gt 0 ]; then
  # Extract the symbols that fired
  SIGNAL_SYMBOLS=$(echo "$SCAN_OUTPUT" | grep "FIRES" | awk '{print $1}' | tr '\n' ' ')
  MSG="$FIRES signal(s) fire today: $SIGNAL_SYMBOLS — pull IBKR quote and run decision_today.py"
  python -m paper_trading.notify "$MSG" 2>/dev/null || true
  echo "[NOTIFICATION SENT] $MSG" >> "$LOGFILE"
else
  echo "[no signals — silent]" >> "$LOGFILE"
fi

# Tail of output goes to syslog-ish dump too
echo "---END $(date)---" >> "$LOGFILE"
