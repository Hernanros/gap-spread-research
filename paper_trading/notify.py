"""Push notifications. Free, dual-channel.

1. macOS local notification (osascript) — banner appears on your Mac
2. ntfy.sh push to phone — if NTFY_TOPIC env var is set

Setup for phone push (one-time, free):
  1. Install ntfy app on phone (iOS App Store or Google Play, free)
  2. Pick a unique topic name (e.g., "hr-gap-trader-x7k2")
  3. Subscribe in the app to that topic
  4. Add `NTFY_TOPIC=hr-gap-trader-x7k2` to your .env file
  5. Done — push notifications now reach your phone

Usage:
  from paper_trading.notify import notify
  notify("R1 signal fires on ES", title="Gap Trader")
"""
import os
import subprocess


def notify(message: str, title: str = "Gap Trader", priority: str = "default"):
    """Fire-and-forget notification. macOS banner + optional ntfy.sh phone push."""
    # 1. macOS local
    try:
        # Escape double-quotes for AppleScript
        safe_msg = message.replace('"', '\\"')
        safe_title = title.replace('"', '\\"')
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{safe_msg}" with title "{safe_title}" sound name "Glass"'],
            timeout=5, check=False,
        )
    except Exception:
        pass

    # 2. ntfy.sh push (if configured)
    topic = os.environ.get("NTFY_TOPIC")
    if topic:
        try:
            import urllib.request
            req = urllib.request.Request(
                f"https://ntfy.sh/{topic}",
                data=message.encode("utf-8"),
                headers={
                    "Title": title,
                    "Priority": priority,
                    "Tags": "chart_with_upwards_trend",
                },
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5).read()
        except Exception:
            pass


if __name__ == "__main__":
    import sys
    msg = " ".join(sys.argv[1:]) or "Test notification from gap-trader"
    notify(msg)
    print("Notification sent.")
