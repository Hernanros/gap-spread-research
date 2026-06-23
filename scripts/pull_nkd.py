"""Pull NKD (Nikkei 225 USD futures) 1-min OHLCV from Databento — cost ~$6.26."""
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

import os
import databento as db
import pandas as pd

c = db.Historical()

print("Confirming cost before fetch...")
cost = c.metadata.get_cost(
    dataset="GLBX.MDP3",
    symbols=["NKD.c.0"],
    stype_in="continuous",
    schema="ohlcv-1m",
    start="2019-01-01",
    end="2026-06-22",
)
print(f"  Cost: ${cost:.4f}")
assert cost < 10.0, f"Cost {cost} exceeded safety threshold $10"

print("\nFetching NKD 1-min OHLCV...")
data = c.timeseries.get_range(
    dataset="GLBX.MDP3",
    symbols=["NKD.c.0"],
    stype_in="continuous",
    schema="ohlcv-1m",
    start="2019-01-01",
    end="2026-06-22",
)
df = data.to_df()
print(f"\nFetched {len(df):,} rows")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 3 rows:")
print(df.head(3))
print(f"\nLast 3 rows:")
print(df.tail(3))

# Match the format of existing ES/NQ/RTY/YM CSVs:
#   ts_event,open,high,low,close,volume
#   2019-01-01 23:00:00+00:00,...
out = df[["open", "high", "low", "close", "volume"]].copy()
out.index.name = "ts_event"

# Ensure UTC tz-aware string format
if out.index.tz is None:
    out.index = out.index.tz_localize("UTC")

out_path = "data/NKD_1m_2019_2026.csv"
out.to_csv(out_path)
print(f"\nWritten: {out_path}")
print(f"File size: {os.path.getsize(out_path)/1024/1024:.1f} MB")
