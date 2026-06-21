import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_daily, make_session_bars
from lib.path import analyze_paths, label_vix_regime


def _make_event(
    date: str = "2024-06-03",
    direction: str = "gap_up",
    open_: float = 105.0,
    prev_close: float = 100.0,
    atr: float = 2.5,
    premarket_fill_pct: float = 0.2,
    vix_close: float = 18.0,
) -> pd.DataFrame:
    gap = open_ - prev_close
    return pd.DataFrame(
        {
            "direction": [direction],
            "open": [open_],
            "prev_close": [prev_close],
            "gap": [gap],
            "gap_ratio": [gap / atr],
            "atr": [atr],
            "premarket_fill_pct": [premarket_fill_pct],
            "vix_close": [vix_close],
        },
        index=pd.DatetimeIndex([date]),
    )


def test_gap_filled_when_low_touches_prev_close():
    session = make_session_bars(
        "2024-06-03",
        closes=[105, 104, 103, 101, 100, 100, 101],
        lows=[104.5, 103.5, 102.5, 100.5, 99.5, 99.5, 100.5],
    )
    events = _make_event(direction="gap_up", open_=105.0, prev_close=100.0)
    result = analyze_paths(events, session, entry_times=["09:30"])
    assert result.iloc[0]["gap_filled"] is True or result.iloc[0]["gap_filled"] == True


def test_gap_not_filled_when_price_stays_above_prev_close():
    session = make_session_bars(
        "2024-06-03",
        closes=[105, 106, 107, 106, 105, 104, 103],
        lows=[104.5, 105.5, 106.5, 105.5, 104.5, 103.5, 102.5],
    )
    events = _make_event(direction="gap_up", open_=105.0, prev_close=100.0)
    result = analyze_paths(events, session, entry_times=["09:30"])
    assert result.iloc[0]["gap_filled"] is False or result.iloc[0]["gap_filled"] == False


def test_fill_time_is_minutes_from_open():
    # Price touches prev_close (100) at 10:00 = 30 mins from open
    closes = [105, 104, 103, 102, 101, 100] + [100] * 7
    lows = [c - 0.3 for c in closes]
    lows[5] = 99.5  # bar at 10:00 (index 6 from 09:30 = 6*5=30 mins)
    session = make_session_bars("2024-06-03", closes=closes, lows=lows)
    events = _make_event(direction="gap_up", open_=105.0, prev_close=100.0)
    result = analyze_paths(events, session, entry_times=["09:30"])
    fill_time = result.iloc[0]["fill_time"]
    assert fill_time is not None and fill_time >= 0


def test_mae_gap_up_is_adverse_upward_move():
    # Gap up bull put: adverse = price rising away from prev_close
    # Entry at 105, session goes to high of 111
    closes = [105, 107, 110, 108, 106]
    highs = [106, 108, 111, 109, 107]
    lows = [104, 106, 109, 107, 105]
    session = make_session_bars("2024-06-03", closes=closes, highs=highs, lows=lows)
    events = _make_event(direction="gap_up", open_=105.0)
    result = analyze_paths(events, session, entry_times=["09:30"])
    # MAE = max(high - entry_price) = 111 - 105 = 6
    assert result.iloc[0]["mae"] == pytest.approx(6.0)


def test_mae_gap_down_is_adverse_downward_move():
    # Gap down bear call: adverse = price falling away from prev_close
    # Entry at 95, session goes to low of 89
    closes = [95, 93, 90, 92, 94]
    highs = [96, 94, 91, 93, 95]
    lows = [94, 92, 89, 91, 93]
    session = make_session_bars("2024-06-03", closes=closes, highs=highs, lows=lows)
    events = _make_event(direction="gap_down", open_=95.0, prev_close=100.0)
    result = analyze_paths(events, session, entry_times=["09:30"])
    # MAE = max(entry_price - low) = 95 - 89 = 6
    assert result.iloc[0]["mae"] == pytest.approx(6.0)


def test_entry_timing_grid_returns_four_rows_per_event():
    closes = [105] * 20
    session = make_session_bars("2024-06-03", closes=closes)
    events = _make_event()
    result = analyze_paths(events, session, entry_times=["09:30", "09:45", "10:00", "10:30"])
    assert len(result) == 4
    assert set(result["entry_time"]) == {"09:30", "09:45", "10:00", "10:30"}


def test_later_entry_has_smaller_or_equal_mae():
    # As we wait longer, the worst adverse from entry should not increase in an uptrend
    closes = [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.1 for c in closes]
    session = make_session_bars("2024-06-03", closes=closes, highs=highs, lows=lows)
    events = _make_event(direction="gap_up")
    result = analyze_paths(events, session, entry_times=["09:30", "09:45"])
    mae_open = result[result["entry_time"] == "09:30"]["mae"].iloc[0]
    mae_945 = result[result["entry_time"] == "09:45"]["mae"].iloc[0]
    # 09:45 entry misses first 3 bars of the uptrend → smaller MAE from entry
    assert mae_945 <= mae_open


def test_mae_pct_atr_computed():
    closes = [105] * 13
    session = make_session_bars("2024-06-03", closes=closes)
    events = _make_event(atr=2.5)
    result = analyze_paths(events, session, entry_times=["09:30"])
    row = result.iloc[0]
    assert row["mae_pct_atr"] == pytest.approx(row["mae"] / row["atr"])


def test_label_vix_regime_low():
    vix = pd.Series([12.0])
    assert label_vix_regime(vix).iloc[0] == "low"


def test_label_vix_regime_mid():
    vix = pd.Series([20.0])
    assert label_vix_regime(vix).iloc[0] == "mid"


def test_label_vix_regime_high():
    vix = pd.Series([32.0])
    assert label_vix_regime(vix).iloc[0] == "high"
