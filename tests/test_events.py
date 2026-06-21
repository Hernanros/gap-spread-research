import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_daily
from lib.events import compute_atr, detect_events, _fill_pct


# --- ATR ---

def test_atr_first_value_nan():
    df = make_daily([100.0] * 20)
    atr = compute_atr(df, period=14)
    assert pd.isna(atr.iloc[0])


def test_atr_all_positive_after_warmup():
    df = make_daily([100.0] * 20)
    atr = compute_atr(df, period=14)
    assert (atr.dropna() > 0).all()


def test_atr_larger_range_yields_larger_atr():
    df_narrow = make_daily([100.0] * 20, highs=[101.0] * 20, lows=[99.0] * 20)
    df_wide = make_daily([100.0] * 20, highs=[105.0] * 20, lows=[95.0] * 20)
    assert compute_atr(df_wide).iloc[-1] > compute_atr(df_narrow).iloc[-1]


# --- detect_events ---

def test_detects_gap_up():
    closes = [100.0] * 20
    opens = closes[:]
    opens[-1] = 108.0  # large gap up on last day
    df = make_daily(closes, opens=opens)
    events = detect_events(df, pd.DataFrame())
    assert "gap_up" in events["direction"].values


def test_detects_gap_down():
    closes = [100.0] * 20
    opens = closes[:]
    opens[-1] = 92.0  # large gap down on last day
    df = make_daily(closes, opens=opens)
    events = detect_events(df, pd.DataFrame())
    assert "gap_down" in events["direction"].values


def test_no_event_small_gap():
    closes = [100.0] * 20
    opens = closes[:]
    opens[-1] = 100.05  # tiny gap, well below 1 ATR
    df = make_daily(closes, opens=opens)
    events = detect_events(df, pd.DataFrame())
    assert len(events) == 0


def test_gap_ratio_correct():
    closes = [100.0] * 20
    opens = closes[:]
    opens[-1] = 108.0
    df = make_daily(closes, opens=opens)
    events = detect_events(df, pd.DataFrame())
    gap_up = events[events["direction"] == "gap_up"].iloc[0]
    assert gap_up["gap"] == pytest.approx(8.0)
    assert gap_up["gap_ratio"] == pytest.approx(gap_up["gap"] / gap_up["atr"])


# --- _fill_pct ---

def _make_events_row(direction, open_, prev_close, gap, pm_high, pm_low):
    return pd.DataFrame(
        {
            "direction": [direction],
            "open": [open_],
            "prev_close": [prev_close],
            "gap": [gap],
            "premarket_high": [pm_high],
            "premarket_low": [pm_low],
        },
        index=pd.DatetimeIndex(["2024-06-03"]),
    )


def test_fill_pct_gap_up_full_fill():
    ev = _make_events_row("gap_up", open_=105, prev_close=100, gap=5,
                          pm_high=106, pm_low=100)
    assert _fill_pct(ev).iloc[0] == pytest.approx(1.0)


def test_fill_pct_gap_up_half_fill():
    ev = _make_events_row("gap_up", open_=105, prev_close=100, gap=5,
                          pm_high=106, pm_low=102.5)
    assert _fill_pct(ev).iloc[0] == pytest.approx(0.5)


def test_fill_pct_gap_up_no_fill_clipped():
    # pre-market low above open → no retracement at all
    ev = _make_events_row("gap_up", open_=105, prev_close=100, gap=5,
                          pm_high=107, pm_low=106)
    assert _fill_pct(ev).iloc[0] == pytest.approx(0.0)


def test_fill_pct_gap_down_full_fill():
    # gap down: open=95, prev_close=100, gap=-5
    # pm high = 100 → fully filled
    ev = _make_events_row("gap_down", open_=95, prev_close=100, gap=-5,
                          pm_high=100, pm_low=94)
    assert _fill_pct(ev).iloc[0] == pytest.approx(1.0)


def test_fill_pct_gap_down_half_fill():
    ev = _make_events_row("gap_down", open_=95, prev_close=100, gap=-5,
                          pm_high=97.5, pm_low=94)
    assert _fill_pct(ev).iloc[0] == pytest.approx(0.5)


def test_fill_pct_capped_at_one():
    # pre-market low went below prev_close → clip to 1.0
    ev = _make_events_row("gap_up", open_=105, prev_close=100, gap=5,
                          pm_high=106, pm_low=98)  # below prev_close
    assert _fill_pct(ev).iloc[0] == pytest.approx(1.0)
