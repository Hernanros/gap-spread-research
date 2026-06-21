import pandas as pd
import numpy as np
import pytest
from lib.spreads import (
    _bs_put,
    _bs_call,
    price_spread,
    simulate_exits,
    compute_metrics,
)


# --- Black-Scholes ---

def test_bs_put_intrinsic_at_zero_time():
    assert _bs_put(S=95, K=100, T=0, r=0.05, sigma=0.20) == pytest.approx(5.0)


def test_bs_put_zero_when_otm_at_zero_time():
    assert _bs_put(S=105, K=100, T=0, r=0.05, sigma=0.20) == pytest.approx(0.0)


def test_bs_call_intrinsic_at_zero_time():
    assert _bs_call(S=105, K=100, T=0, r=0.05, sigma=0.20) == pytest.approx(5.0)


def test_bs_call_zero_when_otm_at_zero_time():
    assert _bs_call(S=95, K=100, T=0, r=0.05, sigma=0.20) == pytest.approx(0.0)


def test_bs_put_positive_for_positive_time():
    val = _bs_put(S=100, K=100, T=1 / 252, r=0.05, sigma=0.20)
    assert val > 0


def test_bs_put_atm_greater_than_otm():
    atm = _bs_put(S=100, K=100, T=1 / 252, r=0.05, sigma=0.20)
    otm = _bs_put(S=100, K=90, T=1 / 252, r=0.05, sigma=0.20)
    assert atm > otm


# --- price_spread ---

def test_price_spread_gap_up_positive_credit():
    result = price_spread(
        S=105, K_short=100, atr=5, direction="gap_up",
        vix=20, r=0.05, dte=1, width_atr=0.5,
    )
    assert result["credit"] >= 0
    assert result["K_long"] < result["K_short"]
    assert result["spread_width"] == pytest.approx(2.5)


def test_price_spread_gap_down_positive_credit():
    result = price_spread(
        S=95, K_short=100, atr=5, direction="gap_down",
        vix=20, r=0.05, dte=1, width_atr=0.5,
    )
    assert result["credit"] >= 0
    assert result["K_long"] > result["K_short"]


def test_price_spread_higher_vix_higher_credit():
    low_vix = price_spread(S=105, K_short=100, atr=5, direction="gap_up", vix=15)
    high_vix = price_spread(S=105, K_short=100, atr=5, direction="gap_up", vix=30)
    assert high_vix["credit"] > low_vix["credit"]


# --- simulate_exits ---

def _make_path_df(**kwargs) -> pd.DataFrame:
    defaults = dict(
        date=pd.Timestamp("2024-06-03"),
        ticker="SPY",
        direction="gap_up",
        entry_time="09:30",
        entry_price=105.0,
        prev_close=100.0,
        atr=5.0,
        mae=1.0,
        eod_close=103.0,
        vix_close=20.0,
        gap_filled=True,
        gap_ratio=2.0,
        premarket_fill_pct=0.3,
    )
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


def test_eod_win_when_price_stays_above_short_strike():
    # Gap up, EOD close above prev_close → spread expires worthless → full credit
    df = _make_path_df(direction="gap_up", prev_close=100, eod_close=103, mae=0.5)
    exits = simulate_exits(df, profit_targets=[], stop_multiples=[])
    eod = exits[exits["exit_strategy"] == "EOD"].iloc[0]
    assert eod["win"] is True or eod["win"] == True
    assert eod["pnl"] > 0


def test_eod_loss_when_price_breaches_short_strike():
    # Gap up, EOD close well below prev_close (deep in the spread)
    df = _make_path_df(direction="gap_up", prev_close=100, eod_close=95, mae=6.0, atr=5.0)
    exits = simulate_exits(df, profit_targets=[], stop_multiples=[])
    eod = exits[exits["exit_strategy"] == "EOD"].iloc[0]
    assert eod["pnl"] < eod["credit"]


def test_stop_loss_triggers_caps_loss():
    # MAE larger than 1× credit → stop triggers
    df = _make_path_df(mae=100.0)  # absurdly large MAE to guarantee trigger
    exits = simulate_exits(df, profit_targets=[], stop_multiples=[1])
    sl = exits[exits["exit_strategy"] == "SL1x"].iloc[0]
    credit = sl["credit"]
    assert sl["pnl"] == pytest.approx(-credit, abs=1e-6)


def test_profit_target_win_when_mae_small():
    # Small MAE → price barely moved → PT50 hit
    df = _make_path_df(mae=0.01)
    exits = simulate_exits(df, profit_targets=[0.50], stop_multiples=[])
    pt = exits[exits["exit_strategy"] == "PT50"].iloc[0]
    credit = pt["credit"]
    assert pt["pnl"] == pytest.approx(0.50 * credit, abs=1e-6)


def test_all_exit_strategies_present():
    df = _make_path_df()
    exits = simulate_exits(df, profit_targets=[0.25, 0.50, 0.75], stop_multiples=[1, 2, 3])
    strategies = set(exits["exit_strategy"])
    assert strategies == {"EOD", "PT25", "PT50", "PT75", "SL1x", "SL2x", "SL3x"}


# --- compute_metrics ---

def test_compute_metrics_win_rate():
    rows = [
        _make_path_df(eod_close=103).iloc[0].to_dict(),  # win
        _make_path_df(eod_close=103).iloc[0].to_dict(),  # win
        _make_path_df(eod_close=95, mae=10.0).iloc[0].to_dict(),  # likely loss
    ]
    path_df = pd.DataFrame(rows)
    exits = simulate_exits(path_df, profit_targets=[], stop_multiples=[])
    metrics = compute_metrics(exits)
    eod = metrics[
        (metrics["exit_strategy"] == "EOD") & (metrics["entry_time"] == "09:30")
    ].iloc[0]
    assert 0 < eod["win_rate"] <= 1.0


def test_compute_metrics_profit_factor_positive():
    rows = [_make_path_df(eod_close=103).iloc[0].to_dict() for _ in range(5)]
    rows += [_make_path_df(eod_close=95, mae=10.0).iloc[0].to_dict()]
    path_df = pd.DataFrame(rows)
    exits = simulate_exits(path_df, profit_targets=[], stop_multiples=[])
    metrics = compute_metrics(exits)
    eod = metrics[metrics["exit_strategy"] == "EOD"].iloc[0]
    assert eod["profit_factor"] > 0
