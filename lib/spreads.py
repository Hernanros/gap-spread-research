from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

PROFIT_TARGETS: list[float] = [0.25, 0.50, 0.75]
STOP_MULTIPLES: list[int] = [1, 2, 3]
RISK_FREE_RATE: float = 0.05
SPREAD_WIDTH_ATR: float = 0.5
DTE: int = 1


def _bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def price_spread(
    S: float,
    K_short: float,
    atr: float,
    direction: str,
    vix: float,
    r: float = RISK_FREE_RATE,
    dte: int = DTE,
    width_atr: float = SPREAD_WIDTH_ATR,
) -> dict:
    """Price a credit spread using Black-Scholes.

    Args:
        S: Entry price (underlying at time of entry).
        K_short: Short strike = prev_close (ATM for the spread).
        atr: ATR value for computing spread width.
        direction: 'gap_up' → bull put spread; 'gap_down' → bear call spread.
        vix: VIX close, used as annualized IV proxy (vix / 100).
        r: Risk-free rate.
        dte: Days to expiration.
        width_atr: Spread width as multiple of ATR.

    Returns:
        dict with keys: credit, spread_width, K_short, K_long, sigma.

    Note: All P&L values from this function are SYNTHETIC (Black-Scholes + VIX proxy).
    """
    sigma = vix / 100.0
    T = dte / 252.0
    spread_width = width_atr * atr

    if direction == "gap_up":
        K_long = K_short - spread_width
        credit = _bs_put(S, K_short, T, r, sigma) - _bs_put(S, K_long, T, r, sigma)
    else:
        K_long = K_short + spread_width
        credit = _bs_call(S, K_short, T, r, sigma) - _bs_call(S, K_long, T, r, sigma)

    return {
        "credit": max(credit, 0.0),
        "spread_width": spread_width,
        "K_short": K_short,
        "K_long": K_long,
        "sigma": sigma,
    }


def simulate_exits(
    path_df: pd.DataFrame,
    profit_targets: Sequence[float] = PROFIT_TARGETS,
    stop_multiples: Sequence[int] = STOP_MULTIPLES,
    r: float = RISK_FREE_RATE,
    dte: int = DTE,
    width_atr: float = SPREAD_WIDTH_ATR,
) -> pd.DataFrame:
    """Simulate exit strategies for each row in path_df.

    Each event × entry_time row produces N exit rows
    (1 EOD + len(profit_targets) PTs + len(stop_multiples) SLs).

    Note: P&L values are SYNTHETIC — Black-Scholes + VIX/100 as IV proxy.
    """
    records: list[dict] = []

    for _, row in path_df.iterrows():
        vix = float(row.get("vix_close", 20.0))
        if np.isnan(vix):
            vix = 20.0

        sp = price_spread(
            S=float(row["entry_price"]),
            K_short=float(row["prev_close"]),
            atr=float(row["atr"]),
            direction=str(row["direction"]),
            vix=vix,
            r=r,
            dte=dte,
            width_atr=width_atr,
        )
        credit = sp["credit"]
        width = sp["spread_width"]
        mae = float(row["mae"])
        eod_close = float(row["eod_close"])
        prev_close = float(row["prev_close"])
        direction = str(row["direction"])

        # EOD P&L: intrinsic value of short option at close
        if direction == "gap_up":
            intrinsic = min(max(prev_close - eod_close, 0.0), width)
        else:
            intrinsic = min(max(eod_close - prev_close, 0.0), width)
        eod_pnl = credit - intrinsic

        def _record(exit_name: str, pnl: float) -> dict:
            return {
                "date": row["date"],
                "ticker": row.get("ticker", ""),
                "direction": direction,
                "entry_time": row["entry_time"],
                "exit_strategy": exit_name,
                "credit": credit,
                "spread_width": width,
                "sigma": sp["sigma"],
                "mae": mae,
                "pnl": pnl,
                "win": bool(pnl > 0),
                "gap_ratio": float(row.get("gap_ratio", np.nan)),
                "premarket_fill_pct": float(row.get("premarket_fill_pct", np.nan)),
                "vix_close": vix,
                "gap_filled": bool(row.get("gap_filled", False)),
            }

        records.append(_record("EOD", eod_pnl))

        for pt in profit_targets:
            target_hit = (mae < (1 - pt) * width) if width > 0 else False
            pnl = pt * credit if target_hit else eod_pnl
            records.append(_record(f"PT{int(pt * 100)}", pnl))

        for sl in stop_multiples:
            stop_level = sl * credit
            pnl = -stop_level if mae > stop_level else eod_pnl
            records.append(_record(f"SL{sl}x", pnl))

    return pd.DataFrame(records)


def compute_metrics(exits_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate win rate, expectancy, and profit factor per exit × entry_time × direction."""
    records: list[dict] = []
    group_cols = ["exit_strategy", "entry_time", "direction"]

    for keys, group in exits_df.groupby(group_cols, observed=True):
        exit_strat, entry_time, direction = keys
        wins = group[group["win"]]
        losses = group[~group["win"]]
        n = len(group)
        win_rate = len(wins) / n
        avg_win = float(wins["pnl"].mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses["pnl"].mean()) if len(losses) > 0 else 0.0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        total_loss = float(losses["pnl"].sum())
        profit_factor = float(wins["pnl"].sum()) / abs(total_loss) if total_loss != 0 else np.inf
        avg_credit = float(group["credit"].mean())

        records.append(
            {
                "exit_strategy": exit_strat,
                "entry_time": entry_time,
                "direction": direction,
                "n_trades": n,
                "win_rate": round(win_rate, 3),
                "avg_win": round(avg_win, 4),
                "avg_loss": round(avg_loss, 4),
                "expectancy": round(expectancy, 4),
                "expectancy_per_dollar": round(expectancy / avg_credit, 3) if avg_credit > 0 else 0.0,
                "profit_factor": profit_factor,
                "total_pnl": round(float(group["pnl"].sum()), 4),
                "mae_p50": round(float(group["mae"].quantile(0.50)), 4),
                "mae_p90": round(float(group["mae"].quantile(0.90)), 4),
                "mae_p99": round(float(group["mae"].quantile(0.99)), 4),
                "mae_worst": round(float(group["mae"].max()), 4),
            }
        )

    return pd.DataFrame(records).sort_values(group_cols).reset_index(drop=True)
