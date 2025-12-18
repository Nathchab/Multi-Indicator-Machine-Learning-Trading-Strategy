from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd

#Standard assumption for annualizing daily financial returns
TRADING_DAYS_PER_YEARS = 252

@dataclass
class BacktestResult:
    equity_curve : pd.Series
    benchmark_curve : pd.Series
    strategy_returns : pd.Series
    benchmark_returns : pd.Series
    stats : Dict[str, float]

def _compute_equity_curve(returns : pd.Series, starting_capital : float = 1.0)-> pd.Series:
    """
    Compute equity curve from simple returns
    """
    #Missing returns are treated as flat periods to preserve continuity of the equity curve
    returns = returns.fillna(0.0)
    equity = (1.0 + returns).cumprod() * starting_capital
    return equity

def _compute_performance_stats(returns: pd.Series, freq : int = TRADING_DAYS_PER_YEARS)-> Dict[str, float]:
    """
    Compute basic perfomance statistics
    """
    returns = returns.dropna()
    #Graceful handling of empty return series avoids silent failures in downstream evaluation
    if returns.empty:
        return {
            "total_return" : np.nan,
            "annual_return" : np.nan,
            "annual_vol" : np.nan,
            "sharpe" : np.nan,
            "max_drawdown" : np.nan,
            "hit_rate" : np.nan,
            "avg_gain" : np.nan,
            "avg_loss" : np.nan,
            "n_trades" : 0,
        }
    total_return = (1+ returns).prod() - 1
    mean_ret = returns.mean()
    vol = returns.std()

    #Returns and volatility are annualized to make results comparable across strategies and time horizons
    annual_return = (1 + mean_ret) ** freq - 1 if mean_ret is not None else np.nan
    annual_vol = vol * np.sqrt(freq) if vol is not None else np.nan
    sharpe = annual_return / annual_vol if (annual_vol is not None and annual_vol != 0) else np.nan

    #Maximum drawdown captures worst peak-to-trough loss, a key risk metric for trading strategies
    equity = _compute_equity_curve(returns)
    rolling_max = equity.cummax()
    drawdown = (equity/rolling_max) - 1.0
    max_drawdown = drawdown.min()

    wins = (returns > 0).sum()
    losses = (returns < 0).sum()
    hit_rate = wins / (wins + losses) if (wins + losses) > 0 else np.nan

    avg_gain = returns[returns > 0].mean() if wins > 0 else np.nan
    avg_loss = returns[returns < 0].mean() if losses > 0 else np.nan

    stats = {
        "total_return" : float(total_return),
        "annual_return" : float(annual_return),
        "annual_vol" : float(annual_vol),
        "sharpe" : float(sharpe),
        "max_drawdown" : float(max_drawdown),
        "hit_rate" : float(hit_rate),
        "avg_gain" : float(avg_gain) if not np.isnan(avg_gain) else np.nan,
        "avg_loss" : float(avg_loss) if not np.isnan(avg_loss) else np.nan,
        "n_trades" : int(wins + losses),
    }

    return stats

def backtest_signals(returns : pd.Series, signals : pd.Series, trading_cost_bps : float = 0.0, starting_capital : float = 1.0, freq : int = TRADING_DAYS_PER_YEARS)-> BacktestResult:
    """
    Backtest a strategy defined by position signals
    """
    #Signals and returns are aligned on common timestamps to avoid implicit look-ahead or misalignment bias
    data = pd.concat({"returns" : returns, "signal" : signals}, axis=1).dropna()
    if data.empty:
        raise ValueError("No overlapping data between returns and signals")
    position = data["signal"].astype(float)
    position_prev = position.shift(1).fillna(0.0)
    #Turnover measures position changes and directly drives transaction costs
    turnover = (position - position_prev).abs()

    #Trading costs are expressed in basis points and converted to decimal rates
    cost_rate = trading_cost_bps / 1e4
    trading_costs = turnover * cost_rate

    #Strategy returns account for both market exposure and trading frictions
    strat_returns = position * data["returns"] - trading_costs

    benchmark_returns = data["returns"].copy()

    equity_curve = _compute_equity_curve(strat_returns, starting_capital)
    benchmark_curve = _compute_equity_curve(benchmark_returns, starting_capital)

    stats = _compute_performance_stats(strat_returns, freq=freq)
    bench_stats = _compute_performance_stats(benchmark_returns, freq=freq)

    stats_with_bench = {
        **{f"strategy_{k}": v for k, v in stats.items()},
        **{f"benchmark_{k}": v for k, v in bench_stats.items()}
    }

    return BacktestResult(equity_curve=equity_curve, benchmark_curve=benchmark_curve, strategy_returns=strat_returns, benchmark_returns=benchmark_returns, stats=stats_with_bench,)
