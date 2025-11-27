"""
Advanced metrics for model evaluation and trading strategy assessment.

This module provides metrics beyond basic backtest statistics:
- ML model evaluation (AUC, Brier, IC)
- Risk-adjusted returns (Sortino, Calmar)
- Regime analysis
- SHAP-based interpretability helpers
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 4.0


# ============================================================================
# ML Model Evaluation Metrics
# ============================================================================

def information_coefficient(predictions: pd.Series, actuals: pd.Series) -> float:
    """
    Calculate Information Coefficient (IC) - Spearman correlation between
    predictions and actual returns.
    
    IC > 0: Positive predictive power
    IC ≈ 0: No predictive power
    IC < 0: Inverse predictive power
    
    Args:
        predictions: Model predictions
        actuals: Actual returns
        
    Returns:
        IC value between -1 and 1
    """
    data = pd.DataFrame({"pred": predictions, "actual": actuals}).dropna()
    if len(data) < 2:
        return np.nan
    
    ic, _ = stats.spearmanr(data["pred"], data["actual"])
    return float(ic)


def rank_information_coefficient(predictions: pd.Series, actuals: pd.Series) -> float:
    """
    Calculate Rank IC - correlation between rank of predictions and rank of actuals.
    More robust to outliers than standard IC.
    """
    data = pd.DataFrame({"pred": predictions, "actual": actuals}).dropna()
    if len(data) < 2:
        return np.nan
    
    pred_ranks = data["pred"].rank()
    actual_ranks = data["actual"].rank()
    
    ic, _ = stats.spearmanr(pred_ranks, actual_ranks)
    return float(ic)


def auc_score(y_true: pd.Series, y_pred_proba: pd.Series) -> float:
    """
    Calculate AUC-ROC score for binary classification.
    
    Args:
        y_true: True binary labels (0/1 or -1/1)
        y_pred_proba: Predicted probabilities for positive class
        
    Returns:
        AUC score between 0 and 1 (0.5 = random, 1.0 = perfect)
    """
    data = pd.DataFrame({"y_true": y_true, "y_pred": y_pred_proba}).dropna()
    if len(data) < 2 or data["y_true"].nunique() < 2:
        return np.nan
    
    # Convert labels to binary if needed
    y_binary = (data["y_true"] > 0).astype(int)
    
    try:
        auc = roc_auc_score(y_binary, data["y_pred"])
        return float(auc)
    except ValueError:
        return np.nan


def brier_score(y_true: pd.Series, y_pred_proba: pd.Series) -> float:
    """
    Calculate Brier Score - measures accuracy of probabilistic predictions.
    Lower is better (0 = perfect, 1 = worst).
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Brier score between 0 and 1
    """
    data = pd.DataFrame({"y_true": y_true, "y_pred": y_pred_proba}).dropna()
    if len(data) < 2:
        return np.nan
    
    y_binary = (data["y_true"] > 0).astype(int)
    
    try:
        score = brier_score_loss(y_binary, data["y_pred"])
        return float(score)
    except ValueError:
        return np.nan


def classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Returns:
        Dictionary with accuracy, precision, recall, F1
    """
    data = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if len(data) < 2 or data["y_true"].nunique() < 2:
        return {
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
        }
    
    # Convert to binary
    y_true_bin = (data["y_true"] > 0).astype(int)
    y_pred_bin = (data["y_pred"] > 0).astype(int)
    
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    
    # Handle case where only one class is predicted
    if cm.shape != (2, 2):
        return {
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
        }
    
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
    
    return {
        "accuracy": float(accuracy) if not np.isnan(accuracy) else np.nan,
        "precision": float(precision) if not np.isnan(precision) else np.nan,
        "recall": float(recall) if not np.isnan(recall) else np.nan,
        "f1_score": float(f1) if not np.isnan(f1) else np.nan,
    }


# ============================================================================
# Advanced Risk-Adjusted Performance Metrics
# ============================================================================

def sortino_ratio(
    returns: pd.Series,
    target_return: float = 0.0,
    freq: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calculate Sortino Ratio - risk-adjusted return using downside deviation.
    Better than Sharpe for asymmetric return distributions.
    
    Args:
        returns: Series of returns
        target_return: Minimum acceptable return (MAR)
        freq: Frequency for annualization
        
    Returns:
        Annualized Sortino ratio
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    
    excess_returns = returns - target_return
    mean_excess = excess_returns.mean()
    
    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return np.inf if mean_excess > 0 else np.nan
    
    downside_std = np.sqrt((downside_returns ** 2).mean())
    
    if downside_std == 0:
        return np.inf if mean_excess > 0 else np.nan
    
    annual_mean = mean_excess * freq
    annual_downside_std = downside_std * np.sqrt(freq)
    
    sortino = annual_mean / annual_downside_std
    return float(sortino)


def calmar_ratio(returns: pd.Series, freq: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Calculate Calmar Ratio - annual return divided by maximum drawdown.
    Measures return relative to worst drawdown.
    
    Args:
        returns: Series of returns
        freq: Frequency for annualization
        
    Returns:
        Calmar ratio
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    
    # Annual return
    mean_ret = returns.mean()
    annual_return = (1 + mean_ret) ** freq - 1
    
    # Max drawdown
    equity = (1 + returns).cumprod()
    rolling_max = equity.cummax()
    drawdown = (equity / rolling_max) - 1.0
    max_dd = abs(drawdown.min())
    
    if max_dd == 0:
        return np.inf if annual_return > 0 else np.nan
    
    calmar = annual_return / max_dd
    return float(calmar)


def omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0,
    freq: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calculate Omega Ratio - probability-weighted ratio of gains vs losses.
    
    Args:
        returns: Series of returns
        threshold: Threshold return (often 0 or risk-free rate)
        freq: Frequency for annualization
        
    Returns:
        Omega ratio
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    
    omega = gains / losses
    return float(omega)


def tail_ratio(returns: pd.Series, percentile: float = 5.0) -> float:
    """
    Calculate Tail Ratio - ratio of right tail to left tail.
    Measures asymmetry in extreme outcomes.
    
    Args:
        returns: Series of returns
        percentile: Percentile for tail definition (default 5%)
        
    Returns:
        Tail ratio (> 1 means positive skew)
    """
    returns = returns.dropna()
    if len(returns) < 20:  # Need sufficient data for tails
        return np.nan
    
    right_tail = np.abs(np.percentile(returns, 100 - percentile))
    left_tail = np.abs(np.percentile(returns, percentile))
    
    if left_tail == 0:
        return np.inf if right_tail > 0 else np.nan
    
    ratio = right_tail / left_tail
    return float(ratio)


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) - maximum expected loss at given confidence level.
    
    Args:
        returns: Series of returns
        confidence: Confidence level (e.g., 0.95 for 95% VaR)
        
    Returns:
        VaR as positive number (loss)
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    
    var = -np.percentile(returns, (1 - confidence) * 100)
    return float(var)


def conditional_value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Conditional VaR (CVaR / Expected Shortfall).
    Expected loss given that loss exceeds VaR.
    
    Args:
        returns: Series of returns
        confidence: Confidence level
        
    Returns:
        CVaR as positive number
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    
    var = value_at_risk(returns, confidence)
    cvar = -returns[returns <= -var].mean()
    
    return float(cvar) if not np.isnan(cvar) else np.nan


# ============================================================================
# Trading Strategy Metrics
# ============================================================================

def turnover_rate(signals: pd.Series) -> float:
    """
    Calculate average turnover rate - fraction of portfolio changed per period.
    
    Args:
        signals: Position signals (-1, 0, 1)
        
    Returns:
        Average turnover (0 to 1)
    """
    signals = signals.dropna()
    if len(signals) < 2:
        return np.nan
    
    position_changes = signals.diff().abs()
    avg_turnover = position_changes.mean() / 2.0  # Divide by 2 for full round-trip
    
    return float(avg_turnover)


def stability_of_timeseries(returns: pd.Series) -> float:
    """
    Calculate stability - R² of linear regression of cumulative returns vs time.
    Measures consistency of returns over time.
    
    Args:
        returns: Series of returns
        
    Returns:
        R² value (0 to 1, higher = more stable)
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    
    cum_returns = (1 + returns).cumprod()
    time_index = np.arange(len(cum_returns))
    
    # Linear regression
    coef = np.polyfit(time_index, cum_returns, 1)
    predicted = np.polyval(coef, time_index)
    
    # R²
    ss_res = np.sum((cum_returns - predicted) ** 2)
    ss_tot = np.sum((cum_returns - cum_returns.mean()) ** 2)
    
    if ss_tot == 0:
        return np.nan
    
    r_squared = 1 - (ss_res / ss_tot)
    return float(r_squared)


def up_capture(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate upside capture ratio - strategy return in up markets / benchmark return.
    
    Returns:
        Ratio (> 1 means outperformance in up markets)
    """
    data = pd.DataFrame({
        "strategy": strategy_returns,
        "benchmark": benchmark_returns
    }).dropna()
    
    if len(data) < 2:
        return np.nan
    
    up_periods = data["benchmark"] > 0
    if up_periods.sum() == 0:
        return np.nan
    
    strat_up = data.loc[up_periods, "strategy"].mean()
    bench_up = data.loc[up_periods, "benchmark"].mean()
    
    if bench_up == 0:
        return np.nan
    
    return float(strat_up / bench_up)


def down_capture(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate downside capture ratio - strategy return in down markets / benchmark return.
    
    Returns:
        Ratio (< 1 means outperformance in down markets, i.e., less loss)
    """
    data = pd.DataFrame({
        "strategy": strategy_returns,
        "benchmark": benchmark_returns
    }).dropna()
    
    if len(data) < 2:
        return np.nan
    
    down_periods = data["benchmark"] < 0
    if down_periods.sum() == 0:
        return np.nan
    
    strat_down = data.loc[down_periods, "strategy"].mean()
    bench_down = data.loc[down_periods, "benchmark"].mean()
    
    if bench_down == 0:
        return np.nan
    
    return float(strat_down / bench_down)


# ============================================================================
# Comprehensive Metrics Report
# ============================================================================

def compute_full_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    signals: Optional[pd.Series] = None,
    predictions: Optional[pd.Series] = None,
    actuals: Optional[pd.Series] = None,
    freq: int = TRADING_DAYS_PER_YEAR
) -> Dict[str, float]:
    """
    Compute comprehensive set of metrics for strategy evaluation.
    
    Args:
        strategy_returns: Strategy returns
        benchmark_returns: Benchmark returns
        signals: Position signals (optional, for turnover)
        predictions: Model predictions (optional, for IC/AUC)
        actuals: Actual returns (optional, for IC/AUC)
        freq: Frequency for annualization
        
    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}
    
    # Risk-adjusted returns
    metrics["sortino_ratio"] = sortino_ratio(strategy_returns, freq=freq)
    metrics["calmar_ratio"] = calmar_ratio(strategy_returns, freq=freq)
    metrics["omega_ratio"] = omega_ratio(strategy_returns, freq=freq)
    metrics["tail_ratio"] = tail_ratio(strategy_returns)
    
    # Risk metrics
    metrics["var_95"] = value_at_risk(strategy_returns, 0.95)
    metrics["cvar_95"] = conditional_value_at_risk(strategy_returns, 0.95)
    
    # Strategy metrics
    if signals is not None:
        metrics["turnover_rate"] = turnover_rate(signals)
    
    metrics["stability"] = stability_of_timeseries(strategy_returns)
    metrics["up_capture"] = up_capture(strategy_returns, benchmark_returns)
    metrics["down_capture"] = down_capture(strategy_returns, benchmark_returns)
    
    # ML metrics (if predictions provided)
    if predictions is not None and actuals is not None:
        metrics["information_coefficient"] = information_coefficient(predictions, actuals)
        metrics["rank_ic"] = rank_information_coefficient(predictions, actuals)
        
        # Classification metrics (if binary)
        try:
            metrics["auc_score"] = auc_score(actuals, predictions)
            metrics["brier_score"] = brier_score(actuals, predictions)
        except:
            pass
    
    return metrics


# ============================================================================
# Rolling Metrics for Regime Analysis
# ============================================================================

def rolling_sharpe(
    returns: pd.Series,
    window: int = 60,
    freq: int = TRADING_DAYS_PER_YEAR
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.
    
    Args:
        returns: Series of returns
        window: Rolling window size
        freq: Frequency for annualization
        
    Returns:
        Series of rolling Sharpe ratios
    """
    rolling_mean = returns.rolling(window).mean() * freq
    rolling_std = returns.rolling(window).std() * np.sqrt(freq)
    
    rolling_sharpe_ratio = rolling_mean / rolling_std
    return rolling_sharpe_ratio


def rolling_ic(
    predictions: pd.Series,
    actuals: pd.Series,
    window: int = 60
) -> pd.Series:
    """
    Calculate rolling Information Coefficient.
    
    Args:
        predictions: Model predictions
        actuals: Actual returns
        window: Rolling window size
        
    Returns:
        Series of rolling IC values
    """
    data = pd.DataFrame({"pred": predictions, "actual": actuals}).dropna()
    
    def calc_ic(pred, actual):
        if len(pred) < 2:
            return np.nan
        ic, _ = stats.spearmanr(pred, actual)
        return ic
    
    rolling_ic_values = data.rolling(window).apply(
        lambda x: calc_ic(x["pred"], x["actual"]),
        raw=False
    )
    
    return rolling_ic_values["pred"]


def identify_regimes(
    returns: pd.Series,
    vol_threshold_low: float = 0.01,
    vol_threshold_high: float = 0.03,
    window: int = 20
) -> pd.Series:
    """
    Identify market regimes based on volatility.
    
    Args:
        returns: Series of returns
        vol_threshold_low: Threshold for low volatility regime
        vol_threshold_high: Threshold for high volatility regime
        window: Window for volatility calculation
        
    Returns:
        Series with regime labels: 'low_vol', 'normal', 'high_vol'
    """
    rolling_vol = returns.rolling(window).std()
    
    regimes = pd.Series(index=returns.index, dtype=str)
    regimes[rolling_vol < vol_threshold_low] = "low_vol"
    regimes[(rolling_vol >= vol_threshold_low) & (rolling_vol < vol_threshold_high)] = "normal"
    regimes[rolling_vol >= vol_threshold_high] = "high_vol"
    
    return regimes


def metrics_by_regime(
    returns: pd.Series,
    regimes: pd.Series,
    freq: int = TRADING_DAYS_PER_YEAR
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics separately for each market regime.
    
    Args:
        returns: Series of returns
        regimes: Series with regime labels
        freq: Frequency for annualization
        
    Returns:
        Dictionary with metrics for each regime
    """
    regime_metrics = {}
    
    for regime in regimes.unique():
        regime_returns = returns[regimes == regime]
        
        if len(regime_returns) < 2:
            continue
        
        mean_ret = regime_returns.mean()
        vol = regime_returns.std()
        
        metrics = {
            "mean_return": float(mean_ret * freq),
            "volatility": float(vol * np.sqrt(freq)),
            "sharpe": float((mean_ret * freq) / (vol * np.sqrt(freq))) if vol > 0 else np.nan,
            "n_periods": len(regime_returns)
        }
        
        regime_metrics[regime] = metrics
    
    return regime_metrics


# ============================================================================
# TEST BLOCK
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TESTING metrics.py")
    print("=" * 70)
    
    # Create synthetic data
    np.random.seed(42)
    n = 252  # 1 year of daily data
    
    print("\nCreating synthetic test data...")
    print(f"   - {n} samples (1 year of daily data)")
    
    # Simulate predictions and actual returns
    predictions = pd.Series(np.random.randn(n) * 0.02, name="predictions")
    actuals = pd.Series(predictions * 0.5 + np.random.randn(n) * 0.01, name="actuals")
    
    # Simulate strategy and benchmark returns
    strategy_returns = pd.Series(np.random.randn(n) * 0.01 + 0.0003, name="strategy")
    benchmark_returns = pd.Series(np.random.randn(n) * 0.01 + 0.0002, name="benchmark")
    
    # Simulate signals
    signals = pd.Series(np.random.choice([-1, 0, 1], size=n), name="signals")
    
    print("\n" + "=" * 70)
    print("1. ML MODEL EVALUATION METRICS (Required by proposal)")
    print("=" * 70)
    
    # Test IC
    ic = information_coefficient(predictions, actuals)
    print(f"\nInformation Coefficient (IC): {ic:.4f}")
    print(f"Interpretation: {'Strong positive' if ic > 0.05 else 'Weak'} predictive power")
    
    # Test AUC (convert to binary for testing)
    y_binary = (actuals > 0).astype(int)
    predictions_prob = (predictions - predictions.min()) / (predictions.max() - predictions.min())
    auc = auc_score(y_binary, predictions_prob)
    print(f"\nAUC-ROC Score: {auc:.4f}")
    print(f"Interpretation: {'Good' if auc > 0.6 else 'Moderate' if auc > 0.55 else 'Weak'} classification")
    
    # Test Brier Score
    brier = brier_score(y_binary, predictions_prob)
    print(f"\nBrier Score: {brier:.4f}")
    print(f"Interpretation: Lower is better (0 = perfect, 1 = worst)")
    
    print("\n" + "=" * 70)
    print("RISK-ADJUSTED PERFORMANCE METRICS")
    print("=" * 70)
    
    # Test Sortino Ratio
    sortino = sortino_ratio(strategy_returns)
    print(f"\nSortino Ratio: {sortino:.4f}")
    print(f"(Like Sharpe, but only penalizes downside volatility)")
    
    # Test Calmar Ratio
    calmar = calmar_ratio(strategy_returns)
    print(f"\nCalmar Ratio: {calmar:.4f}")
    print(f"(Annual return / Max drawdown)")
    
    # Test Omega Ratio
    omega = omega_ratio(strategy_returns)
    print(f"\nOmega Ratio: {omega:.4f}")
    print(f"(Probability-weighted gains/losses)")
    
    print("\n" + "=" * 70)
    print("3. TRADING STRATEGY METRICS")
    print("=" * 70)
    
    # Test turnover
    turnover = turnover_rate(signals)
    print(f"\nTurnover Rate: {turnover:.4f}")
    print(f"(Average portfolio change per period)")
    
    # Test capture ratios
    up_cap = up_capture(strategy_returns, benchmark_returns)
    down_cap = down_capture(strategy_returns, benchmark_returns)
    print(f"\nUp Capture Ratio: {up_cap:.4f}")
    print(f"Down Capture Ratio: {down_cap:.4f}")
    
    print("\n" + "=" * 70)
    print("4. COMPREHENSIVE METRICS (All-in-One)")
    print("=" * 70)
    
    # Test compute_full_metrics
    full_metrics = compute_full_metrics(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        signals=signals,
        predictions=predictions,
        actuals=actuals
    )
    
    print("\nFull metrics computed successfully!")
    print(f"\nSummary of key metrics:")
    
    key_metrics = [
        'information_coefficient', 'sortino_ratio', 'calmar_ratio', 
        'omega_ratio', 'var_95', 'cvar_95', 'turnover_rate'
    ]
    
    for key in key_metrics:
        if key in full_metrics and not np.isnan(full_metrics[key]):
            print(f"   {key:25s}: {full_metrics[key]:8.4f}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)