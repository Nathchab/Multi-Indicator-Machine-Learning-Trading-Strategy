"""
main script to compare OLS and ML models for SPY trading strategy.

Pipeline :
1. Download market data (SPY, VIX, risk-free rate)
2. Engineer features
3. Split train / test by date
4. Train models (OLS, RandomForest, XGBoost, LightGBM)
5. Generate trading signals from predictions
6. Backtest all strategies with transaction costs
7. Print performance comparison table
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np 
import pandas as pd 
from scipy.stats import spearmanr

from src.data.fetcher import (
    get_single_ticker,
    get_vix,
    get_risk_free_rate
)
from src.data.features import FeatureEngineer, prepare_model_data
from src.models.baseline import OLSModel
from src.models.ml_models import (
    make_random_forest,
    make_lightgbm,
    make_xgboost
)
from src.evaluation.backtest import backtest_signals

def load_data(start : str, end : str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    print("\n1) Loading market data...")
    spy = get_single_ticker("SPY", start, end)
    vix = get_vix(start, end)
    rf = get_risk_free_rate(start, end)

    print(f"SPY : {spy.index.min().date()} -> {spy.index.max().date()}")
    print(f"VIX and risk-free rate loaded")
    
    fe = FeatureEngineer()
    print("\n2) Creating features...")
    df = fe.create_all_features(spy, vix=vix, rf=rf)
    
    df = df.dropna()

    print(f"Final feature DataFrame shape : {df.shape}")

    X, y_reg, _ = prepare_model_data(df, fe, dropna=True)

    common_idx = X.index.intersection(y_reg.index)
    X = X.loc[common_idx]
    y_reg = y_reg.loc[common_idx]

    print(f"Model dataset shape : X ={X.shape}, y={y_reg.shape}")
    return df, X, y_reg

def train_test_split_by_date(
    X: pd.DataFrame, y : pd.Series, split_date : str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    print("\n3) Train / test split...")
    train_idx = X.index < split_date
    test_idx = X.index >= split_date

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    print(f"Train size: {X_train.shape[0]} samples")
    print(f"Test size : {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test

def train_models(
    X_train : pd.DataFrame, y_train : pd.Series
) -> Dict[str, object]:
    print("\n4) Training models...")
    models: Dict[str, object] = {
        "OLS" : OLSModel(use_hac=True),
        "Random forest" : make_random_forest(),
        "XGBoost": make_xgboost(),
        "LightGBM" : make_lightgbm()
    }
    for name, model in models.items():
        print(f"-> Training {name}...")
        model.fit(X_train, y_train)
        print(f"{name} trained")

    return models

def make_predictions(
    models: Dict[str, object], X_test: pd.DataFrame, y_test: pd.Series
)-> Dict[str, Dict[str, object]]:
    """
        Run predictions and basic stats (R², IC) for each model.

        Returns dict:
        {
            model_name: {
                "model": fitted_model,
                "pred": pd.Series,
                "r2_train": float | None,
                "r2_test": float | None,
                "ic": float,
                "ic_pval": float,
            }
        }
    """
    print("\n5) Evaluating models (predictive power)...")
    results : Dict[str, Dict[str, object]] = {}
    
    for name, model in models.items():
        print(f"\n -> {name}")

        try:
            r2_test = model.score(X_test, y_test)
        except Exception:
            r2_test = None
        
        r2_train = None

        y_pred = model.predict(X_test)
        y_pred = pd.Series(y_pred, index=y_test.index, name="y_pred")

        ic, pval = spearmanr(y_pred, y_test)

        if r2_test is not None:
            print(f"R-squared (test) : {r2_test:.4f}")
        results[name] = {
            "model" : model,
            "pred" : y_pred,
            "r2_train" : r2_train,
            "r2_test" : r2_test,
            "ic" : ic,
            "ic_pval" : pval
        }
    return results

def predictions_to_signals(
    predictions : pd.Series, threshold: float = 0.0
)-> pd.Series:
    sig = (predictions > threshold).astype(int)
    sig.name = "signal"
    return sig

def backtest_all_strategies(
    y_test : pd.Series, model_results: Dict[str, Dict[str, object]]
)-> pd.DataFrame:
    print("\n6) Backtesting all strategies (with transaction costs)...")
    performance_summary: Dict[str, Dict[str, float]] = {}
    for name, res in model_results.items():
        print(f"\nBacktesting {name}...")
        pred = res["pred"]

        signals = predictions_to_signals(pred, threshold=0.0)

        bt = backtest_signals(
            returns=y_test,
            signals=signals, 
            trading_cost_bps=1.0,
            starting_capital=1.0
        )
        ret = bt.strategy_returns
        equity = (1 + ret).cumprod()

        if ret.std() > 0:
            sharpe = (ret.mean() / ret.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        total_return = (equity.iloc[-1] -1) * 100
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min() * 100
        num_trades = signals.diff().abs().sum()
        time_in_mkt = signals.mean() * 100

        performance_summary[name] = {
            "Total Return (%)": total_return,
            "Sharpe": sharpe,
            "Max Drawdown (%)": max_dd,
            "Time in Market (%)": time_in_mkt,
            "Num Trades": num_trades,
            "IC (test)": res["ic"],
        }

        print(f"Total Return : {total_return:.2f}%")
        print(f"Sharpe       : {sharpe:.3f}")
        print(f"Max DD       : {max_dd:.2f}%")
        print(f"TiM          : {time_in_mkt:.1f}%")

        print("\nBacktesting buy & hold...")
        bh_equity = (1 + y_test).cumprod()
        bh_total_return = (bh_equity.iloc[-1] - 1) * 100
        bh_running_max = bh_equity.expanding().max()
        bh_dd = (bh_equity - bh_running_max) / bh_running_max
        bh_max_dd = bh_dd.min() * 100

        bh_sharpe = (y_test.mean() / y_test.std()) * np.sqrt(252) if y_test.std() > 0 else 0.0

        performance_summary["Buy & Hold"] = {
        "Total Return (%)": bh_total_return,
        "Sharpe": bh_sharpe,
        "Max Drawdown (%)": bh_max_dd,
        "Time in Market (%)": 100.0,
        "Num Trades": 0.0,
        "IC (test)": np.nan,
    }

    perf_df = pd.DataFrame(performance_summary).T
    return perf_df
def main()-> None:
    print("="*80)
    print("SPY TRADING STRATEGY - OLS vs ML MODELS")
    print("="*80)

    df, X, y = load_data(start="2015-01-01", end="2024-01-01")

    X_train, X_test, y_train, y_test = train_test_split_by_date(
        X, y, split_date="2020-01-01"
    )

    models = train_models(X_train, y_train)

    models_results = make_predictions(models, X_test, y_test)

    perf_df = backtest_all_strategies(y_test, models_results)

    print("\n" + "=" *80)
    print("FINAL PERFORMANCE SUMMARY")
    print("=" * 80)
    print(perf_df.round(3))

if __name__ == "__main__":
    main()