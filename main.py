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
    #Load and prepare data with features
    print("\n1) Loading market data...")
    spy = get_single_ticker("SPY", start, end, use_cache=True)
    vix = get_vix(start, end, use_cache=True)
    rf = get_risk_free_rate(start, end, use_cache=True)

    print(f"SPY : {len(spy)} rows")
    print(f"VIX : {len(vix)} rows")
    print(f"RF : {len(rf)} rows")

    fe = FeatureEngineer()
    print("\n2) Creating features...")
    features_df = fe.create_all_features(spy, vix=vix, rf=rf)
    
    print(f"Features created: {len(features_df.columns)} columns")
    print(f"Date range: {features_df.index.min().date()} to {features_df.index.max().date()}")

    X, y_reg, y_clf = prepare_model_data(features_df, fe, dropna=True)

    common_idx = X.index.intersection(y_reg.index).intersection(y_clf.index)
    X = X.loc[common_idx]
    y_reg = y_reg.loc[common_idx]
    y_clf = y_clf.loc[common_idx]

    print(f"\n   After final alignment:")
    print(f"   - X shape: {X.shape}")
    print(f"   - y_reg shape: {y_reg.shape}")
    print(f"   - y_clf shape: {y_clf.shape}")
    print(f"   - Indices match: {X.index.equals(y_reg.index) and X.index.equals(y_clf.index)}")
    
    return features_df, X, y_reg, y_clf

def train_test_split_by_date(
    X: pd.DataFrame,
    y_reg : pd.Series,
    y_clf: pd.Series,
    split_date : str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    #split data by date (respects temporal ordering)
    print("\n3) Train / test split...")

    assert X.index.equals(y_reg.index) and X.index.equals(y_clf.index)

    train_mask = X.index < split_date
    test_mask = X.index >= split_date
    
    X_train = X.loc[train_mask]
    y_train_reg = y_reg.loc[train_mask]
    y_train_clf = y_clf.loc[train_mask]
    
    X_test = X.loc[test_mask]
    y_test_reg = y_reg.loc[test_mask]
    y_test_clf = y_clf.loc[test_mask]

    print(f"Train period: {X_train.index[0].date()} to {X_train.index[-1].date()}")
    print(f"- Observations: {len(X_train)}")
    print(f"Test period: {X_test.index[0].date()} to {X_test.index[-1].date()}")
    print(f"- Observations: {len(X_test)}")
    print(f"Train/test ratio: {len(X_train) / (len(X_train) + len(X_test)) * 100:.1f}% / {len(X_test) / (len(X_train) + len(X_test)) * 100:.1f}%")
    
    return X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf

def train_models(
    X_train : pd.DataFrame, 
    y_train_reg : pd.Series,
    y_train_clf: pd.Series
) -> Dict[str, object]:
    #train all models
    print("\n4) Training models...")
    models: Dict[str, object] = {}

    print("Training OLS...")
    try:
        ols = OLSModel(use_hac=True, maxlags=5)
        try:
            ols.fit(X_train, y_train_reg)
        except TypeError:
            ols.fit(X_train, y_train_reg, add_constant=True, standardize=True)
        models["OLS"] = ols
        print("OLS trained")
    except Exception as e:
        print(f"OLS training failed: {e}")

    #random forest
    print("Training random forest")
    rf = make_random_forest(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42
    )
    rf.fit(X_train, y_train_reg)
    models["Random Forest"] = rf
    print("Random Forest trained")
    
    #XGBoost
    print("Training XGBoost")
    try:
        xgb = make_xgboost(
            learning_rate=0.05,
            n_estimators=400,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        xgb.fit(X_train, y_train_reg)
        models["XGBoost"] = xgb
        print("XGBoost trained")
    except ImportError:
        print("XGBoost not available")

    #LightGBM
    print("Training LightGBM")
    try:
        lgbm = make_lightgbm(
            learning_rate=0.05,
            n_estimators=400,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        lgbm.fit(X_train, y_train_reg)
        models["LightGBM"] = lgbm
        print("LightGBM trained")
    except ImportError:
        print("LightGBM not available")
    return models

def make_predictions(
    models: Dict[str, object], 
    X_train: pd.DataFrame,
    X_test: pd.DataFrame, 
    y_train: pd.Series,
    y_test: pd.Series
) -> Dict[str, Dict[str, object]]:
    """
    Run predictions and calculate metrics (R-squared, IC) for each model.
    """
    print("\n5) Evaluating models (predictive power)...")
    results: Dict[str, Dict[str, object]] = {}
    
    for name, model in models.items():
        print(f"\n   -> {name}")

        #Train score
        try:
            r2_train = model.score(X_train, y_train)
        except Exception as e:
            r2_train = np.nan
        
        #Test score
        try:
            r2_test = model.score(X_test, y_test)
        except Exception as e:
            r2_test = np.nan
        
        #Predictions
        y_pred_raw = model.predict(X_test)
        
        #Force conversion to Series with X_test index
        if isinstance(y_pred_raw, pd.Series):
            y_pred = pd.Series(y_pred_raw.values, index=X_test.index, name="pred")
        else:
            #NumPy array or other - convert
            y_pred = pd.Series(y_pred_raw, index=X_test.index, name="pred")
        
        #Information Coefficient
        ic, pval = spearmanr(y_pred, y_test)
        
        #Display
        if not np.isnan(r2_train):
            print(f"R-squared (train): {r2_train:.4f}")
        if not np.isnan(r2_test):
            print(f"R-squared (test):  {r2_test:.4f}")
        print(f"IC (test):  {ic:.4f} (p-value: {pval:.6f})")
        
        results[name] = {
            "model": model,
            "pred": y_pred,
            "r2_train": r2_train,
            "r2_test": r2_test,
            "ic": ic,
            "ic_pval": pval
        }
    
    return results

def predictions_to_signals(
    predictions : pd.Series, threshold_quantile: float = 0.7
)-> pd.Series:

    threshold_value = predictions.quantile(threshold_quantile)
    signals = pd.Series(0, index=predictions.index, name="signal")
    signals[predictions > threshold_value] = 1
    return signals

def compare_feature_sets(
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf, stochastic_cols: list
)-> pd.DataFrame:
    """
    Compare model performance with and without stochastic features
    """
    print("\n" + "="*80)
    print("Feature set comparison : Baseline vs Stochastic Feature")
    print("="*80)

    results_comparison = []
    print("\n1) Training models without stochastic features...")

    stochastic_cols_present = [col for col in stochastic_cols if col in X_train.columns]
    
    valid_rows = X_train[stochastic_cols_present].notna().all(axis=1)
    
    X_train_subset = X_train[valid_rows]
    y_train_reg_subset = y_train_reg[valid_rows]
    y_train_clf_subset = y_train_clf[valid_rows]
    
    print(f"Using {len(X_train_subset)} samples for comparison ({(~valid_rows).sum()} rows with NaN removed)")
    
    X_train_baseline = X_train_subset.drop(columns=stochastic_cols, errors='ignore')
    X_test_baseline = X_test.drop(columns=stochastic_cols, errors='ignore')


    models_baseline = train_models(X_train_baseline, y_train_reg_subset, y_train_clf_subset)
    results_baseline = make_predictions(
        models_baseline, X_train_baseline, X_test_baseline, y_train_reg_subset, y_test_reg
    )
    perf_baseline = backtest_all_strategies(y_test_reg, results_baseline, signal_threshold=0.70)


    # Stochastic models (use original X_train, X_test, y_train_reg, y_test_reg)
    models_stochastic = train_models(X_train_subset, y_train_reg_subset, y_train_clf_subset)
    results_stochastic = make_predictions(
        models_stochastic, X_train_subset, X_test, y_train_reg_subset, y_test_reg
    )
    perf_stochastic = backtest_all_strategies(y_test_reg, results_stochastic, signal_threshold=0.70)

    print("\n" + "="*80)
    print("Comparison results")
    print("="*80)

    comparison_data = []

    for model_name in perf_baseline.index:
        if model_name == "Buy & Hold" :
            continue
        baseline_sharpe = perf_baseline.loc[model_name, "Sharpe Ratio"]
        stochastic_sharpe = perf_stochastic.loc[model_name, "Sharpe Ratio"]

        baseline_ic = perf_baseline.loc[model_name, "IC (test)"]
        stochastic_ic = perf_stochastic.loc[model_name, "IC (test)"]

        baseline_return = perf_baseline.loc[model_name, "Total Return (%)"]
        stochastic_return = perf_stochastic.loc[model_name, "Total Return (%)"]

        sharpe_improvement = ((stochastic_sharpe - baseline_sharpe) / abs(baseline_sharpe) * 100) if not np.isnan(baseline_sharpe) and baseline_sharpe != 0 else np.nan
        ic_improvement = ((stochastic_ic - baseline_ic) / abs(baseline_ic) * 100) if not np.isnan(baseline_ic) and baseline_ic != 0 else np.nan
        return_improvement = stochastic_return - baseline_return

        comparison_data.append({
            "Model": model_name,
            "Baseline Sharpe" : baseline_sharpe, 
            "Stochastic Sharpe": stochastic_sharpe,
            "Sharpe delta (%)" : sharpe_improvement,
            "Baseline IC" : baseline_ic,
            "Stochastic IC" : stochastic_ic,
            "IC delta (%)" : ic_improvement,
            "Return delta (%)" : return_improvement
        })
        comparison_df = pd.DataFrame(comparison_data)

        print("\n")
        print(comparison_df.round(3))

        print("\n" + "="*80)
        print("Summary")
        print("="*80)

        avg_sharpe_improvement = comparison_df["Sharpe delta (%)"].mean()
        avg_ic_improvement = comparison_df["IC delta (%)"].mean()

        print(f"\nAverage Sharpe improvement : {avg_sharpe_improvement:.2f}%")
        print(f"Average IC improvement : {avg_ic_improvement:.2f}%")

        if avg_sharpe_improvement > 0:
            print("\n Stochastic feature improve performance")
        else: 
            print("\n Stochastic feature decrease performance")
        
        return comparison_df, perf_baseline, perf_stochastic


def optimize_threshold(
    y_test: pd.Series, 
    model_results: Dict[str, Dict[str, object]], 
    thresholds: list = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    ) -> Tuple[float, pd.DataFrame]:
    """Test multiple thresholds and find best Sharpe."""
    print("\n" + "="*80)
    print("Thresold optimization")
    print("="*80)
    
    best_sharpe = -np.inf
    best_threshold = None
    results = []
    
    for threshold in thresholds:
        print(f"\nTesting threshold = {threshold:.0%}...")
        perf_df = backtest_all_strategies(y_test, model_results, signal_threshold=threshold)   
        
        #best model's Sharpe
        ml_models = perf_df.drop("Buy & Hold", errors='ignore')
        if len(ml_models) > 0:
            sharpe_series = ml_models["Sharpe Ratio"].dropna()
            if len(sharpe_series) == 0:
                continue
            best_model = sharpe_series.idxmax()
            sharpe = ml_models.loc[best_model, "Sharpe Ratio"] if "Sharpe Ratio" in ml_models.columns else ml_models.loc[best_model, "Sharpe"]
            time_in_market = ml_models.loc[best_model, "Time in Market (%)"]
            
            print(f"  Best model: {best_model}")
            print(f"  Sharpe: {sharpe:.3f}")
            print(f"  Time in Market: {time_in_market:.1f}%")
            
            results.append({
                "Threshold": threshold,
                "Best Model": best_model,
                "Sharpe": sharpe,
                "Time in Market (%)": time_in_market
            })
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_threshold = threshold
    
    results_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("Threshold optimization")
    print("="*80)
    print(results_df.round(3))
    if best_threshold is None:
        print("\nNo valid threshold found, using default 0.70")
        best_threshold = 0.70
    else:
        print(f"\nBest threshold: {best_threshold:.0%} (Sharpe = {best_sharpe:.3f})")

    return best_threshold, results_df

def backtest_all_strategies(
    y_test : pd.Series, model_results: Dict[str, Dict[str, object]],
    signal_threshold: float = 0.7
)-> pd.DataFrame:
    print("\n6) Backtesting all strategies (with transaction costs)...")

    performance_summary: Dict[str, Dict[str, float]] = {}

    for name, res in model_results.items():
        print(f"\nBacktesting {name}...")
        pred = res["pred"]

        signals = predictions_to_signals(pred, threshold_quantile=signal_threshold) #change the time in the market to see if it changes the perfomance 

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

        #total return
        total_return = (equity.iloc[-1] -1) * 100

        #max drawdown
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min() * 100

        #trade stats
        num_trades = signals.diff().abs().sum()
        time_in_mkt = signals.mean() * 100

        #win rate
        wins = (ret > 0).sum()
        losses = (ret < 0).sum()
        win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0

        performance_summary[name] = {
            "Total Return (%)": total_return,
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": max_dd,
            "Win Rate (%)": win_rate,
            "Num Trades": num_trades,
            "Time in Market (%)": time_in_mkt,
            "IC (test)": res["ic"],
            "IC p-value": res["ic_pval"],
            "Overfit": res["r2_train"] - res["r2_test"] if not np.isnan(res["r2_train"]) else np.nan
        }

        print(f"Total Return: {total_return:.2f}%")
        print(f"Sharpe: {sharpe:.3f}")
        print(f"Max DD: {max_dd:.2f}%")
        print(f"Time in Market: {time_in_mkt:.1f}%")

        # Buy & Hold benchmark
    print("\n   -> Backtesting Buy & Hold...")
    bh_equity = (1 + y_test).cumprod()
    bh_total_return = (bh_equity.iloc[-1] - 1) * 100
    bh_running_max = bh_equity.expanding().max()
    bh_dd = (bh_equity - bh_running_max) / bh_running_max
    bh_max_dd = bh_dd.min() * 100
    bh_sharpe = (y_test.mean() / y_test.std()) * np.sqrt(252) if y_test.std() > 0 else 0.0
    bh_wins = (y_test > 0).sum()
    bh_losses = (y_test < 0).sum()
    bh_win_rate = (bh_wins / (bh_wins + bh_losses)) * 100 if (bh_wins + bh_losses) > 0 else 0

    performance_summary["Buy & Hold"] = {
        "Total Return (%)": bh_total_return,
        "Sharpe Ratio": bh_sharpe,
        "Max Drawdown (%)": bh_max_dd,
        "Win Rate (%)": bh_win_rate,
        "Num Trades": 0.0,
        "Time in Market (%)": 100.0,
        "IC (test)": np.nan,
        "IC p-value": np.nan,
        "Overfit": np.nan
}

    perf_df = pd.DataFrame(performance_summary).T
    return perf_df

def main()-> None:
    #Main execution function
    print("=" * 80)
    print("SPY TRADING STRATEGY - OLS vs ML MODELS")
    print("=" * 80)
    print("\nReproducing analysis from ML comparison notebook...")
    print("Configuration:")
    print("- Data: 2015-01-01 to 2024-01-01")
    print("- Split: 2020-01-01")
    print("- Signal threshold: 70th percentile")
    print("- Transaction costs: 1 basis point")

    # Load data
    features_df, X, y_reg, y_clf = load_data(
        start="2015-01-01", 
        end="2024-01-01"
    )
    common_idx = X.index.intersection(y_reg.index).intersection(y_clf.index)
    X = X.loc[common_idx]
    y_reg = y_reg.loc[common_idx]
    y_clf = y_clf.loc[common_idx]
    print(f"\nAfter alignment: X shape={X.shape}, y_reg shape={y_reg.shape}, y_clf shape={y_clf.shape}")

    # Split
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split_by_date(
    X, y_reg, y_clf, split_date="2020-01-01"
    )

    #feature comparison
    fe = FeatureEngineer()
    stochastic_cols = fe.get_stochastic_feature_names()
    
    comparison_df, perf_baseline, perf_stochastic = compare_feature_sets(
    X_train, X_test,
    y_train_reg,
    y_test_reg, 
    y_train_clf, 
    y_test_clf,
    stochastic_cols
)
    print("\n" + "="*80)
    print("Using stochastic feature for threshold optimization")
    print("="*80)

    #train
    models = train_models(X_train, y_train_reg, y_train_clf)

    #predict
    models_results = make_predictions(models, X_train, X_test, y_train_reg, y_test_reg)

    #Optimize threshold
    best_threshold, optimization_results = optimize_threshold(
        y_test_reg, 
        models_results,
        thresholds=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    )

    print("\n" + "="*80)
    print(f"Final backtest with optimal threshold ({best_threshold:.0%})")
    print("="*80)
    perf_df = backtest_all_strategies(
        y_test_reg, 
        models_results, 
        signal_threshold=best_threshold
    )

    #display results
    print("\n" + "=" * 80)
    print("FINAL PERFORMANCE SUMMARY")
    print("=" * 80)
    print(perf_df.round(3))

    # Key insights
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Check which models are available
    available_models = perf_df.index.tolist()
    print(f"\nModels evaluated: {', '.join(available_models)}")

    # Best IC (excluding Buy & Hold)
    ic_data = perf_df.drop("Buy & Hold", errors='ignore')["IC (test)"].dropna()
    if len(ic_data) > 0:
        best_ic_model = ic_data.idxmax()
        print(f"\n✓ Best Information Coefficient: {best_ic_model}")
        print(f"  IC = {perf_df.loc[best_ic_model, 'IC (test)']:.4f}")
        if 'IC p-value' in perf_df.columns:
            print(f"  p-value = {perf_df.loc[best_ic_model, 'IC p-value']:.6f}")

    # Best Sharpe
    best_sharpe_model = perf_df["Sharpe Ratio"].idxmax()
    print(f"\n✓ Best Sharpe Ratio: {best_sharpe_model}")
    print(f"  Sharpe = {perf_df.loc[best_sharpe_model, 'Sharpe Ratio']:.3f}")
    if "Buy & Hold" in perf_df.index:
        print(f"  vs Buy & Hold = {perf_df.loc['Buy & Hold', 'Sharpe Ratio']:.3f}")

    # ML vs OLS comparison (only if OLS exists)
    if "OLS" in perf_df.index and len(ic_data) > 1:
        ols_ic = perf_df.loc["OLS", "IC (test)"]
        ml_ics = ic_data.drop("OLS", errors='ignore')
        
        if len(ml_ics) > 0 and not np.isnan(ols_ic) and ols_ic != 0:
            best_ml_ic = ml_ics.max()
            best_ml_name = ml_ics.idxmax()
            improvement = ((best_ml_ic - ols_ic) / abs(ols_ic) * 100)
            
            print(f"\n✓ ML vs OLS:")
            print(f"  OLS IC: {ols_ic:.4f}")
            print(f"  Best ML IC: {best_ml_ic:.4f} ({best_ml_name})")
            print(f"  Improvement: {improvement:.1f}%")
    elif "OLS" not in perf_df.index:
        print("\nOLS model not available for comparison")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()


