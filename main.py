import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from src.data.fetcher import (
    get_single_ticker,
    get_vix,
    get_risk_free_rate,
)
from src.data.features import FeatureEngineer, prepare_model_data
from src.models.ml_models import make_random_forest, make_xgboost, make_lightgbm
from src.evaluation.backtest import backtest_signals


def predictions_to_signals(predictions, threshold_quantile=0.4):
    """Convert predictions to binary trading signals."""
    threshold_value = predictions.quantile(threshold_quantile)
    signals = (predictions > threshold_value).astype(int)
    return signals


def evaluate_model(model, X_train, X_test, y_train, y_test, threshold=0.4):
    """
    Train model, generate predictions, backtest, and return metrics.
    
    Returns:
        dict with equity curve, returns, and performance metrics
    """
    #Train
    model.fit(X_train, y_train)
    
    #Predict
    y_pred = pd.Series(model.predict(X_test), index=X_test.index)
    
    #Information Coefficient
    ic, ic_pval = spearmanr(y_pred, y_test)
    
    #Generate signals
    signals = predictions_to_signals(y_pred, threshold_quantile=threshold)
    
    #Backtest
    results = backtest_signals(
        returns=y_test,
        signals=signals,
        trading_cost_bps=1.0
    )
    
    #Calculate metrics
    equity = (1 + results.strategy_returns).cumprod()
    total_return = (equity.iloc[-1] - 1) * 100
    sharpe = (results.strategy_returns.mean() / results.strategy_returns.std()) * np.sqrt(252)
    
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min() * 100
    
    #Time in market
    time_in_market = signals.mean() * 100
    
    #Number of trades
    num_trades = signals.diff().abs().sum()
    
    return {
        'equity': equity,
        'returns': results.strategy_returns,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'ic': ic,
        'ic_pval': ic_pval,
        'time_in_market': time_in_market,
        'num_trades': num_trades,
        'signals': signals
    }


def main():
    print("=" * 80)
    print(" " * 20 + "SPY ML TRADING STRATEGY")
    print(" " * 15 + "Multi-Model Comparison & Analysis")
    print("=" * 80)

    #1. Load Data
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING")
    print("="*80)
    
    START_DATE = "2015-01-01"
    END_DATE   = "2024-01-01"
    
    print(f"Period: {START_DATE} to {END_DATE}")
    print("Loading SPY, VIX, and Risk-Free Rate...")

    spy = get_single_ticker("SPY", START_DATE, END_DATE)
    vix = get_vix(START_DATE, END_DATE)
    rf  = get_risk_free_rate(START_DATE, END_DATE)
    
    print(f"SPY data loaded: {len(spy)} rows")
    print(f"VIX data loaded: {len(vix)} rows")
    print(f"RF data loaded: {len(rf)} rows")

    #2. Feature Engineering
    print("\n" + "="*80)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*80)
    
    fe = FeatureEngineer()
    df = fe.create_all_features(spy, vix=vix, rf=rf)
    
    X, y_reg, _ = prepare_model_data(df, fe)
    
    print(f"\n Features created:")
    print(f"  - Total samples: {len(X)}")
    print(f"  - Number of features: {X.shape[1]}")
    print(f"  - Feature names: {list(X.columns[:5])}... (showing first 5)")

    #3. Tain/Test Split
    print("\n" + "="*80)
    print("STEP 3: TRAIN/TEST SPLIT")
    print("="*80)
    
    split_date = "2020-01-01"
    X_train = X.loc[:split_date]
    X_test = X.loc[split_date:]
    y_train = y_reg.loc[X_train.index]
    y_test = y_reg.loc[X_test.index]

    print(f"Split date: {split_date}")
    print(f"Train period: {X_train.index[0].date()} to {X_train.index[-1].date()}")
    print(f"Test period:  {X_test.index[0].date()} to {X_test.index[-1].date()}")
    print(f"Train samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test samples:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    #4. Train multiple model 
    print("\n" + "="*80)
    print("STEP 4: TRAINING & EVALUATING MODELS")
    print("="*80)
    
    models = {
        "Random Forest": make_random_forest(n_estimators=300, max_depth=10, random_state=42),
        "XGBoost": make_xgboost(n_estimators=400, max_depth=5, learning_rate=0.05, random_state=42),
        "LightGBM": make_lightgbm(n_estimators=400, num_leaves=31, learning_rate=0.05, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n Training {name}...")
        results[name] = evaluate_model(model, X_train, X_test, y_train, y_test, threshold=0.4)
        print(f" Complete | Return: {results[name]['total_return']:.2f}% | Sharpe: {results[name]['sharpe']:.3f}")

    #5. Buy & Hold benchmark
    print("\n Calculating Buy & Hold benchmark...")
    bh_equity = (1 + y_test).cumprod()
    bh_return = (bh_equity.iloc[-1] - 1) * 100
    bh_sharpe = (y_test.mean() / y_test.std()) * np.sqrt(252)
    
    bh_running_max = bh_equity.expanding().max()
    bh_drawdown = (bh_equity - bh_running_max) / bh_running_max
    bh_max_dd = bh_drawdown.min() * 100
    
    results["Buy & Hold"] = {
        'equity': bh_equity,
        'total_return': bh_return,
        'sharpe': bh_sharpe,
        'max_drawdown': bh_max_dd,
        'time_in_market': 100.0,
        'num_trades': 0,
        'ic': np.nan,
        'ic_pval': np.nan
    }
    print(f" Complete | Return: {bh_return:.2f}% | Sharpe: {bh_sharpe:.3f}")

    #6. Performance comparison
    print("\n" + "="*80)
    print("STEP 5: PERFORMANCE COMPARISON")
    print("="*80)
    
    comparison_df = pd.DataFrame({
        'Strategy': list(results.keys()),
        'Total Return (%)': [r['total_return'] for r in results.values()],
        'Sharpe Ratio': [r['sharpe'] for r in results.values()],
        'Max Drawdown (%)': [r['max_drawdown'] for r in results.values()],
        'IC (test)': [r.get('ic', np.nan) for r in results.values()],
        'IC p-value': [r.get('ic_pval', np.nan) for r in results.values()],
        'Time in Market (%)': [r['time_in_market'] for r in results.values()],
        'Num Trades': [r['num_trades'] for r in results.values()]
    })
    
    print("\n" + comparison_df.to_string(index=False))

    #7. Best model identification
    print("\n" + "="*80)
    print("STEP 6: BEST MODEL ANALYSIS")
    print("="*80)
    
    # Exclude Buy & Hold from ranking
    ml_models = comparison_df[comparison_df['Strategy'] != 'Buy & Hold'].copy()
    
    # Best by Sharpe
    best_sharpe_idx = ml_models['Sharpe Ratio'].idxmax()
    best_sharpe = ml_models.loc[best_sharpe_idx, 'Strategy']
    
    # Best by Return
    best_return_idx = ml_models['Total Return (%)'].idxmax()
    best_return = ml_models.loc[best_return_idx, 'Strategy']
    
    # Best by IC
    best_ic_idx = ml_models['IC (test)'].idxmax()
    best_ic = ml_models.loc[best_ic_idx, 'Strategy']
    
    print(f"\n RANKINGS:")
    print(f"  Best Sharpe Ratio:      {best_sharpe} ({ml_models.loc[best_sharpe_idx, 'Sharpe Ratio']:.3f})")
    print(f"  Best Total Return:      {best_return} ({ml_models.loc[best_return_idx, 'Total Return (%)']:.2f}%)")
    print(f"  Best IC (predictive):   {best_ic} ({ml_models.loc[best_ic_idx, 'IC (test)']:.4f})")
    
    # Overall winner (by Sharpe - most important metric)
    print(f"\nOVERALL WINNER: {best_sharpe}")
    print(f"Reason: Highest risk-adjusted returns (Sharpe Ratio)")
    
    # Outperformance vs Buy & Hold
    bh_return_val = comparison_df[comparison_df['Strategy'] == 'Buy & Hold']['Total Return (%)'].values[0]
    winner_return = ml_models.loc[best_sharpe_idx, 'Total Return (%)']
    outperformance = winner_return - bh_return_val
    
    print(f"\nvs Buy & Hold:")
    print(f"   {best_sharpe}: {winner_return:.2f}%")
    print(f"   Buy & Hold: {bh_return_val:.2f}%")
    print(f"   Outperformance: {outperformance:+.2f}%")

    #8. Visualization
    print("\n" + "="*80)
    print("STEP 7: GENERATING VISUALIZATION")
    print("="*80)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    colors = {'Random Forest': '#2E86AB', 'XGBoost': '#A23B72', 
              'LightGBM': '#F18F01', 'Buy & Hold': '#C73E1D'}
    
    for name, res in results.items():
        axes[0].plot(res['equity'].index, res['equity'].values, 
                    label=name, linewidth=2, color=colors.get(name, 'gray'))
    
    axes[0].set_title('Equity Curves Comparison (2020-2024)', 
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Equity (Starting Capital = $1)', fontsize=11)
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    for name, res in results.items():
        if name == 'Buy & Hold':
            dd = bh_drawdown * 100
        else:
            equity = res['equity']
            running_max = equity.expanding().max()
            dd = (equity - running_max) / running_max * 100
        
        axes[1].plot(dd.index, dd.values, label=name, 
                    linewidth=2, color=colors.get(name, 'gray'), alpha=0.4)
    
    axes[1].set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Drawdown (%)', fontsize=11)
    axes[1].set_xlabel('Date', fontsize=11)
    axes[1].legend(loc='lower left', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("Chart saved as 'model_comparison.png'")
    
    #9. Final summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print(f"Data loaded: {START_DATE} to {END_DATE}")
    print(f"Features engineered: {X.shape[1]} features")
    print(f"Models trained: {len(models)} ML models")
    print(f"Best model: {best_sharpe} (Sharpe: {ml_models.loc[best_sharpe_idx, 'Sharpe Ratio']:.3f})")
    print(f"Outperformance: {outperformance:+.2f}% vs Buy & Hold")
    print(f"Visualization saved: model_comparison.png")
    print("\n" + "="*80)
    print("PIPELINE EXECUTED SUCCESSFULLY")
    print("="*80)


if __name__ == "__main__":
    main()