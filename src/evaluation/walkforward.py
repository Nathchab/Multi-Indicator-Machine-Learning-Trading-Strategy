import numpy as np
import pandas as pd
from typing import Dict
from src.evaluation.backtest import backtest_signals

def walk_forward_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, object],
    train_window: int = 500,
    test_window: int = 20,
    signal_threshold: float = 0.7, 
    verbose: bool = True
):
    """
    Walk-forward backtest:
    - Rolling training window (e.g., 500 days)
    - Rolling test window (e.g., 20 days)
    - Model retrained at each iteration
    - Returns concatenated predictions for each model
    - This setup mimics a realistic live trading scenario, where models are periodically retrained on expanding information.
    """
    #Store out-of-sample strategy returns for each rolling test window to reconstruct a continuous equity curve
    n = len(X)
    all_strategy_returns = {name: [] for name in models.keys()}

    start = 0
    end_train = train_window
    end_test = train_window + test_window

    if verbose:
        print("\n============================================")
        print(f" WALK-FORWARD BACKTEST ({train_window}d train / {test_window}d test)")
        print("============================================")

    #Iteratively move the training and testing windows forward to avoid any look-ahead bias
    while end_test <= n:

        #Rolling windows
        X_train = X.iloc[start:end_train]
        y_train = y.iloc[start:end_train]
        X_test = X.iloc[end_train:end_test]

        if verbose:
            print(f"\nTraining: {X_train.index[0].date()} {X_train.index[-1].date()}")
            print(f"Testing : {X_test.index[0].date()} {X_test.index[-1].date()}")

        #Train + Predict for each model
        for name, model in models.items():
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            pred = pd.Series(pred, index=X_test.index)

            #Using a quantile-based threshold ensures a stable exposure level and avoids dependence on absolute prediction magnitudes
            threshold_value = pred.quantile(signal_threshold)
            signals = (pred > threshold_value).astype(int)

            y_test = y.loc[X_test.index]

            bt = backtest_signals(
                returns=y_test,
                signals=signals,
                trading_cost_bps=1.0,
                starting_capital=1.0
            )

            all_strategy_returns[name].append(bt.strategy_returns)

        #Windows are advanced by the test size to maintain a non-overlapping out-of-sample evaluation
        start += test_window
        end_train += test_window
        end_test += test_window

    #Concatenate all out-of-sample returns to form a single time series per model
    final_returns = {
    name: pd.concat(ret_list).sort_index()
    for name, ret_list in all_strategy_returns.items()
    }

    return final_returns
