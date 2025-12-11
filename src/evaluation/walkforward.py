import numpy as np
import pandas as pd
from typing import Dict

def walk_forward_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, object],
    train_window: int = 500,
    test_window: int = 20,
    verbose: bool = True
):
    """
    Walk-forward backtest:
    - Rolling training window (e.g., 500 days)
    - Rolling test window (e.g., 20 days)
    - Model retrained at each iteration
    - Returns concatenated predictions for each model
    """

    n = len(X)
    all_predictions = {name: [] for name in models.keys()}

    start = 0
    end_train = train_window
    end_test = train_window + test_window

    if verbose:
        print("\n============================================")
        print(f" WALK-FORWARD BACKTEST ({train_window}d train / {test_window}d test)")
        print("============================================")

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
            all_predictions[name].append(
                pd.Series(pred, index=X_test.index)
            )

        #Roll the windows
        start += test_window
        end_train += test_window
        end_test += test_window

    #Concatenate all predictions
    final_predictions = {
        name: pd.concat(pred_list).sort_index()
        for name, pred_list in all_predictions.items()
    }

    return final_predictions