from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np 
import pandas as pd 

from sklearn.ensemble import RandomForestRegressor

try: 
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

@dataclass
class MLModelWrapper:
    """
    Generic wrapper to have the same interface as OLSModel
    """
    model : Any
    name : str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLModelWrapper":
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        y_pred = self.model.predict(X)
        return pd.Series(y_pred, index=X.index, name="y_pred")
    
    def get_feature_importance(self)-> Optional[pd.Series]:
        """
        Get feature importances if avaliable
        """
        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(
                self.model.feature_importances_,
                index=self.model.feature_names_in_
            ).sort_values(ascending=False)
        return None

    def score(self, X: pd.DataFrame, y: pd.Series)-> float:
        """
        Calculate R-squared score
        """
        return self.model.score(X, y)
        
def make_random_forest(
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 5,
    random_state: int = 42,
)-> MLModelWrapper:
    #Minimum leaf size is increased to reduce overfitting in noisy financial data

    """
    Create random forest model wrapper
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=random_state
    )
    return MLModelWrapper(model=rf, name="RandomForest")

def make_xgboost(
    learning_rate: float = 0.05,
    n_estimators: int = 400,
    max_depth : int= 3,
    subsample : float = 0.8,
    colsample_bytree : float = 0.8,
    random_state : int = 42
) -> MLModelWrapper:
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed")
    xgb = XGBRegressor(
        learning_rate=learning_rate,
        n_estimators = n_estimators,
        max_depth = max_depth,
        subsample = subsample,
        colsample_bytree = colsample_bytree,
        objective = "reg:squarederror",
        random_state = random_state,
        n_jobs = -1
    )
    return MLModelWrapper(model=xgb, name="XGBoost")

def make_lightgbm(
    learning_rate : float = 0.05,
    n_estimators : int = 400,
    num_leaves : int = 31,
    subsample : float = 0.8,
    colsample_bytree : float = 0.8,
    random_state : int = 42
) -> MLModelWrapper:
    if LGBMRegressor is None:
        raise ImportError("lightbgm is not installed")
    lgbm = LGBMRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        subsample = subsample,
        colsample_bytree = colsample_bytree,
        verbose = -1,
        random_state=random_state,
        n_jobs = -1
    )
    return MLModelWrapper(model=lgbm, name="LightGBM")

if __name__ == "__main__":
    print("Testing ML Models...")
    
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
    y = pd.Series(X['a'] * 2 + X['b'] - X['c'] + np.random.randn(100) * 0.1)
    
    rf = make_random_forest()
    rf.fit(X, y)
    print(f"RandomForest R-squared: {rf.score(X, y):.4f}")
    print(f"Feature importance:\n{rf.get_feature_importance()}")
    
    print("\nML models module working!")