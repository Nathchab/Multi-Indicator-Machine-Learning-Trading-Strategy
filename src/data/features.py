"""
MVP version includes essential features.
- Returns (1d, 5, 20d)
- Momentum (SMA 20, 50, crossover)
- Volatility (20d realized vol)
- RSI (14d)
- Target variable (next day return)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional

class FeatureEngineer:
    def __init__(self):
        pass

    def add_returns(
        self, 
        df: pd.DataFrame, 
        periods: list[int] = [1, 5, 20],
    ) -> pd.DataFrame:
        """Add return features to the DataFrame."""
        df = df.copy()
        for period in periods:
            ret = df["close"].pct_change(period)
            df[f"return_{period}d"] = ret.shift(1)
            log_ret = np.log(df["close"] / df["close"].shift(period))
            df[f"log_return_{period}d"] = log_ret.shift(1)
        return df

    def add_momentum(
        self, 
        df: pd.DataFrame, 
        windows: list[int] = [20, 50] 
    ) -> pd.DataFrame:
        """Add momentum indicators (Simple Moving Averages).
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data
        windows : list[int]
            List of window sizes
        """
        df = df.copy()
        for window in windows:
            sma = df["close"].rolling(window).mean()
            df[f"sma_{window}d"] = sma.shift(1)
            df[f'price_to_sma_{window}'] = (df['close'] / sma - 1).shift(1)
        
        if 20 in windows and 50 in windows:
            df['ma_cross_20_50'] = (df['sma_20d'] / df['sma_50d'] - 1).shift(1)
        return df

    def add_volatility(
        self, 
        df: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """Add realized volatility feature.
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with returns
        window : int
            Rolling window size
        """
        df = df.copy()
        
        if "return_1d" not in df.columns:
            df["return_1d_temp"] = df["close"].pct_change()
            return_col = "return_1d_temp"
        else:
            return_col = "return_1d"
            df["return_1d_temp"] = df["close"].pct_change()
            return_col = "return_1d_temp"
        
        vol = df[return_col].rolling(window).std() * np.sqrt(252)
        df[f"volatility_{window}d"] = vol.shift(1)
        hl_range = (df['high'] - df['low']) / df['close']
        df[f'hl_range_{window}d'] = hl_range.rolling(window).mean().shift(1)  
        
        if 'return_1d_temp' in df.columns:
            df = df.drop(columns=['return_1d_temp'])
        
        return df

    def add_rsi(
        self, 
        df: pd.DataFrame, 
        window: int = 14
    ) -> pd.DataFrame:
        """Add Relative Strength Index (RSI) feature.
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with returns
        window : int
            RSI period
        """
        df = df.copy()

        delta = df['close'].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        df[f'rsi_{window}'] = rsi.shift(1)
        
        return df
        
    def add_volume_features(
        self, 
        df: pd.DataFrame, 
        window: int = 20
    ) -> pd.DataFrame:
        """
        Add volume-base features
        Parameters 
        ----------
        df : pd.DataFrame
            OHLCV data
        window : int
            Rolling window size
        """

        df = df.copy()
            
        vol_ma = df['volume'].rolling(window).mean()
        df[f'volume_ma_{window}'] = vol_ma.shift(1)
        
        df[f'volume_ratio_{window}'] = (df['volume'] / vol_ma).shift(1)
        
        return df

    def create_target(
            self, 
            df: pd.DataFrame, 
            horizon: int = 1,
            threshold: float = 0.0
        ) -> pd.DataFrame:
        """
        Create target variable (forward returns).
        
        CRITICAL: Target is NOT lagged (it's what we're predicting).
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data
        horizon : int
            Forward-looking periods (default=1 for next-day prediction)
        threshold : float
            Classification threshold for binary target
        """
        df = df.copy()
        df[f"target_return_{horizon}d"] = (
        df["close"].pct_change(horizon).shift(-horizon)
    )
        df[f'target_direction_{horizon}d'] = (
        df[f"target_return_{horizon}d"] > threshold
    ).astype(int)
            
        return df
    
    def create_all_features(
        self, 
        df: pd.DataFrame,
        vix: Optional[pd.Series] = None,
        rf: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Create full feature set (MVP version).
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data (single ticker with date index)
        vix : pd.Series, optional
            VIX data (with date index)
        rf : pd.Series, optional
            Risk-free rate data (with date index)
        """
        print("Creating features...")
        df = self.add_returns(df)
        print("Returns added")
        
        df = self.add_momentum(df)
        print("Momentum added")
        
        df = self.add_volatility(df)
        print("Volatility added")
        
        df = self.add_rsi(df)
        print("RSI added")
        
        df = self.add_volume_features(df)
        print("Volume features added")

        if vix is not None:
                df = df.join(vix, how='left')
                df['vix_change'] = df['vix'].pct_change().shift(1)
                print("✓ VIX added")
        
        if rf is not None:
            df = df.join(rf, how='left')
            if 'return_1d' in df.columns:
                df['excess_return_1d'] = df['return_1d'] - (df['rf'] / 252)
            print("Risk-free rate added")
        
        df = self.create_target(df)
        print("Target created")    
        return df

    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """Get list of feature column names (excluding target and OHLCV).
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with features
        """
        exclude = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
        exclude += [col for col in df.columns if col.startswith('target_')]

        features = [col for col in df.columns if col not in exclude]
        
        return features

    def check_for_leakage(self, df: pd.DataFrame) -> dict:
        """
        Check for potential data leakage.
        """
        diagnostics = {}
        if 'target_return_1d' in df.columns:
            features = self.get_feature_names(df)
        
        correlations = df[features+['target_return_1d']].corr()['target_return_1d'].drop('target_return_1d')
        suspicious = correlations[correlations.abs() > 0.9]
        
        diagnostics['max_correlation'] = correlations.abs().max()
        diagnostics['suspicious_features'] = suspicious.to_dict()
        diagnostics['has_leakage'] = len(suspicious) > 0 
        return diagnostics
    
def prepare_model_data(
    df: pd.DataFrame,
    feature_engineer: FeatureEngineer,
    dropna: bool = True
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for modeling (features and target)
    Parameters 
    ----------
    df : pd.DataFrame
        DataFrame with features and target
    feature_engineer : FeatureEngineer
        Feature engineer instance
    dropna : bool
        Whether to drop rows with NaN
    """
    feature_names = feature_engineer.get_feature_names(df)

    X = df[feature_names].copy()
    y_reg = df['target_return_1d'].copy()
    y_clf = df['target_direction_1d'].copy()

    if dropna:
        valid_idx = X.notna().all(axis=1) & y_reg.notna()
        
        X = X[valid_idx]
        y_reg = y_reg[valid_idx]
        y_clf = y_clf[valid_idx]
        
        print(f"Data prepared: {len(X)} valid samples")
        print(f"Dropped {(~valid_idx).sum()} rows with NaN")

    return X, y_reg, y_clf

if __name__ == "__main__":

    from src.data.fetcher import get_single_ticker
    
    print("Testing FeatureEngineer...")
    
    spy = get_single_ticker('SPY', '2020-01-01', '2024-01-01')
    
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(spy)
    
    print(f"\nCreated {len(features_df.columns)} columns")
    print(f"Date range: {features_df.index.min()} to {features_df.index.max()}")
    
    diagnostics = engineer.check_for_leakage(features_df)
    print(f"\nMax correlation with target: {diagnostics['max_correlation']:.3f}")
    if diagnostics['has_leakage']:
        print("WARNING: Potential data leakage detected!")
        print(diagnostics['suspicious_features'])
    else:
        print("No obvious data leakage detected")

    X, y_reg, y_clf = prepare_model_data(features_df, engineer)
    print(f"\nModel data ready:")
    print(f"- Features: {X.shape}")
    print(f"- Target (regression): {y_reg.shape}")
    print(f"- Target (classification): {y_clf.shape}")
    
    print("\n" + "="*50)
    print("Feature names:")
    print("="*50)
    for feat in engineer.get_feature_names(features_df):
        print(f"  - {feat}")