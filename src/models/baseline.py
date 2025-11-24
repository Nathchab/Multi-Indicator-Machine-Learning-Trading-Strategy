"""
Baseline using OLS regression
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from typing import Tuple, List, Optional, Dict
import warnings

class OLSModel:
    """
    OLS baseline model for predicting returns 

    This model provides a statistically interpretable baseline with :
    - Coefficients significance testing (t-tests)
    - Model diagnostics (R-squared, AIC, BIC)
    - Robust standard errors (HAC for time series)
    - Full transparency on linear relationships
    """
    def __init__(self, use_hac:bool = True, maxlags: int = 5):
        """
        use_hac : bool
            Whether to use Heteroskedasticity and Autocorrelation Consistent
            (HAC) standard errors. Recommended for time series.
        maxlags : int
            Maximum number of lags for HAC covariance (Newey-West)
        """
        self.use_hac = use_hac
        self.maxlags = maxlags
        self.model = None
        self.results = None
        self.features_names = None
        self.X_train_mean = None
        self.X_train_std = None

    def fit(self, X: pd.DataFrame, y: pd.Series, add_constant: bool = True, standardize: bool = False) -> "OLSModel":
        """
        Fit the OLS model
        """
        self.features_names = X.columns.tolist()
        X_work = X.copy()

        if standardize:
            self.X_train_mean = X_work.mean()
            self.X_train_std = X_work.std()
            X_work = (X_work - self.X_train_mean) / self.X_train_std
        else:
            self.X_train_mean = None
            self.X_train_std = None
        
        if add_constant:
            X_work = sm.add_constant(X_work)
        self.model = sm.OLS(y, X_work, missing='drop')

        if self.use_hac:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.results = self.model.fit(cov_type='HAC', cov_kwds={'maxlags': self.maxlags})
        
        else:
            self.results = self.model.fit()

        return self

    def predict(self, X: pd.DataFrame, add_constant: bool = True) -> np.ndarray:

        """
        Make predictions using the fitted model
        """
        if self.results is None:
            raise ValueError("Model must be fitted before making predictions'.")
        if self.X_train_mean is not None:
            X_transformed = (X - self.X_train_mean) / self.X_train_std
        else:
            X_transformed = X.copy()
        
        if add_constant:
            X_transformed = sm.add_constant(X_transformed)
        return self.results.predict(X_transformed)

    def summary(self) -> None:
        """
        Print the summary of the fitted model
        """
        if self.results is None:
            raise ValueError("Model must be fitted before getting summary'.")
        
        print("="*80)
        print("OLS Regression Results")
        print("="*80)
        print(self.results.summary())
        print("\n" + "="*80)
        print("Model fit statistics")
        print("="*80)
        print(f"R-squared : {self.results.rsquared:.6f}")
        print(f"Adj. R-squared : {self.results.rsquared_adj:.6f}")
        print(f"AIC : {self.results.aic:.2f}")
        print(f"BIC : {self.results.bic:.2f}")
        print(f"F-statistic : {self.results.fvalue:.4f}")
        print(f"Prob (F-statistic) : {self.results.f_pvalue:.6f}")

    def get_significant_features(self, alpha: float = 0.05)-> List[str]:
        """
        Get List of statistically significant features
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")
        pvalues = self.results.pvalues
        significant = pvalues[pvalues < alpha].index.tolist()

        if 'const' in significant:
            significant.remove('const')
        return significant

    def get_coefficients(self) -> pd.DataFrame:
        """
        Get coefficients estimates with statistics
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")
        
        coef_df = pd.DataFrame({"coefficient" : self.results.params, 
                                "std_error" : self.results.bse,
                                "t_statistic" : self.results.tvalues, 
                                "p_value" : self.results.pvalues,
                                'ci_lower': self.results.conf_int()[0],
                                'ci_upper': self.results.conf_int()[1]})

        coef_df["significance"] = coef_df["p_value"].apply(lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "")

        return coef_df
    
    def diagnostics(self) -> Dict:
        """
        Run Diagnostic tests on the model
        """
        if self.results is None: 
            raise ValueError("Model must be fitted first")
        
        diagnostics = {
            "r_squared" : self.results.rsquared,
            "adj_r_squared" : self.results.rsquared_adj,
            "aic" : self.results.aic,
            "bic" : self.results.bic, 
            "f_statistic" : self.results.fvalue,
            "f_pvalue" : self.results.f_pvalue,
            "condition_number" : self.results.condition_number,
            "n_observations" : int(self.results.nobs),
            "n_features" : len(self.feature_names) if self.feature_names else 0
        }
        return diagnostics

class RollingOLS:
    """
    Rolling window OLS for handling non-stationarity in time series

    This approach retrains the model periodically to adapt to changing market regimes, addressing the non-stationarity problem
    """
    def __init__ (self, window: int = 252, use_hac: bool = True, min_periods: int = None):
        self.window = window
        self.use_hac = use_hac
        self.min_periods = min_periods or window
        self.predictions = []
        self.coefficients = []
        self.feature_names = None

    def fit_predict(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, pd.DataFrame]:
        """ 
        Fit rolling OLS and make prediction
        """
        self.feature_names = X.columns.tolist()
        predictions = np.full(len(y), np.nan)
        coeffictients = []

        X_with_const = sm.add_constant(X)

        print(f"Running rolling OLS with window={self.window}...")
        print(f"Total periods : {len(X)}, Training starts at periods {self.window}")

        for i in range (self.window, len(X)):
            if i % 100 == 0:
                print(f"Processing period {i}/{len(X)}...")
            X_train = X_with_const.iloc[i-self.window:i]
            y_train = y.iloc[i-self.window:i]

            try:
                model = sm.OLS(y_train, X_train, missing="drop")

                if self.use_hac:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        results = model.fit(cov_type="HAC", cov_kwds={"maxlags" : 5})
                else:
                    results = model.fit()
                
                X_test = X_with_const.iloc[[i]]
                predictions[i] = results.predict(X_test).values[0]
                coef_dict = results.params.to_dict()
                coef_dict["date"] = X.index[i]
                cooefficients.append(coef_dict)
            except Exception as e:
                continue 
            self.predictions = predictions
            self.coefficients = pd.DataFrame(coeffictients)
            if "date" in self.coefficients.columns:
                self.coefficients = self.coefficients.set_index("date")
            
            print(f"Rolling OLS complete. Generated {(~np.isnan(predictions)).sum()} predictions")
            return predictions, self.coefficients

class FeatureSelector:
    """
    Select features for OLS based on statistical criteria
    """
    @staticmethod
    def select_by_correlation(X: pd.DataFrame, y: pd.Series, threshold: float = 0.02, max_features: int = None)-> List[str]:
        """
        Select features based on correlation with target

        Parameters
            ----------
            X : pd.DataFrame
                Features
            y : pd.Series
                Target
            threshold : float
                Minimum absolute correlation
            max_features : int
                Maximum number of features to select
        """

        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        selected = correlations[correlations >= threshold].index.tolist()
        if max_features:
            selected = selected[:max_features]
        return selected
    @staticmethod
    def select_by_vif(X: pd.DataFrame, threshold: float = 10) -> List[str]:
        """
        Remove features with high multicollinearity (VIF > threshold).
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
        threshold : float
            VIF threshold
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        selected = X.columns.tolist()
        while True:
            X_subset = X[selected]
            vif_data = pd.DataFrame({"feature" : selected, "VIF" : [variance_inflation_factor(X_subset.value, i)]})
            max_vif = vif_data["VIF"].max()
            if max_vif <= threshold:
                break
            
            worst_feature = vif_data.loc[vif_data["VIF"].idmax(), "feature"]
            selected.remove(worst_feature)
            print(f"Remove {worst_feature} (VIF={max_vif:.2f})")
        return selected

if __name__ == "__main__":

    print("Testing OLS Model...")
    
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    X = pd.DataFrame({
        'momentum': np.random.randn(n),
        'volatility': np.random.randn(n),
        'rsi': np.random.randn(n)
    }, index=dates)
    
    y = pd.Series(
        0.1 * X['momentum'] - 0.05 * X['volatility'] + np.random.randn(n) * 0.02,
        index=dates,
        name='returns'
    )
    
    model = OLSModel(use_hac=True)
    model.fit(X, y)
    
    print("\n" + "="*80)
    model.summary()
    
    print("\n" + "="*80)
    print("COEFFICIENT ESTIMATES")
    print("="*80)
    print(model.get_coefficients())
    
    print("\n" + "="*80)
    print(f"Significant features (alpha=0.05): {model.get_significant_features()}")
    
    predictions = model.predict(X.head(10))
    print(f"\nFirst 10 predictions: {predictions}")
    
    print("\n✓ OLS Model test completed successfully!")