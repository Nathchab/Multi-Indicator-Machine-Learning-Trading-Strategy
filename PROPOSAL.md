# Project Proposal

## Project Title
Multi-Indicator Machine Learning Trading Strategy with Econometric Baseline

## Category
Data Science · Econometrics · Machine Learning · Quantitative Finance

## Problem Statement / Motivation

Financial markets are noisy and non-stationary, making it difficult to identify reliable trading signals from price and volume data. Many common trading strategies rely on simple rule-based indicators (e.g., moving average crossovers), but these often fail across different market regimes.

This project aims to evaluate whether statistically grounded models and machine learning methods can better identify predictive patterns in market data and translate them into improved trading performance. The goal is not only to make predictions but to verify their statistical validity and economic usefulness.

The core work consists of statistical modeling, machine learning, and performance evaluation.

## Planned Approach and Technologies

**Data:**
- Daily OHLCV data for selected ETFs (e.g., SPY, QQQ, TLT, GLD) along with VIX and risk-free rates
- All models are multivariate, using multiple features concurrently

**Econometric Baseline (OLS):**

Linear regression model (OLS) predicting next-day excess returns based on lagged indicators such as momentum, volatility, RSI, moving average spreads, and VIX changes.

Using statsmodels:
- Coefficient interpretation (β's)
- Hypothesis testing (p-values, t-stats)
- Model comparison (Adjusted R², AIC/BIC)
- Robust (HAC) errors for time-series autocorrelation
- This establishes a transparent, statistically interpretable baseline before applying machine learning

**Machine Learning Models:**

Evaluate non-linear methods such as Random Forest and Gradient Boosting (XGBoost / LightGBM), using:
- TimeSeriesSplit / walk-forward validation
- Out-of-sample metrics: AUC, Brier Score, Information Coefficient (IC)
- SHAP values for model interpretability
- Only if the ML model demonstrates predictive value beyond OLS do we proceed to simulation

**Trading Simulation (Application Phase):**

A backtesting engine (backtrader / vectorbt) will allocate positions based on ML signal confidence, incorporating transaction costs and tracking:
- Sharpe & Sortino Ratios
- Max Drawdown
- Turnover & Stability across market regimes

## Expected Challenges and How to Address Them

- **Overfitting**: Early stopping, regularization, feature reduction via SHAP
- **Non-stationarity**: Rolling retraining + regime-based evaluation
- **Leakage**: All features lagged; validation respects time ordering

## Success Criteria

We will benchmark the model-based strategy against:
- Buy-and-Hold, and
- The trailing average return (historical mean)

We expect Buy-and-Hold to perform strongly, and the goal is not to outperform it, but to use it as a unit of measurement to evaluate whether combining multiple signals provides any incremental predictive value. Success means demonstrating that the multivariate models offer meaningful insight or structure.

## Stretch Goals (if time permits)

- Stochastic calculus-based volatility features (GBM drift/σ estimation)
