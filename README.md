# Multi-Indicator-Machine-Learning-Trading-Strategy

## Research Question

**Can machine learning models extract predictive signals from market and macro-financial indicators ?**

This project compares multiple ML models against a Buy & Hold baseline using out-of-sample backtesting with realistic transaction costs (1 basis point).

---

## Quick Start

### Prerequisites

- Python 3.11+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/Nathchab/Multi-Indicator-Machine-Learning-Trading-Strategy.git
cd Multi-Indicator-Machine-Learning-Trading-Strategy

# Create conda environment
conda env create -f environment.yml
conda activate trading-strategy

# Or use pip
pip install -r requirements.txt
```

### Run the Analysis
```bash

python main.py

jupyter notebook
```

**Expected Output:**
- Multi-model performance comparison table (RF, XGBoost, LightGBM, Buy & Hold)
- Best model identification with justification
- Equity curve visualization saved as `model_comparison.png`
- Detailed metrics: Total Return, Sharpe Ratio, Max Drawdown, IC, Time in Market
- Outperformance analysis vs Buy & Hold baseline

---

## Notebooks Guide

The `notebooks/` folder contains 6 Jupyter notebooks for comprehensive analysis:

### 1. Data Exploration (`01_data_exploration.ipynb`)
**Purpose:** Exploratory Data Analysis and data quality validation

- Data quality checks and missing value analysis
- Feature distributions and summary statistics
- Correlation analysis between features
- Price and return patterns
- Outlier detection
- Time series stationarity tests

**Key outputs:** Correlation heatmaps, distribution plots, data quality report

---

### 2. Model Comparison (`02_model_comparison.ipynb`)
**Purpose:** Train and compare multiple ML models

- Train Random Forest, XGBoost, LightGBM, and OLS baseline
- Compare predictive metrics (R², IC, RMSE, AUC)
- Feature importance analysis
- Cross-validation results
- Hyperparameter impact study
- Model selection criteria

**Key outputs:** Model comparison table, feature importance plots, prediction scatter plots

---

### 3. Threshold Optimization (`03_threshold_optimization.ipynb`)
**Purpose:** Optimize signal generation thresholds

- Test multiple threshold strategies (quantile-based, fixed, adaptive)
- Sharpe Ratio optimization
- Risk-return tradeoff analysis
- Time-in-market vs performance
- Sensitivity analysis
- Optimal threshold identification

**Key outputs:** Threshold performance curves, optimal threshold recommendation

---

### 4. Stochastic Features Analysis (`04_stochastic_features.ipynb`)
**Purpose:** Analyze impact of stochastic calculus features

- GBM framework (Geometric Brownian Motion)
- Drift (μ) and volatility (σ) estimation at multiple timeframes
- Before/after comparison (with vs without stochastic features)
- Feature impact on Sharpe Ratio and IC
- Regime detection (volatility regimes, drift mean reversion)
- Vol-of-vol indicators

**Key outputs:** Feature impact comparison, regime visualization, performance delta

---

### 5. Walk-Forward Validation (`05_walk_forward_validation.ipynb`)
**Purpose:** Robust out-of-sample testing with rolling windows

- Rolling window backtests (500d train / 20d test)
- Performance stability over time
- Model retraining frequency analysis
- Strategy comparison under different market regimes
- Realistic transaction cost simulation (1 bps)
- Overfitting detection

**Key outputs:** Walk-forward equity curves, rolling performance metrics, stability analysis

---

### 6. OLS Baseline Backtest (`backtest_ols.ipynb`)
**Purpose:** Backtest OLS (Ordinary Least Squares) baseline model

- Simple linear regression benchmark
- Compare OLS vs ML models
- Assess value added by non-linear models
- Performance attribution analysis
- Statistical significance tests
- Economic vs statistical significance

**Key outputs :** OLS performance metrics, comparison with ML models

---

### Recommended Order

1. **Start here :** `01_data_exploration.ipynb` - Understand the data
2. **Then :** `02_model_comparison.ipynb` - See which models work best
3. **Optimize :** `03_threshold_optimization.ipynb` - Fine-tune signal generation
4. **Validate :** `05_walk_forward_validation.ipynb` - Ensure robustness
5. **Deep dive :** `04_stochastic_features.ipynb` - Understand advanced features
6. **Baseline :** `backtest_ols.ipynb` - Compare with simple benchmark

**All notebooks are self-contained** and can be run independently!

## Project Structure

```
Multi-Indicator-Machine-Learning-Trading-Strategy/
├── main.py                          # Main entry point - Model comparison
├── src/
│   ├── data/
│   │   ├── fetcher.py              # Data download (yfinance, FRED)
│   │   └── features.py             # Feature engineering (technical + stochastic)
│   ├── models/
│   │   ├── baseline.py             # OLS baseline model
│   │   └── ml_models.py            # RF, XGBoost, LightGBM
│   ├── evaluation/
│   │   ├── backtest.py             # Backtesting engine
│   │   └── walkforward.py          # Walk-forward validation
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA and data quality checks
│   ├── 02_model_comparison.ipynb   # Model training and evaluation
│   ├── 03_threshold_optimization   # Optimization of the threshold
│   └── 04_stochastic_features.ipynb      # Stochastic calculus features
│   └── 05_walk_forward_validation.ipynb  # Robust out-of-sample testing
│   └── backtest_ols.ipynb                # Backtesting OLS
├── test/
│   ├── test_fetcher.py             # Data fetching tests
├── environment.yml                 # Conda dependencies
├── requirements.txt                # Pip dependencies
├── README.md                       # This file
└── PROPOSAL.md                     # Original project proposal
```

---

## Methodology

### 1. Data Collection
- **Asset :** SPY (S&P 500 ETF)
- **Period :** 2015-01-01 to 2024-01-01
- **Features :** technical + stochastic calculus indicators
- **Sources :** Yahoo Finance (prices), FRED (VIX, risk-free rate)

### 2. Feature Engineering

**Technical Indicators :**
- Returns (1d, 5d, 20d log returns)
- Momentum (SMA 20/50, price-to-SMA ratios)
- Volatility (20d realized vol, Garman-Klass)
- RSI (14-period)
- Volume indicators

**Stochastic Calculus Features (GBM-based):**
- Drift estimation (μ at 20/60/120d windows)
- Volatility estimation (σ at 20/60/120d windows)
- Drift/Vol ratios (Sharpe-like indicators)
- Volatility-of-volatility (regime detection)
- Mean reversion indicators

### 3. Train/Test Split
- **Train :** 2015-01-01 to 2019-12-31 (5 years)
- **Test :** 2020-01-01 to 2024-01-01 (4 years)
- **Method :** Temporal split (no look-ahead bias)

### 4. Models Tested

| Model | Hyperparameters | Purpose |
|-------|----------------|---------|
| **Random Forest** | n_estimators=300, max_depth=10 | Ensemble baseline |
| **XGBoost** | n_estimators=400, learning_rate=0.05 | Gradient boosting |
| **LightGBM** | n_estimators=400, num_leaves=31 | Fast gradient boosting |
| **OLS (Baseline)** | HAC standard errors | Linear benchmark |

### 5. Signal Generation
- **Method :** Quantile thresholding (top 30% predictions → BUY)
- **Alternative strategies tested :** Conservative (>5bps), VIX-adjusted

### 6. Backtesting
- **Transaction costs :** 1 basis point (realistic)
- **Position sizing :** Binary (100% in or out)
- **Benchmark :** Buy & Hold SPY

### 7. Evaluation Metrics
- **Sharpe Ratio** (primary metric - risk-adjusted return)
- Total Return
- Max Drawdown
- Information Coefficient (predictive power)
- Time in Market
- Win Rate

### 8. Validation
- **Walk-Forward :** 500-day train / 20-day test rolling windows
- **Purpose :** Ensure results are not due to overfitting

---

## Features Deep Dive

### Why Stochastic Calculus Features ?

Traditional technical indicators capture **price patterns**, but stochastic features capture **market regimes**:

```python
# Geometric Brownian Motion framework
dS = μ·S·dt + σ·S·dW

# We estimate μ (drift) and σ (volatility) at multiple timeframes
# This helps identify:
# - High vol regimes → Reduce exposure
# - Drift changes → Trend reversals
# - Vol-of-vol spikes → Market stress
```

**Impact :** Stochastic features improved Sharpe by **+33.29%** (see notebooks for details).

---

**Tests automatically configure PYTHONPATH** - no manual setup needed!

---

## Configuration

### Customizing Parameters

Edit parameters in `main.py`:

```python
# Date range
START_DATE = "2015-01-01"
END_DATE   = "2024-01-01"

# Train/test split
split_date = "2020-01-01"

# Signal threshold
threshold = 0.4

# Transaction costs (in basis points)
trading_cost_bps = 1.0
```

### Model Hyperparameters

Modify in `src/models/ml_models.py`:

```python
def make_random_forest(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
):
    # ...
```

### Core Dependencies
```
python>=3.11
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
yfinance>=0.2.28
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.11.0
```

### Optional Dependencies
```
jupyter>=1.0.0          # For notebooks
```

See `requirements.txt` or `environment.yml` for full list.

---

## Acknowledgments

- **Data Sources:** Yahoo Finance (yfinance), Federal Reserve Economic Data (FRED)
- **Inspiration:** Quantitative finance research on ML for trading
- **Academic Context:** HEC Lausanne - Datascience and andvanced programming 

---

