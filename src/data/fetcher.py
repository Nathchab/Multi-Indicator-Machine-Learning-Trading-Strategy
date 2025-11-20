from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


DATA_DIR = Path("data") / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _ticker_to_filename(ticker: str, interval: str) -> Path:
    """Convert a ticker and interval to a standardized filename."""
    return DATA_DIR / f"{ticker}_{interval}.csv"

def download_prices(
    tickers: Iterable[str],
    start: str,
    end: str,
    interval: str = "1d",
    source: str = "yahoo",
    cache: bool = True,
) -> pd.DataFrame:
    """Download historical price data for given tickers.

    Args:
        tickers: An iterable of ticker symbols.
        start: Start date in 'YYYY-MM-DD' format.
        end: End date in 'YYYY-MM-DD' format.
        interval: Data interval (e.g., '1d', '1h').
        source: Data source (currently only 'yahoo' is supported).
        cache: Whether to cache the downloaded data locally.

    Returns:
        A DataFrame containing the historical price data.
    """
    tickers = list(tickers)
    if source != "yahoo":
        raise NotImplementedError("Currently, only 'yahoo' source is supported.") 
    all_data = []
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")

        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )   
        if df.empty:
            print(f"No data found for {ticker}. Skipping.")
            continue
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
        
        df = df.reset_index()
        
        date_col = [col for col in df.columns if 'date' in col.lower()][0]
        df = df.rename(columns={date_col: 'date'})
        
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        
        df['ticker'] = ticker
        
        df = df.drop_duplicates(subset=['ticker', 'date'], keep='first')
        
        all_data.append(df)
        
        if cache:
            filepath = _ticker_to_filename(ticker, interval)
            df.to_csv(filepath, index=False)
            print(f"✓ Data for {ticker} cached at {filepath}.")
    
    if not all_data:
        raise ValueError("No data was fetched for the provided tickers.")
    
    data = pd.concat(all_data, ignore_index=True)
    
    data['date'] = pd.to_datetime(data['date'])
    
    n_before = len(data)
    data = data.drop_duplicates(subset=['ticker', 'date'], keep='first')
    n_after = len(data)
    
    if n_before != n_after:
        print(f"Removed {n_before - n_after} duplicate rows")

    data = data.set_index(['ticker', 'date']).sort_index()
    
    return data

def load_cached_prices(
    tickers: Iterable[str],
    interval: str = "1d",
) -> pd.DataFrame:
    """Load cached historical price data for given tickers."""

    tickers = list(tickers)
    all_data = []

    for ticker in tickers:
        filepath = _ticker_to_filename(ticker, interval)
        if not filepath.exists():
            raise FileNotFoundError(f"Cached data for {ticker} not found at {filepath}.")
        
        df = pd.read_csv(filepath, parse_dates=["date"]) 
        all_data.append(df)
    
    data = pd.concat(all_data, ignore_index=True)
    data = data.drop_duplicates(subset=['ticker', 'date'], keep='first')
    data = data.set_index(["ticker", "date"]).sort_index()
    return data

def get_prices(
    tickers: Iterable[str],
    start: str,
    end: str,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Get prices, using cache if available, otherwise download.
    """
    if use_cache:
        try:
            print("Attempting to load from cache...")
            data = load_cached_prices(tickers, interval)
            data_start = data.index.get_level_values('date').min()
            data_end = data.index.get_level_values('date').max()
            
            if data_start <= pd.Timestamp(start) and data_end >= pd.Timestamp(end):
                print("✓ Loaded from cache successfully.")
                return data.loc[(slice(None), slice(start, end)), :]
            else:
                print("Cached data doesn't cover requested period. Downloading...")
        except FileNotFoundError:
            print("Cache not found. Downloading...")
    
    return download_prices(tickers, start, end, interval, cache=use_cache)


def get_single_ticker(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to get data for a single ticker.
    Returns a DataFrame with single index (date).
    """
    data = get_prices([ticker], start, end, interval, use_cache)
    return data.xs(ticker, level='ticker')

def get_vix(start: str, end: str, use_cache: bool = True)-> pd.Series:
    """
    Get VIX index data.

    Returns:
        pd.Series with date index
    """
    data = get_single_ticker("^VIX", start, end, use_cache=use_cache)
    return data["close"].rename("VIX")

def get_risk_free_rate(start: str, end: str, use_cache: bool = True) ->pd.Series:
    """
    Get risk-free rate data (13 week Treasury bill rate).

    Returns:
        pd.Series with date index (annualized rate in decimal form)
    """
    data = get_single_ticker("^IRX", start, end, use_cache=use_cache)
    return (data["close"] / 100).rename("rf")

def get_market_data(
    tickers: Iterable[str],
    start: str,
    end: str,
    include_vix: bool = True,
    include_rf: bool = True,
    use_cache: bool = True,
)-> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
    """
    Get market data needed for the project.

    Returns:
        Tuple of (prices DataFrame, VIX Series or None, risk-free rate Series or None)
    """
    prices = get_prices(tickers, start, end, use_cache=use_cache)
    vix = get_vix(start, end, use_cache) if include_vix else None
    rf = get_risk_free_rate(start, end, use_cache) if include_rf else None
    return prices, vix, rf