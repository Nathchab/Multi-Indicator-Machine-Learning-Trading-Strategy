import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.data.fetcher import get_market_data, get_single_ticker

def test_basic_fetch():
    """Test basic data fetching."""
    print("=" * 50)
    print("Testing single ticker fetch...")
    print("=" * 50)
    
    spy = get_single_ticker('SPY', '2020-01-01', '2024-01-01')
    print(f"\n SPY data shape: {spy.shape}")
    print(f" Date range: {spy.index.min()} to {spy.index.max()}")
    print(f" Columns: {spy.columns.tolist()}")
    print(f"\nFirst few rows:\n{spy.head()}")
    
    print("\n" + "=" * 50)
    print("Testing market data fetch...")
    print("=" * 50)
    
    prices, vix, rf = get_market_data(
        ['SPY', 'QQQ'],
        '2020-01-01',
        '2024-01-01',
        use_cache=False  
    )
    
    print(f"\nPrices shape: {prices.shape}")
    print(f"VIX shape: {vix.shape}")
    print(f"RF shape: {rf.shape}")
    
    
    print(f"\n Missing values in prices: {prices.isna().sum().sum()}")
    print(f" Missing values in VIX: {vix.isna().sum()}")
    print(f" Missing values in RF: {rf.isna().sum()}")
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)

if __name__ == "__main__":
    test_basic_fetch()