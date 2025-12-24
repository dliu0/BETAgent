"""
Ticker history fetcher with flexible period and interval control.
Supports daily/hourly data with easy CSV export and auto-refresh capabilities.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Literal
from src.getdata import StockDataStream
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TickerHistoryFetcher:
    """
    Fetch ticker price history with flexible time periods and intervals.
    Returns data as DataFrame (dates as rows, tickers as columns).
    """
    
    def __init__(self,
                 tickers: List[str],
                 interval: Literal["1m", "5m", "15m", "30m", "60m", "1h", "1d", "1wk", "1mo"] = "1d",
                 period: Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"] = "1y"):
        """
        Initialize ticker history fetcher.
        
        Args:
            tickers: List of ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
            interval: Data interval - '1m', '5m', '15m', '30m', '60m', '1h', '1d', '1wk', '1mo'
            period: Time period - '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
        """
        self.tickers = tickers
        self.interval = interval
        self.period = period
        self.data = None
        self.last_fetch_time = None
        
        logger.info(f"Initialized TickerHistoryFetcher for {len(tickers)} tickers")
    
    def _period_to_dates(self) -> tuple[str, str]:
        """Convert period string to start and end dates."""
        end_date = datetime.now()
        
        period_map = {
            "1d": timedelta(days=1),
            "5d": timedelta(days=5),
            "1mo": timedelta(days=30),
            "3mo": timedelta(days=90),
            "6mo": timedelta(days=180),
            "1y": timedelta(days=365),
            "2y": timedelta(days=730),
            "5y": timedelta(days=1825),
            "10y": timedelta(days=3650),
            "ytd": timedelta(days=(end_date - datetime(end_date.year, 1, 1)).days),
            "max": timedelta(days=36500),  # ~100 years
        }
        
        if self.period not in period_map:
            raise ValueError(f"Invalid period: {self.period}")
        
        start_date = end_date - period_map[self.period]
        
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    
    def fetch(self) -> pd.DataFrame:
        """
        Fetch ticker history data.
        
        Returns:
            DataFrame with dates as rows, tickers as columns
        """
        start_date, end_date = self._period_to_dates()
        
        logger.info(f"Fetching {self.interval} data from {start_date} to {end_date}")
        
        # Use StockDataStream to fetch data
        stream = StockDataStream(
            self.tickers,
            start_date=start_date,
            end_date=end_date,
            interval=self.interval
        )
        
        self.data = stream.fetch_data()
        self.last_fetch_time = datetime.now()
        
        logger.info(f"Fetched {len(self.data)} rows × {len(self.data.columns)} columns")
        return self.data
    
    def get_data(self) -> pd.DataFrame:
        """
        Get current data, fetching if necessary.
        
        Returns:
            DataFrame with dates as rows, tickers as columns
        """
        if self.data is None:
            self.fetch()
        return self.data
    
    def refresh(self) -> pd.DataFrame:
        """
        Refresh data to get the most recent values.
        
        Returns:
            Updated DataFrame
        """
        logger.info("Refreshing data...")
        return self.fetch()
    
    def update_period(self, period: str) -> pd.DataFrame:
        """
        Change the time period and fetch new data.
        
        Args:
            period: New period string
        
        Returns:
            DataFrame with new period data
        """
        self.period = period
        return self.fetch()
    
    def update_interval(self, interval: str) -> pd.DataFrame:
        """
        Change the time interval and fetch new data.
        
        Args:
            interval: New interval string ('1d', '1h', etc.)
        
        Returns:
            DataFrame with new interval data
        """
        self.interval = interval
        return self.fetch()
    
    def to_csv(self, filepath: str, include_index: bool = True) -> None:
        """
        Export data to CSV file.
        
        Args:
            filepath: Path where to save the CSV
            include_index: Whether to include the date index
        """
        if self.data is None:
            self.fetch()
        
        self.data.to_csv(filepath, index=include_index)
        logger.info(f"Data exported to {filepath}")
    
    def summary(self) -> dict:
        """
        Get summary information about fetched data.
        
        Returns:
            Dictionary with data shape, date range, and tickers
        """
        if self.data is None:
            return {"status": "No data fetched yet"}
        
        return {
            "shape": self.data.shape,
            "rows": len(self.data),
            "columns": len(self.data.columns),
            "start_date": self.data.index[0],
            "end_date": self.data.index[-1],
            "tickers": list(self.data.columns),
            "last_fetch": self.last_fetch_time,
            "interval": self.interval,
            "period": self.period
        }


# Convenience functions
def fetch_tickers(tickers: List[str],
                  interval: str = "1d",
                  period: str = "1y") -> pd.DataFrame:
    """
    Quick function to fetch ticker data with default parameters.
    
    Args:
        tickers: List of ticker symbols
        interval: Data interval (default: '1d')
        period: Time period (default: '1y')
    
    Returns:
        DataFrame with dates as rows, tickers as columns
    """
    fetcher = TickerHistoryFetcher(tickers, interval, period)
    return fetcher.fetch()


def fetch_and_export(tickers: List[str],
                     filepath: str,
                     interval: str = "1d",
                     period: str = "1y",
                     include_index: bool = True) -> pd.DataFrame:
    """
    Fetch ticker data and immediately export to CSV.
    
    Args:
        tickers: List of ticker symbols
        filepath: Where to save the CSV
        interval: Data interval (default: '1d')
        period: Time period (default: '1y')
        include_index: Whether to include dates in CSV
    
    Returns:
        DataFrame that was exported
    """
    fetcher = TickerHistoryFetcher(tickers, interval, period)
    data = fetcher.fetch()
    fetcher.to_csv(filepath, include_index)
    return data


# Example usage
if __name__ == "__main__":
    # Example 1: Fetch daily data for last year
    print("=" * 60)
    print("Example 1: Daily data for last year")
    print("=" * 60)
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    
    fetcher = TickerHistoryFetcher(tickers, interval="1d", period="1y")
    df = fetcher.fetch()
    
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nLast 5 rows:")
    print(df.tail())
    
    # Export to CSV
    fetcher.to_csv("ticker_history_daily.csv")
    print(f"\nExported to ticker_history_daily.csv")
    
    # Example 2: Fetch hourly data for last 5 days
    print("\n" + "=" * 60)
    print("Example 2: Hourly data for last 5 days")
    print("=" * 60)
    
    fetcher_hourly = TickerHistoryFetcher(tickers, interval="1h", period="5d")
    df_hourly = fetcher_hourly.fetch()
    
    print(f"\nData shape: {df_hourly.shape}")
    print(f"Date range: {df_hourly.index[0]} to {df_hourly.index[-1]}")
    print(f"\nFirst 5 rows:")
    print(df_hourly.head())
    
    # Export hourly data
    fetcher_hourly.to_csv("ticker_history_hourly.csv")
    print(f"\nExported to ticker_history_hourly.csv")
    
    # Example 3: Summary statistics
    print("\n" + "=" * 60)
    print("Example 3: Summary information")
    print("=" * 60)
    print(fetcher.summary())
    
    # Example 4: Dynamic updates
    print("\n" + "=" * 60)
    print("Example 4: Dynamic period/interval updates")
    print("=" * 60)
    
    # Change to 3 months of data
    df_3mo = fetcher.update_period("3mo")
    print(f"\nUpdated to 3 months: {df_3mo.shape}")
    
    # Change to hourly interval
    df_hourly_updated = fetcher.update_interval("1h")
    print(f"Updated to hourly: {df_hourly_updated.shape}")
    
    # Refresh to latest data
    df_refreshed = fetcher.refresh()
    print(f"Refreshed to latest: {df_refreshed.shape}")
