import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataStream:
    """
    A streaming data source for stock price data from Yahoo Finance.
    Pulls closing prices for multiple stocks with each column representing a stock
    and each row representing a trading day.
    """
    
    def __init__(self, 
                 tickers: List[str],
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 interval: str = "1d"):
        """
        Initialize the stock data stream.
        
        Args:
            tickers: List of stock tickers (e.g., ['AAPL', 'MSFT', 'GOOGL'])
            start_date: Start date (YYYY-MM-DD). Defaults to 1 year ago.
            end_date: End date (YYYY-MM-DD). Defaults to today.
            interval: Data interval - '1d', '1wk', '1mo' (default: '1d')
        """
        self.tickers = tickers
        self.interval = interval
        
        # Set default dates
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.last_update = None
        
        logger.info(f"Initialized StockDataStream with {len(tickers)} tickers")
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch stock price data for all tickers.
        
        Returns:
            DataFrame with closing prices (rows=dates, columns=tickers)
        """
        try:
            logger.info(f"Fetching data for {len(self.tickers)} stocks from {self.start_date} to {self.end_date}")
            
            # Download data for all tickers at once
            data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                progress=False
            )
            
            # Handle case where only one ticker returns a Series instead of DataFrame
            if len(self.tickers) == 1:
                data = data[['Close']].rename(columns={'Close': self.tickers[0]})
            else:
                # Extract only Close prices
                data = data['Close']
            
            # Sort by date ascending
            data = data.sort_index()
            
            # Remove any rows with NaN values (missing data days)
            data = data.dropna()
            
            self.data = data
            self.last_update = datetime.now()
            
            logger.info(f"Successfully fetched {len(data)} rows of data for {len(data.columns)} stocks")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def get_latest_data(self, num_days: Optional[int] = None) -> pd.DataFrame:
        """
        Get the latest data, optionally limited to recent days.
        
        Args:
            num_days: Number of most recent days to return (None = all data)
        
        Returns:
            DataFrame with closing prices
        """
        if self.data is None:
            self.fetch_data()
        
        data = self.data
        if num_days:
            data = data.tail(num_days)
        
        return data
    
    def update_data(self, new_end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Update data stream with new data since last fetch.
        Useful for continuously pulling new data.
        
        Args:
            new_end_date: End date for update (defaults to today)
        
        Returns:
            Updated DataFrame
        """
        if new_end_date is None:
            new_end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Update end date and fetch new data
        old_end = self.end_date
        self.end_date = new_end_date
        
        logger.info(f"Updating data from {old_end} to {new_end_date}")
        return self.fetch_data()
    
    def get_data_as_1d(self, 
                       flatten_order: str = 'C',
                       include_dates: bool = False) -> np.ndarray | Tuple[np.ndarray, List]:
        """
        Convert 2D stock data to 1D array for algorithm processing.
        
        Args:
            flatten_order: 'C' (row-major/C-order) or 'F' (column-major/Fortran-order)
            include_dates: If True, also return the date index
        
        Returns:
            1D numpy array (or tuple with dates if include_dates=True)
        """
        if self.data is None:
            self.fetch_data()
        
        data_1d = self.data.values.flatten(order=flatten_order)
        
        if include_dates:
            return data_1d, self.data.index.tolist()
        
        return data_1d
    
    def get_stock_by_ticker(self, ticker: str) -> pd.Series:
        """
        Get historical data for a single stock.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Series with closing prices for that stock
        """
        if self.data is None:
            self.fetch_data()
        
        if ticker not in self.data.columns:
            raise ValueError(f"Ticker {ticker} not in data")
        
        return self.data[ticker]
    
    def to_csv(self, filepath: str) -> None:
        """Save current data to CSV file."""
        if self.data is None:
            self.fetch_data()
        
        self.data.to_csv(filepath)
        logger.info(f"Data saved to {filepath}")
    
    def summary_stats(self) -> pd.DataFrame:
        """Get summary statistics for all stocks."""
        if self.data is None:
            self.fetch_data()
        
        return self.data.describe()


# Example predefined stock lists
STOCKS_TECH_100 = [
    "AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "GOOG", "AMZN", "META", "AVGO", "QCOM",
    "INTC", "AMD", "CRM", "NFLX", "ADBE", "INTU", "SNPS", "CDNS", "MU", "KLAC",
    "LRCX", "ASML", "ARM", "MCHP", "AMAT", "LSCC", "MXIM", "PSTG", "SPLK", "OKTA",
    "ZM", "CRWD", "SNOW", "DDOG", "HACK", "PAYC", "TEAM", "NOW", "VMWR", "FTNT",
    "JNPR", "PALO", "PFPT", "CNUV", "PING", "CHKP", "RAMP", "PD", "SEMR", "DBX",
    "NET", "UPST", "SQ", "PYPL", "HOOD", "COIN", "GSAT", "AKAM", "KLAY", "CDW",
    "MANH", "ODFL", "JKHY", "VRSK", "FARO", "IT", "CHTR", "ADXN", "ATVI", "RBLX",
    "ZS", "CSCO", "EWBC", "FNWK", "GDDY", "HUBS", "ORCL", "HPE", "IBM", "UIS",
    "PRG", "AZPN", "SMAR", "ALTR", "ROKU", "PLTR", "U", "MSTR", "RIOT", "CIFR",
    "MINING", "VTNR", "CLSK", "MARA", "HUT", "BTBT", "GFAI", "SCAM", "DMRC", "WHG"
]

STOCKS_FINANCE_100 = [
    "JPM", "BLK", "BAC", "WFC", "GS", "MS", "BK", "USB", "AXP", "PYPL",
    "SQ", "HOOD", "COIN", "NVDT", "SCHW", "TROW", "IVZ", "ANTM", "UNH", "CI",
    "COG", "MRK", "JNJ", "PFE", "LLY", "ABBV", "CVS", "ABIBB", "APD", "D",
    "NEE", "DUK", "SO", "AWK", "XEL", "AEE", "AEP", "EXC", "PPL", "EVRG",
    "IDA", "WEC", "OKE", "MPC", "PSX", "VLO", "CTRA", "EOG", "COP", "PXD",
    "CVX", "XOM", "HAL", "RIG", "FANG", "BKR", "BDI", "DHI", "TOL", "PHM",
    "KBH", "LEN", "MDC", "NRZ", "AGR", "IRM", "CCI", "EQIX", "WELL", "PLD",
    "AMT", "ARE", "AVB", "EQR", "ESS", "FRT", "MAA", "NLY", "STAG", "VICI",
    "PLD", "PSA", "STOR", "SRC", "PTC", "PSTG", "DELL", "HPE", "CCS", "INCY",
    "SBAC", "SITE", "SNR", "COIN", "HON", "GE", "CMI", "CAT", "PCAR", "ITT"
]


def create_default_stream(num_stocks: int = 100) -> StockDataStream:
    """
    Create a default stock data stream with commonly traded stocks.
    
    Args:
        num_stocks: Number of stocks to include (max 100)
    
    Returns:
        Initialized StockDataStream object
    """
    tickers = STOCKS_TECH_100[:num_stocks]
    return StockDataStream(tickers)
