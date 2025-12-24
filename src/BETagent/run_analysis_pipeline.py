"""
Analysis pipeline: Automatically fetch ticker data and run BET algorithm.
Orchestrates the flow from data retrieval → CSV creation → algorithm execution.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from gettickerhistory import TickerHistoryFetcher
from bet import DDMbet
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# IMPORTANT: Import your runBET function here
# Replace the placeholder import below with the actual location of runBET
# ============================================================================

def runBET(dataX, dataY, plot=False, find_min=False, max_depth=8, p_value_threshold=0.05):
    """
    Placeholder for your BET algorithm.
    REPLACE THIS with your actual runBET function by uncommenting the import above.
    """
    # Use the class from bet.py to run BET. This keeps a simple function shim
    # so older call sites that expect runBET(...) continue to work.
    ddm = DDMbet()
    return ddm.runBET(dataX, dataY, plot=plot, find_min=find_min, max_depth=max_depth, p_value_threshold=p_value_threshold)


class BETAnalysisPipeline:
    """
    Automates the pipeline: fetch ticker data → create CSV → run BET algorithm.
    """
    
    def __init__(self, 
                 tickers: list,
                 interval: str = "1d",
                 period: str = "1y",
                 output_dir: str = "bet_analysis_outputs"):
        """
        Initialize the analysis pipeline.
        
        Args:
            tickers: List of ticker symbols to fetch
            interval: Data interval ('1d', '1h', etc.)
            period: Time period ('1y', '3mo', etc.)
            output_dir: Directory to save CSVs and results
        """
        self.tickers = tickers
        self.interval = interval
        self.period = period
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.fetcher = None
        self.data = None
        self.csv_path = None
        self.results = {}
        
        logger.info(f"Initialized BETAnalysisPipeline for {len(tickers)} tickers")
    
    def fetch_and_save_data(self, filename: str = None) -> pd.DataFrame:
        """
        Fetch ticker data and save to CSV.
        
        Args:
            filename: Custom filename (defaults to ticker_history_<timestamp>.csv)
        
        Returns:
            DataFrame with fetched data
        """
        logger.info(f"Fetching {self.interval} data for period {self.period}...")
        
        self.fetcher = TickerHistoryFetcher(
            self.tickers,
            interval=self.interval,
            period=self.period
        )
        self.data = self.fetcher.fetch()
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            interval_label = self.interval.replace(" ", "")
            filename = f"ticker_history_{interval_label}_{timestamp}.csv"
        
        self.csv_path = self.output_dir / filename
        self.fetcher.to_csv(str(self.csv_path))
        
        logger.info(f"Data saved to {self.csv_path}")
        logger.info(f"Data shape: {self.data.shape}")
        
        return self.data
    
    def load_data_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load previously created CSV file.
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            DataFrame
        """
        self.data = pd.read_csv(csv_path, index_col=0)
        self.csv_path = csv_path
        logger.info(f"Loaded data from {csv_path}")
        logger.info(f"Data shape: {self.data.shape}")
        return self.data
    
    def extract_1d_arrays(self, ticker_x: str = None, ticker_y: str = None) -> tuple:
        """
        Extract 1D arrays from the CSV data for algorithm input.
        
        Args:
            ticker_x: Ticker symbol for dataX (defaults to first ticker)
            ticker_y: Ticker symbol for dataY (defaults to second ticker, or first if only one)
        
        Returns:
            Tuple of (dataX, dataY) as 1D numpy arrays
        """
        if self.data is None:
            raise ValueError("No data loaded. Call fetch_and_save_data() first.")
        
        available_tickers = list(self.data.columns)
        
        # Default assignments
        if ticker_x is None:
            ticker_x = available_tickers[0]
        if ticker_y is None:
            ticker_y = available_tickers[1] if len(available_tickers) > 1 else available_tickers[0]
        
        # Validate tickers exist
        if ticker_x not in available_tickers:
            raise ValueError(f"Ticker {ticker_x} not found in data. Available: {available_tickers}")
        if ticker_y not in available_tickers:
            raise ValueError(f"Ticker {ticker_y} not found in data. Available: {available_tickers}")
        
        # Extract as 1D arrays
        dataX = self.data[ticker_x].values.astype(float)
        dataY = self.data[ticker_y].values.astype(float)
        
        logger.info(f"Extracted dataX from {ticker_x}: shape {dataX.shape}")
        logger.info(f"Extracted dataY from {ticker_y}: shape {dataY.shape}")
        
        return dataX, dataY
    
    def run_algorithm(self,
                     ticker_x: str = None,
                     ticker_y: str = None,
                     plot: bool = False,
                     find_min: bool = False,
                     max_depth: int = 8,
                     p_value_threshold: float = 0.05,
                     save_result: bool = True,
                     result_filename: str = None) -> dict:
        """
        Extract data and run the BET algorithm.
        
        Args:
            ticker_x: Ticker for X data (defaults to first ticker)
            ticker_y: Ticker for Y data (defaults to second ticker)
            plot: Whether to plot results
            find_min: Whether to find minimum
            max_depth: Maximum depth parameter for algorithm
            p_value_threshold: P-value threshold for algorithm
        
        Returns:
            Algorithm results/output
        """
        logger.info("Extracting data arrays...")
        dataX, dataY = self.extract_1d_arrays(ticker_x, ticker_y)

        logger.info("Running BET algorithm...")
        # Use DDMbet implementation from bet.py
        ddm = DDMbet()
        # Prepare default result filename early so plots can be saved next to results
        tx = ticker_x or self.data.columns[0]
        ty = ticker_y or self.data.columns[1 if len(self.data.columns) > 1 else 0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_result_filename = result_filename or f"bet_result_{tx}_{ty}_{timestamp}"

        # Decide where to save the plot: if save_result is requested, save into output_dir
        plot_path = None
        if plot:
            if save_result:
                plot_path = str(self.output_dir / f"{default_result_filename}.png")
            elif self.csv_path:
                plot_dir = Path(self.csv_path).parent
                plot_name = f"bet_plot_{tx}_{ty}.png"
                plot_path = str(plot_dir / plot_name)

        result = ddm.runBET(
            dataX,
            dataY,
            plot=plot,
            find_min=find_min,
            max_depth=max_depth,
            p_value_threshold=p_value_threshold,
            plot_save_path=plot_path,
            show_plot=plot and (plot_path is None)
        )
        
        self.results = {
            "ticker_x": tx,
            "ticker_y": ty,
            "data_source": str(self.csv_path),
            "algorithm_params": {
                "plot": plot,
                "find_min": find_min,
                "max_depth": max_depth,
                "p_value_threshold": p_value_threshold
            },
            "algorithm_output": result,
            "timestamp": datetime.now().isoformat()
        }
        
        # Optionally save the algorithm output to files
        if save_result:
            # Use default_result_filename if result_filename wasn't provided
            result_filename = result_filename or default_result_filename

            # Save TXT with repr of full results
            txt_path = self.output_dir / f"{result_filename}.txt"
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(repr(self.results))
                logger.info(f"Saved BET result text to {txt_path}")
            except Exception as e:
                logger.exception(f"Failed to save TXT result: {e}")

            # Save JSON with structured output (convert numpy arrays)
            json_path = self.output_dir / f"{result_filename}.json"
            try:
                struct = {
                    "ticker_x": self.results["ticker_x"],
                    "ticker_y": self.results["ticker_y"],
                    "data_source": self.results["data_source"],
                    "algorithm_params": self.results["algorithm_params"],
                    "timestamp": self.results["timestamp"],
                    "p_value": float(result[0]) if hasattr(result, '__len__') else float(result),
                    "min_count": int(result[1]) if hasattr(result, '__len__') else None,
                    "min_cross": (result[2].tolist() if hasattr(result[2], 'tolist') else list(result[2])) if hasattr(result, '__len__') else None
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(struct, f, indent=2)
                logger.info(f"Saved BET result JSON to {json_path}")
            except Exception as e:
                logger.exception(f"Failed to save JSON result: {e}")

        logger.info("Algorithm execution completed")
        return result

    def run_pairwise_analysis(self,
                              max_depth: int = 2,
                              nprocess: int = 1,
                              save_csv: bool = True,
                              csv_filename: str = None,
                              save_plots: bool = False,
                              plot_dir: str = None) -> pd.DataFrame:
        """
        Run BET across all column pairs in the loaded data, save results and optional plots.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call fetch_and_save_data() or load_data_from_csv() first.")

        ddm = DDMbet()
        logger.info("Running pairwise BET across all columns...")
        results_df = ddm.test_variable_pairs(self.data, max_depth=max_depth, nprocess=nprocess)

        # Save CSV
        if save_csv:
            if csv_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_filename = f"bet_pairwise_results_{timestamp}.csv"
            out_path = self.output_dir / csv_filename
            results_df.to_csv(out_path, index=False)
            logger.info(f"Pairwise results saved to {out_path}")

        # Save plots for each pair (optional)
        if save_plots:
            plot_output_dir = Path(plot_dir) if plot_dir else (self.output_dir / "plots")
            plot_output_dir.mkdir(parents=True, exist_ok=True)
            for _, row in results_df.iterrows():
                varx = row['VarX']
                vary = row['VarY']
                # Extract arrays
                dataX = self.data[varx].values.astype(float)
                dataY = self.data[vary].values.astype(float)
                safe_name = f"{varx}__{vary}".replace('/', '_')
                plot_path = plot_output_dir / f"bet_{safe_name}.png"
                try:
                    ddm.runBET(dataX, dataY, plot=True, max_depth=max_depth, p_value_threshold=0.05, plot_save_path=str(plot_path), show_plot=False)
                except Exception as e:
                    logger.exception(f"Failed to create plot for {varx}-{vary}: {e}")

        return results_df
    
    def run_full_pipeline(self,
                         fetch_new: bool = True,
                         csv_path: str = None,
                         ticker_x: str = None,
                         ticker_y: str = None,
                         plot: bool = False,
                         find_min: bool = False,
                         max_depth: int = 8,
                         p_value_threshold: float = 0.05,
                         save_result: bool = True,
                         result_filename: str = None) -> dict:
        """
        Run the complete pipeline: fetch data → create CSV → run algorithm.
        
        Args:
            fetch_new: If True, fetch new data. If False, use existing CSV.
            csv_path: Path to existing CSV (required if fetch_new=False)
            ticker_x: Ticker for X data
            ticker_y: Ticker for Y data
            plot: Algorithm parameter
            find_min: Algorithm parameter
            max_depth: Algorithm parameter
            p_value_threshold: Algorithm parameter
        
        Returns:
            Algorithm results
        """
        logger.info("=" * 70)
        logger.info("Starting BET Analysis Pipeline")
        logger.info("=" * 70)
        
        # Step 1: Fetch or load data
        if fetch_new:
            logger.info(f"\nStep 1: Fetching ticker data...")
            self.fetch_and_save_data()
        else:
            if csv_path is None:
                raise ValueError("csv_path required when fetch_new=False")
            logger.info(f"\nStep 1: Loading data from {csv_path}...")
            self.load_data_from_csv(csv_path)
        
        # Step 2: Run algorithm
        logger.info(f"\nStep 2: Running BET algorithm...")
        result = self.run_algorithm(
            ticker_x=ticker_x,
            ticker_y=ticker_y,
            plot=plot,
            find_min=find_min,
            max_depth=max_depth,
            p_value_threshold=p_value_threshold,
            save_result=save_result,
            result_filename=result_filename
        )
        
        logger.info("\n" + "=" * 70)
        logger.info("Pipeline completed successfully")
        logger.info("=" * 70)
        
        return result
    
    def get_results_summary(self) -> dict:
        """Get summary of latest analysis results."""
        return self.results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def analyze_tickers_quick(tickers: list,
                         ticker_x: str = None,
                         ticker_y: str = None,
                         interval: str = "1d",
                         period: str = "1y",
                         plot: bool = False,
                         find_min: bool = False,
                         max_depth: int = 8,
                         p_value_threshold: float = 0.05) -> dict:
    """
    Quick function to run full pipeline in one call.
    
    Args:
        tickers: List of ticker symbols
        ticker_x: Ticker for X data (defaults to first)
        ticker_y: Ticker for Y data (defaults to second)
        interval: Data interval
        period: Time period
        plot: Algorithm parameter
        find_min: Algorithm parameter
        max_depth: Algorithm parameter
        p_value_threshold: Algorithm parameter
    
    Returns:
        Algorithm results
    """
    pipeline = BETAnalysisPipeline(tickers, interval, period)
    
    return pipeline.run_full_pipeline(
        fetch_new=True,
        ticker_x=ticker_x,
        ticker_y=ticker_y,
        plot=plot,
        find_min=find_min,
        max_depth=max_depth,
        p_value_threshold=p_value_threshold
    )


def analyze_csv_file(csv_path: str,
                    ticker_x: str = None,
                    ticker_y: str = None,
                    plot: bool = False,
                    find_min: bool = False,
                    max_depth: int = 8,
                    p_value_threshold: float = 0.05,
                    save_result: bool = True) -> dict:
    """
    Analyze an existing CSV file with the BET algorithm.
    
    Args:
        csv_path: Path to CSV file created by gettickerhistory
        ticker_x: Ticker for X data (defaults to first column)
        ticker_y: Ticker for Y data (defaults to second column)
        plot: Algorithm parameter
        find_min: Algorithm parameter
        max_depth: Algorithm parameter
        p_value_threshold: Algorithm parameter
    
    Returns:
        Algorithm results
    """
    pipeline = BETAnalysisPipeline([])
    pipeline.load_data_from_csv(csv_path)
    # Run single pair algorithm on loaded CSV and return results
    return pipeline.run_algorithm(
        ticker_x=ticker_x,
        ticker_y=ticker_y,
        plot=plot,
        find_min=find_min,
        max_depth=max_depth,
        p_value_threshold=p_value_threshold,
        save_result=save_result
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BET Analysis Pipeline - Setup Instructions")
    print("=" * 70)
    
    print("""
BEFORE RUNNING THIS SCRIPT:

1. Locate your runBET function and note its module path
   Example: src.algorithm, your_module, etc.

2. In this file (run_analysis_pipeline.py), uncomment and modify the import:
   
   from your_module import runBET
   
   Replace 'your_module' with the actual location of your runBET function.

3. Remove or comment out the placeholder function.

4. Then you can use the pipeline in three ways:

   OPTION A: Full pipeline - fetch new data and analyze
   ─────────────────────────────────────────────────
   from run_analysis_pipeline import BETAnalysisPipeline
   
   pipeline = BETAnalysisPipeline(
       tickers=["AAPL", "MSFT", "GOOGL"],
       interval="1d",
       period="1y"
   )
   
   results = pipeline.run_full_pipeline(
       ticker_x="AAPL",
       ticker_y="MSFT",
       plot=False,
       find_min=False,
       max_depth=8,
       p_value_threshold=0.05
   )
   
   
   OPTION B: Quick one-liner
   ──────────────────────────
   from run_analysis_pipeline import analyze_tickers_quick
   
   results = analyze_tickers_quick(
       tickers=["AAPL", "MSFT", "GOOGL"],
       ticker_x="AAPL",
       ticker_y="MSFT",
       period="3mo"
   )
   
   
   OPTION C: Analyze existing CSV
   ──────────────────────────────
   from run_analysis_pipeline import analyze_csv_file
   
   results = analyze_csv_file(
       csv_path="ticker_history_daily.csv",
       ticker_x="AAPL",
       ticker_y="MSFT"
   )

""")
    
    print("=" * 70)
    print("Update the import at the top of this file to get started!")
    print("=" * 70)
