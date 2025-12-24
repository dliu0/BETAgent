"""Test file for BET Finance analysis pipeline."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Import from src.BETagent
from src.BETagent.gettickerhistory import TickerHistoryFetcher, fetch_tickers, fetch_and_export
from src.BETagent.run_analysis_pipeline import analyze_csv_file
from src import analyze_bet_outputs, make_gemini_invoker


def main():
    """Example usage of TickerHistoryFetcher for flexible ticker history data."""
    
    # Define tickers to fetch
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    
    print("=" * 70)
    print("TickerHistoryFetcher - Example Usage")
    print("=" * 70)
    
    # Example 1: Daily data for last year
    print("\n1. Fetching daily data for last year...")
    fetcher = TickerHistoryFetcher(tickers, interval="1d", period="1y")
    daily_data = fetcher.fetch()
    
    print(f"   Shape: {daily_data.shape}")
    print(f"   Date range: {daily_data.index[0].date()} to {daily_data.index[-1].date()}")
    print(f"\n   Last 5 rows:")
    print(daily_data.tail())
    
    # Export daily data to CSV
    fetcher.to_csv("ticker_history_daily.csv")
    print("\n   ✓ Exported to 'ticker_history_daily.csv'")
    
    # Automatically analyze the CSV with BET and show plots
    print("\nRunning BET analysis on the daily CSV (plot=True)...")
    try:
        result = analyze_csv_file(
            csv_path="ticker_history_daily.csv",
            ticker_x="AAPL",
            ticker_y="MSFT",
            plot=True,
            find_min=False,
            max_depth=8,
            p_value_threshold=0.05,
            save_result=True
        )
        print("\nBET result:\n", result)
    except Exception as e:
        print("Error running BET analysis:", e)
    
    # Example 2: Hourly data for last 5 days
    print("\n" + "-" * 70)
    print("2. Fetching hourly data for last 5 days...")
    fetcher_hourly = TickerHistoryFetcher(tickers, interval="1h", period="5d")
    hourly_data = fetcher_hourly.fetch()
    
    print(f"   Shape: {hourly_data.shape}")
    print(f"   Date range: {hourly_data.index[0]} to {hourly_data.index[-1]}")
    print(f"\n   First 5 rows:")
    print(hourly_data.head())
    
    # Export hourly data to CSV
    fetcher_hourly.to_csv("ticker_history_hourly.csv")
    print("\n   ✓ Exported to 'ticker_history_hourly.csv'")
    
    # Example 3: Quick fetch with convenience function
    print("\n" + "-" * 70)
    print("3. Quick fetch using convenience function (last 3 months)...")
    quick_data = fetch_tickers(tickers, interval="1d", period="3mo")
    
    print(f"   Shape: {quick_data.shape}")
    print(f"\n   Summary statistics:")
    print(quick_data.describe().round(2))
    
    # Example 4: Fetch and export in one call
    print("\n" + "-" * 70)
    print("4. Fetch and export in one call (last 6 months)...")
    data = fetch_and_export(
        tickers,
        "ticker_history_6months.csv",
        interval="1d",
        period="6mo"
    )
    print(f"   Shape: {data.shape}")
    print("   ✓ Exported to 'ticker_history_6months.csv'")
    
    # Example 5: Dynamic updates
    print("\n" + "-" * 70)
    print("5. Dynamic updates - changing period and interval...")
    
    # Update to 3 months
    data_3mo = fetcher.update_period("3mo")
    print(f"   Updated to 3 months: {data_3mo.shape}")
    
    # Update to hourly
    data_hourly_new = fetcher.update_interval("1h")
    print(f"   Updated to hourly: {data_hourly_new.shape}")
    
    # Refresh to get latest data
    data_latest = fetcher.refresh()
    print(f"   Refreshed to latest: {data_latest.shape}")
    
    # Example 6: Get summary information
    print("\n" + "-" * 70)
    print("6. Summary information about fetched data...")
    summary = fetcher.summary()
    for key, value in summary.items():
        if key != "tickers":
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {', '.join(value)}")
    
    print("\n" + "=" * 70)
    print("Examples complete! CSV files are ready for analysis.")
    print("=" * 70)

    # Example 7: Run AI analysis on BET outputs (uses stub if no LLM provided)
    print("\n" + "-" * 70)
    print("7. AI analysis on BET outputs (saved results folder)")
    try:
        # Try to create a Gemini invoker (requires `google-generativeai` and
        # `GOOGLE_API_KEY` environment variable). If unavailable, fall back to stub.
        try:
            invoker = make_gemini_invoker()
        except Exception as e:
            print("Gemini invoker unavailable, using local stub:", e)
            invoker = None

        prompt = '''Act as a Senior Principal Researcher in Complex Systems & Causal Discovery.
I have performed a Binary Expansion Test (BET) on two variables, and the algorithm has rejected the null hypothesis of independence, indicating a STRONG non-linear dependence.

Your goal is to interpret this mathematical finding and provide a causal explanation for why these two specific variables are coupled.

### 1. The Data & Statistical Evidence
* **Variable X:** {variable_x_name}
* **Variable Y:** {variable_y_name}
* **Linear Correlation (Pearson):** {pearson_score} (Note: If low, the relationship is hidden/non-linear).
* **BET P-Value:** {p_value} (Probability of independence. < 0.01 implies strong coupling).
* **Symmetry Count:** {min_count} (Low count implies a "Forbidden Zone" or structural barrier).
* **Interaction Pattern (Bit Depth):** {interaction_vector}
    * *Guidance:* Activity in early bits (Index 0-2) implies a "Macro/Coarse" relationship. Activity in later bits (Index 3+) implies a "Micro/Fine-grained" relationship (e.g., volatility clustering or high-frequency coupling).

### 2. Your Research Task
Using your internal knowledge base and available search tools, investigate the structural, physical, or institutional links between {variable_x_name} and {variable_y_name}.

**Generate 3 Distinct Scientific Hypotheses** for why this non-linear relationship exists.

**For each hypothesis, you must:**
1.  **Name the Mechanism:** Give it a scientific title (e.g., "Liquidity Spillover," "Common Mode Failure," "Regulatory Arbitrage").
2.  **Explain the Physics/Logic:** Specifically explain how the *Bit Interaction* supports this. (e.g., "The interaction at Bit 3 suggests the coupling is driven by volatility, not price trends").
3.  **Provide Evidence:** You **MUST** cite a real paper, news event, or economic report that confirms this link exists.

### 3. Output Format
Return your answer in the following structured format:

**Hypothesis 1: [Name]**
* **Mechanism:** [Explanation]
* **Why BET detected it:** [Connect to the bits/p-value]
* **Source:** [Citation/Link]

(Repeat for Hypothesis 2 and 3)
'''

        analyze_bet_outputs("bet_analysis_outputs", prompt=prompt, llm_invoke=invoker)
    except Exception as e:
        print("AI analysis skipped:", e)


if __name__ == "__main__":
    main()
