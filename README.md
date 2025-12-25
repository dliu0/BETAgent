# BET Agent: Binary Expansion Test with AI-Powered Causal Analysis

## Overview

**BET Agent** is an automated pipeline for discovering and interpreting non-linear dependencies between variables using the **Binary Expansion Test (BET)** algorithm combined with **AI-powered causal analysis**.

The workflow:
1. **Automated Data Collection** — Fetch financial data (stocks, cryptocurrencies, etc.) or load your own datasets
2. **BET Analysis** — Detect hidden non-linear coupling between two variables using the Binary Expansion Test
3. **AI Interpretation** — Send results to Google Gemini for qualitative causal explanation and hypothesis generation
4. **Saved Reports** — Get both quantitative (JSON/CSV) and qualitative (AI-generated) analysis in a single output folder
---

## Features

- **Automated ticker data fetching** (via yfinance) with configurable intervals and periods
- **BET algorithm** implementation for detecting non-linear dependencies
- **AI-powered causal research** — generates 3 scientific hypotheses with evidence citations
- **Multiple data source support** — CSV, Excel, Parquet, JSON, or direct DataFrames
- **Visualization** — automatic plots of relationships
- **Reproducible analysis** — all results timestamped and organized in output folders

---

## Requirements

### Python
- Python 3.10 or higher

### Dependencies
All dependencies are listed in `pyproject.toml`:

```
matplotlib>=3.10.8
pandas>=2.3.3
yfinance>=0.2.32
numpy>=1.24.0
google-genai>=0.1.0
openpyxl>=3.10.0        # For Excel support
pyarrow>=14.0.0         # For Parquet support
```

### API Keys
- **Google Gemini API Key** — Required for AI analysis
  - Get one at: https://ai.google.dev/
  - Set environment variable: `GOOGLE_API_KEY=your_key_here`

---

## Installation

1. **Clone the repository**
   ```bash
   git clone <repo_url>
   cd BETAgent
   ```

2. **Install dependencies**
   ```bash
   pip install -e .
   ```
   Or with UV (faster):
   ```bash
   uv sync
   ```

3. **Set your API key**
   
   PowerShell (persistent):
   ```powershell
   [Environment]::SetEnvironmentVariable("GOOGLE_API_KEY","your_api_key_here","User")
   ```
   
   Bash/Linux/Mac:
   ```bash
   export GOOGLE_API_KEY="your_api_key_here"
   ```

---

## Architecture

```
BETAgent/
├── src/
│   ├── __init__.py                  # Package exports
│   ├── BETagent/
│   │   ├── ai_agent.py              # AI invoker & analysis logic
│   │   ├── gettickerhistory.py      # Data fetching (yfinance wrapper)
│   │   ├── run_analysis_pipeline.py # BET algorithm & plotting
│   │   ├── bet.py                   # Core BET math
│   │   └── getdata.py               # Data utilities
│   └── BETfinance/
│       └── BETfinance.py            # Alternative entry point
├── tests/
│   ├── __init__.py
│   └── test_finance.py              # Test suite
└── pyproject.toml                   # Dependencies & metadata
```

---

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError`, ensure:
1. You're running from the project root directory
2. Dependencies are installed: `pip install -e .`
3. Python path includes project root

### Gemini API Errors
- **404 Not Found** — Model name may be outdated. Check available models at https://ai.google.dev/
- **Authentication Failed** — Verify `GOOGLE_API_KEY` is set: `echo $env:GOOGLE_API_KEY` (PowerShell)

### Data Fetching Issues
- yfinance may be temporarily unavailable. Try again in a few moments
- Ensure tickers are valid (e.g., "AAPL", not "apple")

---

## Project Status

**Draft Version** — Core functionality working mainly for finance. Future improvements:
- [ ] General use case for data outside of finance
- [ ] Improvement to LLM prompt and research

---

## License

MIT License — See LICENSE file for details

---

## Citation

If you use this project, please cite:

```bibtex
@software{betagent2025,
  title={BET Agent: Automated Non-Linear Dependency Discovery with AI Causal Analysis},
  author={Darren Liu},
  year={2025},
  url={https://github.com/dliu0/BETAgent}
}
```

---

## Acknowledgments

- **BET Algorithm** — Based on Binary Expansion Test methodology for detecting non-linear dependencies by Prof. [Kai Zhang](https://zhangk.web.unc.edu/).

    [Reference paper: Zhang, K. (2019). BET on Independence, Journal of the American Statistical Association, 114, 1620-1637](https://zhangk.web.unc.edu/wp-content/uploads/sites/14612/2019/04/2019-BET-on-Independence.pdf)
- **Google Gemini** — For AI-powered causal research and hypothesis generation
- **yfinance** — For financial data fetching
