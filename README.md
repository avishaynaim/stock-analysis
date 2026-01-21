# Stock Analysis Pro

A professional-grade stock analysis system with ML-powered forecasting, technical analysis, and portfolio tracking.

## Features

- **Dashboard** - Real-time stock analysis with composite scoring
- **Technical Analysis** - RSI, MACD, Bollinger Bands, and more
- **ML Forecasting** - Per-ticker trained models using ensemble methods (RF + GB + XGBoost)
- **Pro Analysis** - Risk metrics, entry/exit signals, backtesting, factor analysis
- **S&P 500 Rankings** - Comprehensive market analysis
- **Stock Screener** - Filter stocks by custom criteria
- **Portfolio Tracker** - Track your holdings and performance

## Live Demo

Access the live app: [Your Railway URL]

## Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/stock-analysis.git
cd stock-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run src/stock_analysis/gui/app.py
```

## Deployment on Railway

1. Push to GitHub
2. Connect Railway to your GitHub repo
3. Railway will auto-detect the Procfile and deploy

## Tech Stack

- **Frontend**: Streamlit + Plotly
- **ML**: scikit-learn, XGBoost
- **Data**: yfinance, pandas
- **Deployment**: Railway

## License

MIT
