"""
Stock Analysis GUI - Main Application

A comprehensive stock analysis dashboard with:
- Interactive price charts
- Technical indicators visualization
- Scoring and probability analysis
- ML-powered forecasting
- Stock screening
- Portfolio comparison
"""

# Fix Python path for Railway deployment
import sys
import os
from pathlib import Path

# Add src directory to path
app_dir = Path(__file__).resolve().parent.parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional CSS styling
st.markdown("""
<style>
    /* Dark theme variables */
    :root {
        --bg-primary: #0e1117;
        --bg-secondary: #1a1f2e;
        --bg-tertiary: #262d40;
        --accent-primary: #00d4ff;
        --accent-secondary: #7c3aed;
        --accent-success: #10b981;
        --accent-warning: #f59e0b;
        --accent-danger: #ef4444;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border-color: #334155;
    }

    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }

    .sub-header {
        font-size: 1rem;
        color: var(--text-secondary);
        margin-top: 0;
    }

    /* Pro metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%);
        border: 1px solid var(--border-color);
        padding: 1.25rem;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }

    /* Score colors */
    .score-high { color: var(--accent-success); font-weight: 700; }
    .score-medium { color: var(--accent-warning); font-weight: 700; }
    .score-low { color: var(--accent-danger); font-weight: 700; }

    /* Enhanced metrics */
    .stMetric > div {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(71, 85, 105, 0.5);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }

    .stMetric > div:hover {
        border-color: var(--accent-primary);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.15);
    }

    /* Signal banner styles */
    .signal-banner {
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 24px;
        border-left: 4px solid;
        backdrop-filter: blur(10px);
    }

    .signal-strong-buy {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%);
        border-left-color: var(--accent-success);
    }

    .signal-buy {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.12) 0%, rgba(34, 197, 94, 0.04) 100%);
        border-left-color: #22c55e;
    }

    .signal-neutral {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(245, 158, 11, 0.04) 100%);
        border-left-color: var(--accent-warning);
    }

    .signal-avoid {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(239, 68, 68, 0.04) 100%);
        border-left-color: var(--accent-danger);
    }

    /* Pro badge */
    .pro-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    .badge-cached {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
    }

    .badge-trained {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }

    .badge-retrained {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }

    /* Card containers */
    .pro-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(71, 85, 105, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(71, 85, 105, 0.3);
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(124, 58, 237, 0.2) 100%);
        border-color: var(--accent-primary);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%);
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.35);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(71, 85, 105, 0.3);
    }

    [data-testid="stSidebar"] .stMarkdown h1 {
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Data table styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 10px;
        border: 1px solid rgba(71, 85, 105, 0.3);
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d4ff 0%, #7c3aed 100%);
    }

    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #7c3aed 0%, #00d4ff 100%);
    }

    /* Chart container */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(71, 85, 105, 0.5) 50%, transparent 100%);
        margin: 32px 0;
    }

    /* Info/Warning boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
    }

    /* Spinner */
    .stSpinner > div {
        border-color: var(--accent-primary) transparent transparent transparent;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_stock_data(symbol: str):
    """Load stock data with caching."""
    from stock_analysis.data.provider import DataProvider

    provider = DataProvider()
    try:
        price_data = provider.get_prices(symbol)
        return price_data.data
    except Exception as e:
        st.error(f"Failed to load data for {symbol}: {e}")
        return None


@st.cache_data(ttl=3600)
def compute_analysis(symbol: str, prices_json: str, benchmark_json: str | None):
    """Compute stock analysis with caching."""
    from stock_analysis.scoring.scorer import StockScorer

    prices = pd.read_json(prices_json)
    benchmark = pd.read_json(benchmark_json) if benchmark_json else None

    scorer = StockScorer()
    return scorer.analyze(symbol, prices, benchmark)


@st.cache_data(ttl=3600)
def compute_indicators(prices_json: str):
    """Compute all indicators with caching."""
    from stock_analysis.indicators.engine import IndicatorEngine

    prices = pd.read_json(prices_json)
    engine = IndicatorEngine()
    return engine.compute_all(prices)


def create_price_chart(prices: pd.DataFrame, symbol: str, indicators: dict) -> go.Figure:
    """Create interactive price chart with indicators."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(f"{symbol} Price", "RSI", "MACD", "Volume"),
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=prices.index,
            open=prices["open"],
            high=prices["high"],
            low=prices["low"],
            close=prices["adj_close"],
            name="Price",
            increasing_line_color="#00c853",
            decreasing_line_color="#ff5252",
        ),
        row=1, col=1,
    )

    # Add EMAs
    for period, color in [(8, "#ff9800"), (21, "#2196f3"), (50, "#9c27b0")]:
        ema = prices["adj_close"].ewm(span=period, adjust=False).mean()
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=ema,
                name=f"EMA {period}",
                line=dict(color=color, width=1),
            ),
            row=1, col=1,
        )

    # Bollinger Bands
    sma_20 = prices["adj_close"].rolling(20).mean()
    std_20 = prices["adj_close"].rolling(20).std()
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20

    fig.add_trace(
        go.Scatter(
            x=prices.index, y=upper_band,
            name="BB Upper",
            line=dict(color="rgba(128, 128, 128, 0.3)", dash="dash"),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=prices.index, y=lower_band,
            name="BB Lower",
            line=dict(color="rgba(128, 128, 128, 0.3)", dash="dash"),
            fill="tonexty",
            fillcolor="rgba(128, 128, 128, 0.1)",
        ),
        row=1, col=1,
    )

    # RSI
    delta = prices["adj_close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    fig.add_trace(
        go.Scatter(x=prices.index, y=rsi, name="RSI", line=dict(color="#1f77b4")),
        row=2, col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    ema12 = prices["adj_close"].ewm(span=12, adjust=False).mean()
    ema26 = prices["adj_close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    colors = ["#00c853" if val >= 0 else "#ff5252" for val in macd_hist]
    fig.add_trace(
        go.Bar(x=prices.index, y=macd_hist, name="MACD Hist", marker_color=colors),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=prices.index, y=macd_line, name="MACD", line=dict(color="#1f77b4")),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=prices.index, y=signal_line, name="Signal", line=dict(color="#ff7043")),
        row=3, col=1,
    )

    # Volume
    colors = ["#00c853" if prices["adj_close"].iloc[i] >= prices["open"].iloc[i] else "#ff5252"
              for i in range(len(prices))]
    fig.add_trace(
        go.Bar(x=prices.index, y=prices["volume"], name="Volume", marker_color=colors),
        row=4, col=1,
    )

    # Layout
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)

    return fig


def create_score_gauge(score: float, title: str) -> go.Figure:
    """Create a gauge chart for scores."""
    if score >= 70:
        color = "#00c853"
    elif score >= 50:
        color = "#ffc107"
    else:
        color = "#ff5252"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title, "font": {"size": 16}},
        number={"font": {"size": 32}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, 30], "color": "rgba(255, 82, 82, 0.3)"},
                {"range": [30, 50], "color": "rgba(255, 193, 7, 0.3)"},
                {"range": [50, 70], "color": "rgba(255, 235, 59, 0.3)"},
                {"range": [70, 100], "color": "rgba(0, 200, 83, 0.3)"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))

    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_probability_chart(prob_up: float) -> go.Figure:
    """Create probability visualization."""
    fig = go.Figure()

    # Probability bar
    fig.add_trace(go.Bar(
        x=["Bearish", "Bullish"],
        y=[1 - prob_up, prob_up],
        marker_color=["#ff5252", "#00c853"],
        text=[f"{(1-prob_up)*100:.1f}%", f"{prob_up*100:.1f}%"],
        textposition="auto",
    ))

    fig.update_layout(
        title="Probability Estimate",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        height=300,
        showlegend=False,
    )

    return fig


def create_forecast_chart(prices: pd.DataFrame, forecast_days: int = 30) -> go.Figure:
    """Create price forecast visualization."""
    # Simple forecast using trend + volatility
    recent_prices = prices["adj_close"].iloc[-60:]
    returns = recent_prices.pct_change().dropna()

    mean_return = returns.mean()
    std_return = returns.std()

    last_price = prices["adj_close"].iloc[-1]
    last_date = prices.index[-1]

    # Generate forecast dates
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq="B")

    # Monte Carlo simulation
    np.random.seed(42)
    n_simulations = 100
    simulations = np.zeros((n_simulations, forecast_days))

    for i in range(n_simulations):
        sim_returns = np.random.normal(mean_return, std_return, forecast_days)
        sim_prices = last_price * np.cumprod(1 + sim_returns)
        simulations[i] = sim_prices

    # Calculate percentiles
    p10 = np.percentile(simulations, 10, axis=0)
    p25 = np.percentile(simulations, 25, axis=0)
    p50 = np.percentile(simulations, 50, axis=0)
    p75 = np.percentile(simulations, 75, axis=0)
    p90 = np.percentile(simulations, 90, axis=0)

    fig = go.Figure()

    # Historical prices
    fig.add_trace(go.Scatter(
        x=prices.index[-90:],
        y=prices["adj_close"].iloc[-90:],
        name="Historical",
        line=dict(color="#1f77b4", width=2),
    ))

    # Forecast bands
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=p90, name="90th %ile",
        line=dict(color="rgba(0, 200, 83, 0.2)"),
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=p10, name="10th %ile",
        line=dict(color="rgba(0, 200, 83, 0.2)"),
        fill="tonexty", fillcolor="rgba(0, 200, 83, 0.1)",
    ))

    fig.add_trace(go.Scatter(
        x=forecast_dates, y=p75, name="75th %ile",
        line=dict(color="rgba(0, 200, 83, 0.3)"),
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=p25, name="25th %ile",
        line=dict(color="rgba(0, 200, 83, 0.3)"),
        fill="tonexty", fillcolor="rgba(0, 200, 83, 0.2)",
    ))

    # Median forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=p50, name="Forecast (Median)",
        line=dict(color="#00c853", width=2, dash="dash"),
    ))

    fig.update_layout(
        title=f"Price Forecast ({forecast_days} Business Days)",
        xaxis_title="Date",
        yaxis_title="Price",
        height=400,
        showlegend=True,
    )

    return fig


def create_indicator_heatmap(indicators: dict) -> go.Figure:
    """Create indicator signal heatmap."""
    # Group indicators by signal
    bullish = []
    bearish = []
    neutral = []

    # RSI
    rsi = indicators.get("rsi", 50)
    if rsi < 30:
        bullish.append(("RSI", rsi, "Oversold"))
    elif rsi > 70:
        bearish.append(("RSI", rsi, "Overbought"))
    else:
        neutral.append(("RSI", rsi, "Neutral"))

    # MACD
    macd_hist = indicators.get("macd_histogram", 0)
    if macd_hist > 0:
        bullish.append(("MACD", macd_hist, "Bullish"))
    else:
        bearish.append(("MACD", macd_hist, "Bearish"))

    # Stochastic
    stoch_k = indicators.get("stoch_k", 50)
    if stoch_k < 20:
        bullish.append(("Stochastic", stoch_k, "Oversold"))
    elif stoch_k > 80:
        bearish.append(("Stochastic", stoch_k, "Overbought"))
    else:
        neutral.append(("Stochastic", stoch_k, "Neutral"))

    # ADX
    adx = indicators.get("adx", 20)
    trend_dir = indicators.get("trend_direction", 0)
    if adx > 25 and trend_dir > 0:
        bullish.append(("ADX Trend", adx, "Strong Up"))
    elif adx > 25 and trend_dir < 0:
        bearish.append(("ADX Trend", adx, "Strong Down"))
    else:
        neutral.append(("ADX Trend", adx, "Weak/No Trend"))

    # Create summary
    total = len(bullish) + len(bearish) + len(neutral)

    fig = go.Figure(go.Pie(
        values=[len(bullish), len(neutral), len(bearish)],
        labels=["Bullish", "Neutral", "Bearish"],
        marker_colors=["#00c853", "#ffc107", "#ff5252"],
        hole=0.4,
    ))

    fig.update_layout(
        title="Indicator Signals Summary",
        height=300,
        annotations=[dict(text=f"{len(bullish)}/{total}", x=0.5, y=0.5, font_size=20, showarrow=False)],
    )

    return fig


def main():
    """Main application."""
    # Sidebar
    st.sidebar.markdown("# 📈 Stock Analysis Pro")
    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Technical Analysis", "Forecasting", "Pro Analysis", "S&P 500 Rankings", "Stock Screener", "Compare Stocks", "Portfolio Tracker"],
        index=0,
    )

    st.sidebar.markdown("---")

    # Stock input
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()
    benchmark = st.sidebar.text_input("Benchmark", value="SPY").upper()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Actions")

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Load data
    with st.spinner(f"Loading data for {symbol}..."):
        prices = load_stock_data(symbol)
        benchmark_prices = load_stock_data(benchmark) if benchmark else None

    if prices is None:
        st.error(f"Could not load data for {symbol}. Please check the symbol and try again.")
        return

    # Compute analysis
    with st.spinner("Computing analysis..."):
        try:
            prices_json = prices.to_json()
            benchmark_json = benchmark_prices.to_json() if benchmark_prices is not None else None
            analysis = compute_analysis(symbol, prices_json, benchmark_json)
            indicators = compute_indicators(prices_json)
        except Exception as e:
            st.error(f"Analysis error: {e}")
            analysis = None
            indicators = {}

    # Page routing
    if page == "Dashboard":
        render_dashboard(symbol, prices, analysis, indicators)
    elif page == "Technical Analysis":
        render_technical_analysis(symbol, prices, indicators)
    elif page == "Forecasting":
        render_forecasting(symbol, prices, indicators)
    elif page == "Pro Analysis":
        render_pro_analysis(symbol, prices, indicators, analysis)
    elif page == "S&P 500 Rankings":
        render_sp500_rankings()
    elif page == "Stock Screener":
        render_screener()
    elif page == "Compare Stocks":
        render_comparison(benchmark_prices)
    elif page == "Portfolio Tracker":
        render_portfolio_tracker()


def render_dashboard(symbol: str, prices: pd.DataFrame, analysis, indicators: dict):
    """Render main dashboard."""
    st.markdown(f"<h1 class='main-header'>📊 {symbol} Dashboard</h1>", unsafe_allow_html=True)

    if analysis is None:
        st.warning("Analysis not available")
        return

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    current_price = prices["adj_close"].iloc[-1]
    prev_price = prices["adj_close"].iloc[-2]
    price_change = (current_price - prev_price) / prev_price * 100

    with col1:
        st.metric("Price", f"${current_price:.2f}", f"{price_change:+.2f}%")

    with col2:
        st.metric("Composite Score", f"{analysis.composite_score.value:.1f}", analysis.composite_score.components.get("rating", "N/A"))

    with col3:
        st.metric("Technical", f"{analysis.technical_score.value:.1f}")

    with col4:
        st.metric("Momentum", f"{analysis.momentum_score.value:.1f}")

    with col5:
        prob_up = analysis.probability.get("prob_up", 0.5)
        st.metric("P(Up)", f"{prob_up*100:.1f}%", analysis.probability.get("signal", "neutral"))

    st.markdown("---")

    # Charts row
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Price Chart")
        fig = create_price_chart(prices.iloc[-252:], symbol, indicators)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Score Overview")

        # Gauge charts
        fig1 = create_score_gauge(analysis.composite_score.value, "Composite")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = create_probability_chart(analysis.probability.get("prob_up", 0.5))
        st.plotly_chart(fig2, use_container_width=True)

    # Analysis summary
    st.markdown("---")
    st.subheader("Analysis Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Technical Analysis")
        st.write(f"**Score:** {analysis.technical_score.value:.1f}/100")
        st.write(f"**Signal:** {analysis.technical_score.interpretation}")
        st.write(f"**RSI:** {indicators.get('rsi', 'N/A'):.1f}" if isinstance(indicators.get('rsi'), float) else "RSI: N/A")
        st.write(f"**MACD:** {'Bullish' if indicators.get('macd_histogram', 0) > 0 else 'Bearish'}")

    with col2:
        st.markdown("#### Momentum Analysis")
        st.write(f"**Score:** {analysis.momentum_score.value:.1f}/100")
        st.write(f"**Signal:** {analysis.momentum_score.interpretation}")

        # Returns
        returns = {}
        for period in [5, 21, 63]:
            if len(prices) > period:
                ret = (prices["adj_close"].iloc[-1] / prices["adj_close"].iloc[-period-1] - 1) * 100
                returns[period] = ret
        st.write(f"**5D Return:** {returns.get(5, 0):+.2f}%")
        st.write(f"**1M Return:** {returns.get(21, 0):+.2f}%")
        st.write(f"**3M Return:** {returns.get(63, 0):+.2f}%")

    with col3:
        st.markdown("#### Risk Analysis")
        st.write(f"**Score:** {analysis.risk_score.value:.1f}/100")
        st.write(f"**Signal:** {analysis.risk_score.interpretation}")
        st.write(f"**Volatility:** {indicators.get('annualized_volatility', 0):.1f}%" if isinstance(indicators.get('annualized_volatility'), float) else "Volatility: N/A")
        st.write(f"**Sharpe:** {indicators.get('sharpe_ratio', 0):.2f}" if isinstance(indicators.get('sharpe_ratio'), float) else "Sharpe: N/A")


def render_technical_analysis(symbol: str, prices: pd.DataFrame, indicators: dict):
    """Render technical analysis page."""
    st.markdown(f"<h1 class='main-header'>🔧 Technical Analysis: {symbol}</h1>", unsafe_allow_html=True)

    # Indicator selection
    st.sidebar.markdown("### Indicator Settings")
    show_ema = st.sidebar.checkbox("Show EMAs", value=True)
    show_bb = st.sidebar.checkbox("Show Bollinger Bands", value=True)
    show_volume = st.sidebar.checkbox("Show Volume", value=True)

    # Main chart
    st.subheader("Interactive Chart")
    fig = create_price_chart(prices.iloc[-252:], symbol, indicators)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Indicators table
    st.subheader("All Indicators")

    # Group indicators
    groups = {
        "Trend": ["ema_8", "ema_21", "ema_50", "macd_line", "macd_signal", "macd_histogram", "adx", "plus_di", "minus_di"],
        "Momentum": ["rsi", "stoch_k", "stoch_d", "roc_10", "williams_r", "cci"],
        "Volatility": ["atr", "atr_pct", "bollinger_bandwidth", "historical_volatility", "annualized_volatility"],
        "Volume": ["obv", "volume_sma_ratio", "mfi"],
        "Structure": ["pivot_point", "resistance_1", "support_1"],
    }

    tabs = st.tabs(list(groups.keys()) + ["All Indicators"])

    for i, (group_name, group_indicators) in enumerate(groups.items()):
        with tabs[i]:
            data = []
            for ind_name in group_indicators:
                if ind_name in indicators:
                    val = indicators[ind_name]
                    if isinstance(val, float):
                        data.append({"Indicator": ind_name, "Value": f"{val:.4f}"})
                    else:
                        data.append({"Indicator": ind_name, "Value": str(val)})

            if data:
                st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
            else:
                st.info(f"No {group_name.lower()} indicators computed")

    # All indicators tab
    with tabs[-1]:
        data = []
        for name, value in sorted(indicators.items()):
            if isinstance(value, float):
                data.append({"Indicator": name, "Value": f"{value:.4f}"})
            elif isinstance(value, (int, str)):
                data.append({"Indicator": name, "Value": str(value)})

        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True, height=500)

    # Signal summary
    st.markdown("---")
    st.subheader("Signal Summary")

    col1, col2 = st.columns(2)

    with col1:
        fig = create_indicator_heatmap(indicators)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Key Signals")

        # RSI Signal
        rsi = indicators.get("rsi", 50)
        rsi_signal = "🟢 Oversold (Buy)" if rsi < 30 else "🔴 Overbought (Sell)" if rsi > 70 else "🟡 Neutral"
        st.write(f"**RSI ({rsi:.1f}):** {rsi_signal}")

        # MACD Signal
        macd_hist = indicators.get("macd_histogram", 0)
        macd_signal = "🟢 Bullish" if macd_hist > 0 else "🔴 Bearish"
        st.write(f"**MACD:** {macd_signal}")

        # Trend Signal
        adx = indicators.get("adx", 20)
        trend_strength = "Strong" if adx > 25 else "Weak"
        st.write(f"**Trend Strength (ADX {adx:.1f}):** {trend_strength}")

        # Volume Signal
        vol_ratio = indicators.get("volume_sma_ratio", 1)
        vol_signal = "🟢 High Volume" if vol_ratio > 1.5 else "🔴 Low Volume" if vol_ratio < 0.5 else "🟡 Normal"
        st.write(f"**Volume:** {vol_signal}")


def render_forecasting(symbol: str, prices: pd.DataFrame, indicators: dict):
    """Render ML-powered forecasting page."""
    st.markdown(f"<h1 class='main-header'>🤖 ML-Powered Forecasting: {symbol}</h1>", unsafe_allow_html=True)

    st.markdown("""
    **Per-Ticker Machine Learning** - Each stock gets its own trained model based on its unique
    historical patterns. Uses Random Forest + Gradient Boosting ensemble trained specifically
    on this ticker's indicator patterns that led to significant gains.
    """)

    # ML Settings
    st.sidebar.markdown("### ML Settings")
    forward_days = st.sidebar.slider("Forecast Horizon (days)", 5, 60, 20)
    gain_threshold = st.sidebar.slider("Big Gain Threshold (%)", 5, 30, 15) / 100
    force_retrain = st.sidebar.checkbox("Force Model Retrain", value=False,
                                        help="Retrain model even if cached version exists")

    # Run ML forecast
    with st.spinner(f"Loading/Training ML model for {symbol}..."):
        try:
            from stock_analysis.ml.engine import MLForecastEngine
            from stock_analysis.ml.model_storage import get_model_storage
            from stock_analysis.indicators.engine import IndicatorEngine

            # Compute full indicator DataFrame
            engine = IndicatorEngine()
            indicator_df = engine.compute_all(prices, return_dataframe=True)

            # Create ML engine with per-ticker caching
            ml_engine = MLForecastEngine(
                forward_days=forward_days,
                big_gain_threshold=gain_threshold,
                use_model_cache=True,
            )

            # Get forecast (will use cached model if available)
            forecast = ml_engine.forecast(prices, indicator_df, ticker=symbol, force_retrain=force_retrain)

            # Get backtest results
            backtest = ml_engine.get_backtest_results(prices, indicator_df)
            performance = ml_engine.get_performance_summary(backtest)

            # Get model storage info
            model_storage = get_model_storage()
            model_info = model_storage.get_model_info(symbol)

        except Exception as e:
            st.error(f"ML Forecasting error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    # Main Signal Display
    st.markdown("---")

    # Signal banner
    signal_colors = {
        "Strong Buy": "#00c853",
        "Buy": "#4caf50",
        "Neutral": "#ffc107",
        "Avoid": "#ff5252",
    }
    signal_color = signal_colors.get(forecast.signal_strength, "#666")

    # Model source badge
    model_source_badges = {
        "cached": ("💾 Cached Model", "#2196f3"),
        "trained": ("🆕 Newly Trained", "#4caf50"),
        "retrained": ("🔄 Retrained", "#ff9800"),
    }
    model_badge, badge_color = model_source_badges.get(forecast.model_source, ("Model", "#666"))

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {signal_color}22, {signal_color}44);
                padding: 20px; border-radius: 15px; border-left: 5px solid {signal_color}; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h2 style="margin: 0; color: {signal_color};">Signal: {forecast.signal_strength}</h2>
            <span style="background: {badge_color}; color: white; padding: 5px 12px; border-radius: 15px; font-size: 0.85em;">
                {model_badge}
            </span>
        </div>
        <p style="margin: 5px 0; font-size: 1.1em;">Confidence: {forecast.confidence_level}</p>
        <p style="margin: 5px 0;">Probability of {gain_threshold*100:.0f}%+ gain: <b>{forecast.prediction.probability*100:.1f}%</b></p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "ML Prediction",
            f"{forecast.prediction.probability*100:.1f}%",
            "Big Gain" if forecast.prediction.prediction == 1 else "No Signal"
        )

    with col2:
        st.metric(
            "Historical Pattern Rate",
            f"{forecast.similar_patterns['big_gain_rate']*100:.0f}%",
            f"Avg: {forecast.similar_patterns['avg_return']*100:+.1f}%"
        )

    with col3:
        st.metric(
            "Model Confidence",
            f"{forecast.prediction.confidence*100:.0f}%",
            forecast.confidence_level
        )

    with col4:
        st.metric(
            "Model ROC-AUC",
            f"{forecast.model_metrics.roc_auc:.3f}",
            f"CV: {forecast.model_metrics.cv_mean:.3f}"
        )

    st.markdown("---")

    # Key Insights
    st.subheader("Key Insights")
    for insight in forecast.key_insights:
        st.info(insight)

    st.markdown("---")

    # Two column layout
    col1, col2 = st.columns(2)

    with col1:
        # Feature Importance Chart
        st.subheader("Top Predictive Indicators")

        top_features = forecast.feature_importance.head(15)
        fig = go.Figure(go.Bar(
            x=top_features["importance"],
            y=top_features["feature"],
            orientation="h",
            marker_color="#1f77b4",
        ))
        fig.update_layout(
            height=400,
            yaxis=dict(autorange="reversed"),
            xaxis_title="Importance",
            margin=dict(l=200),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Similar Historical Patterns
        st.subheader("Similar Historical Patterns")

        similar_data = []
        for i, (date, ret) in enumerate(zip(
            forecast.similar_patterns["similar_dates"][:10],
            forecast.similar_patterns["similar_returns"][:10]
        )):
            similar_data.append({
                "Date": str(date)[:10],
                "Return": f"{ret*100:+.1f}%",
                "Outcome": "Big Gain" if ret >= gain_threshold else "Gain" if ret > 0 else "Loss"
            })

        st.dataframe(pd.DataFrame(similar_data), use_container_width=True, hide_index=True)

        # Pattern stats
        st.markdown(f"""
        **Pattern Statistics:**
        - Win Rate: {forecast.similar_patterns['gain_rate']*100:.0f}%
        - Big Gain Rate: {forecast.similar_patterns['big_gain_rate']*100:.0f}%
        - Best Outcome: {forecast.similar_patterns['best_return']*100:+.1f}%
        - Worst Outcome: {forecast.similar_patterns['worst_return']*100:+.1f}%
        """)

    st.markdown("---")

    # Backtest Results
    st.subheader("Model Backtest Performance")

    if "error" not in performance:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Signals", performance["total_signals"])

        with col2:
            st.metric("Win Rate", f"{performance['win_rate']*100:.1f}%")

        with col3:
            st.metric("Big Gain Hit Rate", f"{performance['big_gain_hit_rate']*100:.1f}%")

        with col4:
            st.metric("Avg Return/Signal", f"{performance['avg_return']*100:+.1f}%")

        # Backtest chart
        st.subheader("Strategy vs Buy & Hold")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=backtest.index,
            y=backtest["cumulative_return"] * 100,
            name="Buy & Hold",
            line=dict(color="#1f77b4", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=backtest.index,
            y=backtest["strategy_cumulative"] * 100,
            name="ML Strategy",
            line=dict(color="#00c853", width=2),
        ))

        # Mark signals
        signals = backtest[backtest["signal"] == 1]
        fig.add_trace(go.Scatter(
            x=signals.index,
            y=signals["strategy_cumulative"] * 100,
            mode="markers",
            name="Buy Signals",
            marker=dict(color="#ff9800", size=8, symbol="triangle-up"),
        ))

        fig.update_layout(
            height=400,
            yaxis_title="Cumulative Return (%)",
            xaxis_title="Date",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Performance comparison
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Strategy Total Return",
                f"{performance['total_return']*100:+.1f}%",
            )
        with col2:
            st.metric(
                "Buy & Hold Return",
                f"{performance['buy_hold_return']*100:+.1f}%",
            )

    st.markdown("---")

    # Model Details
    with st.expander("Model Details"):
        # Per-ticker model info
        st.markdown(f"### Per-Ticker Model for {symbol}")

        if model_info:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **Model Info:**
                - Trained At: {model_info.trained_at[:19]}
                - Data Range: {model_info.data_start} to {model_info.data_end}
                - Training Samples: {model_info.n_samples:,}
                - Features: {model_info.n_features}
                """)
            with col2:
                st.markdown(f"""
                **Model Settings:**
                - Forward Days: {model_info.forward_days}
                - Gain Threshold: {model_info.gain_threshold*100:.0f}%
                - Big Gain Threshold: {model_info.big_gain_threshold*100:.0f}%
                - Model Source: {forecast.model_source}
                """)
        else:
            st.info("Model was trained fresh for this session")

        st.markdown("---")

        st.markdown(f"""
        **Model Architecture:**
        - Ensemble of Random Forest (200 trees) + Gradient Boosting (100 trees)
        - Trained specifically on **{symbol}**'s historical patterns
        - Features include:
            - Raw indicator values
            - Rate of change patterns
            - Z-scores and momentum
            - Cross-indicator interactions

        **Training Metrics:**
        - Accuracy: {forecast.model_metrics.accuracy*100:.1f}%
        - Precision: {forecast.model_metrics.precision*100:.1f}%
        - Recall: {forecast.model_metrics.recall*100:.1f}%
        - F1 Score: {forecast.model_metrics.f1*100:.1f}%
        - ROC-AUC: {forecast.model_metrics.roc_auc:.3f}
        - Cross-Validation Mean: {forecast.model_metrics.cv_mean:.3f} (+/- {forecast.model_metrics.cv_std:.3f})
        """)

        # Confusion matrix
        if forecast.model_metrics.confusion_matrix is not None:
            st.markdown("**Confusion Matrix:**")
            cm = forecast.model_metrics.confusion_matrix
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=["No Big Gain", "Big Gain"],
                y=["No Big Gain", "Big Gain"],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
            ))
            fig.update_layout(
                height=300,
                xaxis_title="Predicted",
                yaxis_title="Actual",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Monte Carlo still available
    st.markdown("---")
    st.subheader("Monte Carlo Price Simulation")

    forecast_days_mc = st.slider("Monte Carlo Forecast Days", 5, 90, 30)
    fig = create_forecast_chart(prices, forecast_days_mc)
    st.plotly_chart(fig, use_container_width=True)


def render_screener():
    """Render stock screener page."""
    st.markdown("<h1 class='main-header'>🔍 Stock Screener</h1>", unsafe_allow_html=True)

    st.markdown("Screen stocks based on technical and fundamental criteria.")

    # Screening criteria
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Score Filters")
        min_composite = st.slider("Min Composite Score", 0, 100, 60)
        min_technical = st.slider("Min Technical Score", 0, 100, 50)

    with col2:
        st.markdown("#### Momentum Filters")
        min_momentum = st.slider("Min Momentum Score", 0, 100, 50)
        min_probability = st.slider("Min P(Up) %", 0, 100, 55)

    with col3:
        st.markdown("#### Risk Filters")
        min_risk_score = st.slider("Min Risk Score", 0, 100, 40)

    # Stock universe
    st.markdown("---")
    st.subheader("Stock Universe")

    universe_input = st.text_area(
        "Enter symbols (comma-separated)",
        value="AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, V, JNJ",
    )

    if st.button("Run Screener", type="primary"):
        symbols = [s.strip().upper() for s in universe_input.split(",")]

        with st.spinner(f"Screening {len(symbols)} stocks..."):
            from stock_analysis.data.provider import DataProvider
            from stock_analysis.scoring.scorer import StockScorer

            provider = DataProvider()
            scorer = StockScorer()

            results = []
            progress = st.progress(0)

            for i, sym in enumerate(symbols):
                try:
                    price_data = provider.get_prices(sym)
                    benchmark_data = provider.get_prices("SPY")

                    analysis = scorer.analyze(sym, price_data.data, benchmark_data.data)

                    # Apply filters
                    if (analysis.composite_score.value >= min_composite and
                        analysis.technical_score.value >= min_technical and
                        analysis.momentum_score.value >= min_momentum and
                        analysis.risk_score.value >= min_risk_score and
                        analysis.probability.get("prob_up", 0) * 100 >= min_probability):

                        results.append({
                            "Symbol": sym,
                            "Price": f"${price_data.data['adj_close'].iloc[-1]:.2f}",
                            "Composite": f"{analysis.composite_score.value:.1f}",
                            "Rating": analysis.composite_score.components.get("rating", "N/A"),
                            "Technical": f"{analysis.technical_score.value:.1f}",
                            "Momentum": f"{analysis.momentum_score.value:.1f}",
                            "Risk": f"{analysis.risk_score.value:.1f}",
                            "P(Up)": f"{analysis.probability.get('prob_up', 0.5)*100:.1f}%",
                            "Signal": analysis.probability.get("signal", "neutral"),
                        })

                except Exception as e:
                    st.warning(f"Failed to analyze {sym}: {e}")

                progress.progress((i + 1) / len(symbols))

            progress.empty()

            if results:
                # Sort by composite score
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values("Composite", ascending=False)

                st.success(f"Found {len(results)} stocks matching criteria!")
                st.dataframe(results_df, use_container_width=True, hide_index=True)

                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results (CSV)",
                    csv,
                    "screener_results.csv",
                    "text/csv",
                )
            else:
                st.warning("No stocks match the specified criteria.")


def render_comparison(benchmark_prices: pd.DataFrame | None):
    """Render stock comparison page."""
    st.markdown("<h1 class='main-header'>⚖️ Stock Comparison</h1>", unsafe_allow_html=True)

    # Stock selection
    symbols_input = st.text_input(
        "Enter symbols to compare (comma-separated)",
        value="AAPL, MSFT, GOOGL",
    )

    symbols = [s.strip().upper() for s in symbols_input.split(",")]

    if st.button("Compare", type="primary"):
        with st.spinner("Loading and analyzing stocks..."):
            from stock_analysis.data.provider import DataProvider
            from stock_analysis.scoring.scorer import StockScorer

            provider = DataProvider()
            scorer = StockScorer()

            analyses = []
            price_data = {}

            for sym in symbols:
                try:
                    pd_data = provider.get_prices(sym)
                    price_data[sym] = pd_data.data

                    benchmark_data = provider.get_prices("SPY")
                    analysis = scorer.analyze(sym, pd_data.data, benchmark_data.data)
                    analyses.append(analysis)

                except Exception as e:
                    st.warning(f"Failed to load {sym}: {e}")

            if not analyses:
                st.error("Could not analyze any stocks.")
                return

            # Comparison table
            st.subheader("Comparison Table")

            comparison_data = []
            for a in analyses:
                comparison_data.append({
                    "Symbol": a.symbol,
                    "Price": f"${a.price:.2f}",
                    "Composite": a.composite_score.value,
                    "Rating": a.composite_score.components.get("rating", "N/A"),
                    "Technical": a.technical_score.value,
                    "Momentum": a.momentum_score.value,
                    "Risk": a.risk_score.value,
                    "P(Up)": f"{a.probability.get('prob_up', 0.5)*100:.1f}%",
                    "Signal": a.probability.get("signal", "neutral"),
                })

            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Price comparison chart
            st.subheader("Normalized Price Comparison")

            fig = go.Figure()
            for sym, prices in price_data.items():
                # Normalize to 100
                normalized = prices["adj_close"] / prices["adj_close"].iloc[0] * 100
                fig.add_trace(go.Scatter(
                    x=prices.index[-252:],
                    y=normalized.iloc[-252:],
                    name=sym,
                    mode="lines",
                ))

            fig.update_layout(
                yaxis_title="Normalized Price (Start = 100)",
                xaxis_title="Date",
                height=500,
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)

            # Score comparison radar chart
            st.subheader("Score Comparison")

            categories = ["Composite", "Technical", "Momentum", "Risk"]

            fig = go.Figure()

            for a in analyses:
                fig.add_trace(go.Scatterpolar(
                    r=[a.composite_score.value, a.technical_score.value,
                       a.momentum_score.value, a.risk_score.value],
                    theta=categories,
                    fill="toself",
                    name=a.symbol,
                ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=500,
            )

            st.plotly_chart(fig, use_container_width=True)


def render_sp500_rankings():
    """Render S&P 500 rankings page with comprehensive analysis."""
    st.markdown("<h1 class='main-header'>🏆 S&P 500 Rankings</h1>", unsafe_allow_html=True)
    st.markdown("Comprehensive analysis and ranking of all S&P 500 stocks")

    # Import required modules
    from stock_analysis.data.provider import DataProvider
    from stock_analysis.data.universe import UniverseManager
    from stock_analysis.scoring.scorer import StockScorer
    from stock_analysis.indicators.engine import IndicatorEngine

    # Settings
    st.sidebar.markdown("### Ranking Settings")
    max_stocks = st.sidebar.slider("Stocks to Analyze", 20, 100, 50)
    sort_by = st.sidebar.selectbox(
        "Sort By",
        ["Composite Score", "Technical Score", "Momentum Score", "Probability", "1M Return", "3M Return"],
        index=0,
    )
    sector_filter = st.sidebar.multiselect(
        "Filter Sectors",
        ["Technology", "Healthcare", "Financials", "Consumer Discretionary",
         "Communication Services", "Industrials", "Consumer Staples",
         "Energy", "Utilities", "Real Estate", "Materials"],
    )

    # Get S&P 500 tickers
    universe = UniverseManager()

    @st.cache_data(ttl=86400, show_spinner=False)
    def get_sp500_list():
        return universe.get_universe("SP500")

    with st.spinner("Fetching S&P 500 constituents..."):
        sp500_tickers = get_sp500_list()
        st.info(f"Found {len(sp500_tickers)} S&P 500 stocks")

    # Analyze button
    if st.button("🚀 Analyze S&P 500", type="primary"):
        provider = DataProvider()
        scorer = StockScorer()
        engine = IndicatorEngine()

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Limit to max_stocks for performance
        tickers_to_analyze = sp500_tickers[:max_stocks]

        for i, ticker in enumerate(tickers_to_analyze):
            progress_bar.progress((i + 1) / len(tickers_to_analyze))
            status_text.text(f"Analyzing {ticker}... ({i+1}/{len(tickers_to_analyze)})")

            try:
                # Get prices (uses local storage cache)
                price_data = provider.get_prices(ticker)
                prices = price_data.data

                if len(prices) < 50:
                    continue

                # Compute analysis
                analysis = scorer.analyze(ticker, prices)

                # Compute returns
                close = prices["adj_close"]
                ret_1m = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else 0
                ret_3m = (close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close) > 63 else 0
                ret_ytd = (close.iloc[-1] / close.iloc[0] - 1) * 100

                # Get sector info
                try:
                    info = provider.get_ticker_info(ticker)
                    sector = info.sector or "Unknown"
                    name = info.name or ticker
                except Exception:
                    sector = "Unknown"
                    name = ticker

                # Get key indicators
                indicators = engine.compute_all(prices)

                results.append({
                    "Ticker": ticker,
                    "Name": name[:30] if name else ticker,
                    "Sector": sector,
                    "Price": close.iloc[-1],
                    "Composite": analysis.composite_score.value,
                    "Technical": analysis.technical_score.value,
                    "Momentum": analysis.momentum_score.value,
                    "Risk": analysis.risk_score.value,
                    "P(Up)": analysis.probability.get("prob_up", 0.5) * 100,
                    "Signal": analysis.probability.get("signal", "neutral"),
                    "1M Return": ret_1m,
                    "3M Return": ret_3m,
                    "YTD Return": ret_ytd,
                    "RSI": indicators.get("rsi", 50),
                    "MACD Hist": indicators.get("macd_histogram", 0),
                    "Vol 20d": indicators.get("annualized_volatility", 0),
                })

            except Exception as e:
                continue

        progress_bar.empty()
        status_text.empty()

        if not results:
            st.error("No stocks could be analyzed. Please try again.")
            return

        # Create DataFrame
        df = pd.DataFrame(results)

        # Apply sector filter
        if sector_filter:
            df = df[df["Sector"].isin(sector_filter)]

        # Sort
        sort_col_map = {
            "Composite Score": "Composite",
            "Technical Score": "Technical",
            "Momentum Score": "Momentum",
            "Probability": "P(Up)",
            "1M Return": "1M Return",
            "3M Return": "3M Return",
        }
        sort_col = sort_col_map.get(sort_by, "Composite")
        df = df.sort_values(sort_col, ascending=False)

        # Store in session state
        st.session_state["sp500_results"] = df

    # Display results if available
    if "sp500_results" in st.session_state:
        df = st.session_state["sp500_results"]

        st.markdown("---")

        # Summary metrics
        st.subheader("Market Overview")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            bullish = len(df[df["P(Up)"] > 55])
            st.metric("Bullish Stocks", bullish, f"{bullish/len(df)*100:.0f}%")

        with col2:
            avg_score = df["Composite"].mean()
            st.metric("Avg Composite Score", f"{avg_score:.1f}")

        with col3:
            avg_1m = df["1M Return"].mean()
            st.metric("Avg 1M Return", f"{avg_1m:+.1f}%")

        with col4:
            avg_3m = df["3M Return"].mean()
            st.metric("Avg 3M Return", f"{avg_3m:+.1f}%")

        with col5:
            best = df.iloc[0]["Ticker"]
            st.metric("Top Stock", best, f"{df.iloc[0]['Composite']:.0f}")

        st.markdown("---")

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Rankings Table", "📈 Charts", "🎯 Top Picks", "📉 Sector Analysis"])

        with tab1:
            # Full rankings table
            st.subheader("Full Rankings")

            # Format the dataframe for display
            display_df = df.copy()
            display_df["Price"] = display_df["Price"].apply(lambda x: f"${x:.2f}")
            display_df["Composite"] = display_df["Composite"].apply(lambda x: f"{x:.1f}")
            display_df["Technical"] = display_df["Technical"].apply(lambda x: f"{x:.1f}")
            display_df["Momentum"] = display_df["Momentum"].apply(lambda x: f"{x:.1f}")
            display_df["P(Up)"] = display_df["P(Up)"].apply(lambda x: f"{x:.1f}%")
            display_df["1M Return"] = display_df["1M Return"].apply(lambda x: f"{x:+.1f}%")
            display_df["3M Return"] = display_df["3M Return"].apply(lambda x: f"{x:+.1f}%")
            display_df["RSI"] = display_df["RSI"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")

            st.dataframe(
                display_df[["Ticker", "Name", "Sector", "Price", "Composite", "Technical",
                           "Momentum", "P(Up)", "Signal", "1M Return", "3M Return", "RSI"]],
                use_container_width=True,
                hide_index=True,
            )

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Full Data (CSV)",
                csv,
                "sp500_rankings.csv",
                "text/csv",
            )

        with tab2:
            # Visualization charts
            st.subheader("Score Distribution")

            col1, col2 = st.columns(2)

            with col1:
                # Score histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df["Composite"], nbinsx=20, name="Composite Score"))
                fig.update_layout(
                    title="Composite Score Distribution",
                    xaxis_title="Score",
                    yaxis_title="Count",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Probability histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df["P(Up)"], nbinsx=20, name="P(Up)", marker_color="green"))
                fig.add_vline(x=50, line_dash="dash", line_color="red")
                fig.update_layout(
                    title="Probability Distribution",
                    xaxis_title="P(Up) %",
                    yaxis_title="Count",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Scatter plot: Score vs Return
            st.subheader("Score vs Performance")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["Composite"],
                y=df["1M Return"],
                mode="markers",
                text=df["Ticker"],
                marker=dict(
                    size=10,
                    color=df["P(Up)"],
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="P(Up)"),
                ),
            ))
            fig.update_layout(
                title="Composite Score vs 1-Month Return",
                xaxis_title="Composite Score",
                yaxis_title="1M Return (%)",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Top 20 bar chart
            st.subheader("Top 20 by Composite Score")
            top20 = df.head(20)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top20["Ticker"],
                y=top20["Composite"],
                marker_color=top20["P(Up)"],
                marker_colorscale="RdYlGn",
                text=top20["Composite"].apply(lambda x: f"{x:.0f}"),
                textposition="outside",
            ))
            fig.update_layout(
                title="Top 20 Stocks by Composite Score",
                yaxis_title="Composite Score",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # Top picks with detailed cards
            st.subheader("Top 10 Picks")

            top10 = df.head(10)

            for i, row in top10.iterrows():
                with st.expander(f"#{list(top10.index).index(i)+1} {row['Ticker']} - {row['Name']}", expanded=(list(top10.index).index(i) < 3)):
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Price", f"${row['Price']:.2f}")
                        st.metric("1M Return", f"{row['1M Return']:+.1f}%")

                    with col2:
                        st.metric("Composite Score", f"{row['Composite']:.1f}")
                        st.metric("Technical Score", f"{row['Technical']:.1f}")

                    with col3:
                        st.metric("P(Up)", f"{row['P(Up)']:.1f}%")
                        st.metric("Signal", row["Signal"])

                    with col4:
                        st.metric("Sector", row["Sector"])
                        st.metric("RSI", f"{row['RSI']:.1f}" if pd.notna(row['RSI']) else "N/A")

                    # Quick insight
                    signal_color = "green" if row["P(Up)"] > 55 else "red" if row["P(Up)"] < 45 else "orange"
                    st.markdown(f"""
                    **Quick Analysis:** {row['Ticker']} shows a composite score of {row['Composite']:.1f}
                    with a {row['P(Up)']:.0f}% probability of upward movement.
                    The stock has returned {row['1M Return']:+.1f}% over the past month and {row['3M Return']:+.1f}% over 3 months.
                    """)

        with tab4:
            # Sector analysis
            st.subheader("Performance by Sector")

            # Aggregate by sector
            sector_stats = df.groupby("Sector").agg({
                "Ticker": "count",
                "Composite": "mean",
                "P(Up)": "mean",
                "1M Return": "mean",
                "3M Return": "mean",
            }).round(2)
            sector_stats.columns = ["Count", "Avg Score", "Avg P(Up)", "Avg 1M Ret", "Avg 3M Ret"]
            sector_stats = sector_stats.sort_values("Avg Score", ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Sector Statistics**")
                st.dataframe(sector_stats, use_container_width=True)

            with col2:
                # Sector bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=sector_stats.index,
                    y=sector_stats["Avg Score"],
                    marker_color=sector_stats["Avg P(Up)"],
                    marker_colorscale="RdYlGn",
                    text=sector_stats["Avg Score"].apply(lambda x: f"{x:.1f}"),
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Average Composite Score by Sector",
                    yaxis_title="Avg Composite Score",
                    height=400,
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Sector treemap
            st.subheader("Sector Composition")
            fig = go.Figure(go.Treemap(
                labels=df["Ticker"],
                parents=df["Sector"],
                values=[1] * len(df),
                marker=dict(
                    colors=df["Composite"],
                    colorscale="RdYlGn",
                    showscale=True,
                ),
                textinfo="label+value",
            ))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Click 'Analyze S&P 500' to start the analysis. Data and ML models will be cached locally for faster future access.")

        # Show storage stats
        from stock_analysis.data.storage import get_storage
        from stock_analysis.ml.model_storage import get_model_storage

        storage = get_storage()
        data_stats = storage.get_stats()

        model_storage = get_model_storage()
        model_stats = model_storage.get_stats()

        st.markdown("### Local Data Storage")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cached Tickers", data_stats["ticker_count"])
        with col2:
            st.metric("Total Data Rows", f"{data_stats['total_rows']:,}")
        with col3:
            st.metric("Data Size", f"{data_stats['total_size_mb']:.1f} MB")

        st.markdown("### Per-Ticker ML Models")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trained Models", model_stats["model_count"])
        with col2:
            st.metric("Avg ROC-AUC", f"{model_stats['avg_roc_auc']:.3f}" if model_stats['avg_roc_auc'] > 0 else "N/A")
        with col3:
            st.metric("Models Size", f"{model_stats['total_size_mb']:.1f} MB")


def render_pro_analysis(symbol: str, prices: pd.DataFrame, indicators: dict, analysis):
    """Render professional analysis page with advanced tools."""
    st.markdown(f"<h1 class='main-header'>⚡ Pro Analysis: {symbol}</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(124, 58, 237, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%);
                border: 1px solid rgba(124, 58, 237, 0.3); border-radius: 12px; padding: 16px; margin-bottom: 24px;">
        <span style="color: #7c3aed; font-weight: 600;">PRO FEATURES</span>
        <span style="color: #94a3b8;"> — Advanced analysis tools for professional traders</span>
    </div>
    """, unsafe_allow_html=True)

    # Tabs for different pro tools
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Risk Analysis", "🎯 Entry/Exit Signals", "📈 Backtesting", "🔬 Factor Analysis", "📉 Drawdown Analysis"
    ])

    close = prices["adj_close"]

    with tab1:
        st.subheader("Advanced Risk Metrics")

        col1, col2, col3, col4 = st.columns(4)

        # Calculate risk metrics
        returns = close.pct_change().dropna()
        annual_vol = returns.std() * np.sqrt(252) * 100
        max_drawdown = ((close / close.cummax()) - 1).min() * 100
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # Calculate VaR and CVaR
        var_95 = returns.quantile(0.05) * 100
        cvar_95 = returns[returns <= returns.quantile(0.05)].mean() * 100

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (returns.mean() * 252) / downside_std if downside_std > 0 else 0

        # Calmar ratio
        calmar = (returns.mean() * 252 * 100) / abs(max_drawdown) if max_drawdown != 0 else 0

        with col1:
            st.metric("Annual Volatility", f"{annual_vol:.1f}%")
            st.metric("Max Drawdown", f"{max_drawdown:.1f}%")

        with col2:
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            st.metric("Sortino Ratio", f"{sortino:.2f}")

        with col3:
            st.metric("VaR (95%)", f"{var_95:.2f}%")
            st.metric("CVaR (95%)", f"{cvar_95:.2f}%")

        with col4:
            st.metric("Calmar Ratio", f"{calmar:.2f}")
            # Beta calculation if we had benchmark
            st.metric("Win Rate", f"{(returns > 0).mean() * 100:.1f}%")

        # Risk visualization
        st.markdown("---")
        st.subheader("Returns Distribution")

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Daily Returns Distribution", "Rolling Volatility"))

        # Returns histogram
        fig.add_trace(
            go.Histogram(x=returns * 100, nbinsx=50, name="Returns", marker_color="#7c3aed"),
            row=1, col=1
        )
        fig.add_vline(x=var_95, line_dash="dash", line_color="red", row=1, col=1)

        # Rolling volatility
        rolling_vol = returns.rolling(21).std() * np.sqrt(252) * 100
        fig.add_trace(
            go.Scatter(x=rolling_vol.index[-252:], y=rolling_vol.iloc[-252:], name="Rolling Vol (21d)", line=dict(color="#00d4ff")),
            row=1, col=2
        )

        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Entry & Exit Signal Generator")

        # Signal parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            rsi_oversold = st.slider("RSI Oversold", 20, 40, 30)
            rsi_overbought = st.slider("RSI Overbought", 60, 80, 70)
        with col2:
            bb_lower_pct = st.slider("BB Lower %", 0.0, 0.3, 0.2)
            bb_upper_pct = st.slider("BB Upper %", 0.7, 1.0, 0.8)
        with col3:
            volume_spike = st.slider("Volume Spike Threshold", 1.0, 3.0, 1.5)

        # Generate signals
        rsi = indicators.get("rsi", 50)
        bb_pct = indicators.get("bb_percent", 0.5) if isinstance(indicators.get("bb_percent"), float) else 0.5
        vol_ratio = indicators.get("volume_sma_ratio", 1)

        entry_signals = []
        exit_signals = []

        # Check entry conditions
        if rsi < rsi_oversold:
            entry_signals.append(f"🟢 RSI Oversold ({rsi:.1f})")
        if bb_pct < bb_lower_pct:
            entry_signals.append(f"🟢 Below BB Lower ({bb_pct:.2f})")
        if vol_ratio > volume_spike and rsi < 50:
            entry_signals.append(f"🟢 Volume Spike on Weakness ({vol_ratio:.1f}x)")

        # Check exit conditions
        if rsi > rsi_overbought:
            exit_signals.append(f"🔴 RSI Overbought ({rsi:.1f})")
        if bb_pct > bb_upper_pct:
            exit_signals.append(f"🔴 Above BB Upper ({bb_pct:.2f})")
        if vol_ratio > volume_spike and rsi > 50:
            exit_signals.append(f"🔴 Volume Spike on Strength ({vol_ratio:.1f}x)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📥 Entry Signals")
            if entry_signals:
                for signal in entry_signals:
                    st.success(signal)
            else:
                st.info("No entry signals at current levels")

        with col2:
            st.markdown("### 📤 Exit Signals")
            if exit_signals:
                for signal in exit_signals:
                    st.warning(signal)
            else:
                st.info("No exit signals at current levels")

        # Support/Resistance levels
        st.markdown("---")
        st.subheader("Key Support & Resistance Levels")

        high_52w = prices["high"].iloc[-252:].max()
        low_52w = prices["low"].iloc[-252:].min()
        current = close.iloc[-1]

        pivot = (prices["high"].iloc[-1] + prices["low"].iloc[-1] + close.iloc[-1]) / 3
        r1 = 2 * pivot - prices["low"].iloc[-1]
        s1 = 2 * pivot - prices["high"].iloc[-1]
        r2 = pivot + (prices["high"].iloc[-1] - prices["low"].iloc[-1])
        s2 = pivot - (prices["high"].iloc[-1] - prices["low"].iloc[-1])

        levels_data = pd.DataFrame({
            "Level": ["52W High", "R2", "R1", "Pivot", "Current", "S1", "S2", "52W Low"],
            "Price": [high_52w, r2, r1, pivot, current, s1, s2, low_52w],
            "Distance": [
                f"{(high_52w/current - 1)*100:+.1f}%",
                f"{(r2/current - 1)*100:+.1f}%",
                f"{(r1/current - 1)*100:+.1f}%",
                f"{(pivot/current - 1)*100:+.1f}%",
                "—",
                f"{(s1/current - 1)*100:+.1f}%",
                f"{(s2/current - 1)*100:+.1f}%",
                f"{(low_52w/current - 1)*100:+.1f}%",
            ]
        })
        levels_data["Price"] = levels_data["Price"].apply(lambda x: f"${x:.2f}")
        st.dataframe(levels_data, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Strategy Backtester")

        # Strategy selection
        strategy = st.selectbox(
            "Select Strategy",
            ["RSI Mean Reversion", "Moving Average Crossover", "Bollinger Band Breakout", "MACD Signal"]
        )

        lookback = st.slider("Backtest Period (days)", 60, 504, 252)

        if st.button("Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                # Simple strategy backtests
                test_prices = prices.iloc[-lookback:].copy()
                test_close = test_prices["adj_close"]
                test_returns = test_close.pct_change()

                if strategy == "RSI Mean Reversion":
                    delta = test_close.diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rsi_series = 100 - (100 / (1 + gain/loss))
                    signals = ((rsi_series < 30).astype(int) - (rsi_series > 70).astype(int))

                elif strategy == "Moving Average Crossover":
                    ema_8 = test_close.ewm(span=8).mean()
                    ema_21 = test_close.ewm(span=21).mean()
                    signals = (ema_8 > ema_21).astype(int)

                elif strategy == "Bollinger Band Breakout":
                    sma = test_close.rolling(20).mean()
                    std = test_close.rolling(20).std()
                    upper = sma + 2 * std
                    lower = sma - 2 * std
                    signals = ((test_close > upper).astype(int) - (test_close < lower).astype(int))

                else:  # MACD Signal
                    ema12 = test_close.ewm(span=12).mean()
                    ema26 = test_close.ewm(span=26).mean()
                    macd = ema12 - ema26
                    signal_line = macd.ewm(span=9).mean()
                    signals = (macd > signal_line).astype(int)

                # Calculate strategy returns
                strategy_returns = signals.shift(1) * test_returns
                strategy_cumulative = (1 + strategy_returns).cumprod()
                buy_hold_cumulative = (1 + test_returns).cumprod()

                # Metrics
                total_return = (strategy_cumulative.iloc[-1] - 1) * 100
                bh_return = (buy_hold_cumulative.iloc[-1] - 1) * 100
                n_trades = signals.diff().abs().sum() / 2
                win_rate = (strategy_returns[strategy_returns != 0] > 0).mean() * 100

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Strategy Return", f"{total_return:+.1f}%")
                with col2:
                    st.metric("Buy & Hold", f"{bh_return:+.1f}%")
                with col3:
                    st.metric("# Trades", f"{n_trades:.0f}")
                with col4:
                    st.metric("Win Rate", f"{win_rate:.1f}%" if not np.isnan(win_rate) else "N/A")

                # Performance chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=strategy_cumulative.index, y=strategy_cumulative * 100,
                    name=strategy, line=dict(color="#7c3aed", width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=buy_hold_cumulative.index, y=buy_hold_cumulative * 100,
                    name="Buy & Hold", line=dict(color="#00d4ff", width=2)
                ))
                fig.update_layout(
                    title="Strategy vs Buy & Hold",
                    yaxis_title="Value ($100 start)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Factor Exposure Analysis")

        # Calculate factor exposures
        returns = close.pct_change().dropna()

        # Momentum factor
        momentum_1m = close.pct_change(21).iloc[-1] * 100
        momentum_3m = close.pct_change(63).iloc[-1] * 100
        momentum_6m = close.pct_change(126).iloc[-1] * 100

        # Volatility factor
        vol_20 = returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
        vol_60 = returns.rolling(60).std().iloc[-1] * np.sqrt(252) * 100

        # Mean reversion factor
        z_score = (close.iloc[-1] - close.rolling(50).mean().iloc[-1]) / close.rolling(50).std().iloc[-1]

        # Create factor chart
        factors = {
            "Momentum 1M": min(max(momentum_1m / 20 * 50 + 50, 0), 100),
            "Momentum 3M": min(max(momentum_3m / 30 * 50 + 50, 0), 100),
            "Momentum 6M": min(max(momentum_6m / 50 * 50 + 50, 0), 100),
            "Low Volatility": 100 - min(vol_20, 100),
            "Mean Reversion": 50 - z_score * 25,
        }

        fig = go.Figure(go.Bar(
            x=list(factors.keys()),
            y=list(factors.values()),
            marker_color=["#10b981" if v > 50 else "#ef4444" for v in factors.values()],
            text=[f"{v:.0f}" for v in factors.values()],
            textposition="outside"
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="Factor Scores (50 = Neutral)",
            yaxis_range=[0, 100],
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Factor details
        st.markdown("### Factor Details")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Momentum Factors:**
            - 1-Month Return: {momentum_1m:+.1f}%
            - 3-Month Return: {momentum_3m:+.1f}%
            - 6-Month Return: {momentum_6m:+.1f}%
            """)
        with col2:
            st.markdown(f"""
            **Risk Factors:**
            - 20-Day Volatility: {vol_20:.1f}%
            - 60-Day Volatility: {vol_60:.1f}%
            - Z-Score (50d): {z_score:.2f}
            """)

    with tab5:
        st.subheader("Drawdown Analysis")

        # Calculate drawdowns
        cummax = close.cummax()
        drawdown = (close / cummax - 1) * 100

        # Find major drawdowns
        dd_threshold = -5  # 5% drawdown threshold

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=("Price with Drawdown Periods", "Drawdown %"),
                           row_heights=[0.6, 0.4])

        # Price chart
        fig.add_trace(
            go.Scatter(x=close.index[-252:], y=close.iloc[-252:], name="Price", line=dict(color="#00d4ff")),
            row=1, col=1
        )

        # Drawdown chart
        fig.add_trace(
            go.Scatter(x=drawdown.index[-252:], y=drawdown.iloc[-252:], name="Drawdown",
                      fill="tozeroy", line=dict(color="#ef4444"), fillcolor="rgba(239, 68, 68, 0.3)"),
            row=2, col=1
        )
        fig.add_hline(y=-10, line_dash="dash", line_color="orange", row=2, col=1)
        fig.add_hline(y=-20, line_dash="dash", line_color="red", row=2, col=1)

        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown statistics
        current_dd = drawdown.iloc[-1]
        max_dd = drawdown.min()
        avg_dd = drawdown[drawdown < 0].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Drawdown", f"{current_dd:.1f}%")
        with col2:
            st.metric("Max Drawdown", f"{max_dd:.1f}%")
        with col3:
            st.metric("Avg Drawdown", f"{avg_dd:.1f}%" if not np.isnan(avg_dd) else "0%")


def render_portfolio_tracker():
    """Render portfolio tracking page."""
    st.markdown("<h1 class='main-header'>💼 Portfolio Tracker</h1>", unsafe_allow_html=True)

    st.markdown("""
    Track your portfolio performance and get AI-powered insights.
    """)

    # Portfolio input
    st.subheader("Enter Your Holdings")

    # Initialize session state for portfolio
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        new_symbol = st.text_input("Symbol", key="new_symbol").upper()
    with col2:
        new_shares = st.number_input("Shares", min_value=0.0, value=0.0, key="new_shares")
    with col3:
        new_cost = st.number_input("Cost Basis ($)", min_value=0.0, value=0.0, key="new_cost")
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Add Position"):
            if new_symbol and new_shares > 0:
                st.session_state.portfolio.append({
                    "symbol": new_symbol,
                    "shares": new_shares,
                    "cost_basis": new_cost
                })
                st.rerun()

    # Display and analyze portfolio
    if st.session_state.portfolio:
        st.markdown("---")
        st.subheader("Current Holdings")

        from stock_analysis.data.provider import DataProvider
        provider = DataProvider()

        portfolio_data = []
        total_value = 0
        total_cost = 0

        for position in st.session_state.portfolio:
            try:
                price_data = provider.get_prices(position["symbol"])
                current_price = price_data.data["adj_close"].iloc[-1]
                market_value = current_price * position["shares"]
                cost_value = position["cost_basis"] * position["shares"]
                gain_loss = market_value - cost_value
                gain_pct = (gain_loss / cost_value * 100) if cost_value > 0 else 0

                portfolio_data.append({
                    "Symbol": position["symbol"],
                    "Shares": position["shares"],
                    "Cost Basis": f"${position['cost_basis']:.2f}",
                    "Current Price": f"${current_price:.2f}",
                    "Market Value": f"${market_value:,.2f}",
                    "Gain/Loss": f"${gain_loss:+,.2f}",
                    "Return": f"{gain_pct:+.1f}%"
                })

                total_value += market_value
                total_cost += cost_value

            except Exception as e:
                st.warning(f"Could not fetch data for {position['symbol']}: {e}")

        if portfolio_data:
            df = pd.DataFrame(portfolio_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Portfolio summary
            st.markdown("---")
            st.subheader("Portfolio Summary")

            total_gain = total_value - total_cost
            total_return = (total_gain / total_cost * 100) if total_cost > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Value", f"${total_value:,.2f}")
            with col2:
                st.metric("Total Cost", f"${total_cost:,.2f}")
            with col3:
                st.metric("Total Gain/Loss", f"${total_gain:+,.2f}")
            with col4:
                st.metric("Total Return", f"{total_return:+.1f}%")

            # Portfolio allocation pie chart
            values = [float(row["Market Value"].replace("$", "").replace(",", "")) for row in portfolio_data]
            symbols = [row["Symbol"] for row in portfolio_data]

            fig = go.Figure(go.Pie(
                labels=symbols,
                values=values,
                hole=0.4,
                marker_colors=["#7c3aed", "#00d4ff", "#10b981", "#f59e0b", "#ef4444", "#6366f1"]
            ))
            fig.update_layout(title="Portfolio Allocation", height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Clear portfolio button
        if st.button("Clear Portfolio"):
            st.session_state.portfolio = []
            st.rerun()

    else:
        st.info("Add positions above to start tracking your portfolio.")


if __name__ == "__main__":
    main()
