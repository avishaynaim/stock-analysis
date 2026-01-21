"""
Relative Strength indicators (Group G).

Indicators:
- Relative Strength vs Benchmark
- Relative Strength Ratio Line
- Mansfield Relative Strength
- Performance Ranking
"""

import numpy as np
import pandas as pd

from stock_analysis.indicators.registry import indicator


@indicator(
    name="relative_strength_benchmark",
    group="relative",
    description="Relative Strength vs Benchmark",
    parameters={"periods": [21, 63, 252]},
    required_periods=252,
    requires_benchmark=True,
)
def compute_relative_strength_benchmark(
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame | None = None,
    periods: list[int] | None = None,
) -> dict[str, float]:
    """Compute relative strength vs benchmark at multiple horizons."""
    if periods is None:
        periods = [21, 63, 252]

    close = prices["adj_close"]

    if len(close) < max(periods):
        return {}

    result = {}

    # If no benchmark, compute absolute returns only
    if benchmark_prices is None or benchmark_prices.empty:
        for period in periods:
            if len(close) >= period:
                stock_return = (close.iloc[-1] / close.iloc[-period] - 1) * 100
                result[f"return_{period}d"] = stock_return
                result[f"rs_{period}d"] = stock_return  # Same as return without benchmark
        return result

    bench_close = benchmark_prices["adj_close"]

    # Align data
    common_idx = close.index.intersection(bench_close.index)
    if len(common_idx) < max(periods):
        # Fall back to absolute returns
        for period in periods:
            if len(close) >= period:
                stock_return = (close.iloc[-1] / close.iloc[-period] - 1) * 100
                result[f"return_{period}d"] = stock_return
                result[f"rs_{period}d"] = stock_return
        return result

    close = close.loc[common_idx]
    bench_close = bench_close.loc[common_idx]

    for period in periods:
        if len(close) >= period:
            stock_return = (close.iloc[-1] / close.iloc[-period] - 1) * 100
            bench_return = (bench_close.iloc[-1] / bench_close.iloc[-period] - 1) * 100
            excess_return = stock_return - bench_return

            result[f"return_{period}d"] = stock_return
            result[f"bench_return_{period}d"] = bench_return
            result[f"rs_{period}d"] = excess_return

    # Composite RS score (average of excess returns)
    rs_values = [result.get(f"rs_{p}d", 0) for p in periods if f"rs_{p}d" in result]
    if rs_values:
        result["rs_composite"] = np.mean(rs_values)
        result["outperforming"] = 1 if result["rs_composite"] > 0 else 0

    return result


@indicator(
    name="rs_ratio_line",
    group="relative",
    description="Relative Strength Ratio Line",
    parameters={"ma_period": 50},
    required_periods=100,
    requires_benchmark=True,
)
def compute_rs_ratio_line(
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame | None = None,
    ma_period: int = 50,
) -> dict[str, float]:
    """Compute RS ratio line (price/benchmark) with trend analysis."""
    close = prices["adj_close"]

    if len(close) < ma_period + 10:
        return {}

    if benchmark_prices is None or benchmark_prices.empty:
        return {"rs_ratio": 1.0, "rs_trend": 0}

    bench_close = benchmark_prices["adj_close"]

    # Align data
    common_idx = close.index.intersection(bench_close.index)
    if len(common_idx) < ma_period + 10:
        return {"rs_ratio": 1.0, "rs_trend": 0}

    close = close.loc[common_idx]
    bench_close = bench_close.loc[common_idx]

    # Calculate RS ratio line
    rs_ratio = close / bench_close

    # RS ratio moving average
    rs_ma = rs_ratio.rolling(window=ma_period).mean()

    current_ratio = rs_ratio.iloc[-1]
    current_ma = rs_ma.iloc[-1]

    # RS momentum (rate of change of ratio)
    rs_roc = (rs_ratio.iloc[-1] / rs_ratio.iloc[-21] - 1) * 100 if len(rs_ratio) > 21 else 0

    # RS trend (above/below MA)
    rs_trend = 1 if current_ratio > current_ma else -1

    # RS acceleration (second derivative proxy)
    if len(rs_ratio) > 10:
        recent_slope = rs_ratio.iloc[-1] - rs_ratio.iloc[-5]
        prior_slope = rs_ratio.iloc[-5] - rs_ratio.iloc[-10]
        rs_acceleration = 1 if recent_slope > prior_slope else -1
    else:
        rs_acceleration = 0

    return {
        "rs_ratio": current_ratio,
        "rs_ratio_ma": current_ma,
        "rs_trend": rs_trend,
        "rs_roc_21d": rs_roc,
        "rs_acceleration": rs_acceleration,
        "rs_improving": 1 if rs_trend == 1 and rs_acceleration == 1 else 0,
    }


@indicator(
    name="mansfield_rs",
    group="relative",
    description="Mansfield Relative Strength",
    parameters={"ma_period": 52},
    required_periods=100,
    requires_benchmark=True,
)
def compute_mansfield_rs(
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame | None = None,
    ma_period: int = 52,
) -> dict[str, float]:
    """Compute Mansfield Relative Strength (zero-centered)."""
    close = prices["adj_close"]

    if len(close) < ma_period + 10:
        return {}

    if benchmark_prices is None or benchmark_prices.empty:
        return {"mansfield_rs": 0.0}

    bench_close = benchmark_prices["adj_close"]

    # Align data
    common_idx = close.index.intersection(bench_close.index)
    if len(common_idx) < ma_period + 10:
        return {"mansfield_rs": 0.0}

    close = close.loc[common_idx]
    bench_close = bench_close.loc[common_idx]

    # Calculate RS ratio
    rs_ratio = close / bench_close

    # Moving average of RS ratio
    rs_ma = rs_ratio.rolling(window=ma_period).mean()

    # Mansfield RS = ((RS / RS_MA) - 1) * 100
    mansfield = ((rs_ratio / rs_ma) - 1) * 100

    current_mansfield = mansfield.iloc[-1]

    # Trend of Mansfield RS
    if len(mansfield) > 10:
        mansfield_trend = 1 if mansfield.iloc[-1] > mansfield.iloc[-10] else -1
    else:
        mansfield_trend = 0

    return {
        "mansfield_rs": current_mansfield,
        "mansfield_positive": 1 if current_mansfield > 0 else 0,
        "mansfield_trend": mansfield_trend,
        "mansfield_strong": 1 if current_mansfield > 10 else 0,
        "mansfield_weak": 1 if current_mansfield < -10 else 0,
    }


@indicator(
    name="momentum_score",
    group="relative",
    description="Multi-period Momentum Score",
    parameters={},
    required_periods=252,
)
def compute_momentum_score(
    prices: pd.DataFrame,
) -> dict[str, float]:
    """Compute composite momentum score across multiple timeframes."""
    close = prices["adj_close"]

    if len(close) < 252:
        return {}

    # Calculate returns at different horizons
    returns = {}
    periods = [21, 63, 126, 252]

    for period in periods:
        if len(close) >= period:
            ret = (close.iloc[-1] / close.iloc[-period] - 1) * 100
            returns[period] = ret

    if not returns:
        return {}

    # Weighted average (more weight to recent)
    weights = {21: 0.4, 63: 0.3, 126: 0.2, 252: 0.1}
    weighted_sum = sum(returns.get(p, 0) * weights[p] for p in periods)

    # Momentum consistency (all positive or all negative)
    signs = [1 if returns.get(p, 0) > 0 else -1 for p in periods if p in returns]
    consistency = abs(sum(signs)) / len(signs) if signs else 0

    # Momentum acceleration (short-term vs long-term)
    if 21 in returns and 63 in returns:
        acceleration = returns[21] - returns[63] / 3  # Normalize
    else:
        acceleration = 0

    return {
        "momentum_score": weighted_sum,
        "momentum_1m": returns.get(21, 0),
        "momentum_3m": returns.get(63, 0),
        "momentum_6m": returns.get(126, 0),
        "momentum_12m": returns.get(252, 0),
        "momentum_consistency": consistency,
        "momentum_acceleration": acceleration,
        "momentum_positive": 1 if weighted_sum > 0 else 0,
    }


@indicator(
    name="drawdown_analysis",
    group="relative",
    description="Drawdown Analysis",
    parameters={"lookback": 252},
    required_periods=252,
)
def compute_drawdown_analysis(
    prices: pd.DataFrame,
    lookback: int = 252,
) -> dict[str, float]:
    """Compute drawdown metrics."""
    close = prices["adj_close"]

    if len(close) < lookback:
        return {}

    # Rolling maximum
    rolling_max = close.rolling(window=lookback, min_periods=1).max()

    # Current drawdown
    drawdown = (close - rolling_max) / rolling_max

    current_dd = drawdown.iloc[-1]

    # Maximum drawdown in period
    max_dd = drawdown.iloc[-lookback:].min()

    # Days since high
    high_idx = close.iloc[-lookback:].idxmax()
    current_idx = close.index[-1]
    days_since_high = (current_idx - high_idx).days if hasattr(current_idx - high_idx, "days") else len(close) - close.iloc[-lookback:].values.argmax() - 1

    # Recovery ratio (current price vs max drawdown low)
    if max_dd < 0:
        dd_low = close.iloc[-lookback:].min()
        period_high = close.iloc[-lookback:].max()
        recovery_ratio = (close.iloc[-1] - dd_low) / (period_high - dd_low) if period_high != dd_low else 1
    else:
        recovery_ratio = 1

    return {
        "current_drawdown": current_dd * 100,
        "max_drawdown": max_dd * 100,
        "days_since_high": days_since_high,
        "recovery_ratio": recovery_ratio,
        "in_drawdown": 1 if current_dd < -0.05 else 0,
        "near_high": 1 if current_dd > -0.03 else 0,
    }


@indicator(
    name="risk_adjusted_returns",
    group="relative",
    description="Risk-Adjusted Return Metrics",
    parameters={"period": 252, "risk_free_rate": 0.05},
    required_periods=252,
)
def compute_risk_adjusted_returns(
    prices: pd.DataFrame,
    period: int = 252,
    risk_free_rate: float = 0.05,
) -> dict[str, float]:
    """Compute Sharpe, Sortino, and Calmar ratios."""
    close = prices["adj_close"]

    if len(close) < period:
        return {}

    # Daily returns
    returns = close.pct_change().dropna()

    if len(returns) < period:
        return {}

    returns_period = returns.iloc[-period:]

    # Annualized return
    total_return = close.iloc[-1] / close.iloc[-period] - 1
    ann_return = (1 + total_return) ** (252 / period) - 1

    # Annualized volatility
    ann_vol = returns_period.std() * np.sqrt(252)

    # Sharpe Ratio
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0

    # Sortino Ratio (downside deviation)
    negative_returns = returns_period[returns_period < 0]
    downside_dev = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else ann_vol
    sortino = (ann_return - risk_free_rate) / downside_dev if downside_dev > 0 else 0

    # Calmar Ratio (return / max drawdown)
    rolling_max = close.rolling(window=period, min_periods=1).max()
    drawdown = (close - rolling_max) / rolling_max
    max_dd = abs(drawdown.iloc[-period:].min())
    calmar = ann_return / max_dd if max_dd > 0 else 0

    return {
        "annualized_return": ann_return * 100,
        "annualized_volatility": ann_vol * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "risk_reward_favorable": 1 if sharpe > 1 else 0,
    }
