# Backtesting & Evaluation Framework

## 1. Overview

This document defines the research-grade backtesting system for validating the stock analysis models. The framework emphasizes statistical rigor, realistic assumptions, and comprehensive failure mode analysis.

### Design Principles

1. **No Lookahead Bias:** Strict point-in-time data access
2. **Realistic Assumptions:** Account for transaction costs, slippage, liquidity
3. **Statistical Rigor:** Proper significance testing and confidence intervals
4. **Comprehensive Metrics:** Beyond returns - risk, consistency, robustness
5. **Failure Analysis:** Understand when and why the system fails

### Framework Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BACKTESTING FRAMEWORK ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ DATA LAYER                                                          │   │
│  │ • Point-in-time price data                                          │   │
│  │ • Universe membership history                                       │   │
│  │ • Corporate actions (splits, dividends)                             │   │
│  │ • Fundamental data with release dates                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ SIGNAL GENERATION                                                   │   │
│  │ • Walk-forward model training                                       │   │
│  │ • Score/probability computation                                     │   │
│  │ • Signal filtering and ranking                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PORTFOLIO CONSTRUCTION                                              │   │
│  │ • Position sizing                                                   │   │
│  │ • Risk constraints                                                  │   │
│  │ • Rebalancing logic                                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ EXECUTION SIMULATION                                                │   │
│  │ • Transaction costs                                                 │   │
│  │ • Slippage modeling                                                 │   │
│  │ • Liquidity constraints                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ EVALUATION & ANALYSIS                                               │   │
│  │ • Performance metrics                                               │   │
│  │ • Statistical significance                                          │   │
│  │ • Calibration analysis                                              │   │
│  │ • Failure mode identification                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Walk-Forward Backtesting Mechanics

### 2.1 Core Walk-Forward Engine

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward backtest."""

    # Time windows
    initial_train_days: int = 756       # 3 years initial training
    retrain_frequency_days: int = 21    # Retrain monthly
    embargo_days: int = 21              # Gap to prevent leakage

    # Expanding vs. rolling
    expanding_window: bool = True       # Expanding training window

    # Trading frequency
    rebalance_frequency: str = 'weekly' # 'daily', 'weekly', 'monthly'

    # Universe
    universe: str = 'SP500'
    min_price: float = 5.0              # Minimum stock price
    min_volume: float = 500_000         # Minimum daily dollar volume

    # Signal thresholds
    entry_score_threshold: float = 7.0  # Minimum score to enter
    exit_score_threshold: float = 4.0   # Exit when score drops below

    # Dates
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class WalkForwardBacktester:
    """
    Walk-forward backtesting engine.

    Simulates realistic trading with proper temporal separation.
    """

    def __init__(
        self,
        config: WalkForwardConfig,
        model_factory: Callable,
        data_provider,
        portfolio_constructor,
        execution_simulator
    ):
        self.config = config
        self.model_factory = model_factory
        self.data = data_provider
        self.portfolio = portfolio_constructor
        self.execution = execution_simulator

        self.results = None
        self.model_versions = []

    def run(self) -> 'BacktestResults':
        """
        Execute full walk-forward backtest.
        """
        # Initialize
        dates = self.data.get_trading_dates(
            self.config.start_date,
            self.config.end_date
        )

        current_model = None
        last_train_date = None
        positions = {}
        equity_curve = []
        trades = []
        daily_returns = []

        # Main loop
        for i, date in enumerate(dates):
            # Check if retraining needed
            if self._should_retrain(date, last_train_date):
                current_model = self._train_model(date)
                last_train_date = date
                self.model_versions.append({
                    'date': date,
                    'model': current_model
                })

            # Check if rebalancing day
            if self._is_rebalance_day(date):
                # Generate signals
                universe = self._get_universe(date)
                signals = self._generate_signals(
                    current_model, universe, date
                )

                # Construct target portfolio
                target_positions = self.portfolio.construct(
                    signals=signals,
                    current_positions=positions,
                    date=date
                )

                # Execute trades
                executed_trades = self.execution.execute(
                    current_positions=positions,
                    target_positions=target_positions,
                    date=date
                )

                trades.extend(executed_trades)
                positions = target_positions

            # Mark to market
            portfolio_value = self._mark_to_market(positions, date)
            equity_curve.append({
                'date': date,
                'value': portfolio_value,
                'positions': len(positions)
            })

            # Compute daily return
            if len(equity_curve) > 1:
                daily_return = (
                    equity_curve[-1]['value'] / equity_curve[-2]['value'] - 1
                )
                daily_returns.append({
                    'date': date,
                    'return': daily_return
                })

        # Build results
        self.results = BacktestResults(
            config=self.config,
            equity_curve=pd.DataFrame(equity_curve),
            trades=pd.DataFrame(trades),
            daily_returns=pd.DataFrame(daily_returns),
            model_versions=self.model_versions
        )

        return self.results

    def _should_retrain(
        self,
        current_date: datetime,
        last_train_date: Optional[datetime]
    ) -> bool:
        """Check if model should be retrained."""
        if last_train_date is None:
            return True

        days_since_train = (current_date - last_train_date).days
        return days_since_train >= self.config.retrain_frequency_days

    def _train_model(self, as_of_date: datetime):
        """Train model using data up to as_of_date."""
        # Get training data (respecting embargo)
        train_end = as_of_date - timedelta(days=self.config.embargo_days)

        if self.config.expanding_window:
            train_start = None  # Use all available data
        else:
            train_start = train_end - timedelta(days=self.config.initial_train_days)

        X, y, features = self.data.get_training_data(
            start_date=train_start,
            end_date=train_end
        )

        # Train model
        model = self.model_factory()
        model.fit(X, y, features)

        return model

    def _is_rebalance_day(self, date: datetime) -> bool:
        """Check if this is a rebalancing day."""
        if self.config.rebalance_frequency == 'daily':
            return True
        elif self.config.rebalance_frequency == 'weekly':
            return date.weekday() == 4  # Friday
        elif self.config.rebalance_frequency == 'monthly':
            # Last trading day of month
            next_day = date + timedelta(days=1)
            return next_day.month != date.month
        return False

    def _get_universe(self, date: datetime) -> List[str]:
        """Get tradeable universe for date."""
        universe = self.data.get_universe_members(
            universe=self.config.universe,
            date=date
        )

        # Apply filters
        filtered = []
        for ticker in universe:
            price = self.data.get_price(ticker, date)
            volume = self.data.get_dollar_volume(ticker, date)

            if (price >= self.config.min_price and
                volume >= self.config.min_volume):
                filtered.append(ticker)

        return filtered

    def _generate_signals(
        self,
        model,
        universe: List[str],
        date: datetime
    ) -> pd.DataFrame:
        """Generate trading signals for universe."""
        signals = []

        for ticker in universe:
            # Get features as of date
            features = self.data.get_features_as_of(ticker, date)

            if features is None:
                continue

            # Generate score
            score = model.score(features)

            signals.append({
                'ticker': ticker,
                'date': date,
                'score': score.score,
                'probability': score.breakdown.get('edge', {}).get('probability', 0.5),
                'signal': 'LONG' if score.score >= self.config.entry_score_threshold else 'NONE'
            })

        return pd.DataFrame(signals)

    def _mark_to_market(
        self,
        positions: Dict[str, float],
        date: datetime
    ) -> float:
        """Calculate portfolio value."""
        total = 0.0

        for ticker, shares in positions.items():
            price = self.data.get_price(ticker, date)
            if price is not None:
                total += shares * price

        return total
```

### 2.2 Walk-Forward Simulation Modes

```python
class SimulationMode:
    """Different simulation modes for various research purposes."""

    @staticmethod
    def signal_only(config: WalkForwardConfig) -> WalkForwardConfig:
        """
        Signal-only mode: No portfolio construction.
        Just evaluate signal quality.
        """
        config.rebalance_frequency = 'daily'
        return config

    @staticmethod
    def paper_trading(config: WalkForwardConfig) -> WalkForwardConfig:
        """
        Paper trading mode: Realistic execution simulation.
        """
        config.rebalance_frequency = 'weekly'
        return config

    @staticmethod
    def stress_test(config: WalkForwardConfig) -> WalkForwardConfig:
        """
        Stress test mode: Focus on drawdown periods.
        """
        # Include known stress periods
        config.stress_periods = [
            ('2008-09-01', '2009-03-31'),  # GFC
            ('2020-02-15', '2020-04-15'),  # COVID
            ('2022-01-01', '2022-10-31'),  # 2022 Bear
        ]
        return config


class IncrementalBacktester:
    """
    Incremental backtester for efficient parameter sweeps.

    Caches intermediate results to speed up repeated runs.
    """

    def __init__(self, base_config: WalkForwardConfig):
        self.base_config = base_config
        self.signal_cache = {}
        self.model_cache = {}

    def run_parameter_sweep(
        self,
        parameter_grid: Dict[str, List],
        n_jobs: int = -1
    ) -> pd.DataFrame:
        """
        Run backtest across parameter combinations.
        """
        from itertools import product

        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())

        results = []

        for combo in product(*param_values):
            params = dict(zip(param_names, combo))

            # Create config with parameters
            config = self._apply_params(self.base_config, params)

            # Run backtest
            backtester = WalkForwardBacktester(config, ...)
            result = backtester.run()

            results.append({
                **params,
                'sharpe': result.metrics.sharpe_ratio,
                'cagr': result.metrics.cagr,
                'max_dd': result.metrics.max_drawdown,
                'win_rate': result.metrics.win_rate
            })

        return pd.DataFrame(results)

    def _apply_params(
        self,
        base_config: WalkForwardConfig,
        params: Dict
    ) -> WalkForwardConfig:
        """Apply parameters to config."""
        import copy
        config = copy.deepcopy(base_config)

        for key, value in params.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config
```

### 2.3 Event-Driven Simulation

```python
@dataclass
class Event:
    """Base event class."""
    timestamp: datetime
    event_type: str


@dataclass
class MarketEvent(Event):
    """Market data update."""
    ticker: str
    price: float
    volume: int


@dataclass
class SignalEvent(Event):
    """Trading signal."""
    ticker: str
    signal: str  # 'LONG', 'SHORT', 'EXIT'
    score: float
    confidence: float


@dataclass
class OrderEvent(Event):
    """Order to be executed."""
    ticker: str
    direction: str  # 'BUY', 'SELL'
    quantity: int
    order_type: str  # 'MARKET', 'LIMIT'
    limit_price: Optional[float] = None


@dataclass
class FillEvent(Event):
    """Order fill confirmation."""
    ticker: str
    direction: str
    quantity: int
    fill_price: float
    commission: float


class EventDrivenBacktester:
    """
    Event-driven backtesting engine.

    More realistic simulation of order flow and execution.
    """

    def __init__(self):
        self.event_queue = []
        self.handlers = {}

    def register_handler(
        self,
        event_type: str,
        handler: Callable
    ):
        """Register event handler."""
        self.handlers[event_type] = handler

    def run(self, events: List[Event]):
        """Process all events."""
        self.event_queue = sorted(events, key=lambda e: e.timestamp)

        while self.event_queue:
            event = self.event_queue.pop(0)

            handler = self.handlers.get(event.event_type)
            if handler:
                new_events = handler(event)
                if new_events:
                    self.event_queue.extend(new_events)
                    self.event_queue.sort(key=lambda e: e.timestamp)
```

---

## 3. Portfolio Construction Assumptions

### 3.1 Portfolio Construction Framework

```python
@dataclass
class PortfolioConfig:
    """Portfolio construction configuration."""

    # Capital
    initial_capital: float = 1_000_000
    max_gross_exposure: float = 1.0     # 100% invested

    # Position limits
    max_positions: int = 50
    min_position_weight: float = 0.01   # 1% minimum
    max_position_weight: float = 0.05   # 5% maximum

    # Risk limits
    max_sector_weight: float = 0.25     # 25% per sector
    max_single_stock_risk: float = 0.02 # 2% VaR per position

    # Turnover
    max_daily_turnover: float = 0.20    # 20% per day
    min_holding_period_days: int = 5    # Minimum holding

    # Sizing method
    sizing_method: str = 'equal_weight' # 'equal_weight', 'score_weighted', 'risk_parity'


class PortfolioConstructor:
    """
    Construct portfolio from signals with constraints.
    """

    def __init__(self, config: PortfolioConfig):
        self.config = config

    def construct(
        self,
        signals: pd.DataFrame,
        current_positions: Dict[str, float],
        date: datetime,
        risk_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Construct target portfolio.

        Args:
            signals: DataFrame with ticker, score, probability
            current_positions: Current holdings {ticker: shares}
            date: Current date
            risk_data: Optional risk metrics per ticker

        Returns:
            Target positions {ticker: shares}
        """
        # Filter to actionable signals
        candidates = signals[signals['signal'] == 'LONG'].copy()

        if len(candidates) == 0:
            return {}

        # Sort by score
        candidates = candidates.sort_values('score', ascending=False)

        # Limit to max positions
        candidates = candidates.head(self.config.max_positions)

        # Compute weights based on sizing method
        weights = self._compute_weights(candidates, risk_data)

        # Apply constraints
        weights = self._apply_constraints(weights, current_positions)

        # Convert to shares
        positions = self._weights_to_shares(weights, date)

        return positions

    def _compute_weights(
        self,
        candidates: pd.DataFrame,
        risk_data: Optional[Dict]
    ) -> Dict[str, float]:
        """Compute position weights."""
        weights = {}

        if self.config.sizing_method == 'equal_weight':
            n = len(candidates)
            for ticker in candidates['ticker']:
                weights[ticker] = 1.0 / n

        elif self.config.sizing_method == 'score_weighted':
            total_score = candidates['score'].sum()
            for _, row in candidates.iterrows():
                weights[row['ticker']] = row['score'] / total_score

        elif self.config.sizing_method == 'risk_parity':
            if risk_data is None:
                # Fall back to equal weight
                return self._compute_weights(candidates, None)

            # Inverse volatility weighting
            inv_vol = {}
            for ticker in candidates['ticker']:
                vol = risk_data.get(ticker, {}).get('volatility', 0.20)
                inv_vol[ticker] = 1.0 / vol

            total_inv_vol = sum(inv_vol.values())
            for ticker, iv in inv_vol.items():
                weights[ticker] = iv / total_inv_vol

        return weights

    def _apply_constraints(
        self,
        weights: Dict[str, float],
        current_positions: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply portfolio constraints."""
        # Clip individual positions
        for ticker in weights:
            weights[ticker] = np.clip(
                weights[ticker],
                self.config.min_position_weight,
                self.config.max_position_weight
            )

        # Renormalize to target exposure
        total = sum(weights.values())
        if total > 0:
            scale = self.config.max_gross_exposure / total
            weights = {k: v * scale for k, v in weights.items()}

        return weights

    def _weights_to_shares(
        self,
        weights: Dict[str, float],
        date: datetime
    ) -> Dict[str, float]:
        """Convert weights to share quantities."""
        positions = {}
        capital = self.config.initial_capital

        for ticker, weight in weights.items():
            target_value = capital * weight
            price = self.data.get_price(ticker, date)

            if price and price > 0:
                shares = int(target_value / price)
                positions[ticker] = shares

        return positions
```

### 3.2 Transaction Cost Model

```python
@dataclass
class TransactionCostConfig:
    """Transaction cost configuration."""

    # Commission
    commission_per_share: float = 0.005     # $0.005 per share
    min_commission: float = 1.0             # $1 minimum
    max_commission_pct: float = 0.01        # 1% maximum

    # Spread
    spread_bps: float = 5.0                 # 5 bps half-spread

    # Market impact
    impact_coefficient: float = 0.1         # Impact = coef * sqrt(shares/ADV)
    impact_exponent: float = 0.5

    # Slippage
    slippage_bps: float = 2.0               # 2 bps random slippage


class TransactionCostModel:
    """
    Model realistic transaction costs.
    """

    def __init__(self, config: TransactionCostConfig):
        self.config = config

    def estimate_cost(
        self,
        ticker: str,
        shares: int,
        price: float,
        adv: float,
        side: str  # 'BUY' or 'SELL'
    ) -> Dict[str, float]:
        """
        Estimate total transaction cost.

        Returns:
            Dict with breakdown of costs
        """
        trade_value = abs(shares * price)

        # Commission
        commission = max(
            self.config.min_commission,
            min(
                abs(shares) * self.config.commission_per_share,
                trade_value * self.config.max_commission_pct
            )
        )

        # Spread cost (half spread)
        spread_cost = trade_value * self.config.spread_bps / 10000

        # Market impact
        participation_rate = abs(shares) / adv if adv > 0 else 0
        impact_cost = (
            trade_value *
            self.config.impact_coefficient *
            (participation_rate ** self.config.impact_exponent)
        )

        # Random slippage
        slippage = trade_value * self.config.slippage_bps / 10000

        total_cost = commission + spread_cost + impact_cost + slippage

        return {
            'commission': commission,
            'spread': spread_cost,
            'impact': impact_cost,
            'slippage': slippage,
            'total': total_cost,
            'total_bps': total_cost / trade_value * 10000 if trade_value > 0 else 0
        }

    def get_execution_price(
        self,
        mid_price: float,
        side: str,
        cost_estimate: Dict[str, float]
    ) -> float:
        """
        Get effective execution price including costs.
        """
        spread_impact = cost_estimate['spread'] + cost_estimate['impact']
        price_impact_pct = spread_impact / (mid_price * 100)  # Approximate

        if side == 'BUY':
            return mid_price * (1 + price_impact_pct)
        else:
            return mid_price * (1 - price_impact_pct)
```

### 3.3 Liquidity Constraints

```python
class LiquidityModel:
    """
    Model liquidity constraints on trading.
    """

    def __init__(
        self,
        max_participation_rate: float = 0.10,  # Max 10% of ADV
        max_position_adv_days: float = 5.0     # Max 5 days to liquidate
    ):
        self.max_participation = max_participation_rate
        self.max_adv_days = max_position_adv_days

    def get_max_position(
        self,
        ticker: str,
        adv_shares: float,
        price: float
    ) -> float:
        """
        Get maximum position size given liquidity.
        """
        max_by_participation = adv_shares * self.max_participation
        max_by_liquidation = adv_shares * self.max_adv_days

        max_shares = min(max_by_participation, max_by_liquidation)
        max_value = max_shares * price

        return max_value

    def get_execution_days(
        self,
        shares: int,
        adv_shares: float
    ) -> float:
        """
        Estimate days needed to execute order.
        """
        if adv_shares == 0:
            return float('inf')

        daily_capacity = adv_shares * self.max_participation
        return abs(shares) / daily_capacity

    def can_execute_today(
        self,
        shares: int,
        adv_shares: float
    ) -> bool:
        """
        Check if order can be executed in one day.
        """
        return abs(shares) <= adv_shares * self.max_participation
```

---

## 4. Metrics

### 4.1 Performance Metrics

```python
@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    # Returns
    total_return: float
    cagr: float                     # Compound annual growth rate
    mtd_return: float               # Month to date
    ytd_return: float               # Year to date

    # Risk-adjusted
    sharpe_ratio: float             # Annualized
    sortino_ratio: float            # Downside deviation
    calmar_ratio: float             # CAGR / Max DD
    information_ratio: float        # vs benchmark

    # Risk
    volatility: float               # Annualized
    downside_deviation: float
    max_drawdown: float
    max_drawdown_duration_days: int
    avg_drawdown: float
    var_95: float                   # 95% Value at Risk
    cvar_95: float                  # Conditional VaR

    # Win/Loss
    win_rate: float                 # % profitable trades
    profit_factor: float            # Gross profit / Gross loss
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    expectancy: float               # Expected value per trade

    # Consistency
    pct_positive_months: float
    pct_positive_years: float
    best_month: float
    worst_month: float
    best_year: float
    worst_year: float

    # Activity
    total_trades: int
    avg_holding_period_days: float
    turnover_annual: float
    avg_positions: float


class MetricsCalculator:
    """
    Calculate comprehensive performance metrics.
    """

    TRADING_DAYS_PER_YEAR = 252

    def calculate(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.
        """
        returns = equity_curve['value'].pct_change().dropna()

        # Returns
        total_return = equity_curve['value'].iloc[-1] / equity_curve['value'].iloc[0] - 1
        n_years = len(returns) / self.TRADING_DAYS_PER_YEAR
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Risk
        volatility = returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Max drawdown duration
        dd_duration = self._calculate_dd_duration(drawdown)

        # Risk-adjusted
        risk_free_rate = 0.02  # Assume 2% risk-free
        excess_returns = returns - risk_free_rate / self.TRADING_DAYS_PER_YEAR
        sharpe_ratio = (
            excess_returns.mean() / returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)
            if returns.std() > 0 else 0
        )

        sortino_ratio = (
            excess_returns.mean() / downside_deviation * np.sqrt(self.TRADING_DAYS_PER_YEAR)
            if downside_deviation > 0 else 0
        )

        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        # VaR / CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()

        # Win/Loss
        if len(trades) > 0:
            trade_returns = trades['pnl_pct']
            wins = trade_returns[trade_returns > 0]
            losses = trade_returns[trade_returns <= 0]

            win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = losses.mean() if len(losses) > 0 else 0

            gross_profit = wins.sum() if len(wins) > 0 else 0
            gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')
            expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        else:
            win_rate = avg_win = avg_loss = profit_factor = win_loss_ratio = expectancy = 0

        # Monthly/Yearly
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        yearly_returns = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)

        pct_positive_months = (monthly_returns > 0).mean()
        pct_positive_years = (yearly_returns > 0).mean()

        # Information ratio
        if benchmark_returns is not None:
            excess = returns - benchmark_returns
            information_ratio = (
                excess.mean() / excess.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)
                if excess.std() > 0 else 0
            )
        else:
            information_ratio = 0

        return PerformanceMetrics(
            total_return=total_return,
            cagr=cagr,
            mtd_return=monthly_returns.iloc[-1] if len(monthly_returns) > 0 else 0,
            ytd_return=yearly_returns.iloc[-1] if len(yearly_returns) > 0 else 0,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            volatility=volatility,
            downside_deviation=downside_deviation,
            max_drawdown=max_drawdown,
            max_drawdown_duration_days=dd_duration,
            avg_drawdown=drawdown.mean(),
            var_95=var_95,
            cvar_95=cvar_95,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_loss_ratio=win_loss_ratio,
            expectancy=expectancy,
            pct_positive_months=pct_positive_months,
            pct_positive_years=pct_positive_years,
            best_month=monthly_returns.max() if len(monthly_returns) > 0 else 0,
            worst_month=monthly_returns.min() if len(monthly_returns) > 0 else 0,
            best_year=yearly_returns.max() if len(yearly_returns) > 0 else 0,
            worst_year=yearly_returns.min() if len(yearly_returns) > 0 else 0,
            total_trades=len(trades),
            avg_holding_period_days=trades['holding_days'].mean() if len(trades) > 0 else 0,
            turnover_annual=self._calculate_turnover(trades, n_years),
            avg_positions=equity_curve['positions'].mean()
        )

    def _calculate_dd_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        in_dd = drawdown < 0
        dd_groups = (in_dd != in_dd.shift()).cumsum()

        if not in_dd.any():
            return 0

        dd_durations = in_dd.groupby(dd_groups).sum()
        return int(dd_durations.max())

    def _calculate_turnover(
        self,
        trades: pd.DataFrame,
        n_years: float
    ) -> float:
        """Calculate annual turnover."""
        if len(trades) == 0 or n_years == 0:
            return 0

        total_traded = trades['trade_value'].sum()
        avg_capital = 1_000_000  # Placeholder
        return total_traded / avg_capital / n_years
```

### 4.2 Signal Quality Metrics

```python
@dataclass
class SignalMetrics:
    """Metrics for evaluating signal quality."""

    # Accuracy
    hit_rate: float                 # % of signals leading to profit
    precision: float                # True positive rate
    recall: float                   # Sensitivity

    # Calibration
    brier_score: float              # Probability calibration
    calibration_slope: float        # Ideal = 1.0
    calibration_intercept: float    # Ideal = 0.0

    # Information content
    roc_auc: float                  # Area under ROC curve
    avg_precision: float            # Area under PR curve
    information_coefficient: float  # Correlation with returns

    # Timing
    avg_days_to_peak: float         # Days until max gain
    avg_days_to_target: float       # Days to reach target

    # Score distribution
    score_mean: float
    score_std: float
    score_skew: float


class SignalMetricsCalculator:
    """
    Calculate signal quality metrics.
    """

    def calculate(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.DataFrame,
        target_threshold: float = 0.10
    ) -> SignalMetrics:
        """
        Calculate signal metrics.

        Args:
            signals: DataFrame with score, probability
            forward_returns: DataFrame with actual forward returns
            target_threshold: Threshold for "success"
        """
        # Merge signals with returns
        merged = signals.merge(
            forward_returns,
            on=['ticker', 'date'],
            how='inner'
        )

        if len(merged) == 0:
            return self._empty_metrics()

        scores = merged['score'].values
        probabilities = merged['probability'].values
        returns = merged['forward_return'].values
        success = (returns >= target_threshold).astype(int)

        # Hit rate
        high_score = scores >= 7.0
        hit_rate = (returns[high_score] > 0).mean() if high_score.any() else 0

        # ROC AUC
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

        roc_auc = roc_auc_score(success, probabilities) if len(np.unique(success)) > 1 else 0.5
        avg_precision = average_precision_score(success, probabilities) if len(np.unique(success)) > 1 else 0

        # Brier score
        brier = brier_score_loss(success, probabilities)

        # Calibration
        slope, intercept = self._calibration_regression(probabilities, success)

        # Information coefficient
        ic = np.corrcoef(scores, returns)[0, 1] if len(scores) > 1 else 0

        return SignalMetrics(
            hit_rate=hit_rate,
            precision=0,  # Would need threshold
            recall=0,
            brier_score=brier,
            calibration_slope=slope,
            calibration_intercept=intercept,
            roc_auc=roc_auc,
            avg_precision=avg_precision,
            information_coefficient=ic,
            avg_days_to_peak=0,  # Would need intraday data
            avg_days_to_target=0,
            score_mean=scores.mean(),
            score_std=scores.std(),
            score_skew=pd.Series(scores).skew()
        )

    def _calibration_regression(
        self,
        predicted: np.ndarray,
        actual: np.ndarray
    ) -> Tuple[float, float]:
        """Fit calibration regression."""
        from scipy.stats import linregress

        if len(predicted) < 10:
            return 1.0, 0.0

        result = linregress(predicted, actual)
        return result.slope, result.intercept

    def _empty_metrics(self) -> SignalMetrics:
        """Return empty metrics."""
        return SignalMetrics(
            hit_rate=0, precision=0, recall=0,
            brier_score=1.0, calibration_slope=0, calibration_intercept=0,
            roc_auc=0.5, avg_precision=0, information_coefficient=0,
            avg_days_to_peak=0, avg_days_to_target=0,
            score_mean=0, score_std=0, score_skew=0
        )
```

### 4.3 Statistical Significance Testing

```python
class StatisticalTests:
    """
    Statistical significance tests for backtest results.
    """

    @staticmethod
    def sharpe_significance(
        returns: pd.Series,
        null_sharpe: float = 0.0
    ) -> Dict[str, float]:
        """
        Test if Sharpe ratio is significantly different from null.

        Uses Lo (2002) adjustment for serial correlation.
        """
        n = len(returns)
        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            return {'p_value': 1.0, 'significant': False}

        # Annualized Sharpe
        sharpe = mean_return / std_return * np.sqrt(252)

        # Standard error (simplified)
        se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n) * np.sqrt(252)

        # Z-statistic
        z_stat = (sharpe - null_sharpe) / se_sharpe

        # P-value (one-tailed)
        from scipy.stats import norm
        p_value = 1 - norm.cdf(z_stat)

        return {
            'sharpe': sharpe,
            'standard_error': se_sharpe,
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant_5pct': p_value < 0.05,
            'significant_1pct': p_value < 0.01
        }

    @staticmethod
    def bootstrap_confidence_interval(
        returns: pd.Series,
        metric_func: Callable,
        n_bootstrap: int = 10000,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Bootstrap confidence interval for any metric.
        """
        n = len(returns)
        bootstrap_values = []

        for _ in range(n_bootstrap):
            sample = returns.sample(n=n, replace=True)
            value = metric_func(sample)
            bootstrap_values.append(value)

        bootstrap_values = np.array(bootstrap_values)

        alpha = (1 - confidence) / 2
        ci_lower = np.percentile(bootstrap_values, alpha * 100)
        ci_upper = np.percentile(bootstrap_values, (1 - alpha) * 100)

        return {
            'point_estimate': metric_func(returns),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
            'standard_error': bootstrap_values.std()
        }

    @staticmethod
    def multiple_testing_correction(
        p_values: List[float],
        method: str = 'fdr_bh'
    ) -> List[float]:
        """
        Correct for multiple testing.

        Methods:
        - 'bonferroni': Conservative
        - 'fdr_bh': Benjamini-Hochberg FDR
        """
        from scipy.stats import false_discovery_control

        if method == 'bonferroni':
            n = len(p_values)
            return [min(p * n, 1.0) for p in p_values]
        elif method == 'fdr_bh':
            return list(false_discovery_control(p_values))
        else:
            return p_values

    @staticmethod
    def deflated_sharpe_ratio(
        sharpe: float,
        n_trials: int,
        variance_sharpe: float,
        skewness: float = 0,
        kurtosis: float = 3
    ) -> float:
        """
        Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

        Adjusts for multiple testing / strategy selection bias.
        """
        from scipy.stats import norm

        # Expected maximum Sharpe under null
        e_max_sharpe = (
            (1 - np.euler_gamma) * norm.ppf(1 - 1/n_trials) +
            np.euler_gamma * norm.ppf(1 - 1/(n_trials * np.e))
        ) * np.sqrt(variance_sharpe)

        # PSR (Probabilistic Sharpe Ratio)
        psr = norm.cdf(
            (sharpe - e_max_sharpe) /
            np.sqrt(variance_sharpe * (1 + 0.5 * sharpe**2))
        )

        return psr
```

---

## 5. Calibration Analysis

### 5.1 Probability Calibration

```python
class CalibrationAnalyzer:
    """
    Analyze probability calibration of predictions.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def compute_calibration_curve(
        self,
        predicted_proba: np.ndarray,
        actual_outcomes: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute calibration curve.

        Returns:
            DataFrame with bin stats
        """
        # Create bins
        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(predicted_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        results = []
        for i in range(self.n_bins):
            mask = bin_indices == i

            if mask.sum() == 0:
                continue

            bin_predictions = predicted_proba[mask]
            bin_outcomes = actual_outcomes[mask]

            results.append({
                'bin': i,
                'bin_lower': bins[i],
                'bin_upper': bins[i + 1],
                'mean_predicted': bin_predictions.mean(),
                'mean_actual': bin_outcomes.mean(),
                'count': mask.sum(),
                'calibration_error': abs(bin_predictions.mean() - bin_outcomes.mean())
            })

        return pd.DataFrame(results)

    def compute_ece(
        self,
        predicted_proba: np.ndarray,
        actual_outcomes: np.ndarray
    ) -> float:
        """
        Expected Calibration Error.

        Weighted average of per-bin calibration errors.
        """
        curve = self.compute_calibration_curve(predicted_proba, actual_outcomes)

        if len(curve) == 0:
            return 0.0

        total_samples = curve['count'].sum()
        ece = (curve['calibration_error'] * curve['count']).sum() / total_samples

        return ece

    def compute_mce(
        self,
        predicted_proba: np.ndarray,
        actual_outcomes: np.ndarray
    ) -> float:
        """
        Maximum Calibration Error.

        Worst-case bin error.
        """
        curve = self.compute_calibration_curve(predicted_proba, actual_outcomes)

        if len(curve) == 0:
            return 0.0

        return curve['calibration_error'].max()

    def reliability_diagram(
        self,
        predicted_proba: np.ndarray,
        actual_outcomes: np.ndarray
    ) -> Dict:
        """
        Data for reliability diagram visualization.
        """
        curve = self.compute_calibration_curve(predicted_proba, actual_outcomes)

        return {
            'curve': curve,
            'ece': self.compute_ece(predicted_proba, actual_outcomes),
            'mce': self.compute_mce(predicted_proba, actual_outcomes),
            'ideal_line': {'x': [0, 1], 'y': [0, 1]}
        }


class ScoreCalibrationAnalyzer:
    """
    Analyze calibration of 0-10 scores.
    """

    def compute_score_vs_return(
        self,
        scores: np.ndarray,
        returns: np.ndarray,
        score_bins: List[float] = None
    ) -> pd.DataFrame:
        """
        Analyze actual returns by score bucket.
        """
        if score_bins is None:
            score_bins = [0, 2, 4, 6, 8, 10]

        bin_labels = [f'{score_bins[i]}-{score_bins[i+1]}'
                      for i in range(len(score_bins) - 1)]

        score_buckets = pd.cut(
            scores,
            bins=score_bins,
            labels=bin_labels,
            include_lowest=True
        )

        results = []
        for bucket in bin_labels:
            mask = score_buckets == bucket

            if mask.sum() == 0:
                continue

            bucket_returns = returns[mask]

            results.append({
                'score_bucket': bucket,
                'count': mask.sum(),
                'mean_return': bucket_returns.mean(),
                'median_return': np.median(bucket_returns),
                'std_return': bucket_returns.std(),
                'win_rate': (bucket_returns > 0).mean(),
                'high_gain_rate': (bucket_returns >= 0.10).mean()
            })

        return pd.DataFrame(results)

    def score_monotonicity_test(
        self,
        scores: np.ndarray,
        returns: np.ndarray
    ) -> Dict:
        """
        Test if higher scores lead to higher returns.

        Ideally: Score 9-10 > 7-8 > 5-6 > 3-4 > 0-2
        """
        analysis = self.compute_score_vs_return(scores, returns)

        if len(analysis) < 2:
            return {'monotonic': True, 'violations': 0}

        # Check monotonicity
        mean_returns = analysis['mean_return'].values
        violations = sum(
            mean_returns[i] < mean_returns[i + 1]
            for i in range(len(mean_returns) - 1)
        )

        # Spearman correlation for monotonicity
        from scipy.stats import spearmanr
        correlation, p_value = spearmanr(scores, returns)

        return {
            'monotonic': violations == 0,
            'violations': violations,
            'spearman_correlation': correlation,
            'spearman_p_value': p_value,
            'analysis': analysis
        }
```

### 5.2 Regime-Conditional Performance

```python
class RegimeAnalyzer:
    """
    Analyze performance across different market regimes.
    """

    def __init__(self):
        self.regimes = ['BULL', 'BEAR', 'HIGH_VOL', 'LOW_VOL', 'CRISIS']

    def classify_regime(
        self,
        date: datetime,
        market_returns: pd.Series,
        market_volatility: pd.Series
    ) -> str:
        """
        Classify market regime for a date.
        """
        # 3-month lookback
        lookback = 63

        recent_return = market_returns.rolling(lookback).sum().loc[date]
        recent_vol_pctl = market_volatility.rank(pct=True).loc[date]

        if recent_vol_pctl > 0.9:
            return 'CRISIS'
        elif recent_vol_pctl > 0.75:
            return 'HIGH_VOL'
        elif recent_vol_pctl < 0.25:
            return 'LOW_VOL'
        elif recent_return > 0.05:
            return 'BULL'
        elif recent_return < -0.05:
            return 'BEAR'
        else:
            return 'NEUTRAL'

    def performance_by_regime(
        self,
        returns: pd.Series,
        regime_labels: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate performance metrics per regime.
        """
        results = []

        for regime in regime_labels.unique():
            mask = regime_labels == regime
            regime_returns = returns[mask]

            if len(regime_returns) < 20:
                continue

            metrics = {
                'regime': regime,
                'n_days': len(regime_returns),
                'pct_of_sample': mask.mean(),
                'mean_return': regime_returns.mean() * 252,
                'volatility': regime_returns.std() * np.sqrt(252),
                'sharpe': (regime_returns.mean() / regime_returns.std() * np.sqrt(252)
                          if regime_returns.std() > 0 else 0),
                'win_rate': (regime_returns > 0).mean(),
                'max_drawdown': self._max_drawdown(regime_returns)
            }

            results.append(metrics)

        return pd.DataFrame(results)

    def _max_drawdown(self, returns: pd.Series) -> float:
        """Calculate max drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def regime_transition_analysis(
        self,
        returns: pd.Series,
        regime_labels: pd.Series
    ) -> pd.DataFrame:
        """
        Analyze performance during regime transitions.
        """
        transitions = (regime_labels != regime_labels.shift()).cumsum()
        transition_points = regime_labels.groupby(transitions).first()

        results = []
        for i in range(1, len(transition_points)):
            from_regime = transition_points.iloc[i - 1]
            to_regime = transition_points.iloc[i]

            # Performance in days after transition
            transition_idx = (transitions == i).idxmax()
            post_transition = returns.loc[transition_idx:].head(20)

            if len(post_transition) < 5:
                continue

            results.append({
                'from_regime': from_regime,
                'to_regime': to_regime,
                'n_transitions': 1,
                'post_transition_return': post_transition.sum(),
                'post_transition_vol': post_transition.std() * np.sqrt(252)
            })

        return pd.DataFrame(results).groupby(['from_regime', 'to_regime']).agg({
            'n_transitions': 'sum',
            'post_transition_return': 'mean',
            'post_transition_vol': 'mean'
        }).reset_index()
```

---

## 6. Failure Modes

### 6.1 Failure Mode Catalog

```python
@dataclass
class FailureMode:
    """Description of a known failure mode."""
    name: str
    description: str
    detection_method: str
    mitigation: str
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'


FAILURE_MODE_CATALOG = {
    # Data-related failures
    'survivorship_bias': FailureMode(
        name="Survivorship Bias",
        description="Only testing on stocks that survived, excluding delisted/bankrupt",
        detection_method="Compare universe size over time; check for delisting events",
        mitigation="Use point-in-time universe membership; include delisted stocks",
        severity="HIGH"
    ),

    'lookahead_bias': FailureMode(
        name="Lookahead Bias",
        description="Using future information in features or decisions",
        detection_method="Leakage detection tests; review feature timestamps",
        mitigation="Strict point-in-time data access; embargo periods",
        severity="CRITICAL"
    ),

    'stale_data': FailureMode(
        name="Stale Data",
        description="Using outdated data that wouldn't be available in real-time",
        detection_method="Check data freshness vs. assumed availability",
        mitigation="Model realistic data delays (fundamental lag, etc.)",
        severity="MEDIUM"
    ),

    # Model-related failures
    'overfitting': FailureMode(
        name="Overfitting",
        description="Model too complex for data; poor out-of-sample performance",
        detection_method="Train/validation gap; cross-validation variance",
        mitigation="Regularization; simpler models; more data",
        severity="HIGH"
    ),

    'regime_change': FailureMode(
        name="Regime Change",
        description="Model trained on different market regime than current",
        detection_method="Performance by regime; drift detection",
        mitigation="Regime-conditional models; adaptive retraining",
        severity="HIGH"
    ),

    'parameter_sensitivity': FailureMode(
        name="Parameter Sensitivity",
        description="Small parameter changes cause large performance changes",
        detection_method="Parameter sweep analysis; sensitivity testing",
        mitigation="Robust parameters; ensemble across parameters",
        severity="MEDIUM"
    ),

    # Execution-related failures
    'liquidity_assumption': FailureMode(
        name="Unrealistic Liquidity",
        description="Assuming trades execute at unrealistic prices/sizes",
        detection_method="Compare trade sizes vs. ADV; check small-cap exposure",
        mitigation="Liquidity constraints; impact modeling",
        severity="MEDIUM"
    ),

    'transaction_costs': FailureMode(
        name="Understated Transaction Costs",
        description="Not accounting for full execution costs",
        detection_method="Compare simulated vs. realistic cost estimates",
        mitigation="Conservative cost assumptions; sensitivity analysis",
        severity="MEDIUM"
    ),

    # Strategy-related failures
    'crowding': FailureMode(
        name="Strategy Crowding",
        description="Strategy capacity exhausted due to similar strategies",
        detection_method="Factor exposure analysis; correlation with known factors",
        mitigation="Capacity constraints; differentiated signals",
        severity="MEDIUM"
    ),

    'decay': FailureMode(
        name="Alpha Decay",
        description="Signal loses effectiveness over time",
        detection_method="Rolling performance analysis; compare recent vs. historical",
        mitigation="Continuous retraining; new signal research",
        severity="HIGH"
    ),

    'concentration': FailureMode(
        name="Concentration Risk",
        description="Performance driven by few large positions",
        detection_method="Hit rate analysis; position-level attribution",
        mitigation="Position limits; diversification requirements",
        severity="MEDIUM"
    ),

    # Statistical failures
    'multiple_testing': FailureMode(
        name="Multiple Testing Bias",
        description="Many strategies tested, only best shown",
        detection_method="Deflated Sharpe ratio; track all experiments",
        mitigation="Pre-registration; out-of-sample holdout",
        severity="HIGH"
    ),

    'small_sample': FailureMode(
        name="Small Sample Size",
        description="Insufficient data for reliable inference",
        detection_method="Confidence interval width; significance tests",
        mitigation="Longer history; bootstrap analysis",
        severity="MEDIUM"
    )
}
```

### 6.2 Failure Detection System

```python
class FailureDetector:
    """
    Detect potential failure modes in backtest results.
    """

    def __init__(self, backtest_results: 'BacktestResults'):
        self.results = backtest_results
        self.detected_failures = []

    def run_all_checks(self) -> List[Dict]:
        """Run all failure detection checks."""
        self.detected_failures = []

        self._check_survivorship_bias()
        self._check_lookahead_indicators()
        self._check_overfitting()
        self._check_regime_sensitivity()
        self._check_liquidity_realism()
        self._check_concentration_risk()
        self._check_statistical_significance()
        self._check_alpha_decay()

        return self.detected_failures

    def _check_survivorship_bias(self):
        """Check for survivorship bias indicators."""
        # Check if universe size decreases backward in time (suspicious)
        universe_sizes = self.results.equity_curve.groupby(
            pd.Grouper(key='date', freq='M')
        )['positions'].mean()

        if len(universe_sizes) > 12:
            early_avg = universe_sizes.head(6).mean()
            late_avg = universe_sizes.tail(6).mean()

            if early_avg > late_avg * 1.2:
                self.detected_failures.append({
                    'failure_mode': 'survivorship_bias',
                    'severity': 'HIGH',
                    'evidence': f'Early avg positions: {early_avg:.1f}, Late: {late_avg:.1f}',
                    'recommendation': 'Verify point-in-time universe membership'
                })

    def _check_lookahead_indicators(self):
        """Check for suspicious performance patterns indicating lookahead."""
        metrics = self.results.metrics

        # Suspiciously high Sharpe
        if metrics.sharpe_ratio > 3.0:
            self.detected_failures.append({
                'failure_mode': 'lookahead_bias',
                'severity': 'CRITICAL',
                'evidence': f'Sharpe ratio of {metrics.sharpe_ratio:.2f} is suspiciously high',
                'recommendation': 'Review data timestamps and feature construction'
            })

        # Perfect or near-perfect win rate
        if metrics.win_rate > 0.85:
            self.detected_failures.append({
                'failure_mode': 'lookahead_bias',
                'severity': 'HIGH',
                'evidence': f'Win rate of {metrics.win_rate:.1%} is unusually high',
                'recommendation': 'Check for target leakage in features'
            })

    def _check_overfitting(self):
        """Check for overfitting indicators."""
        # Compare in-sample vs out-of-sample if available
        if hasattr(self.results, 'cv_results'):
            is_sharpe = self.results.cv_results.get('train_sharpe', 0)
            oos_sharpe = self.results.cv_results.get('test_sharpe', 0)

            if is_sharpe > 0 and oos_sharpe / is_sharpe < 0.5:
                self.detected_failures.append({
                    'failure_mode': 'overfitting',
                    'severity': 'HIGH',
                    'evidence': f'Train Sharpe: {is_sharpe:.2f}, Test: {oos_sharpe:.2f}',
                    'recommendation': 'Simplify model or increase regularization'
                })

    def _check_regime_sensitivity(self):
        """Check for regime-dependent performance."""
        if hasattr(self.results, 'regime_analysis'):
            regimes = self.results.regime_analysis

            sharpes = regimes['sharpe'].values
            if len(sharpes) >= 3:
                if sharpes.max() > 0 and sharpes.min() < -0.5:
                    self.detected_failures.append({
                        'failure_mode': 'regime_change',
                        'severity': 'HIGH',
                        'evidence': f'Sharpe range: {sharpes.min():.2f} to {sharpes.max():.2f}',
                        'recommendation': 'Consider regime-conditional signals'
                    })

    def _check_liquidity_realism(self):
        """Check if trades are realistic given liquidity."""
        trades = self.results.trades

        if len(trades) == 0:
            return

        # Check for trades exceeding 10% of ADV
        if 'pct_of_adv' in trades.columns:
            high_impact = trades[trades['pct_of_adv'] > 0.10]

            if len(high_impact) > len(trades) * 0.05:
                self.detected_failures.append({
                    'failure_mode': 'liquidity_assumption',
                    'severity': 'MEDIUM',
                    'evidence': f'{len(high_impact)} trades exceed 10% of ADV',
                    'recommendation': 'Add liquidity constraints'
                })

    def _check_concentration_risk(self):
        """Check for excessive concentration in winners."""
        trades = self.results.trades

        if len(trades) < 20:
            return

        # Check if top 10% of trades drive most profit
        trades_sorted = trades.sort_values('pnl', ascending=False)
        top_10_pct = trades_sorted.head(int(len(trades) * 0.1))

        total_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        top_profit = top_10_pct[top_10_pct['pnl'] > 0]['pnl'].sum()

        if total_profit > 0 and top_profit / total_profit > 0.8:
            self.detected_failures.append({
                'failure_mode': 'concentration',
                'severity': 'MEDIUM',
                'evidence': f'Top 10% trades = {top_profit/total_profit:.1%} of profit',
                'recommendation': 'Analyze concentration; consider position limits'
            })

    def _check_statistical_significance(self):
        """Check statistical significance of results."""
        returns = self.results.daily_returns['return']

        sig_test = StatisticalTests.sharpe_significance(returns)

        if not sig_test['significant_5pct']:
            self.detected_failures.append({
                'failure_mode': 'small_sample',
                'severity': 'MEDIUM',
                'evidence': f'Sharpe p-value: {sig_test["p_value"]:.3f}',
                'recommendation': 'Results not statistically significant at 5%'
            })

    def _check_alpha_decay(self):
        """Check for alpha decay over time."""
        returns = self.results.daily_returns.set_index('date')['return']

        # Split into halves
        midpoint = len(returns) // 2
        first_half = returns.iloc[:midpoint]
        second_half = returns.iloc[midpoint:]

        sharpe_1 = first_half.mean() / first_half.std() * np.sqrt(252) if first_half.std() > 0 else 0
        sharpe_2 = second_half.mean() / second_half.std() * np.sqrt(252) if second_half.std() > 0 else 0

        if sharpe_1 > 0.5 and sharpe_2 < sharpe_1 * 0.5:
            self.detected_failures.append({
                'failure_mode': 'decay',
                'severity': 'HIGH',
                'evidence': f'First half Sharpe: {sharpe_1:.2f}, Second half: {sharpe_2:.2f}',
                'recommendation': 'Signal may be decaying; investigate cause'
            })

    def generate_report(self) -> str:
        """Generate failure analysis report."""
        if not self.detected_failures:
            return "No failure modes detected."

        lines = ["## Failure Mode Analysis\n"]

        # Group by severity
        by_severity = {}
        for failure in self.detected_failures:
            sev = failure['severity']
            if sev not in by_severity:
                by_severity[sev] = []
            by_severity[sev].append(failure)

        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            if severity not in by_severity:
                continue

            lines.append(f"\n### {severity} Severity\n")

            for failure in by_severity[severity]:
                lines.append(f"**{failure['failure_mode']}**")
                lines.append(f"- Evidence: {failure['evidence']}")
                lines.append(f"- Recommendation: {failure['recommendation']}")
                lines.append("")

        return "\n".join(lines)
```

### 6.3 Robustness Testing

```python
class RobustnessAnalyzer:
    """
    Test strategy robustness to perturbations.
    """

    def parameter_sensitivity(
        self,
        base_config: WalkForwardConfig,
        parameter: str,
        values: List,
        backtester_factory: Callable
    ) -> pd.DataFrame:
        """
        Test sensitivity to parameter changes.
        """
        results = []

        for value in values:
            config = copy.deepcopy(base_config)
            setattr(config, parameter, value)

            backtester = backtester_factory(config)
            result = backtester.run()

            results.append({
                parameter: value,
                'sharpe': result.metrics.sharpe_ratio,
                'cagr': result.metrics.cagr,
                'max_dd': result.metrics.max_drawdown
            })

        return pd.DataFrame(results)

    def transaction_cost_sensitivity(
        self,
        backtester: WalkForwardBacktester,
        cost_multipliers: List[float] = [0.5, 1.0, 1.5, 2.0, 3.0]
    ) -> pd.DataFrame:
        """
        Test sensitivity to transaction costs.
        """
        results = []

        for mult in cost_multipliers:
            # Adjust cost model
            original_costs = backtester.execution.cost_model.config
            adjusted_config = TransactionCostConfig(
                commission_per_share=original_costs.commission_per_share * mult,
                spread_bps=original_costs.spread_bps * mult,
                impact_coefficient=original_costs.impact_coefficient * mult
            )

            backtester.execution.cost_model = TransactionCostModel(adjusted_config)
            result = backtester.run()

            results.append({
                'cost_multiplier': mult,
                'sharpe': result.metrics.sharpe_ratio,
                'cagr': result.metrics.cagr,
                'net_return': result.metrics.total_return
            })

        return pd.DataFrame(results)

    def time_period_stability(
        self,
        returns: pd.Series,
        window_years: int = 3
    ) -> pd.DataFrame:
        """
        Analyze performance stability across rolling windows.
        """
        window_days = window_years * 252
        results = []

        for i in range(0, len(returns) - window_days, 63):  # Quarterly steps
            window = returns.iloc[i:i + window_days]

            sharpe = window.mean() / window.std() * np.sqrt(252) if window.std() > 0 else 0

            results.append({
                'start_date': window.index[0],
                'end_date': window.index[-1],
                'sharpe': sharpe,
                'return': (1 + window).prod() - 1,
                'volatility': window.std() * np.sqrt(252),
                'max_dd': self._max_drawdown(window)
            })

        df = pd.DataFrame(results)

        # Add stability metrics
        df.attrs['sharpe_mean'] = df['sharpe'].mean()
        df.attrs['sharpe_std'] = df['sharpe'].std()
        df.attrs['pct_positive_sharpe'] = (df['sharpe'] > 0).mean()

        return df

    def _max_drawdown(self, returns: pd.Series) -> float:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
```

---

## 7. Backtest Results Container

```python
@dataclass
class BacktestResults:
    """Complete backtest results container."""

    config: WalkForwardConfig
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    daily_returns: pd.DataFrame
    model_versions: List[Dict]

    # Computed on demand
    _metrics: Optional[PerformanceMetrics] = None
    _signal_metrics: Optional[SignalMetrics] = None
    _regime_analysis: Optional[pd.DataFrame] = None
    _failure_analysis: Optional[List[Dict]] = None

    @property
    def metrics(self) -> PerformanceMetrics:
        if self._metrics is None:
            calculator = MetricsCalculator()
            self._metrics = calculator.calculate(
                self.equity_curve,
                self.trades
            )
        return self._metrics

    @property
    def regime_analysis(self) -> pd.DataFrame:
        if self._regime_analysis is None:
            analyzer = RegimeAnalyzer()
            # Would need market data for full analysis
            self._regime_analysis = pd.DataFrame()
        return self._regime_analysis

    def to_report(self) -> str:
        """Generate comprehensive backtest report."""
        lines = [
            "# Backtest Report",
            f"\n## Configuration",
            f"- Period: {self.config.start_date} to {self.config.end_date}",
            f"- Universe: {self.config.universe}",
            f"- Rebalance: {self.config.rebalance_frequency}",
            "",
            "## Performance Summary",
            f"- Total Return: {self.metrics.total_return:.2%}",
            f"- CAGR: {self.metrics.cagr:.2%}",
            f"- Sharpe Ratio: {self.metrics.sharpe_ratio:.2f}",
            f"- Max Drawdown: {self.metrics.max_drawdown:.2%}",
            f"- Win Rate: {self.metrics.win_rate:.1%}",
            "",
            "## Risk Metrics",
            f"- Volatility: {self.metrics.volatility:.2%}",
            f"- Sortino Ratio: {self.metrics.sortino_ratio:.2f}",
            f"- Calmar Ratio: {self.metrics.calmar_ratio:.2f}",
            f"- 95% VaR: {self.metrics.var_95:.2%}",
            "",
            "## Activity",
            f"- Total Trades: {self.metrics.total_trades}",
            f"- Avg Holding Period: {self.metrics.avg_holding_period_days:.1f} days",
            f"- Annual Turnover: {self.metrics.turnover_annual:.1%}",
        ]

        return "\n".join(lines)

    def save(self, path: str):
        """Save results to disk."""
        import pickle

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'BacktestResults':
        """Load results from disk."""
        import pickle

        with open(path, 'rb') as f:
            return pickle.load(f)
```

---

## 8. Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BACKTESTING FRAMEWORK SUMMARY                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WALK-FORWARD MECHANICS                                                    │
│  ──────────────────────                                                     │
│  • Initial training: 3 years | Retrain: Monthly | Embargo: 21 days         │
│  • Expanding window by default                                             │
│  • Event-driven simulation option                                          │
│                                                                             │
│  PORTFOLIO ASSUMPTIONS                                                     │
│  ─────────────────────                                                      │
│  • Max positions: 50 | Position size: 1-5%                                 │
│  • Sector limit: 25% | Max turnover: 20%/day                               │
│  • Transaction costs: Commission + Spread + Impact + Slippage              │
│  • Liquidity: Max 10% of ADV participation                                 │
│                                                                             │
│  METRICS                                                                   │
│  ───────                                                                    │
│  Performance: CAGR, Sharpe, Sortino, Calmar, Max DD, VaR                   │
│  Win/Loss: Win rate, profit factor, expectancy                             │
│  Signal: ROC-AUC, calibration, IC                                          │
│  Statistical: Significance tests, bootstrap CIs, deflated Sharpe           │
│                                                                             │
│  CALIBRATION                                                               │
│  ───────────                                                                │
│  • Probability calibration curves                                          │
│  • ECE/MCE metrics                                                         │
│  • Score vs. return monotonicity                                           │
│  • Regime-conditional performance                                          │
│                                                                             │
│  FAILURE MODES                                                             │
│  ─────────────                                                              │
│  • Survivorship, lookahead, overfitting                                    │
│  • Regime change, alpha decay, crowding                                    │
│  • Liquidity, costs, concentration                                         │
│  • Multiple testing, small sample                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Assumptions

1. **Market Efficiency:** Reasonable transaction costs capture market impact
2. **Execution:** Orders execute at realistic prices with modeled slippage
3. **Data Quality:** Historical data is accurate and adjusted correctly
4. **Universe Stability:** Point-in-time universe membership is available
5. **Stationarity:** Sufficient regime stability for model learning
6. **Capacity:** Strategy can deploy meaningful capital without degradation
7. **Independence:** Trades are approximately independent for statistics

---

*Document Version: 1.0*
*Last Updated: 2026-01-18*
