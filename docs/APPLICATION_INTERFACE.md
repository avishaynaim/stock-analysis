# Application Interface & Execution Flow

## 1. Overview

This document defines the command-line interface, configuration system, logging framework, and parallelization strategy for the stock analysis system.

### Design Principles

1. **Configuration as Code:** All settings in version-controlled files
2. **Reproducibility:** Every run can be exactly replicated
3. **Progressive Disclosure:** Simple defaults, deep customization available
4. **Fail-Fast:** Validate inputs early, provide clear error messages
5. **Observable:** Comprehensive logging and progress reporting

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       APPLICATION INTERFACE ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ CLI LAYER                                                           │   │
│  │                                                                     │   │
│  │  stock-analysis <command> [options]                                 │   │
│  │                                                                     │   │
│  │  Commands: analyze, scan, backtest, train, export, config           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ CONFIGURATION LAYER                                                 │   │
│  │                                                                     │   │
│  │  config.yaml → Environment → CLI args (precedence order)            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ EXECUTION ENGINE                                                    │   │
│  │                                                                     │   │
│  │  • Job orchestration       • Progress tracking                      │   │
│  │  • Parallel execution      • Resource management                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ OUTPUT LAYER                                                        │   │
│  │                                                                     │   │
│  │  • Logging (file + console)    • Reports (JSON, CSV, HTML)         │   │
│  │  • Progress bars               • Alerts                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. CLI Commands

### 2.1 Command Structure

```python
"""
stock-analysis CLI structure.

Usage:
    stock-analysis <command> [<args>...]
    stock-analysis --version
    stock-analysis --help

Commands:
    analyze     Analyze a single ticker or list of tickers
    scan        Scan full universe for opportunities
    backtest    Run backtesting simulation
    train       Train or retrain ML models
    export      Export data or results
    config      Manage configuration
    data        Data management commands
    report      Generate reports
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import click


class OutputFormat(Enum):
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    TEXT = "text"


class Verbosity(Enum):
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3


# ============================================================================
# Main CLI Group
# ============================================================================

@click.group()
@click.version_option(version="1.0.0")
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to configuration file')
@click.option('--verbose', '-v', count=True,
              help='Increase verbosity (-v, -vv, -vvv)')
@click.option('--quiet', '-q', is_flag=True,
              help='Suppress non-essential output')
@click.option('--log-file', type=click.Path(),
              help='Path to log file')
@click.pass_context
def cli(ctx, config, verbose, quiet, log_file):
    """
    Stock Analysis System - Research-grade equity analysis toolkit.

    Run 'stock-analysis <command> --help' for command-specific help.
    """
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbosity'] = Verbosity.QUIET if quiet else Verbosity(min(verbose, 3))
    ctx.obj['log_file'] = log_file
```

### 2.2 Analyze Command

```python
@cli.command()
@click.argument('tickers', nargs=-1, required=True)
@click.option('--horizon', '-h',
              type=click.Choice(['1d', '5d', '21d', '63d', '252d', 'all']),
              default='all',
              help='Analysis horizon')
@click.option('--as-of', type=click.DateTime(),
              help='Point-in-time analysis date (default: latest)')
@click.option('--output', '-o', type=click.Path(),
              help='Output file path')
@click.option('--format', '-f',
              type=click.Choice(['json', 'csv', 'html', 'text']),
              default='text',
              help='Output format')
@click.option('--include-indicators/--no-indicators',
              default=True,
              help='Include indicator breakdown')
@click.option('--include-probability/--no-probability',
              default=True,
              help='Include probability estimates')
@click.option('--compare-benchmark', type=str,
              help='Compare against benchmark (e.g., SPY)')
@click.pass_context
def analyze(ctx, tickers, horizon, as_of, output, format,
            include_indicators, include_probability, compare_benchmark):
    """
    Analyze one or more tickers.

    Examples:
        stock-analysis analyze AAPL
        stock-analysis analyze AAPL MSFT GOOGL --horizon 21d
        stock-analysis analyze AAPL --as-of 2024-01-15 -o report.json -f json
    """
    from stock_analysis.commands import AnalyzeCommand

    cmd = AnalyzeCommand(
        tickers=list(tickers),
        horizon=horizon,
        as_of_date=as_of,
        output_path=output,
        output_format=OutputFormat(format),
        include_indicators=include_indicators,
        include_probability=include_probability,
        benchmark=compare_benchmark,
        config_path=ctx.obj['config_path'],
        verbosity=ctx.obj['verbosity']
    )

    result = cmd.execute()
    cmd.display(result)


# Example usage and output:
"""
$ stock-analysis analyze AAPL

═══════════════════════════════════════════════════════════════════════════════
                              AAPL - Apple Inc.
═══════════════════════════════════════════════════════════════════════════════

Overall Score: 7.8 / 10  [STRONG]
───────────────────────────────────────────────────────────────────────────────

SUBSCORES
─────────
  Trend:        8.2  ████████░░  Strong uptrend across timeframes
  Momentum:     7.5  ███████░░░  Positive momentum, not overbought
  Volume:       6.8  ██████░░░░  Above average volume
  Rel Strength: 8.5  ████████░░  Outperforming sector and market
  Fundamental:  7.2  ███████░░░  Solid fundamentals
  Edge:         8.0  ████████░░  High probability of gain

PROBABILITY ESTIMATES (21-day horizon)
──────────────────────────────────────
  P(gain > 5%):   62.3%  [Above average]
  P(gain > 10%):  28.7%  [Moderate]
  Confidence:     High (n=847 similar states)

RISK FLAGS
──────────
  ⚠ Elevated sector volatility (VIX > 20)

───────────────────────────────────────────────────────────────────────────────
Analysis as of: 2024-01-18 16:00 ET
"""
```

### 2.3 Scan Command

```python
@cli.command()
@click.option('--universe', '-u',
              type=click.Choice(['SP500', 'SP400', 'SP600', 'RUSSELL1000',
                               'RUSSELL2000', 'NASDAQ100', 'custom']),
              default='SP500',
              help='Universe to scan')
@click.option('--custom-universe', type=click.Path(exists=True),
              help='Path to custom universe file (CSV with ticker column)')
@click.option('--min-score', type=float, default=7.0,
              help='Minimum score threshold')
@click.option('--max-results', '-n', type=int, default=50,
              help='Maximum results to return')
@click.option('--sort-by',
              type=click.Choice(['score', 'probability', 'momentum',
                               'relative_strength', 'volume']),
              default='score',
              help='Sort results by metric')
@click.option('--horizon', '-h',
              type=click.Choice(['5d', '21d', '63d']),
              default='21d',
              help='Target horizon')
@click.option('--sector', type=str, multiple=True,
              help='Filter by sector(s)')
@click.option('--min-price', type=float, default=5.0,
              help='Minimum stock price')
@click.option('--min-volume', type=float, default=500000,
              help='Minimum daily dollar volume')
@click.option('--output', '-o', type=click.Path(),
              help='Output file path')
@click.option('--format', '-f',
              type=click.Choice(['json', 'csv', 'html', 'text']),
              default='text',
              help='Output format')
@click.option('--parallel/--no-parallel', default=True,
              help='Use parallel processing')
@click.option('--workers', '-w', type=int, default=None,
              help='Number of parallel workers (default: auto)')
@click.pass_context
def scan(ctx, universe, custom_universe, min_score, max_results, sort_by,
         horizon, sector, min_price, min_volume, output, format,
         parallel, workers):
    """
    Scan universe for high-scoring opportunities.

    Examples:
        stock-analysis scan --universe SP500 --min-score 8.0
        stock-analysis scan -u NASDAQ100 --sector Technology --horizon 5d
        stock-analysis scan --custom-universe my_watchlist.csv -n 20
    """
    from stock_analysis.commands import ScanCommand

    cmd = ScanCommand(
        universe=universe,
        custom_universe_path=custom_universe,
        min_score=min_score,
        max_results=max_results,
        sort_by=sort_by,
        horizon=horizon,
        sectors=list(sector) if sector else None,
        min_price=min_price,
        min_volume=min_volume,
        output_path=output,
        output_format=OutputFormat(format),
        parallel=parallel,
        n_workers=workers,
        config_path=ctx.obj['config_path'],
        verbosity=ctx.obj['verbosity']
    )

    result = cmd.execute()
    cmd.display(result)


# Example output:
"""
$ stock-analysis scan --universe SP500 --min-score 7.5 -n 10

Scanning S&P 500... ████████████████████████████████████████ 503/503 [00:45]

═══════════════════════════════════════════════════════════════════════════════
                         TOP OPPORTUNITIES (Score ≥ 7.5)
═══════════════════════════════════════════════════════════════════════════════

Rank  Ticker  Score  Trend  Mom   Vol   RS    Edge   P(+10%)  Sector
────  ──────  ─────  ─────  ────  ────  ────  ─────  ───────  ──────────────
  1   NVDA    9.2    9.5    8.8   8.5   9.8   9.0    42.3%    Technology
  2   META    8.8    8.5    8.2   9.0   8.5   9.2    38.7%    Comm Services
  3   AMZN    8.5    8.8    7.5   8.0   8.8   8.5    35.2%    Consumer Disc
  4   MSFT    8.3    8.5    8.0   7.2   8.5   8.2    33.1%    Technology
  5   AAPL    7.8    8.2    7.5   6.8   8.5   8.0    28.7%    Technology
  6   LLY     7.7    7.8    8.0   7.5   8.2   7.5    27.5%    Healthcare
  7   COST    7.6    7.5    7.2   8.2   7.8   7.8    26.8%    Consumer Staples
  8   JPM     7.5    7.2    7.8   7.0   7.5   8.0    25.2%    Financials

Found: 8 opportunities | Universe: 503 | Threshold: 7.5

───────────────────────────────────────────────────────────────────────────────
Scan completed: 2024-01-18 16:30 ET | Duration: 45.2s
"""
```

### 2.4 Backtest Command

```python
@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Backtest configuration file')
@click.option('--start-date', type=click.DateTime(),
              help='Backtest start date')
@click.option('--end-date', type=click.DateTime(),
              help='Backtest end date')
@click.option('--universe', '-u', type=str, default='SP500',
              help='Universe to backtest')
@click.option('--strategy', '-s',
              type=click.Choice(['score_based', 'probability_based',
                               'rule_based', 'ml_ensemble']),
              default='score_based',
              help='Strategy type')
@click.option('--entry-threshold', type=float, default=7.0,
              help='Score threshold to enter')
@click.option('--exit-threshold', type=float, default=4.0,
              help='Score threshold to exit')
@click.option('--rebalance',
              type=click.Choice(['daily', 'weekly', 'monthly']),
              default='weekly',
              help='Rebalance frequency')
@click.option('--initial-capital', type=float, default=1000000,
              help='Initial capital')
@click.option('--max-positions', type=int, default=50,
              help='Maximum positions')
@click.option('--output', '-o', type=click.Path(),
              help='Output directory for results')
@click.option('--report/--no-report', default=True,
              help='Generate HTML report')
@click.option('--save-trades/--no-save-trades', default=True,
              help='Save trade log')
@click.pass_context
def backtest(ctx, config, start_date, end_date, universe, strategy,
             entry_threshold, exit_threshold, rebalance, initial_capital,
             max_positions, output, report, save_trades):
    """
    Run walk-forward backtest.

    Examples:
        stock-analysis backtest --start-date 2020-01-01 --end-date 2023-12-31
        stock-analysis backtest -c backtest_config.yaml
        stock-analysis backtest --strategy ml_ensemble --rebalance monthly
    """
    from stock_analysis.commands import BacktestCommand

    cmd = BacktestCommand(
        config_path=config,
        start_date=start_date,
        end_date=end_date,
        universe=universe,
        strategy=strategy,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        rebalance_frequency=rebalance,
        initial_capital=initial_capital,
        max_positions=max_positions,
        output_dir=output,
        generate_report=report,
        save_trades=save_trades,
        verbosity=ctx.obj['verbosity']
    )

    result = cmd.execute()
    cmd.display(result)
```

### 2.5 Train Command

```python
@cli.command()
@click.option('--model', '-m',
              type=click.Choice(['high_gain_classifier', 'return_regressor',
                               'multi_horizon', 'ensemble', 'all']),
              default='ensemble',
              help='Model to train')
@click.option('--start-date', type=click.DateTime(),
              help='Training data start date')
@click.option('--end-date', type=click.DateTime(),
              help='Training data end date')
@click.option('--universe', '-u', type=str, default='SP500',
              help='Training universe')
@click.option('--validation-split', type=float, default=0.2,
              help='Validation set proportion')
@click.option('--cross-validate/--no-cross-validate', default=True,
              help='Use cross-validation')
@click.option('--n-folds', type=int, default=5,
              help='Number of CV folds')
@click.option('--hyperparameter-search/--no-hyperparameter-search',
              default=False,
              help='Run hyperparameter optimization')
@click.option('--output', '-o', type=click.Path(),
              help='Output directory for models')
@click.option('--save-features/--no-save-features', default=False,
              help='Save computed features')
@click.pass_context
def train(ctx, model, start_date, end_date, universe, validation_split,
          cross_validate, n_folds, hyperparameter_search, output, save_features):
    """
    Train or retrain ML models.

    Examples:
        stock-analysis train --model ensemble
        stock-analysis train -m high_gain_classifier --hyperparameter-search
        stock-analysis train --start-date 2015-01-01 --end-date 2023-12-31
    """
    from stock_analysis.commands import TrainCommand

    cmd = TrainCommand(
        model_type=model,
        start_date=start_date,
        end_date=end_date,
        universe=universe,
        validation_split=validation_split,
        cross_validate=cross_validate,
        n_folds=n_folds,
        hyperparameter_search=hyperparameter_search,
        output_dir=output,
        save_features=save_features,
        config_path=ctx.obj['config_path'],
        verbosity=ctx.obj['verbosity']
    )

    result = cmd.execute()
    cmd.display(result)
```

### 2.6 Data Command

```python
@cli.group()
def data():
    """Data management commands."""
    pass


@data.command('update')
@click.option('--universe', '-u', type=str, default='SP500',
              help='Universe to update')
@click.option('--start-date', type=click.DateTime(),
              help='Start date for data fetch')
@click.option('--data-types', '-t', multiple=True,
              type=click.Choice(['prices', 'fundamentals', 'earnings',
                               'corporate_actions', 'all']),
              default=['all'],
              help='Data types to update')
@click.option('--force/--no-force', default=False,
              help='Force re-download even if data exists')
@click.pass_context
def data_update(ctx, universe, start_date, data_types, force):
    """
    Update local data cache.

    Examples:
        stock-analysis data update --universe SP500
        stock-analysis data update -t prices -t fundamentals
        stock-analysis data update --force
    """
    from stock_analysis.commands import DataUpdateCommand

    cmd = DataUpdateCommand(
        universe=universe,
        start_date=start_date,
        data_types=list(data_types),
        force=force,
        config_path=ctx.obj['config_path'],
        verbosity=ctx.obj['verbosity']
    )

    result = cmd.execute()
    cmd.display(result)


@data.command('status')
@click.option('--universe', '-u', type=str, default=None,
              help='Check specific universe')
@click.pass_context
def data_status(ctx, universe):
    """
    Show data cache status.

    Examples:
        stock-analysis data status
        stock-analysis data status --universe SP500
    """
    from stock_analysis.commands import DataStatusCommand

    cmd = DataStatusCommand(
        universe=universe,
        config_path=ctx.obj['config_path'],
        verbosity=ctx.obj['verbosity']
    )

    result = cmd.execute()
    cmd.display(result)


@data.command('validate')
@click.option('--check-gaps/--no-check-gaps', default=True,
              help='Check for data gaps')
@click.option('--check-splits/--no-check-splits', default=True,
              help='Validate split adjustments')
@click.option('--check-outliers/--no-check-outliers', default=True,
              help='Check for price outliers')
@click.pass_context
def data_validate(ctx, check_gaps, check_splits, check_outliers):
    """
    Validate data integrity.

    Examples:
        stock-analysis data validate
        stock-analysis data validate --check-gaps --no-check-outliers
    """
    from stock_analysis.commands import DataValidateCommand

    cmd = DataValidateCommand(
        check_gaps=check_gaps,
        check_splits=check_splits,
        check_outliers=check_outliers,
        config_path=ctx.obj['config_path'],
        verbosity=ctx.obj['verbosity']
    )

    result = cmd.execute()
    cmd.display(result)
```

### 2.7 Export Command

```python
@cli.command()
@click.argument('export_type',
                type=click.Choice(['scores', 'indicators', 'features',
                                  'backtest', 'model']))
@click.option('--tickers', '-t', multiple=True,
              help='Tickers to export (default: all analyzed)')
@click.option('--start-date', type=click.DateTime(),
              help='Start date')
@click.option('--end-date', type=click.DateTime(),
              help='End date')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output file path')
@click.option('--format', '-f',
              type=click.Choice(['csv', 'parquet', 'json', 'excel']),
              default='csv',
              help='Export format')
@click.option('--compress/--no-compress', default=False,
              help='Compress output')
@click.pass_context
def export(ctx, export_type, tickers, start_date, end_date,
           output, format, compress):
    """
    Export data or results.

    Examples:
        stock-analysis export scores -o scores.csv
        stock-analysis export features -t AAPL -t MSFT -o features.parquet -f parquet
        stock-analysis export backtest -o backtest_results.json -f json
    """
    from stock_analysis.commands import ExportCommand

    cmd = ExportCommand(
        export_type=export_type,
        tickers=list(tickers) if tickers else None,
        start_date=start_date,
        end_date=end_date,
        output_path=output,
        output_format=format,
        compress=compress,
        config_path=ctx.obj['config_path'],
        verbosity=ctx.obj['verbosity']
    )

    result = cmd.execute()
    cmd.display(result)
```

### 2.8 Config Command

```python
@cli.group()
def config():
    """Configuration management commands."""
    pass


@config.command('init')
@click.option('--output', '-o', type=click.Path(), default='config.yaml',
              help='Output path for config file')
@click.option('--template',
              type=click.Choice(['minimal', 'standard', 'full']),
              default='standard',
              help='Configuration template')
@click.pass_context
def config_init(ctx, output, template):
    """
    Initialize configuration file.

    Examples:
        stock-analysis config init
        stock-analysis config init -o my_config.yaml --template full
    """
    from stock_analysis.commands import ConfigInitCommand

    cmd = ConfigInitCommand(
        output_path=output,
        template=template
    )

    result = cmd.execute()
    cmd.display(result)


@config.command('validate')
@click.argument('config_path', type=click.Path(exists=True))
@click.pass_context
def config_validate(ctx, config_path):
    """
    Validate configuration file.

    Examples:
        stock-analysis config validate config.yaml
    """
    from stock_analysis.commands import ConfigValidateCommand

    cmd = ConfigValidateCommand(config_path=config_path)

    result = cmd.execute()
    cmd.display(result)


@config.command('show')
@click.option('--section', '-s', type=str,
              help='Show specific section')
@click.pass_context
def config_show(ctx, section):
    """
    Show current configuration.

    Examples:
        stock-analysis config show
        stock-analysis config show --section data
    """
    from stock_analysis.commands import ConfigShowCommand

    cmd = ConfigShowCommand(
        config_path=ctx.obj['config_path'],
        section=section
    )

    result = cmd.execute()
    cmd.display(result)
```

### 2.9 Report Command

```python
@cli.command()
@click.argument('report_type',
                type=click.Choice(['daily', 'weekly', 'backtest',
                                  'model_performance', 'data_quality']))
@click.option('--output', '-o', type=click.Path(),
              help='Output path')
@click.option('--format', '-f',
              type=click.Choice(['html', 'pdf', 'markdown']),
              default='html',
              help='Report format')
@click.option('--date', type=click.DateTime(),
              help='Report date (default: today)')
@click.option('--open/--no-open', default=True,
              help='Open report after generation')
@click.pass_context
def report(ctx, report_type, output, format, date, open):
    """
    Generate reports.

    Examples:
        stock-analysis report daily
        stock-analysis report weekly -o weekly_report.html
        stock-analysis report backtest -f pdf
    """
    from stock_analysis.commands import ReportCommand

    cmd = ReportCommand(
        report_type=report_type,
        output_path=output,
        output_format=format,
        report_date=date,
        open_after=open,
        config_path=ctx.obj['config_path'],
        verbosity=ctx.obj['verbosity']
    )

    result = cmd.execute()
    cmd.display(result)
```

---

## 3. Configuration File Structure

### 3.1 Configuration Schema

```yaml
# config.yaml - Stock Analysis System Configuration
# ============================================================================
# This file controls all aspects of the analysis system.
# Override any setting via environment variables (prefix: STOCK_ANALYSIS_)
# or CLI arguments.
# ============================================================================

# Version for config compatibility
version: "1.0"

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
data:
  # Data storage paths
  paths:
    base_dir: "${HOME}/.stock-analysis"        # Base data directory
    cache_dir: "${data.paths.base_dir}/cache"  # Cache location
    models_dir: "${data.paths.base_dir}/models"
    output_dir: "${data.paths.base_dir}/output"
    logs_dir: "${data.paths.base_dir}/logs"

  # Data sources
  sources:
    prices:
      provider: "yfinance"                     # yfinance, polygon, alpaca
      api_key: "${POLYGON_API_KEY}"            # API key if required
      rate_limit: 5                            # Requests per second

    fundamentals:
      provider: "yfinance"                     # yfinance, polygon, sec_edgar
      cache_days: 1                            # Days to cache

    earnings:
      provider: "yfinance"
      cache_days: 1

  # Data quality
  quality:
    min_history_days: 252                      # Minimum trading days required
    max_gap_days: 5                            # Maximum allowed data gap
    outlier_threshold: 5.0                     # Std devs for outlier detection
    validate_on_load: true                     # Validate data when loading

# ============================================================================
# UNIVERSE CONFIGURATION
# ============================================================================
universe:
  default: "SP500"

  definitions:
    SP500:
      source: "wikipedia"                      # Auto-fetch S&P 500 constituents
      refresh_days: 7                          # Days between refreshes

    NASDAQ100:
      source: "wikipedia"
      refresh_days: 7

    custom:
      file: "custom_universe.csv"              # Custom ticker list
      ticker_column: "ticker"

  # Filters applied to any universe
  filters:
    min_price: 5.0                             # Minimum stock price
    min_market_cap: 1_000_000_000              # $1B minimum market cap
    min_avg_volume: 500_000                    # Minimum daily dollar volume
    exclude_otc: true                          # Exclude OTC stocks
    exclude_adrs: false                        # Include ADRs

# ============================================================================
# INDICATORS CONFIGURATION
# ============================================================================
indicators:
  # Enabled indicator groups
  groups:
    trend: true
    momentum: true
    volatility: true
    volume: true
    microstructure: true
    regime: true
    relative_strength: true
    structure: true
    fundamentals: true
    sentiment: false                           # Disabled by default

  # Timeframes to compute
  timeframes:
    - "1d"
    - "1w"
    - "1mo"

  # Indicator-specific parameters (override defaults)
  parameters:
    trend:
      sma_periods: [20, 50, 200]
      ema_periods: [12, 26]

    momentum:
      rsi_period: 14
      macd_fast: 12
      macd_slow: 26
      macd_signal: 9

    volatility:
      atr_period: 14
      bollinger_period: 20
      bollinger_std: 2.0

# ============================================================================
# SCORING CONFIGURATION
# ============================================================================
scoring:
  # Subscore weights (must sum to 1.0)
  weights:
    trend: 0.18
    momentum: 0.18
    volume: 0.12
    relative_strength: 0.12
    fundamental: 0.08
    edge: 0.32

  # Risk penalty configuration
  risk_penalties:
    volatility:
      enabled: true
      threshold: 0.90                          # 90th percentile volatility
      max_penalty: 1.0

    drawdown:
      enabled: true
      threshold: -0.15                         # -15% drawdown
      max_penalty: 1.5

    liquidity:
      enabled: true
      threshold: 500_000                       # $500K daily volume
      max_penalty: 0.5

    gap_risk:
      enabled: true
      threshold: 0.05                          # 5% overnight gap
      max_penalty: 0.5

  # Score thresholds
  thresholds:
    exceptional: 9.0
    strong: 7.0
    moderate: 5.0
    weak: 3.0
    poor: 2.0

# ============================================================================
# PROBABILITY ENGINE CONFIGURATION
# ============================================================================
probability:
  # Horizons to estimate
  horizons:
    - days: 5
      threshold: 0.05                          # 5% gain
    - days: 21
      threshold: 0.10                          # 10% gain
    - days: 63
      threshold: 0.15                          # 15% gain

  # Estimator weights
  estimators:
    empirical: 0.40
    supervised: 0.40
    similarity: 0.20

  # Minimum sample requirements
  min_samples:
    empirical: 30
    similarity: 50

  # Similarity search
  similarity:
    n_neighbors: 100
    metric: "euclidean"

# ============================================================================
# ML MODELS CONFIGURATION
# ============================================================================
models:
  # Default model for scoring
  default: "ensemble"

  # Model-specific settings
  high_gain_classifier:
    type: "gradient_boosting"
    params:
      n_estimators: 200
      max_depth: 6
      learning_rate: 0.1
      min_samples_leaf: 50
    calibration: "isotonic"

  return_regressor:
    type: "gradient_boosting"
    params:
      n_estimators: 200
      max_depth: 5
      learning_rate: 0.1

  ensemble:
    models:
      - "high_gain_classifier"
      - "return_regressor"
    weights: [0.6, 0.4]

  # Training settings
  training:
    validation_split: 0.2
    early_stopping_rounds: 20
    seed: 42

# ============================================================================
# BACKTEST CONFIGURATION
# ============================================================================
backtest:
  # Walk-forward settings
  walk_forward:
    initial_train_days: 756                    # 3 years
    retrain_frequency_days: 21                 # Monthly
    embargo_days: 21
    expanding_window: true

  # Portfolio settings
  portfolio:
    initial_capital: 1_000_000
    max_positions: 50
    min_position_weight: 0.01
    max_position_weight: 0.05
    max_sector_weight: 0.25
    sizing_method: "equal_weight"              # equal_weight, score_weighted, risk_parity

  # Transaction costs
  costs:
    commission_per_share: 0.005
    min_commission: 1.0
    spread_bps: 5.0
    impact_coefficient: 0.1
    slippage_bps: 2.0

  # Signal thresholds
  signals:
    entry_score: 7.0
    exit_score: 4.0
    rebalance_frequency: "weekly"

# ============================================================================
# EXECUTION CONFIGURATION
# ============================================================================
execution:
  # Parallelization
  parallel:
    enabled: true
    max_workers: null                          # null = auto (CPU count)
    chunk_size: 50                             # Tickers per chunk
    backend: "multiprocessing"                 # multiprocessing, threading, dask

  # Memory management
  memory:
    max_cache_mb: 2048                         # Maximum cache size
    gc_threshold: 0.8                          # Trigger GC at 80% usage

  # Timeouts
  timeouts:
    data_fetch: 30                             # Seconds
    indicator_compute: 60
    model_inference: 10
    total_analysis: 300

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging:
  # Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
  level: "INFO"

  # Console output
  console:
    enabled: true
    level: "INFO"
    format: "simple"                           # simple, detailed, json
    colors: true

  # File output
  file:
    enabled: true
    level: "DEBUG"
    path: "${data.paths.logs_dir}/stock_analysis.log"
    max_size_mb: 100
    backup_count: 5
    format: "detailed"

  # Component-specific levels
  components:
    data: "INFO"
    indicators: "WARNING"
    models: "INFO"
    backtest: "DEBUG"

# ============================================================================
# REPORTING CONFIGURATION
# ============================================================================
reporting:
  # Default output formats
  defaults:
    format: "text"                             # text, json, csv, html
    include_indicators: true
    include_probability: true

  # HTML report settings
  html:
    template: "default"                        # default, minimal, detailed
    include_charts: true
    chart_library: "plotly"                    # plotly, matplotlib

  # Alert settings
  alerts:
    enabled: false
    email:
      enabled: false
      smtp_server: ""
      recipients: []
    slack:
      enabled: false
      webhook_url: ""

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================
advanced:
  # Feature flags
  features:
    use_gpu: false                             # Use GPU for ML if available
    experimental_indicators: false
    debug_mode: false

  # Database settings
  database:
    type: "duckdb"                             # duckdb, sqlite
    path: "${data.paths.base_dir}/stock_analysis.duckdb"
    read_only: false

  # API settings
  api:
    enabled: false
    host: "127.0.0.1"
    port: 8080
```

### 3.2 Configuration Loader

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import os
import yaml


@dataclass
class ConfigLoader:
    """
    Load and merge configuration from multiple sources.

    Precedence (highest to lowest):
    1. CLI arguments
    2. Environment variables
    3. Config file
    4. Default values
    """

    ENV_PREFIX = "STOCK_ANALYSIS_"
    DEFAULT_CONFIG_PATHS = [
        Path("config.yaml"),
        Path("config.yml"),
        Path.home() / ".stock-analysis" / "config.yaml",
    ]

    def __init__(
        self,
        config_path: Optional[Path] = None,
        cli_overrides: Optional[Dict] = None
    ):
        self.config_path = config_path
        self.cli_overrides = cli_overrides or {}
        self._config: Optional[Dict] = None

    def load(self) -> Dict:
        """Load and merge configuration from all sources."""
        # Start with defaults
        config = self._load_defaults()

        # Merge file config
        file_config = self._load_file_config()
        config = self._deep_merge(config, file_config)

        # Merge environment variables
        env_config = self._load_env_config()
        config = self._deep_merge(config, env_config)

        # Merge CLI overrides
        config = self._deep_merge(config, self.cli_overrides)

        # Resolve variable references
        config = self._resolve_variables(config)

        # Validate
        self._validate(config)

        self._config = config
        return config

    def _load_defaults(self) -> Dict:
        """Load default configuration."""
        return {
            'version': '1.0',
            'data': {
                'paths': {
                    'base_dir': str(Path.home() / '.stock-analysis'),
                },
                'sources': {
                    'prices': {'provider': 'yfinance', 'rate_limit': 5},
                },
                'quality': {
                    'min_history_days': 252,
                    'validate_on_load': True,
                },
            },
            'universe': {
                'default': 'SP500',
                'filters': {
                    'min_price': 5.0,
                    'min_avg_volume': 500_000,
                },
            },
            'scoring': {
                'weights': {
                    'trend': 0.18,
                    'momentum': 0.18,
                    'volume': 0.12,
                    'relative_strength': 0.12,
                    'fundamental': 0.08,
                    'edge': 0.32,
                },
            },
            'execution': {
                'parallel': {
                    'enabled': True,
                    'max_workers': None,
                    'chunk_size': 50,
                },
            },
            'logging': {
                'level': 'INFO',
                'console': {'enabled': True, 'colors': True},
                'file': {'enabled': True},
            },
        }

    def _load_file_config(self) -> Dict:
        """Load configuration from file."""
        config_path = self.config_path

        # Find config file if not specified
        if config_path is None:
            for path in self.DEFAULT_CONFIG_PATHS:
                if path.exists():
                    config_path = path
                    break

        if config_path is None or not config_path.exists():
            return {}

        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}

    def _load_env_config(self) -> Dict:
        """Load configuration from environment variables."""
        config = {}

        for key, value in os.environ.items():
            if not key.startswith(self.ENV_PREFIX):
                continue

            # Convert STOCK_ANALYSIS_DATA_PATHS_BASE_DIR to nested dict
            config_key = key[len(self.ENV_PREFIX):].lower()
            parts = config_key.split('_')

            # Build nested structure
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set value (attempt type conversion)
            current[parts[-1]] = self._parse_env_value(value)

        return config

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value."""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False

        # Number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # List (comma-separated)
        if ',' in value:
            return [v.strip() for v in value.split(',')]

        return value

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if (key in result and
                isinstance(result[key], dict) and
                isinstance(value, dict)):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _resolve_variables(self, config: Dict, root: Optional[Dict] = None) -> Dict:
        """Resolve ${variable} references in config."""
        if root is None:
            root = config

        result = {}

        for key, value in config.items():
            if isinstance(value, dict):
                result[key] = self._resolve_variables(value, root)
            elif isinstance(value, str):
                result[key] = self._resolve_string(value, root)
            elif isinstance(value, list):
                result[key] = [
                    self._resolve_string(v, root) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                result[key] = value

        return result

    def _resolve_string(self, value: str, root: Dict) -> str:
        """Resolve variables in a string."""
        import re

        def replace_var(match):
            var_name = match.group(1)

            # Environment variable
            if var_name in os.environ:
                return os.environ[var_name]

            # Config reference (e.g., data.paths.base_dir)
            parts = var_name.split('.')
            current = root
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return match.group(0)  # Keep original if not found

            return str(current) if not isinstance(current, dict) else match.group(0)

        return re.sub(r'\$\{([^}]+)\}', replace_var, value)

    def _validate(self, config: Dict):
        """Validate configuration."""
        # Check required fields
        required = [
            ('data', 'paths', 'base_dir'),
            ('universe', 'default'),
            ('scoring', 'weights'),
        ]

        for path in required:
            current = config
            for key in path:
                if key not in current:
                    raise ConfigurationError(
                        f"Missing required config: {'.'.join(path)}"
                    )
                current = current[key]

        # Validate weights sum to 1.0
        weights = config.get('scoring', {}).get('weights', {})
        if weights:
            total = sum(weights.values())
            if abs(total - 1.0) > 0.001:
                raise ConfigurationError(
                    f"Scoring weights must sum to 1.0, got {total}"
                )


class ConfigurationError(Exception):
    """Configuration validation error."""
    pass
```

### 3.3 Configuration Access

```python
from typing import Any, Optional
import threading


class Config:
    """
    Global configuration accessor.

    Thread-safe singleton for accessing configuration values.
    """

    _instance: Optional['Config'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._config = {}
        return cls._instance

    @classmethod
    def initialize(cls, config: Dict):
        """Initialize configuration."""
        instance = cls()
        instance._config = config

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Supports dot notation: Config.get('data.paths.base_dir')
        """
        instance = cls()

        parts = key.split('.')
        current = instance._config

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current

    @classmethod
    def require(cls, key: str) -> Any:
        """Get required configuration value (raises if missing)."""
        value = cls.get(key)
        if value is None:
            raise ConfigurationError(f"Required config missing: {key}")
        return value

    @classmethod
    def as_dict(cls) -> Dict:
        """Get full configuration as dictionary."""
        return cls()._config.copy()


# Convenience functions
def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value."""
    return Config.get(key, default)


def require_config(key: str) -> Any:
    """Get required configuration value."""
    return Config.require(key)
```

---

## 4. Logging & Reporting

### 4.1 Logging System

```python
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
from logging.handlers import RotatingFileHandler
import json


class LogFormatter(logging.Formatter):
    """Custom log formatter with color support."""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',
    }

    def __init__(
        self,
        fmt: str = None,
        datefmt: str = None,
        use_colors: bool = True
    ):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"

        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, 'extra'):
            log_data.update(record.extra)

        return json.dumps(log_data)


class LoggingManager:
    """
    Configure and manage logging.
    """

    SIMPLE_FORMAT = "%(asctime)s │ %(levelname)-8s │ %(message)s"
    DETAILED_FORMAT = "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s"

    def __init__(self, config: Dict):
        self.config = config
        self._loggers: Dict[str, logging.Logger] = {}

    def setup(self):
        """Configure logging based on configuration."""
        root_level = getattr(
            logging,
            self.config.get('level', 'INFO').upper()
        )

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Capture all, filter at handlers

        # Remove existing handlers
        root_logger.handlers = []

        # Console handler
        if self.config.get('console', {}).get('enabled', True):
            self._setup_console_handler(root_logger)

        # File handler
        if self.config.get('file', {}).get('enabled', False):
            self._setup_file_handler(root_logger)

        # Set component-specific levels
        for component, level in self.config.get('components', {}).items():
            logger = logging.getLogger(f'stock_analysis.{component}')
            logger.setLevel(getattr(logging, level.upper()))

    def _setup_console_handler(self, logger: logging.Logger):
        """Setup console logging handler."""
        console_config = self.config.get('console', {})

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(
            getattr(logging, console_config.get('level', 'INFO').upper())
        )

        fmt = console_config.get('format', 'simple')
        if fmt == 'json':
            handler.setFormatter(JSONFormatter())
        else:
            format_str = (
                self.DETAILED_FORMAT if fmt == 'detailed'
                else self.SIMPLE_FORMAT
            )
            handler.setFormatter(LogFormatter(
                fmt=format_str,
                datefmt='%H:%M:%S',
                use_colors=console_config.get('colors', True)
            ))

        logger.addHandler(handler)

    def _setup_file_handler(self, logger: logging.Logger):
        """Setup file logging handler."""
        file_config = self.config.get('file', {})

        log_path = Path(file_config.get('path', 'stock_analysis.log'))
        log_path.parent.mkdir(parents=True, exist_ok=True)

        handler = RotatingFileHandler(
            log_path,
            maxBytes=file_config.get('max_size_mb', 100) * 1024 * 1024,
            backupCount=file_config.get('backup_count', 5)
        )
        handler.setLevel(
            getattr(logging, file_config.get('level', 'DEBUG').upper())
        )

        fmt = file_config.get('format', 'detailed')
        if fmt == 'json':
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                fmt=self.DETAILED_FORMAT,
                datefmt='%Y-%m-%d %H:%M:%S'
            ))

        logger.addHandler(handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger."""
        full_name = f'stock_analysis.{name}'

        if full_name not in self._loggers:
            self._loggers[full_name] = logging.getLogger(full_name)

        return self._loggers[full_name]


# Global logging manager
_logging_manager: Optional[LoggingManager] = None


def setup_logging(config: Dict):
    """Setup logging from configuration."""
    global _logging_manager
    _logging_manager = LoggingManager(config)
    _logging_manager.setup()


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a component."""
    if _logging_manager:
        return _logging_manager.get_logger(name)
    return logging.getLogger(f'stock_analysis.{name}')
```

### 4.2 Progress Reporting

```python
from dataclasses import dataclass
from typing import Optional, Callable
import time
import sys


@dataclass
class ProgressState:
    """Progress tracking state."""
    total: int
    current: int = 0
    description: str = ""
    start_time: float = 0

    @property
    def percent(self) -> float:
        return (self.current / self.total * 100) if self.total > 0 else 0

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def rate(self) -> float:
        return self.current / self.elapsed if self.elapsed > 0 else 0

    @property
    def eta(self) -> float:
        if self.rate > 0:
            remaining = self.total - self.current
            return remaining / self.rate
        return 0


class ProgressBar:
    """
    Console progress bar.
    """

    def __init__(
        self,
        total: int,
        description: str = "",
        width: int = 40,
        show_eta: bool = True,
        disable: bool = False
    ):
        self.state = ProgressState(total=total, description=description)
        self.width = width
        self.show_eta = show_eta
        self.disable = disable or not sys.stdout.isatty()
        self.state.start_time = time.time()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def update(self, n: int = 1, description: str = None):
        """Update progress."""
        self.state.current += n
        if description:
            self.state.description = description

        if not self.disable:
            self._render()

    def set_description(self, description: str):
        """Update description."""
        self.state.description = description
        if not self.disable:
            self._render()

    def _render(self):
        """Render progress bar to console."""
        pct = self.state.percent
        filled = int(self.width * pct / 100)
        bar = '█' * filled + '░' * (self.width - filled)

        parts = [
            f"\r{self.state.description}",
            f" {bar}",
            f" {self.state.current}/{self.state.total}",
            f" [{pct:.1f}%]"
        ]

        if self.show_eta and self.state.current > 0:
            eta = self.state.eta
            if eta < 60:
                parts.append(f" [ETA: {eta:.0f}s]")
            else:
                parts.append(f" [ETA: {eta/60:.1f}m]")

        sys.stdout.write(''.join(parts))
        sys.stdout.flush()

    def close(self):
        """Finish progress bar."""
        if not self.disable:
            elapsed = self.state.elapsed
            print(f" [Done in {elapsed:.1f}s]")


class ProgressReporter:
    """
    Multi-level progress reporting.
    """

    def __init__(self, verbosity: int = 1):
        self.verbosity = verbosity
        self._bars = []
        self._callbacks: list[Callable] = []

    def add_callback(self, callback: Callable):
        """Add progress callback."""
        self._callbacks.append(callback)

    def create_bar(
        self,
        total: int,
        description: str = "",
        level: int = 0
    ) -> ProgressBar:
        """Create a progress bar."""
        disable = level > self.verbosity
        bar = ProgressBar(
            total=total,
            description=description,
            disable=disable
        )
        self._bars.append(bar)
        return bar

    def log(self, message: str, level: int = 1):
        """Log a progress message."""
        if level <= self.verbosity:
            print(message)

        for callback in self._callbacks:
            callback({'type': 'log', 'message': message, 'level': level})

    def status(self, message: str):
        """Show status message."""
        if self.verbosity >= 1:
            print(f"→ {message}")

    def success(self, message: str):
        """Show success message."""
        if self.verbosity >= 1:
            print(f"✓ {message}")

    def warning(self, message: str):
        """Show warning message."""
        if self.verbosity >= 0:
            print(f"⚠ {message}")

    def error(self, message: str):
        """Show error message."""
        print(f"✗ {message}", file=sys.stderr)
```

### 4.3 Report Generation

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json


@dataclass
class ReportSection:
    """A section in a report."""
    title: str
    content: Any
    section_type: str = "text"  # text, table, chart, code


class ReportBuilder:
    """
    Build structured reports in multiple formats.
    """

    def __init__(self, title: str):
        self.title = title
        self.sections: List[ReportSection] = []
        self.metadata: Dict[str, Any] = {
            'generated_at': datetime.now().isoformat(),
            'version': '1.0'
        }

    def add_section(
        self,
        title: str,
        content: Any,
        section_type: str = "text"
    ) -> 'ReportBuilder':
        """Add a section to the report."""
        self.sections.append(ReportSection(
            title=title,
            content=content,
            section_type=section_type
        ))
        return self

    def add_text(self, title: str, text: str) -> 'ReportBuilder':
        """Add text section."""
        return self.add_section(title, text, "text")

    def add_table(
        self,
        title: str,
        data: List[Dict],
        columns: Optional[List[str]] = None
    ) -> 'ReportBuilder':
        """Add table section."""
        return self.add_section(title, {
            'data': data,
            'columns': columns or list(data[0].keys()) if data else []
        }, "table")

    def add_metrics(
        self,
        title: str,
        metrics: Dict[str, Any]
    ) -> 'ReportBuilder':
        """Add metrics section."""
        return self.add_section(title, metrics, "metrics")

    def add_chart(
        self,
        title: str,
        chart_type: str,
        data: Any,
        config: Optional[Dict] = None
    ) -> 'ReportBuilder':
        """Add chart section."""
        return self.add_section(title, {
            'chart_type': chart_type,
            'data': data,
            'config': config or {}
        }, "chart")

    def to_text(self) -> str:
        """Render as plain text."""
        lines = [
            "═" * 80,
            self.title.center(80),
            "═" * 80,
            ""
        ]

        for section in self.sections:
            lines.append(f"## {section.title}")
            lines.append("-" * len(section.title))

            if section.section_type == "text":
                lines.append(section.content)

            elif section.section_type == "table":
                lines.append(self._render_text_table(section.content))

            elif section.section_type == "metrics":
                for key, value in section.content.items():
                    if isinstance(value, float):
                        lines.append(f"  {key}: {value:.4f}")
                    else:
                        lines.append(f"  {key}: {value}")

            lines.append("")

        lines.append("-" * 80)
        lines.append(f"Generated: {self.metadata['generated_at']}")

        return "\n".join(lines)

    def _render_text_table(self, table_data: Dict) -> str:
        """Render table as text."""
        data = table_data['data']
        columns = table_data['columns']

        if not data:
            return "(empty)"

        # Calculate column widths
        widths = {col: len(col) for col in columns}
        for row in data:
            for col in columns:
                val = str(row.get(col, ''))
                widths[col] = max(widths[col], len(val))

        # Build table
        lines = []

        # Header
        header = " │ ".join(col.ljust(widths[col]) for col in columns)
        lines.append(header)
        lines.append("─┼─".join("─" * widths[col] for col in columns))

        # Data rows
        for row in data:
            row_str = " │ ".join(
                str(row.get(col, '')).ljust(widths[col])
                for col in columns
            )
            lines.append(row_str)

        return "\n".join(lines)

    def to_json(self) -> str:
        """Render as JSON."""
        return json.dumps({
            'title': self.title,
            'metadata': self.metadata,
            'sections': [
                {
                    'title': s.title,
                    'type': s.section_type,
                    'content': s.content
                }
                for s in self.sections
            ]
        }, indent=2, default=str)

    def to_html(self, template: str = "default") -> str:
        """Render as HTML."""
        from stock_analysis.reporting.html_templates import render_html
        return render_html(self, template)

    def to_csv(self) -> Dict[str, str]:
        """Render tables as CSV files."""
        import csv
        import io

        csv_files = {}

        for i, section in enumerate(self.sections):
            if section.section_type != "table":
                continue

            data = section.content['data']
            columns = section.content['columns']

            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)

            filename = f"{i}_{section.title.lower().replace(' ', '_')}.csv"
            csv_files[filename] = output.getvalue()

        return csv_files

    def save(self, path: Path, format: str = "text"):
        """Save report to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "text":
            path.write_text(self.to_text())
        elif format == "json":
            path.write_text(self.to_json())
        elif format == "html":
            path.write_text(self.to_html())
        elif format == "csv":
            csv_files = self.to_csv()
            for filename, content in csv_files.items():
                (path.parent / filename).write_text(content)
```

---

## 5. Parallelization Strategy

### 5.1 Parallel Execution Framework

```python
from dataclasses import dataclass
from typing import List, Callable, TypeVar, Generic, Optional, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import queue
import threading


T = TypeVar('T')
R = TypeVar('R')


@dataclass
class ParallelConfig:
    """Parallel execution configuration."""
    enabled: bool = True
    max_workers: Optional[int] = None
    chunk_size: int = 50
    backend: str = "multiprocessing"  # multiprocessing, threading
    timeout: Optional[float] = None

    @property
    def workers(self) -> int:
        """Get effective worker count."""
        if not self.enabled:
            return 1
        return self.max_workers or cpu_count()


class ParallelExecutor(Generic[T, R]):
    """
    Parallel task execution with progress tracking.
    """

    def __init__(self, config: ParallelConfig):
        self.config = config
        self._progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable):
        """Set progress callback function."""
        self._progress_callback = callback

    def map(
        self,
        func: Callable[[T], R],
        items: List[T],
        desc: str = "Processing"
    ) -> List[R]:
        """
        Execute function on items in parallel.

        Returns results in same order as input.
        """
        if not self.config.enabled or len(items) <= 1:
            return self._sequential_map(func, items, desc)

        if self.config.backend == "threading":
            return self._threaded_map(func, items, desc)
        else:
            return self._process_map(func, items, desc)

    def _sequential_map(
        self,
        func: Callable[[T], R],
        items: List[T],
        desc: str
    ) -> List[R]:
        """Sequential execution."""
        results = []

        for i, item in enumerate(items):
            result = func(item)
            results.append(result)

            if self._progress_callback:
                self._progress_callback(i + 1, len(items), desc)

        return results

    def _process_map(
        self,
        func: Callable[[T], R],
        items: List[T],
        desc: str
    ) -> List[R]:
        """Process pool execution."""
        results = [None] * len(items)
        completed = 0

        with ProcessPoolExecutor(max_workers=self.config.workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(func, item): idx
                for idx, item in enumerate(items)
            }

            # Collect results as they complete
            for future in as_completed(
                future_to_idx,
                timeout=self.config.timeout
            ):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = ParallelError(idx, items[idx], e)

                completed += 1
                if self._progress_callback:
                    self._progress_callback(completed, len(items), desc)

        return results

    def _threaded_map(
        self,
        func: Callable[[T], R],
        items: List[T],
        desc: str
    ) -> List[R]:
        """Thread pool execution."""
        results = [None] * len(items)
        completed = 0
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
            future_to_idx = {
                executor.submit(func, item): idx
                for idx, item in enumerate(items)
            }

            for future in as_completed(
                future_to_idx,
                timeout=self.config.timeout
            ):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = ParallelError(idx, items[idx], e)

                with lock:
                    completed += 1
                    if self._progress_callback:
                        self._progress_callback(completed, len(items), desc)

        return results

    def map_chunked(
        self,
        func: Callable[[List[T]], List[R]],
        items: List[T],
        desc: str = "Processing"
    ) -> List[R]:
        """
        Execute function on chunks of items.

        More efficient when function has setup overhead.
        """
        chunks = list(self._chunkify(items, self.config.chunk_size))

        chunk_results = self.map(func, chunks, desc)

        # Flatten results
        results = []
        for chunk_result in chunk_results:
            if isinstance(chunk_result, ParallelError):
                results.append(chunk_result)
            else:
                results.extend(chunk_result)

        return results

    def _chunkify(self, items: List[T], chunk_size: int) -> Iterator[List[T]]:
        """Split items into chunks."""
        for i in range(0, len(items), chunk_size):
            yield items[i:i + chunk_size]


@dataclass
class ParallelError:
    """Error during parallel execution."""
    index: int
    item: any
    exception: Exception


class TaskQueue:
    """
    Thread-safe task queue for producer/consumer pattern.
    """

    def __init__(self, maxsize: int = 0):
        self._queue = queue.Queue(maxsize=maxsize)
        self._done = threading.Event()

    def put(self, item: T, block: bool = True, timeout: float = None):
        """Add item to queue."""
        self._queue.put(item, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: float = None) -> Optional[T]:
        """Get item from queue."""
        try:
            return self._queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def task_done(self):
        """Mark task as complete."""
        self._queue.task_done()

    def mark_done(self):
        """Mark queue as finished (no more items)."""
        self._done.set()

    def is_done(self) -> bool:
        """Check if queue is finished."""
        return self._done.is_set() and self._queue.empty()

    def join(self):
        """Wait for all tasks to complete."""
        self._queue.join()
```

### 5.2 Analysis Pipeline

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class AnalysisTask:
    """Single analysis task."""
    ticker: str
    as_of_date: datetime
    horizons: List[str]
    include_indicators: bool = True
    include_probability: bool = True


@dataclass
class AnalysisResult:
    """Analysis result for a single ticker."""
    ticker: str
    score: float
    subscores: Dict[str, float]
    probability: Optional[Dict[str, float]]
    indicators: Optional[Dict[str, Any]]
    risk_flags: List[str]
    timestamp: datetime


class AnalysisPipeline:
    """
    Parallel analysis pipeline for multiple tickers.
    """

    def __init__(
        self,
        config: ParallelConfig,
        data_provider,
        indicator_engine,
        scoring_engine,
        probability_engine
    ):
        self.config = config
        self.data = data_provider
        self.indicators = indicator_engine
        self.scoring = scoring_engine
        self.probability = probability_engine

        self.executor = ParallelExecutor(config)
        self.logger = get_logger('pipeline')

    def analyze_batch(
        self,
        tasks: List[AnalysisTask],
        progress_reporter: Optional[ProgressReporter] = None
    ) -> List[AnalysisResult]:
        """
        Analyze multiple tickers in parallel.
        """
        self.logger.info(f"Starting batch analysis of {len(tasks)} tickers")

        # Set up progress tracking
        if progress_reporter:
            self.executor.set_progress_callback(
                lambda done, total, desc: progress_reporter.create_bar(
                    total, desc
                ).update(1)
            )

        # Execute analysis in parallel
        results = self.executor.map_chunked(
            self._analyze_chunk,
            tasks,
            desc="Analyzing"
        )

        # Handle errors
        successful = []
        errors = []

        for result in results:
            if isinstance(result, ParallelError):
                errors.append(result)
                self.logger.error(
                    f"Error analyzing {result.item.ticker}: {result.exception}"
                )
            else:
                successful.append(result)

        self.logger.info(
            f"Completed: {len(successful)} successful, {len(errors)} errors"
        )

        return successful

    def _analyze_chunk(self, tasks: List[AnalysisTask]) -> List[AnalysisResult]:
        """Analyze a chunk of tickers."""
        # Pre-load data for chunk
        tickers = [t.ticker for t in tasks]
        self.data.preload(tickers)

        results = []
        for task in tasks:
            result = self._analyze_single(task)
            results.append(result)

        return results

    def _analyze_single(self, task: AnalysisTask) -> AnalysisResult:
        """Analyze a single ticker."""
        # Get price data
        prices = self.data.get_prices(task.ticker, as_of=task.as_of_date)

        # Compute indicators
        indicator_values = None
        if task.include_indicators:
            indicator_values = self.indicators.compute_all(
                prices,
                task.as_of_date
            )

        # Compute score
        score_result = self.scoring.score(
            task.ticker,
            task.as_of_date,
            indicator_values
        )

        # Compute probability
        probability = None
        if task.include_probability:
            probability = {}
            for horizon in task.horizons:
                prob = self.probability.estimate(
                    task.ticker,
                    task.as_of_date,
                    horizon
                )
                probability[horizon] = prob

        return AnalysisResult(
            ticker=task.ticker,
            score=score_result.final_score,
            subscores=score_result.subscores,
            probability=probability,
            indicators=indicator_values,
            risk_flags=score_result.risk_flags,
            timestamp=datetime.now()
        )
```

### 5.3 Resource Management

```python
from dataclasses import dataclass
from typing import Dict, Optional
import psutil
import gc
import threading
import time


@dataclass
class ResourceLimits:
    """Resource usage limits."""
    max_memory_mb: int = 4096
    max_cpu_percent: float = 80.0
    gc_threshold: float = 0.8


class ResourceMonitor:
    """
    Monitor and manage system resources.
    """

    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self._stats: Dict[str, float] = {}
        self._lock = threading.Lock()

    def start(self):
        """Start resource monitoring."""
        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=5)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            self._update_stats()
            self._check_limits()
            time.sleep(1)

    def _update_stats(self):
        """Update resource statistics."""
        process = psutil.Process()

        with self._lock:
            self._stats = {
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'memory_percent': process.memory_percent(),
                'cpu_percent': process.cpu_percent(),
                'threads': process.num_threads(),
            }

    def _check_limits(self):
        """Check and enforce resource limits."""
        with self._lock:
            memory_mb = self._stats.get('memory_mb', 0)

        # Trigger GC if memory usage is high
        if memory_mb > self.limits.max_memory_mb * self.limits.gc_threshold:
            gc.collect()

    def get_stats(self) -> Dict[str, float]:
        """Get current resource statistics."""
        with self._lock:
            return self._stats.copy()

    def check_memory_available(self, required_mb: float) -> bool:
        """Check if memory is available for operation."""
        current = self._stats.get('memory_mb', 0)
        return (current + required_mb) <= self.limits.max_memory_mb

    def wait_for_resources(self, timeout: float = 60) -> bool:
        """Wait until resources are available."""
        start = time.time()

        while time.time() - start < timeout:
            with self._lock:
                memory_ok = self._stats.get('memory_mb', 0) < self.limits.max_memory_mb * 0.9
                cpu_ok = self._stats.get('cpu_percent', 0) < self.limits.max_cpu_percent

            if memory_ok and cpu_ok:
                return True

            gc.collect()
            time.sleep(1)

        return False


class MemoryCache:
    """
    LRU cache with memory limits.
    """

    def __init__(self, max_size_mb: int = 1024):
        self.max_size_mb = max_size_mb
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []
        self._sizes: Dict[str, float] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
            return None

    def put(self, key: str, value: Any, size_mb: float):
        """Put item in cache."""
        with self._lock:
            # Evict if needed
            while self._current_size_mb() + size_mb > self.max_size_mb:
                if not self._evict_oldest():
                    break

            self._cache[key] = value
            self._sizes[key] = size_mb
            self._access_order.append(key)

    def _current_size_mb(self) -> float:
        """Get current cache size."""
        return sum(self._sizes.values())

    def _evict_oldest(self) -> bool:
        """Evict oldest item."""
        if not self._access_order:
            return False

        oldest = self._access_order.pop(0)
        del self._cache[oldest]
        del self._sizes[oldest]
        return True

    def clear(self):
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._sizes.clear()
            self._access_order.clear()
```

---

## 6. Execution Flow

### 6.1 Main Entry Point

```python
#!/usr/bin/env python
"""
Stock Analysis System - Main Entry Point
"""

import sys
from pathlib import Path


def main():
    """Main entry point."""
    try:
        # Add src to path if needed
        src_path = Path(__file__).parent.parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from stock_analysis.cli import cli
        from stock_analysis.config import ConfigLoader, Config
        from stock_analysis.logging import setup_logging

        # Load configuration
        loader = ConfigLoader()
        config = loader.load()
        Config.initialize(config)

        # Setup logging
        setup_logging(config.get('logging', {}))

        # Run CLI
        cli(obj={})

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
```

### 6.2 Command Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXECUTION FLOW                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. CLI PARSING                                                             │
│     ┌───────────────────────────────────────────────────────────────────┐   │
│     │ stock-analysis scan --universe SP500 --min-score 7.5 -n 20        │   │
│     └───────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  2. CONFIGURATION LOADING                                                  │
│     ┌───────────────────────────────────────────────────────────────────┐   │
│     │ config.yaml → Environment → CLI args (merge)                      │   │
│     └───────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  3. VALIDATION                                                             │
│     ┌───────────────────────────────────────────────────────────────────┐   │
│     │ • Validate configuration                                          │   │
│     │ • Check data availability                                         │   │
│     │ • Verify model files exist                                        │   │
│     └───────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  4. RESOURCE SETUP                                                         │
│     ┌───────────────────────────────────────────────────────────────────┐   │
│     │ • Initialize logging                                              │   │
│     │ • Start resource monitor                                          │   │
│     │ • Initialize parallel executor                                    │   │
│     │ • Load models into memory                                         │   │
│     └───────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  5. DATA LOADING                                                           │
│     ┌───────────────────────────────────────────────────────────────────┐   │
│     │ • Get universe members (point-in-time)                            │   │
│     │ • Apply filters (price, volume, etc.)                             │   │
│     │ • Pre-fetch price data for universe                               │   │
│     └───────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  6. PARALLEL ANALYSIS                                                      │
│     ┌───────────────────────────────────────────────────────────────────┐   │
│     │ For each chunk of tickers (parallel):                             │   │
│     │   • Compute indicators                                            │   │
│     │   • Generate features                                             │   │
│     │   • Run scoring engine                                            │   │
│     │   • Estimate probabilities                                        │   │
│     └───────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  7. AGGREGATION & RANKING                                                  │
│     ┌───────────────────────────────────────────────────────────────────┐   │
│     │ • Collect results                                                 │   │
│     │ • Sort by specified metric                                        │   │
│     │ • Apply result limit                                              │   │
│     │ • Handle errors                                                   │   │
│     └───────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  8. OUTPUT                                                                 │
│     ┌───────────────────────────────────────────────────────────────────┐   │
│     │ • Format results (text/json/csv/html)                             │   │
│     │ • Write to file if specified                                      │   │
│     │ • Display to console                                              │   │
│     │ • Log summary                                                     │   │
│     └───────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Error Handling

```python
from enum import Enum
from typing import Optional
import traceback


class ExitCode(Enum):
    """Standard exit codes."""
    SUCCESS = 0
    ERROR = 1
    INVALID_ARGS = 2
    CONFIG_ERROR = 3
    DATA_ERROR = 4
    MODEL_ERROR = 5
    INTERRUPTED = 130


class StockAnalysisError(Exception):
    """Base exception for stock analysis errors."""
    exit_code = ExitCode.ERROR

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message)
        self.details = details


class ConfigError(StockAnalysisError):
    """Configuration error."""
    exit_code = ExitCode.CONFIG_ERROR


class DataError(StockAnalysisError):
    """Data access or validation error."""
    exit_code = ExitCode.DATA_ERROR


class ModelError(StockAnalysisError):
    """Model loading or inference error."""
    exit_code = ExitCode.MODEL_ERROR


def handle_error(func):
    """Decorator for error handling in CLI commands."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger('cli')

        try:
            return func(*args, **kwargs)

        except StockAnalysisError as e:
            logger.error(f"{e.__class__.__name__}: {e}")
            if e.details:
                logger.debug(e.details)
            sys.exit(e.exit_code.value)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            sys.exit(ExitCode.INTERRUPTED.value)

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.debug(traceback.format_exc())
            sys.exit(ExitCode.ERROR.value)

    return wrapper
```

---

## 7. Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    APPLICATION INTERFACE SUMMARY                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CLI COMMANDS                                                               │
│  ────────────                                                                │
│  • analyze   - Analyze single/multiple tickers                              │
│  • scan      - Scan universe for opportunities                              │
│  • backtest  - Run walk-forward backtest                                    │
│  • train     - Train/retrain ML models                                      │
│  • export    - Export data/results                                          │
│  • data      - Data management (update, status, validate)                   │
│  • config    - Configuration management                                     │
│  • report    - Generate reports                                             │
│                                                                             │
│  CONFIGURATION                                                              │
│  ─────────────                                                               │
│  • YAML-based configuration file                                            │
│  • Environment variable overrides (STOCK_ANALYSIS_*)                        │
│  • CLI argument overrides                                                   │
│  • Sections: data, universe, indicators, scoring, probability,              │
│              models, backtest, execution, logging, reporting                │
│                                                                             │
│  LOGGING & REPORTING                                                        │
│  ───────────────────                                                         │
│  • Console: Colored output, progress bars                                   │
│  • File: Rotating logs, JSON format option                                  │
│  • Reports: Text, JSON, CSV, HTML formats                                   │
│  • Component-level log configuration                                        │
│                                                                             │
│  PARALLELIZATION                                                            │
│  ───────────────                                                             │
│  • Backend: multiprocessing (default), threading                            │
│  • Auto worker count (CPU cores)                                            │
│  • Chunked execution for efficiency                                         │
│  • Resource monitoring and limits                                           │
│  • Memory-aware caching                                                     │
│                                                                             │
│  EXECUTION FLOW                                                             │
│  ──────────────                                                              │
│  CLI → Config → Validation → Setup → Data → Analysis → Output              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Quick Reference

### Common Commands

```bash
# Analyze single ticker
stock-analysis analyze AAPL

# Analyze multiple tickers
stock-analysis analyze AAPL MSFT GOOGL --horizon 21d

# Scan S&P 500 for opportunities
stock-analysis scan --universe SP500 --min-score 7.5

# Run backtest
stock-analysis backtest --start-date 2020-01-01 --end-date 2023-12-31

# Update data
stock-analysis data update --universe SP500

# Generate report
stock-analysis report daily -o daily_report.html

# Initialize config
stock-analysis config init --template full
```

### Environment Variables

```bash
# Override configuration
export STOCK_ANALYSIS_DATA_PATHS_BASE_DIR=/data/stocks
export STOCK_ANALYSIS_EXECUTION_PARALLEL_MAX_WORKERS=8
export STOCK_ANALYSIS_LOGGING_LEVEL=DEBUG

# API keys
export POLYGON_API_KEY=your_key_here
```

---

*Document Version: 1.0*
*Last Updated: 2026-01-18*
