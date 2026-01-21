"""
Main CLI application for stock analysis.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(
    name="stock-analysis",
    help="Technical analysis and probability-based stock scoring system",
)
console = Console()


@app.command()
def analyze(
    symbol: str = typer.Argument(..., help="Stock symbol to analyze"),
    benchmark: str = typer.Option("SPY", "--benchmark", "-b", help="Benchmark symbol"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: text, json, csv"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Analyze a single stock and show scores."""
    from stock_analysis.data.provider import DataProvider
    from stock_analysis.scoring.scorer import StockScorer

    console.print(f"[bold blue]Analyzing {symbol}...[/bold blue]")

    try:
        # Get data
        provider = DataProvider()
        price_data = provider.get_prices(symbol)  # Default 3 years
        prices = price_data.data  # Extract DataFrame

        benchmark_prices = None
        if benchmark:
            benchmark_data = provider.get_prices(benchmark)
            benchmark_prices = benchmark_data.data

        # Perform analysis
        scorer = StockScorer()
        analysis = scorer.analyze(symbol, prices, benchmark_prices)

        # Output results
        if format == "json":
            result = analysis.to_dict()
            if verbose:
                result["indicators"] = analysis.indicators
            output_text = json.dumps(result, indent=2, default=str)
        elif format == "csv":
            import pandas as pd
            df = pd.DataFrame([analysis.to_dict()])
            output_text = df.to_csv(index=False)
        else:
            output_text = analysis.summary()

        if output:
            Path(output).write_text(output_text)
            console.print(f"[green]Results saved to {output}[/green]")
        else:
            if format == "text":
                _display_analysis(analysis, verbose)
            else:
                console.print(output_text)

    except Exception as e:
        console.print(f"[red]Error analyzing {symbol}: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def screen(
    universe: str = typer.Option("sp500", "--universe", "-u", help="Universe: sp500, nasdaq100, custom"),
    min_score: float = typer.Option(60.0, "--min-score", help="Minimum composite score"),
    min_probability: float = typer.Option(0.55, "--min-prob", help="Minimum probability"),
    top: int = typer.Option(20, "--top", "-t", help="Number of top results"),
    sort_by: str = typer.Option("composite_score", "--sort", "-s", help="Sort by field"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
) -> None:
    """Screen stocks in a universe based on criteria."""
    from stock_analysis.data.provider import DataProvider
    from stock_analysis.data.universe import UniverseManager
    from stock_analysis.scoring.scorer import StockScorer

    console.print(f"[bold blue]Screening {universe} universe...[/bold blue]")

    try:
        # Get universe symbols
        um = UniverseManager()
        symbols = um.get_universe(universe)

        console.print(f"Found {len(symbols)} symbols in {universe}")

        # Get data
        provider = DataProvider()
        price_data = {}

        with console.status("[bold green]Fetching price data...") as status:
            for i, sym in enumerate(symbols):
                try:
                    pd = provider.get_prices(sym)
                    price_data[sym] = pd.data
                    status.update(f"[bold green]Fetched {i+1}/{len(symbols)}: {sym}")
                except Exception:
                    continue

        # Benchmark data
        benchmark_data = provider.get_prices("SPY")
        benchmark_prices = benchmark_data.data

        # Analyze universe
        scorer = StockScorer()

        with console.status("[bold green]Analyzing stocks...") as status:
            analyses = scorer.analyze_universe(
                list(price_data.keys()), price_data, benchmark_prices
            )

        # Screen and sort
        screened = scorer.screen(
            analyses,
            min_composite=min_score,
            min_probability=min_probability,
            max_results=top,
        )

        # Display results
        _display_screen_results(screened)

        if output:
            df = scorer.to_dataframe(screened)
            df.to_csv(output, index=False)
            console.print(f"[green]Results saved to {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def compare(
    symbols: str = typer.Argument(..., help="Comma-separated list of symbols"),
    benchmark: str = typer.Option("SPY", "--benchmark", "-b", help="Benchmark symbol"),
) -> None:
    """Compare multiple stocks side by side."""
    from stock_analysis.data.provider import DataProvider
    from stock_analysis.scoring.scorer import StockScorer

    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    console.print(f"[bold blue]Comparing: {', '.join(symbol_list)}[/bold blue]")

    try:
        provider = DataProvider()
        price_data = {}

        for sym in symbol_list:
            pd = provider.get_prices(sym)
            price_data[sym] = pd.data

        benchmark_data = provider.get_prices(benchmark)
        benchmark_prices = benchmark_data.data

        scorer = StockScorer()
        analyses = scorer.analyze_universe(symbol_list, price_data, benchmark_prices)

        _display_comparison(analyses)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def indicators(
    symbol: str = typer.Argument(..., help="Stock symbol"),
    group: Optional[str] = typer.Option(None, "--group", "-g", help="Indicator group"),
) -> None:
    """Show all indicators for a stock."""
    from stock_analysis.data.provider import DataProvider
    from stock_analysis.indicators.engine import IndicatorEngine

    console.print(f"[bold blue]Computing indicators for {symbol}...[/bold blue]")

    try:
        provider = DataProvider()
        price_data = provider.get_prices(symbol)
        prices = price_data.data

        engine = IndicatorEngine()

        if group:
            ind_results = engine.compute_group(group, prices)
        else:
            ind_results = engine.compute_all(prices)

        # Display as table
        table = Table(title=f"Indicators for {symbol}")
        table.add_column("Indicator", style="cyan")
        table.add_column("Value", style="green")

        for name, value in sorted(ind_results.items()):
            if isinstance(value, float):
                table.add_row(name, f"{value:.4f}")
            else:
                table.add_row(name, str(value))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def backtest(
    symbol: str = typer.Argument(..., help="Stock symbol"),
    strategy: str = typer.Option("score_based", "--strategy", "-s", help="Strategy name"),
    start: Optional[str] = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end: Optional[str] = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
    initial_capital: float = typer.Option(100000.0, "--capital", "-c", help="Initial capital"),
) -> None:
    """Backtest a trading strategy."""
    console.print(f"[bold blue]Backtesting {strategy} on {symbol}...[/bold blue]")
    console.print("[yellow]Backtesting module not yet implemented[/yellow]")


def _display_analysis(analysis, verbose: bool = False) -> None:
    """Display analysis in rich format."""
    # Header
    console.print(Panel(
        f"[bold white]{analysis.symbol}[/bold white] | "
        f"Price: [green]${analysis.price:.2f}[/green] | "
        f"Date: {analysis.date}",
        title="Stock Analysis",
    ))

    # Composite score
    score = analysis.composite_score.value
    rating = analysis.composite_score.components.get("rating", "N/A")
    color = "green" if score >= 60 else "yellow" if score >= 40 else "red"

    console.print(Panel(
        f"[bold {color}]{score:.1f}/100[/bold {color}]\n"
        f"Rating: [bold]{rating}[/bold]\n"
        f"{analysis.composite_score.interpretation}",
        title="Composite Score",
    ))

    # Component scores table
    table = Table(title="Component Scores")
    table.add_column("Component", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Interpretation")

    table.add_row(
        "Technical",
        f"{analysis.technical_score.value:.1f}",
        analysis.technical_score.interpretation,
    )
    table.add_row(
        "Momentum",
        f"{analysis.momentum_score.value:.1f}",
        analysis.momentum_score.interpretation,
    )
    table.add_row(
        "Risk",
        f"{analysis.risk_score.value:.1f}",
        analysis.risk_score.interpretation,
    )

    console.print(table)

    # Probability
    prob_up = analysis.probability.get("prob_up", 0.5) * 100
    confidence = analysis.probability.get("confidence", 0) * 100

    console.print(Panel(
        f"P(Up): [bold green]{prob_up:.1f}%[/bold green]\n"
        f"Confidence: {confidence:.1f}%\n"
        f"Signal: {analysis.probability.get('signal', 'neutral')}",
        title="Probability Estimate",
    ))

    if verbose:
        # Show key indicators
        console.print("\n[bold]Key Indicators:[/bold]")
        key_indicators = [
            "rsi", "macd_histogram", "adx", "atr_pct",
            "return_21d", "volume_sma_ratio", "sharpe_ratio"
        ]
        for ind in key_indicators:
            if ind in analysis.indicators:
                val = analysis.indicators[ind]
                console.print(f"  {ind}: {val:.4f}" if isinstance(val, float) else f"  {ind}: {val}")


def _display_screen_results(analyses) -> None:
    """Display screening results."""
    table = Table(title="Screening Results")
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Symbol", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Rating", justify="center")
    table.add_column("P(Up)", justify="right")
    table.add_column("Signal")

    for i, a in enumerate(analyses, 1):
        prob_up = f"{a.probability.get('prob_up', 0.5)*100:.1f}%"
        table.add_row(
            str(i),
            a.symbol,
            f"{a.composite_score.value:.1f}",
            a.composite_score.components.get("rating", "N/A"),
            prob_up,
            a.probability.get("signal", "neutral"),
        )

    console.print(table)


def _display_comparison(analyses) -> None:
    """Display comparison table."""
    table = Table(title="Stock Comparison")
    table.add_column("Metric", style="cyan")

    for a in analyses:
        table.add_column(a.symbol, justify="right")

    # Add rows
    metrics = [
        ("Price", lambda a: f"${a.price:.2f}"),
        ("Composite", lambda a: f"{a.composite_score.value:.1f}"),
        ("Rating", lambda a: a.composite_score.components.get("rating", "N/A")),
        ("Technical", lambda a: f"{a.technical_score.value:.1f}"),
        ("Momentum", lambda a: f"{a.momentum_score.value:.1f}"),
        ("Risk", lambda a: f"{a.risk_score.value:.1f}"),
        ("P(Up)", lambda a: f"{a.probability.get('prob_up', 0.5)*100:.1f}%"),
        ("Signal", lambda a: a.probability.get("signal", "neutral")),
    ]

    for metric_name, metric_fn in metrics:
        values = [metric_fn(a) for a in analyses]
        table.add_row(metric_name, *values)

    console.print(table)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
