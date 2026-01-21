"""
Output formatters for stock analysis results.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class OutputFormatter(ABC):
    """Base class for output formatters."""

    @abstractmethod
    def format(self, analysis: Any) -> str:
        """Format a single analysis."""
        pass

    @abstractmethod
    def format_multiple(self, analyses: list[Any]) -> str:
        """Format multiple analyses."""
        pass

    def save(self, content: str, path: str | Path) -> None:
        """Save formatted content to file."""
        Path(path).write_text(content)


class TextFormatter(OutputFormatter):
    """Plain text formatter."""

    def format(self, analysis: Any) -> str:
        """Format single analysis as text."""
        lines = [
            "=" * 60,
            f"STOCK ANALYSIS: {analysis.symbol}",
            "=" * 60,
            f"Date: {analysis.date}",
            f"Price: ${analysis.price:.2f}",
            "",
            "-" * 30,
            "COMPOSITE SCORE",
            "-" * 30,
            f"Score: {analysis.composite_score.value:.1f}/100",
            f"Rating: {analysis.composite_score.components.get('rating', 'N/A')}",
            f"Signal: {analysis.composite_score.interpretation}",
            "",
            "-" * 30,
            "COMPONENT SCORES",
            "-" * 30,
            f"Technical:  {analysis.technical_score.value:6.1f}  {analysis.technical_score.interpretation}",
            f"Momentum:   {analysis.momentum_score.value:6.1f}  {analysis.momentum_score.interpretation}",
            f"Risk:       {analysis.risk_score.value:6.1f}  {analysis.risk_score.interpretation}",
            "",
            "-" * 30,
            "PROBABILITY ESTIMATE",
            "-" * 30,
            f"P(Up):      {analysis.probability.get('prob_up', 0.5)*100:6.1f}%",
            f"Confidence: {analysis.probability.get('confidence', 0)*100:6.1f}%",
            f"Signal:     {analysis.probability.get('signal', 'neutral')}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)

    def format_multiple(self, analyses: list[Any]) -> str:
        """Format multiple analyses as text."""
        lines = [
            "=" * 80,
            "STOCK SCREENING RESULTS",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Stocks Analyzed: {len(analyses)}",
            "=" * 80,
            "",
        ]

        # Header
        header = f"{'Rank':<6}{'Symbol':<8}{'Score':>8}{'Rating':>8}{'P(Up)':>8}{'Signal':<15}"
        lines.append(header)
        lines.append("-" * 60)

        # Rows
        for i, a in enumerate(analyses, 1):
            prob_up = f"{a.probability.get('prob_up', 0.5)*100:.1f}%"
            row = f"{i:<6}{a.symbol:<8}{a.composite_score.value:>8.1f}{a.composite_score.components.get('rating', 'N/A'):>8}{prob_up:>8}{a.probability.get('signal', 'neutral'):<15}"
            lines.append(row)

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)


class JSONFormatter(OutputFormatter):
    """JSON formatter."""

    def __init__(self, indent: int = 2, include_indicators: bool = False):
        self.indent = indent
        self.include_indicators = include_indicators

    def format(self, analysis: Any) -> str:
        """Format single analysis as JSON."""
        data = self._analysis_to_dict(analysis)
        return json.dumps(data, indent=self.indent, default=str)

    def format_multiple(self, analyses: list[Any]) -> str:
        """Format multiple analyses as JSON."""
        data = {
            "generated_at": datetime.now().isoformat(),
            "count": len(analyses),
            "results": [self._analysis_to_dict(a) for a in analyses],
        }
        return json.dumps(data, indent=self.indent, default=str)

    def _analysis_to_dict(self, analysis: Any) -> dict:
        """Convert analysis to dictionary."""
        result = analysis.to_dict()

        if self.include_indicators:
            result["indicators"] = analysis.indicators

        result["component_details"] = {
            "technical": {
                "score": analysis.technical_score.value,
                "interpretation": analysis.technical_score.interpretation,
                "components": analysis.technical_score.components,
            },
            "momentum": {
                "score": analysis.momentum_score.value,
                "interpretation": analysis.momentum_score.interpretation,
                "components": analysis.momentum_score.components,
            },
            "risk": {
                "score": analysis.risk_score.value,
                "interpretation": analysis.risk_score.interpretation,
                "components": analysis.risk_score.components,
            },
        }

        return result


class CSVFormatter(OutputFormatter):
    """CSV formatter."""

    def __init__(self, include_indicators: bool = False):
        self.include_indicators = include_indicators

    def format(self, analysis: Any) -> str:
        """Format single analysis as CSV."""
        df = pd.DataFrame([analysis.to_dict()])
        return df.to_csv(index=False)

    def format_multiple(self, analyses: list[Any]) -> str:
        """Format multiple analyses as CSV."""
        records = [a.to_dict() for a in analyses]
        df = pd.DataFrame(records)
        return df.to_csv(index=False)


class HTMLFormatter(OutputFormatter):
    """HTML formatter for web display."""

    def __init__(self, include_css: bool = True):
        self.include_css = include_css

    def format(self, analysis: Any) -> str:
        """Format single analysis as HTML."""
        css = self._get_css() if self.include_css else ""

        score = analysis.composite_score.value
        score_class = "score-high" if score >= 60 else "score-medium" if score >= 40 else "score-low"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Stock Analysis: {analysis.symbol}</title>
    {css}
</head>
<body>
    <div class="container">
        <h1>Stock Analysis: {analysis.symbol}</h1>
        <div class="meta">
            <span class="date">{analysis.date}</span>
            <span class="price">${analysis.price:.2f}</span>
        </div>

        <div class="score-panel {score_class}">
            <div class="score-value">{score:.1f}</div>
            <div class="score-label">Composite Score</div>
            <div class="rating">{analysis.composite_score.components.get('rating', 'N/A')}</div>
            <div class="signal">{analysis.composite_score.interpretation}</div>
        </div>

        <div class="components">
            <h2>Component Scores</h2>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Score</th>
                    <th>Interpretation</th>
                </tr>
                <tr>
                    <td>Technical</td>
                    <td>{analysis.technical_score.value:.1f}</td>
                    <td>{analysis.technical_score.interpretation}</td>
                </tr>
                <tr>
                    <td>Momentum</td>
                    <td>{analysis.momentum_score.value:.1f}</td>
                    <td>{analysis.momentum_score.interpretation}</td>
                </tr>
                <tr>
                    <td>Risk</td>
                    <td>{analysis.risk_score.value:.1f}</td>
                    <td>{analysis.risk_score.interpretation}</td>
                </tr>
            </table>
        </div>

        <div class="probability">
            <h2>Probability Estimate</h2>
            <div class="prob-value">{analysis.probability.get('prob_up', 0.5)*100:.1f}%</div>
            <div class="prob-label">Probability of Positive Return</div>
            <div class="confidence">Confidence: {analysis.probability.get('confidence', 0)*100:.1f}%</div>
        </div>
    </div>
</body>
</html>
"""
        return html

    def format_multiple(self, analyses: list[Any]) -> str:
        """Format multiple analyses as HTML table."""
        css = self._get_css() if self.include_css else ""

        rows = ""
        for i, a in enumerate(analyses, 1):
            score = a.composite_score.value
            score_class = "high" if score >= 60 else "medium" if score >= 40 else "low"
            rows += f"""
            <tr class="{score_class}">
                <td>{i}</td>
                <td><strong>{a.symbol}</strong></td>
                <td>${a.price:.2f}</td>
                <td>{score:.1f}</td>
                <td>{a.composite_score.components.get('rating', 'N/A')}</td>
                <td>{a.probability.get('prob_up', 0.5)*100:.1f}%</td>
                <td>{a.probability.get('signal', 'neutral')}</td>
            </tr>
"""

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Stock Screening Results</title>
    {css}
</head>
<body>
    <div class="container">
        <h1>Stock Screening Results</h1>
        <div class="meta">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            | Stocks: {len(analyses)}
        </div>

        <table class="results-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Symbol</th>
                    <th>Price</th>
                    <th>Score</th>
                    <th>Rating</th>
                    <th>P(Up)</th>
                    <th>Signal</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        return html

    def _get_css(self) -> str:
        """Get CSS styles."""
        return """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        h2 {
            color: #555;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        .meta {
            color: #666;
            margin-bottom: 20px;
        }
        .score-panel {
            text-align: center;
            padding: 30px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .score-high { background: #e8f5e9; }
        .score-medium { background: #fff3e0; }
        .score-low { background: #ffebee; }
        .score-value {
            font-size: 48px;
            font-weight: bold;
        }
        .score-label {
            color: #666;
            margin-top: 10px;
        }
        .rating {
            font-size: 24px;
            font-weight: bold;
            margin-top: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        th {
            background: #f5f5f5;
            font-weight: 600;
        }
        .results-table tbody tr:hover {
            background: #f9f9f9;
        }
        .results-table tr.high td:nth-child(4) { color: #2e7d32; font-weight: bold; }
        .results-table tr.medium td:nth-child(4) { color: #f57c00; font-weight: bold; }
        .results-table tr.low td:nth-child(4) { color: #c62828; font-weight: bold; }
        .probability {
            text-align: center;
            padding: 20px;
            background: #e3f2fd;
            border-radius: 8px;
            margin-top: 20px;
        }
        .prob-value {
            font-size: 36px;
            font-weight: bold;
            color: #1565c0;
        }
    </style>
"""


def get_formatter(format_type: str, **kwargs) -> OutputFormatter:
    """Factory function to get appropriate formatter."""
    formatters = {
        "text": TextFormatter,
        "json": JSONFormatter,
        "csv": CSVFormatter,
        "html": HTMLFormatter,
    }

    formatter_class = formatters.get(format_type.lower())
    if formatter_class is None:
        raise ValueError(f"Unknown format type: {format_type}")

    return formatter_class(**kwargs)
