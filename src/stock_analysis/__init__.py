"""
Stock Analysis System

A research-grade stock analysis toolkit with ML-powered scoring,
probability estimation, and comprehensive technical analysis.
"""

__version__ = "1.0.0"
__author__ = "Stock Analysis Team"

from stock_analysis.core.config import Config, get_config
from stock_analysis.core.logging import get_logger, setup_logging

# Lazy imports for main components
def get_scorer():
    """Get the main stock scorer."""
    from stock_analysis.scoring.scorer import StockScorer
    return StockScorer()


def get_indicator_engine():
    """Get the indicator computation engine."""
    from stock_analysis.indicators.engine import IndicatorEngine
    return IndicatorEngine()


def get_data_provider():
    """Get the data provider."""
    from stock_analysis.data.provider import DataProvider
    return DataProvider()


__all__ = [
    "__version__",
    "Config",
    "get_config",
    "get_logger",
    "setup_logging",
    "get_scorer",
    "get_indicator_engine",
    "get_data_provider",
]
