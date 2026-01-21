"""
Exception hierarchy for the stock analysis system.
"""

from typing import Any, Optional


class StockAnalysisError(Exception):
    """Base exception for all stock analysis errors."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        recoverable: bool = False,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.recoverable = recoverable
        self.details = details or {}

    def __str__(self) -> str:
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "recoverable": self.recoverable,
            "details": self.details,
        }


class DataError(StockAnalysisError):
    """Data loading or validation error."""

    pass


class DataNotFoundError(DataError):
    """Requested data does not exist."""

    def __init__(self, ticker: str, data_type: str = "price", **kwargs: Any):
        message = f"No {data_type} data found for {ticker}"
        super().__init__(message, code="DATA_NOT_FOUND", **kwargs)
        self.ticker = ticker
        self.data_type = data_type


class DataValidationError(DataError):
    """Data failed validation checks."""

    def __init__(self, message: str, issues: Optional[list[str]] = None, **kwargs: Any):
        super().__init__(message, code="DATA_VALIDATION_FAILED", **kwargs)
        self.issues = issues or []


class IndicatorError(StockAnalysisError):
    """Indicator computation error."""

    def __init__(
        self,
        indicator: str,
        message: str,
        **kwargs: Any,
    ):
        full_message = f"Indicator '{indicator}': {message}"
        super().__init__(full_message, code="INDICATOR_ERROR", **kwargs)
        self.indicator = indicator


class FeatureError(StockAnalysisError):
    """Feature engineering error."""

    def __init__(self, feature: str, message: str, **kwargs: Any):
        full_message = f"Feature '{feature}': {message}"
        super().__init__(full_message, code="FEATURE_ERROR", **kwargs)
        self.feature = feature


class ModelError(StockAnalysisError):
    """Model loading or inference error."""

    def __init__(self, model_name: str, message: str, **kwargs: Any):
        full_message = f"Model '{model_name}': {message}"
        super().__init__(full_message, code="MODEL_ERROR", **kwargs)
        self.model_name = model_name


class ModelNotFoundError(ModelError):
    """Model file not found."""

    def __init__(self, model_name: str, path: Optional[str] = None, **kwargs: Any):
        message = f"not found" + (f" at {path}" if path else "")
        super().__init__(model_name, message, **kwargs)
        self.path = path


class ScoringError(StockAnalysisError):
    """Score computation error."""

    def __init__(self, message: str, component: Optional[str] = None, **kwargs: Any):
        full_message = f"Scoring error" + (f" in {component}" if component else "") + f": {message}"
        super().__init__(full_message, code="SCORING_ERROR", **kwargs)
        self.component = component


class ProbabilityError(StockAnalysisError):
    """Probability estimation error."""

    def __init__(self, horizon: str, message: str, **kwargs: Any):
        full_message = f"Probability estimation ({horizon}): {message}"
        super().__init__(full_message, code="PROBABILITY_ERROR", **kwargs)
        self.horizon = horizon


class ConfigError(StockAnalysisError):
    """Configuration error."""

    def __init__(self, message: str, key: Optional[str] = None, **kwargs: Any):
        full_message = f"Configuration error" + (f" for '{key}'" if key else "") + f": {message}"
        super().__init__(full_message, code="CONFIG_ERROR", **kwargs)
        self.key = key


class ValidationError(StockAnalysisError):
    """Input validation error."""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs: Any):
        full_message = f"Validation error" + (f" for '{field}'" if field else "") + f": {message}"
        super().__init__(full_message, code="VALIDATION_ERROR", **kwargs)
        self.field = field


class PipelineError(StockAnalysisError):
    """Pipeline execution error."""

    def __init__(
        self,
        message: str,
        stage: str,
        ticker: Optional[str] = None,
        **kwargs: Any,
    ):
        full_message = f"Pipeline error at {stage}"
        if ticker:
            full_message += f" for {ticker}"
        full_message += f": {message}"
        super().__init__(full_message, code="PIPELINE_ERROR", **kwargs)
        self.stage = stage
        self.ticker = ticker
