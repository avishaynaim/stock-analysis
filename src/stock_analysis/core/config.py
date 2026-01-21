"""
Configuration management for the stock analysis system.
"""

import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DataPathsConfig(BaseModel):
    """Data paths configuration."""

    base_dir: str = Field(default_factory=lambda: str(Path.home() / ".stock-analysis"))
    cache_dir: str = ""
    models_dir: str = ""
    output_dir: str = ""
    logs_dir: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Set derived paths after initialization."""
        if not self.cache_dir:
            self.cache_dir = str(Path(self.base_dir) / "cache")
        if not self.models_dir:
            self.models_dir = str(Path(self.base_dir) / "models")
        if not self.output_dir:
            self.output_dir = str(Path(self.base_dir) / "output")
        if not self.logs_dir:
            self.logs_dir = str(Path(self.base_dir) / "logs")


class DataSourceConfig(BaseModel):
    """Data source configuration."""

    provider: str = "yfinance"
    api_key: Optional[str] = None
    rate_limit: int = 5
    cache_days: int = 1


class DataQualityConfig(BaseModel):
    """Data quality configuration."""

    min_history_days: int = 252
    max_gap_days: int = 5
    outlier_threshold: float = 5.0
    validate_on_load: bool = True


class DataConfig(BaseModel):
    """Data configuration."""

    paths: DataPathsConfig = Field(default_factory=DataPathsConfig)
    sources: dict[str, DataSourceConfig] = Field(
        default_factory=lambda: {"prices": DataSourceConfig()}
    )
    quality: DataQualityConfig = Field(default_factory=DataQualityConfig)


class UniverseFiltersConfig(BaseModel):
    """Universe filter configuration."""

    min_price: float = 5.0
    min_market_cap: Optional[float] = 1_000_000_000
    min_avg_volume: float = 500_000
    exclude_otc: bool = True
    exclude_adrs: bool = False


class UniverseConfig(BaseModel):
    """Universe configuration."""

    default: str = "SP500"
    filters: UniverseFiltersConfig = Field(default_factory=UniverseFiltersConfig)


class IndicatorsConfig(BaseModel):
    """Indicators configuration."""

    groups: dict[str, bool] = Field(
        default_factory=lambda: {
            "trend": True,
            "momentum": True,
            "volatility": True,
            "volume": True,
            "microstructure": True,
            "regime": True,
            "relative_strength": True,
            "structure": True,
            "fundamentals": True,
            "sentiment": False,
        }
    )
    timeframes: list[str] = Field(default_factory=lambda: ["1d", "1w", "1mo"])
    min_periods: int = 50
    min_required: int = 20


class ScoringWeightsConfig(BaseModel):
    """Scoring weights configuration."""

    trend: float = 0.18
    momentum: float = 0.18
    volume: float = 0.12
    relative_strength: float = 0.12
    fundamental: float = 0.08
    edge: float = 0.32


class RiskPenaltyConfig(BaseModel):
    """Individual risk penalty configuration."""

    enabled: bool = True
    threshold: Optional[float] = None
    max_penalty: float = 1.0


class RiskPenaltiesConfig(BaseModel):
    """Risk penalties configuration."""

    volatility: RiskPenaltyConfig = Field(
        default_factory=lambda: RiskPenaltyConfig(threshold=0.90, max_penalty=1.0)
    )
    drawdown: RiskPenaltyConfig = Field(
        default_factory=lambda: RiskPenaltyConfig(threshold=-0.15, max_penalty=1.5)
    )
    liquidity: RiskPenaltyConfig = Field(
        default_factory=lambda: RiskPenaltyConfig(threshold=500_000, max_penalty=0.5)
    )
    gap_risk: RiskPenaltyConfig = Field(
        default_factory=lambda: RiskPenaltyConfig(threshold=0.05, max_penalty=0.5)
    )
    max_total: float = 3.0


class ScoringConfig(BaseModel):
    """Scoring configuration."""

    weights: ScoringWeightsConfig = Field(default_factory=ScoringWeightsConfig)
    risk_penalties: RiskPenaltiesConfig = Field(default_factory=RiskPenaltiesConfig)
    horizon_weights: dict[str, float] = Field(
        default_factory=lambda: {"5d": 0.2, "21d": 0.5, "63d": 0.3}
    )


class ProbabilityHorizonConfig(BaseModel):
    """Probability horizon configuration."""

    days: int
    threshold: float


class ProbabilityConfig(BaseModel):
    """Probability engine configuration."""

    horizons: list[ProbabilityHorizonConfig] = Field(
        default_factory=lambda: [
            ProbabilityHorizonConfig(days=5, threshold=0.05),
            ProbabilityHorizonConfig(days=21, threshold=0.10),
            ProbabilityHorizonConfig(days=63, threshold=0.15),
        ]
    )
    estimators: dict[str, float] = Field(
        default_factory=lambda: {"empirical": 0.40, "supervised": 0.40, "similarity": 0.20}
    )
    min_samples: dict[str, int] = Field(
        default_factory=lambda: {"empirical": 30, "similarity": 50}
    )


class ModelTrainingConfig(BaseModel):
    """Model training configuration."""

    validation_split: float = 0.2
    early_stopping_rounds: int = 20
    seed: int = 42


class ModelsConfig(BaseModel):
    """Models configuration."""

    default: str = "ensemble"
    min_required: int = 1
    training: ModelTrainingConfig = Field(default_factory=ModelTrainingConfig)
    version: str = "1.0.0"


class ParallelConfig(BaseModel):
    """Parallel execution configuration."""

    enabled: bool = True
    max_workers: Optional[int] = None
    chunk_size: int = 50
    backend: str = "multiprocessing"


class ExecutionConfig(BaseModel):
    """Execution configuration."""

    parallel: ParallelConfig = Field(default_factory=ParallelConfig)


class ConsoleLoggingConfig(BaseModel):
    """Console logging configuration."""

    enabled: bool = True
    level: str = "INFO"
    format: str = "simple"
    colors: bool = True


class FileLoggingConfig(BaseModel):
    """File logging configuration."""

    enabled: bool = True
    level: str = "DEBUG"
    path: str = ""
    max_size_mb: int = 100
    backup_count: int = 5


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    console: ConsoleLoggingConfig = Field(default_factory=ConsoleLoggingConfig)
    file: FileLoggingConfig = Field(default_factory=FileLoggingConfig)
    components: dict[str, str] = Field(default_factory=dict)


class ReportingConfig(BaseModel):
    """Reporting configuration."""

    default_format: str = "text"
    include_indicators: bool = True
    include_probability: bool = True


class AppConfig(BaseSettings):
    """Main application configuration."""

    version: str = "1.0"
    data: DataConfig = Field(default_factory=DataConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    indicators: IndicatorsConfig = Field(default_factory=IndicatorsConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    probability: ProbabilityConfig = Field(default_factory=ProbabilityConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    class Config:
        env_prefix = "STOCK_ANALYSIS_"
        env_nested_delimiter = "__"


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
        cli_overrides: Optional[dict[str, Any]] = None,
    ):
        self.config_path = config_path
        self.cli_overrides = cli_overrides or {}

    def load(self) -> AppConfig:
        """Load and merge configuration from all sources."""
        # Load file config
        file_config = self._load_file_config()

        # Resolve variables
        file_config = self._resolve_variables(file_config)

        # Create config from file data
        config = AppConfig(**file_config) if file_config else AppConfig()

        return config

    def _load_file_config(self) -> dict[str, Any]:
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

        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}

    def _resolve_variables(
        self, config: dict[str, Any], root: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
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
                    self._resolve_string(v, root) if isinstance(v, str) else v for v in value
                ]
            else:
                result[key] = value

        return result

    def _resolve_string(self, value: str, root: dict[str, Any]) -> str:
        """Resolve variables in a string."""

        def replace_var(match: re.Match[str]) -> str:
            var_name = match.group(1)

            # Environment variable
            if var_name in os.environ:
                return os.environ[var_name]

            # Special variables
            if var_name == "HOME":
                return str(Path.home())

            # Config reference (e.g., data.paths.base_dir)
            parts = var_name.split(".")
            current: Any = root
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return match.group(0)  # Keep original if not found

            return str(current) if not isinstance(current, dict) else match.group(0)

        return re.sub(r"\$\{([^}]+)\}", replace_var, value)


class Config:
    """
    Global configuration accessor.

    Thread-safe singleton for accessing configuration values.
    """

    _instance: Optional["Config"] = None
    _lock = threading.Lock()
    _config: Optional[AppConfig] = None

    def __new__(cls) -> "Config":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, config: AppConfig) -> None:
        """Initialize configuration."""
        cls._config = config

    @classmethod
    def get_config(cls) -> AppConfig:
        """Get the full configuration object."""
        if cls._config is None:
            # Load default config
            loader = ConfigLoader()
            cls._config = loader.load()
        return cls._config

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Example: Config.get('data.paths.base_dir')
        """
        config = cls.get_config()

        parts = key.split(".")
        current: Any = config

        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current


def get_config(key: Optional[str] = None, default: Any = None) -> Any:
    """
    Get configuration value.

    If key is None, returns the full config object.
    """
    if key is None:
        return Config.get_config()
    return Config.get(key, default)
