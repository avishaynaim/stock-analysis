"""
Per-ticker ML model storage and management.

Trains and stores separate ML models for each ticker,
since different stocks react differently to indicator patterns.
"""

import os
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np

from stock_analysis.core.logging import get_logger

logger = get_logger("ml.model_storage")


@dataclass
class ModelInfo:
    """Metadata about a trained model."""
    ticker: str
    trained_at: str
    n_samples: int
    n_features: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    cv_mean: float
    cv_std: float
    feature_names: list[str]
    target_type: str
    forward_days: int
    gain_threshold: float
    big_gain_threshold: float
    data_start: str
    data_end: str


class ModelStorage:
    """
    Persistent storage for per-ticker ML models.

    Each ticker gets its own trained model stored as a pickle file,
    allowing different stocks to have their own learned patterns.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize model storage.

        Args:
            storage_dir: Directory for storing model files
        """
        self.storage_dir = storage_dir or Path.home() / ".stock-analysis" / "models"
        self.models_dir = self.storage_dir / "tickers"
        self.metadata_file = self.storage_dir / "models_metadata.json"

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Load metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load model metadata: {e}")
        return {"models": {}}

    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self._metadata, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save model metadata: {e}")

    def _get_model_path(self, ticker: str) -> Path:
        """Get path for ticker model file."""
        return self.models_dir / f"{ticker.upper()}_model.pkl"

    def has_model(self, ticker: str) -> bool:
        """Check if we have a trained model for ticker."""
        return self._get_model_path(ticker).exists()

    def get_model_info(self, ticker: str) -> Optional[ModelInfo]:
        """Get metadata about a stored model."""
        ticker = ticker.upper()
        if ticker in self._metadata.get("models", {}):
            return ModelInfo(**self._metadata["models"][ticker])
        return None

    def save_model(
        self,
        ticker: str,
        predictor: Any,  # GainPredictor
        pattern_recognizer: Any,  # PatternRecognizer
        model_info: ModelInfo,
    ) -> None:
        """Save a trained model to disk.

        Args:
            ticker: Stock ticker symbol
            predictor: Trained GainPredictor instance
            pattern_recognizer: Trained PatternRecognizer instance
            model_info: Model metadata
        """
        ticker = ticker.upper()
        path = self._get_model_path(ticker)

        try:
            # Save model components
            model_data = {
                "predictor": {
                    "rf_model": predictor.rf_model,
                    "gb_model": predictor.gb_model,
                    "scaler": predictor.scaler,
                    "feature_names": predictor.feature_names,
                    "metrics": predictor.metrics,
                    "feature_importance": predictor.feature_importance,
                },
                "pattern_recognizer": {
                    "historical_features": pattern_recognizer.historical_features,
                    "historical_returns": pattern_recognizer.historical_returns,
                    "historical_dates": pattern_recognizer.historical_dates,
                    "scaler": pattern_recognizer.scaler,
                    "feature_names": pattern_recognizer.feature_names,
                    "n_neighbors": pattern_recognizer.n_neighbors,
                },
            }

            with open(path, "wb") as f:
                pickle.dump(model_data, f)

            # Update metadata
            if "models" not in self._metadata:
                self._metadata["models"] = {}

            self._metadata["models"][ticker] = asdict(model_info)
            self._save_metadata()

            logger.info(f"Saved model for {ticker} ({model_info.n_samples} samples, AUC: {model_info.roc_auc:.3f})")

        except Exception as e:
            logger.error(f"Failed to save model for {ticker}: {e}")
            raise

    def load_model(
        self,
        ticker: str,
        predictor: Any,  # GainPredictor
        pattern_recognizer: Any,  # PatternRecognizer
    ) -> bool:
        """Load a trained model from disk.

        Args:
            ticker: Stock ticker symbol
            predictor: GainPredictor instance to populate
            pattern_recognizer: PatternRecognizer instance to populate

        Returns:
            True if model was loaded successfully
        """
        path = self._get_model_path(ticker)

        if not path.exists():
            return False

        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

            # Restore predictor
            pred_data = model_data["predictor"]
            predictor.rf_model = pred_data["rf_model"]
            predictor.gb_model = pred_data["gb_model"]
            predictor.scaler = pred_data["scaler"]
            predictor.feature_names = pred_data["feature_names"]
            predictor.metrics = pred_data["metrics"]
            predictor.feature_importance = pred_data["feature_importance"]
            predictor.is_trained = True

            # Restore pattern recognizer
            pr_data = model_data["pattern_recognizer"]
            pattern_recognizer.historical_features = pr_data["historical_features"]
            pattern_recognizer.historical_returns = pr_data["historical_returns"]
            pattern_recognizer.historical_dates = pr_data["historical_dates"]
            pattern_recognizer.scaler = pr_data["scaler"]
            pattern_recognizer.feature_names = pr_data["feature_names"]
            pattern_recognizer.n_neighbors = pr_data["n_neighbors"]

            logger.info(f"Loaded cached model for {ticker}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load model for {ticker}: {e}")
            return False

    def should_retrain(
        self,
        ticker: str,
        current_data_end: datetime,
        min_new_days: int = 5,
    ) -> bool:
        """Check if model should be retrained with new data.

        Args:
            ticker: Stock ticker symbol
            current_data_end: End date of current data
            min_new_days: Minimum new days to trigger retrain

        Returns:
            True if model should be retrained
        """
        info = self.get_model_info(ticker)
        if info is None:
            return True

        try:
            model_data_end = datetime.fromisoformat(info.data_end)
            days_diff = (current_data_end - model_data_end).days
            return days_diff >= min_new_days
        except:
            return True

    def list_models(self) -> list[str]:
        """List all tickers with stored models."""
        tickers = []
        for path in self.models_dir.glob("*_model.pkl"):
            ticker = path.stem.replace("_model", "")
            tickers.append(ticker.upper())
        return sorted(tickers)

    def get_stats(self) -> dict:
        """Get storage statistics."""
        total_size = 0
        model_count = 0

        for path in self.models_dir.glob("*_model.pkl"):
            model_count += 1
            total_size += path.stat().st_size

        # Get average metrics
        avg_auc = 0.0
        avg_accuracy = 0.0
        if self._metadata.get("models"):
            aucs = [m.get("roc_auc", 0) for m in self._metadata["models"].values()]
            accs = [m.get("accuracy", 0) for m in self._metadata["models"].values()]
            avg_auc = sum(aucs) / len(aucs) if aucs else 0
            avg_accuracy = sum(accs) / len(accs) if accs else 0

        return {
            "model_count": model_count,
            "total_size_mb": total_size / (1024 * 1024),
            "avg_roc_auc": avg_auc,
            "avg_accuracy": avg_accuracy,
            "storage_dir": str(self.storage_dir),
        }

    def delete_model(self, ticker: str) -> bool:
        """Delete a ticker's model."""
        ticker = ticker.upper()
        path = self._get_model_path(ticker)

        if path.exists():
            try:
                path.unlink()
                if ticker in self._metadata.get("models", {}):
                    del self._metadata["models"][ticker]
                    self._save_metadata()
                return True
            except Exception as e:
                logger.error(f"Failed to delete model for {ticker}: {e}")
        return False

    def clear_all(self) -> None:
        """Delete all stored models."""
        for path in self.models_dir.glob("*_model.pkl"):
            try:
                path.unlink()
            except:
                pass

        self._metadata = {"models": {}}
        self._save_metadata()


# Global storage instance
_model_storage: Optional[ModelStorage] = None


def get_model_storage() -> ModelStorage:
    """Get global model storage instance."""
    global _model_storage
    if _model_storage is None:
        _model_storage = ModelStorage()
    return _model_storage
