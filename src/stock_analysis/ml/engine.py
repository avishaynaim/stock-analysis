"""
ML Forecast Engine - Orchestrates ML-based stock prediction.

Supports per-ticker model training, where each stock gets its own
trained model based on its unique historical patterns.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, field

from .data_builder import MLDataBuilder, MLDataset
from .models import GainPredictor, PatternRecognizer, PredictionResult, ModelMetrics
from .model_storage import ModelStorage, ModelInfo, get_model_storage


@dataclass
class MLForecast:
    """Complete ML forecast result."""
    symbol: str
    prediction: PredictionResult
    similar_patterns: dict[str, Any]
    model_metrics: ModelMetrics
    feature_importance: pd.DataFrame
    signal_strength: str
    confidence_level: str
    key_insights: list[str]
    model_source: str = "trained"  # "trained", "cached", or "retrained"


class MLForecastEngine:
    """Main engine for ML-based stock forecasting.

    Supports per-ticker model training where each stock gets its own
    trained model, since different stocks react differently to
    indicator patterns.
    """

    def __init__(
        self,
        forward_days: int = 20,
        gain_threshold: float = 0.10,
        big_gain_threshold: float = 0.15,
        use_model_cache: bool = True,
        model_storage: Optional[ModelStorage] = None,
    ):
        self.forward_days = forward_days
        self.gain_threshold = gain_threshold
        self.big_gain_threshold = big_gain_threshold
        self.use_model_cache = use_model_cache
        self.model_storage = model_storage or (get_model_storage() if use_model_cache else None)

        self.data_builder = MLDataBuilder(
            forward_days=forward_days,
            gain_threshold=gain_threshold,
            big_gain_threshold=big_gain_threshold,
        )
        self.predictor = GainPredictor()
        self.pattern_recognizer = PatternRecognizer()

        self.is_trained = False
        self.dataset: Optional[MLDataset] = None
        self.current_ticker: Optional[str] = None
        self._model_source: str = "trained"

    def train(
        self,
        prices: pd.DataFrame,
        indicators: pd.DataFrame,
        target_type: str = "big_gain",
        ticker: Optional[str] = None,
        force_retrain: bool = False,
    ) -> dict[str, Any]:
        """Train the model on a ticker's historical data.

        Args:
            prices: Price data for the ticker
            indicators: Indicator values for the ticker
            target_type: Target type for prediction
            ticker: Ticker symbol (for per-ticker caching)
            force_retrain: Force retraining even if cached model exists

        Returns:
            Dictionary with training metrics
        """
        self.current_ticker = ticker.upper() if ticker else None
        self._model_source = "trained"

        # Try to load cached model if available
        if (
            self.use_model_cache
            and self.model_storage
            and ticker
            and not force_retrain
        ):
            current_data_end = prices.index[-1].to_pydatetime() if hasattr(prices.index[-1], 'to_pydatetime') else datetime.now()

            # Check if cached model is recent enough
            if not self.model_storage.should_retrain(ticker, current_data_end, min_new_days=5):
                # Load cached model
                if self.model_storage.load_model(ticker, self.predictor, self.pattern_recognizer):
                    self.is_trained = True
                    self._model_source = "cached"

                    info = self.model_storage.get_model_info(ticker)
                    return {
                        "n_samples": info.n_samples if info else 0,
                        "n_features": info.n_features if info else 0,
                        "target_distribution": {},
                        "accuracy": info.accuracy if info else 0,
                        "precision": info.precision if info else 0,
                        "recall": info.recall if info else 0,
                        "f1": info.f1 if info else 0,
                        "roc_auc": info.roc_auc if info else 0,
                        "cv_mean": info.cv_mean if info else 0,
                        "cv_std": info.cv_std if info else 0,
                        "model_source": "cached",
                    }

        # Build dataset and train new model
        self.dataset = self.data_builder.build_enhanced_dataset(
            prices=prices,
            indicators=indicators,
            target_type=target_type,
            include_patterns=True,
            include_cross=True,
        )

        # Reset predictor and pattern recognizer for fresh training
        self.predictor = GainPredictor()
        self.pattern_recognizer = PatternRecognizer()

        metrics = self.predictor.train(self.dataset)

        forward_returns = self._calculate_forward_returns(prices)
        self.pattern_recognizer.fit(
            features=self.dataset.features,
            forward_returns=forward_returns.loc[self.dataset.features.index],
        )

        self.is_trained = True

        # Save model to cache
        if self.use_model_cache and self.model_storage and ticker:
            self._save_model_to_cache(ticker, prices, target_type, metrics)
            self._model_source = "retrained" if self.model_storage.has_model(ticker) else "trained"

        return {
            "n_samples": self.dataset.metadata["n_samples"],
            "n_features": self.dataset.metadata["n_features"],
            "target_distribution": self.dataset.metadata["target_distribution"],
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "roc_auc": metrics.roc_auc,
            "cv_mean": metrics.cv_mean,
            "cv_std": metrics.cv_std,
            "model_source": self._model_source,
        }

    def _save_model_to_cache(
        self,
        ticker: str,
        prices: pd.DataFrame,
        target_type: str,
        metrics: ModelMetrics,
    ) -> None:
        """Save trained model to cache storage."""
        try:
            data_start = str(prices.index[0].date() if hasattr(prices.index[0], 'date') else prices.index[0])
            data_end = str(prices.index[-1].date() if hasattr(prices.index[-1], 'date') else prices.index[-1])

            model_info = ModelInfo(
                ticker=ticker.upper(),
                trained_at=datetime.now().isoformat(),
                n_samples=self.dataset.metadata["n_samples"],
                n_features=self.dataset.metadata["n_features"],
                accuracy=metrics.accuracy,
                precision=metrics.precision,
                recall=metrics.recall,
                f1=metrics.f1,
                roc_auc=metrics.roc_auc,
                cv_mean=metrics.cv_mean,
                cv_std=metrics.cv_std,
                feature_names=self.predictor.feature_names,
                target_type=target_type,
                forward_days=self.forward_days,
                gain_threshold=self.gain_threshold,
                big_gain_threshold=self.big_gain_threshold,
                data_start=data_start,
                data_end=data_end,
            )

            self.model_storage.save_model(
                ticker=ticker,
                predictor=self.predictor,
                pattern_recognizer=self.pattern_recognizer,
                model_info=model_info,
            )
        except Exception as e:
            # Don't fail the whole operation if caching fails
            import logging
            logging.getLogger(__name__).warning(f"Failed to cache model for {ticker}: {e}")

    def _calculate_forward_returns(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"] if "close" in prices.columns else prices["adj_close"]
        return close.shift(-self.forward_days) / close - 1

    def forecast(
        self,
        prices: pd.DataFrame,
        indicators: pd.DataFrame,
        ticker: Optional[str] = None,
        force_retrain: bool = False,
    ) -> MLForecast:
        """Generate ML forecast for a ticker.

        Args:
            prices: Price data for the ticker
            indicators: Indicator values for the ticker
            ticker: Ticker symbol (for per-ticker model caching)
            force_retrain: Force retraining even if cached model exists

        Returns:
            MLForecast with prediction and analysis
        """
        # Check if we need to train/load model for this ticker
        if ticker and ticker.upper() != self.current_ticker:
            # Different ticker - need to load/train its specific model
            self.is_trained = False

        if not self.is_trained:
            self.train(prices, indicators, ticker=ticker, force_retrain=force_retrain)

        enhanced_indicators = self._enhance_indicators(indicators)
        latest_features = enhanced_indicators.iloc[[-1]]

        prediction = self.predictor.predict(latest_features)
        similar_patterns = self.pattern_recognizer.find_similar_patterns(latest_features)
        feature_importance = self.predictor.get_feature_importance()

        signal_strength = self._determine_signal(prediction, similar_patterns)
        confidence_level = self._determine_confidence(prediction, similar_patterns)
        key_insights = self._generate_insights(
            prediction, similar_patterns, feature_importance
        )

        return MLForecast(
            symbol=ticker or "",
            prediction=prediction,
            similar_patterns=similar_patterns,
            model_metrics=self.predictor.metrics,
            feature_importance=feature_importance,
            signal_strength=signal_strength,
            confidence_level=confidence_level,
            key_insights=key_insights,
            model_source=self._model_source,
        )

    def _enhance_indicators(self, indicators: pd.DataFrame) -> pd.DataFrame:
        pattern_features = self.data_builder.create_pattern_features(indicators)
        cross_features = self.data_builder.create_cross_features(indicators)

        enhanced = pd.concat([indicators, pattern_features, cross_features], axis=1)

        if self.dataset is not None:
            available_cols = [c for c in self.dataset.feature_names if c in enhanced.columns]
            enhanced = enhanced[available_cols]

        return enhanced

    def _determine_signal(
        self,
        prediction: PredictionResult,
        similar_patterns: dict[str, Any],
    ) -> str:
        prob = prediction.probability
        hist_rate = similar_patterns.get("big_gain_rate", 0)
        combined_score = (prob + hist_rate) / 2

        if combined_score >= 0.6 and prediction.confidence >= 0.5:
            return "Strong Buy"
        elif combined_score >= 0.45:
            return "Buy"
        elif combined_score >= 0.3:
            return "Neutral"
        else:
            return "Avoid"

    def _determine_confidence(
        self,
        prediction: PredictionResult,
        similar_patterns: dict[str, Any],
    ) -> str:
        model_confidence = prediction.confidence
        hist_rate = similar_patterns.get("big_gain_rate", 0)
        agreement = abs(prediction.probability - hist_rate) < 0.2

        if model_confidence >= 0.6 and agreement:
            return "High"
        elif model_confidence >= 0.4 or agreement:
            return "Medium"
        else:
            return "Low"

    def _generate_insights(
        self,
        prediction: PredictionResult,
        similar_patterns: dict[str, Any],
        feature_importance: pd.DataFrame,
    ) -> list[str]:
        insights = []

        prob_pct = prediction.probability * 100
        insights.append(
            f"ML model predicts {prob_pct:.1f}% probability of {self.big_gain_threshold*100:.0f}%+ gain "
            f"in next {self.forward_days} trading days"
        )

        hist_rate = similar_patterns.get("big_gain_rate", 0) * 100
        avg_return = similar_patterns.get("avg_return", 0) * 100
        insights.append(
            f"Similar historical patterns showed {hist_rate:.0f}% big gain rate "
            f"with avg return of {avg_return:+.1f}%"
        )

        top_features = feature_importance.head(3)
        top_names = top_features["feature"].tolist()
        insights.append(
            f"Top signals: {', '.join(top_names)}"
        )

        best = similar_patterns.get("best_return", 0) * 100
        worst = similar_patterns.get("worst_return", 0) * 100
        insights.append(
            f"Historical range: {worst:+.1f}% to {best:+.1f}%"
        )

        return insights

    def get_backtest_results(
        self,
        prices: pd.DataFrame,
        indicators: pd.DataFrame,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        if not self.is_trained:
            self.train(prices, indicators)

        enhanced = self._enhance_indicators(indicators)
        available_features = [f for f in self.dataset.feature_names if f in enhanced.columns]
        X = enhanced[available_features].dropna()

        X_scaled = self.predictor.scaler.transform(X.values)
        rf_proba = self.predictor.rf_model.predict_proba(X_scaled)[:, 1]
        gb_proba = self.predictor.gb_model.predict_proba(X_scaled)[:, 1]
        probabilities = (rf_proba + gb_proba) / 2

        close = prices["close"] if "close" in prices.columns else prices["adj_close"]
        forward_returns = close.shift(-self.forward_days) / close - 1

        backtest = pd.DataFrame(index=X.index)
        backtest["probability"] = probabilities
        backtest["signal"] = (probabilities >= threshold).astype(int)
        backtest["forward_return"] = forward_returns.loc[X.index]
        backtest["actual_big_gain"] = (backtest["forward_return"] >= self.big_gain_threshold).astype(int)
        backtest["strategy_return"] = backtest["signal"] * backtest["forward_return"]
        backtest["cumulative_return"] = (1 + backtest["forward_return"]).cumprod() - 1
        backtest["strategy_cumulative"] = (1 + backtest["strategy_return"]).cumprod() - 1

        return backtest.dropna()

    def get_performance_summary(self, backtest: pd.DataFrame) -> dict[str, Any]:
        signals = backtest[backtest["signal"] == 1]

        if len(signals) == 0:
            return {"error": "No signals generated"}

        wins = (signals["forward_return"] > 0).sum()
        win_rate = wins / len(signals)

        big_gains = signals["actual_big_gain"].sum()
        big_gain_rate = big_gains / len(signals)

        avg_return = signals["forward_return"].mean()
        avg_win = signals[signals["forward_return"] > 0]["forward_return"].mean()
        avg_loss = signals[signals["forward_return"] <= 0]["forward_return"].mean()

        returns_std = signals["forward_return"].std()
        sharpe = avg_return / returns_std if returns_std > 0 else 0

        return {
            "total_signals": len(signals),
            "win_rate": float(win_rate),
            "big_gain_hit_rate": float(big_gain_rate),
            "avg_return": float(avg_return),
            "avg_win": float(avg_win) if not pd.isna(avg_win) else 0,
            "avg_loss": float(avg_loss) if not pd.isna(avg_loss) else 0,
            "sharpe_ratio": float(sharpe),
            "total_return": float(backtest["strategy_cumulative"].iloc[-1]),
            "buy_hold_return": float(backtest["cumulative_return"].iloc[-1]),
        }
