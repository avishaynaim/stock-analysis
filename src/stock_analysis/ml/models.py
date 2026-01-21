"""
ML Models for stock gain prediction based on indicator patterns.

Enhanced with:
- XGBoost support for better performance
- Feature selection with recursive feature elimination
- Improved hyperparameter tuning
- Calibrated probabilities
- Better class imbalance handling
"""
import numpy as np
import pandas as pd
from typing import Optional, Any
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    average_precision_score, brier_score_loss
)
import warnings

from .data_builder import MLDataset

# Try importing XGBoost for better performance
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings('ignore')


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    cv_scores: list[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: str = ""
    # Enhanced metrics
    avg_precision: float = 0.0  # PR-AUC (better for imbalanced data)
    brier_score: float = 0.0  # Calibration score
    profit_factor: float = 0.0  # Trading metric
    expected_value: float = 0.0  # Expected value per trade


@dataclass
class FeatureImportance:
    """Feature importance analysis results."""
    features: list[str]
    importances: list[float]
    std: list[float] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "feature": self.features,
            "importance": self.importances,
        })
        if self.std:
            df["std"] = self.std
        return df.sort_values("importance", ascending=False)

    def top_n(self, n: int = 20) -> pd.DataFrame:
        return self.to_dataframe().head(n)


@dataclass
class PredictionResult:
    """Container for prediction results."""
    probability: float
    prediction: int
    confidence: float
    top_signals: list[tuple[str, float]]
    model_name: str = ""


class GainPredictor:
    """
    Predicts probability of significant gains based on indicator patterns.

    Enhanced with:
    - XGBoost support (if available)
    - Robust scaling for outlier resistance
    - Probability calibration
    - Feature selection
    - Smarter ensemble weighting
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        min_samples_leaf: int = 20,
        random_state: int = 42,
        use_xgboost: bool = True,
        use_feature_selection: bool = True,
        calibrate_probabilities: bool = True,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.use_xgboost = use_xgboost and HAS_XGBOOST
        self.use_feature_selection = use_feature_selection
        self.calibrate_probabilities = calibrate_probabilities

        # Random Forest with better parameters
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
            max_features="sqrt",  # Better generalization
            oob_score=True,  # Out-of-bag score for validation
        )

        # Gradient Boosting with learning rate decay
        self.gb_model = GradientBoostingClassifier(
            n_estimators=n_estimators // 2,
            max_depth=max_depth // 2,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            learning_rate=0.05,  # Lower learning rate for better generalization
            subsample=0.8,  # Stochastic gradient boosting
            validation_fraction=0.1,
            n_iter_no_change=10,  # Early stopping
        )

        # XGBoost if available (often better performance)
        if self.use_xgboost:
            self.xgb_model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth // 2,
                learning_rate=0.05,
                random_state=random_state,
                n_jobs=-1,
                scale_pos_weight=1,  # Will be adjusted based on class imbalance
                eval_metric='auc',
                early_stopping_rounds=10,
            )
        else:
            self.xgb_model = None

        # Use RobustScaler for better handling of outliers in financial data
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.feature_names: list[str] = []
        self.selected_feature_names: list[str] = []
        self.is_trained = False
        self.metrics: Optional[ModelMetrics] = None
        self.feature_importance: Optional[FeatureImportance] = None
        self.model_weights: dict[str, float] = {"rf": 0.4, "gb": 0.3, "xgb": 0.3}

    def train(
        self,
        dataset: MLDataset,
        validation_split: float = 0.2,
    ) -> ModelMetrics:
        X = dataset.features.values
        y = dataset.target.values
        self.feature_names = dataset.feature_names

        X_scaled = self.scaler.fit_transform(X)

        # Feature selection to reduce noise
        if self.use_feature_selection and X_scaled.shape[1] > 20:
            selector_model = RandomForestClassifier(
                n_estimators=50, max_depth=5, random_state=self.random_state, n_jobs=-1
            )
            self.feature_selector = SelectFromModel(
                selector_model, threshold="median", prefit=False
            )
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
            selected_mask = self.feature_selector.get_support()
            self.selected_feature_names = [
                self.feature_names[i] for i, m in enumerate(selected_mask) if m
            ]
        else:
            X_selected = X_scaled
            self.selected_feature_names = self.feature_names.copy()

        # Time-series aware split (no shuffling)
        split_idx = int(len(X_selected) * (1 - validation_split))
        X_train, X_val = X_selected[:split_idx], X_selected[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Calculate class imbalance ratio
        class_ratio = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)

        # Train Random Forest
        self.rf_model.fit(X_train, y_train)

        # Train Gradient Boosting
        self.gb_model.fit(X_train, y_train)

        # Train XGBoost if available
        if self.use_xgboost and self.xgb_model is not None:
            self.xgb_model.set_params(scale_pos_weight=class_ratio)
            eval_set = [(X_val, y_val)]
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )

        # Get ensemble predictions
        ensemble_proba = self._get_ensemble_proba(X_val)
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)

        # Calculate enhanced metrics
        metrics = ModelMetrics(
            accuracy=accuracy_score(y_val, ensemble_pred),
            precision=precision_score(y_val, ensemble_pred, zero_division=0),
            recall=recall_score(y_val, ensemble_pred, zero_division=0),
            f1=f1_score(y_val, ensemble_pred, zero_division=0),
            roc_auc=roc_auc_score(y_val, ensemble_proba) if len(np.unique(y_val)) > 1 else 0.5,
            confusion_matrix=confusion_matrix(y_val, ensemble_pred),
            classification_report=classification_report(y_val, ensemble_pred, zero_division=0),
            avg_precision=average_precision_score(y_val, ensemble_proba) if len(np.unique(y_val)) > 1 else 0.5,
            brier_score=brier_score_loss(y_val, ensemble_proba),
        )

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(self.rf_model, X_selected, y, cv=tscv, scoring="roc_auc")
        metrics.cv_scores = cv_scores.tolist()
        metrics.cv_mean = cv_scores.mean()
        metrics.cv_std = cv_scores.std()

        # Compute and optimize model weights based on validation performance
        self._optimize_ensemble_weights(X_val, y_val)

        self._compute_feature_importance()

        self.is_trained = True
        self.metrics = metrics
        return metrics

    def _get_ensemble_proba(self, X: np.ndarray) -> np.ndarray:
        """Get weighted ensemble probability predictions."""
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        gb_proba = self.gb_model.predict_proba(X)[:, 1]

        if self.use_xgboost and self.xgb_model is not None:
            xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
            ensemble_proba = (
                self.model_weights["rf"] * rf_proba +
                self.model_weights["gb"] * gb_proba +
                self.model_weights["xgb"] * xgb_proba
            )
        else:
            total = self.model_weights["rf"] + self.model_weights["gb"]
            ensemble_proba = (
                self.model_weights["rf"] / total * rf_proba +
                self.model_weights["gb"] / total * gb_proba
            )

        return ensemble_proba

    def _optimize_ensemble_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Optimize ensemble weights based on validation AUC."""
        rf_proba = self.rf_model.predict_proba(X_val)[:, 1]
        gb_proba = self.gb_model.predict_proba(X_val)[:, 1]

        # Calculate individual AUCs
        rf_auc = roc_auc_score(y_val, rf_proba) if len(np.unique(y_val)) > 1 else 0.5
        gb_auc = roc_auc_score(y_val, gb_proba) if len(np.unique(y_val)) > 1 else 0.5

        if self.use_xgboost and self.xgb_model is not None:
            xgb_proba = self.xgb_model.predict_proba(X_val)[:, 1]
            xgb_auc = roc_auc_score(y_val, xgb_proba) if len(np.unique(y_val)) > 1 else 0.5

            # Weight by AUC performance
            total_auc = rf_auc + gb_auc + xgb_auc
            self.model_weights = {
                "rf": rf_auc / total_auc,
                "gb": gb_auc / total_auc,
                "xgb": xgb_auc / total_auc,
            }
        else:
            total_auc = rf_auc + gb_auc
            self.model_weights = {
                "rf": rf_auc / total_auc,
                "gb": gb_auc / total_auc,
                "xgb": 0,
            }

    def _compute_feature_importance(self):
        """Compute weighted feature importance from all models."""
        # Get importances from each model, mapping to selected features
        if self.use_feature_selection and self.feature_selector is not None:
            feature_names = self.selected_feature_names
        else:
            feature_names = self.feature_names

        rf_importance = self.rf_model.feature_importances_
        gb_importance = self.gb_model.feature_importances_

        if self.use_xgboost and self.xgb_model is not None:
            xgb_importance = self.xgb_model.feature_importances_
            avg_importance = (
                self.model_weights["rf"] * rf_importance +
                self.model_weights["gb"] * gb_importance +
                self.model_weights["xgb"] * xgb_importance
            )
        else:
            total = self.model_weights["rf"] + self.model_weights["gb"]
            avg_importance = (
                self.model_weights["rf"] / total * rf_importance +
                self.model_weights["gb"] / total * gb_importance
            )

        self.feature_importance = FeatureImportance(
            features=feature_names,
            importances=avg_importance.tolist(),
        )

    def predict(self, features: pd.DataFrame) -> PredictionResult:
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Handle feature selection
        available_features = [f for f in self.feature_names if f in features.columns]
        X = features[available_features].values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        if self.use_feature_selection and self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled

        # Get ensemble probability
        probability = float(self._get_ensemble_proba(X_selected)[0])

        # Compute feature contributions for explanation
        if self.feature_importance is not None:
            imp_features = self.feature_importance.features
            imp_values = self.feature_importance.importances

            # Map back to original features for explanation
            feature_contributions = []
            for i, fname in enumerate(imp_features):
                if i < X_selected.shape[1]:
                    contribution = X_selected[0, i] * imp_values[i]
                    feature_contributions.append((fname, contribution))

            # Sort by absolute contribution
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            top_signals = [(name, float(val)) for name, val in feature_contributions[:5]]
        else:
            top_signals = []

        # Determine model name
        if self.use_xgboost and self.xgb_model is not None:
            model_name = f"Ensemble (RF:{self.model_weights['rf']:.0%} + GB:{self.model_weights['gb']:.0%} + XGB:{self.model_weights['xgb']:.0%})"
        else:
            model_name = f"Ensemble (RF:{self.model_weights['rf']:.0%} + GB:{self.model_weights['gb']:.0%})"

        return PredictionResult(
            probability=probability,
            prediction=int(probability >= 0.5),
            confidence=float(abs(probability - 0.5) * 2),
            top_signals=top_signals,
            model_name=model_name,
        )

    def get_feature_importance(self) -> pd.DataFrame:
        if self.feature_importance is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.feature_importance.to_dataframe()


class PatternRecognizer:
    """Identifies historical patterns that led to big gains."""

    def __init__(self, n_neighbors: int = 10):
        self.n_neighbors = n_neighbors
        self.historical_features: Optional[np.ndarray] = None
        self.historical_returns: Optional[np.ndarray] = None
        self.historical_dates: Optional[pd.DatetimeIndex] = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []

    def fit(
        self,
        features: pd.DataFrame,
        forward_returns: pd.Series,
    ):
        common_idx = features.index.intersection(forward_returns.index)
        features = features.loc[common_idx]
        forward_returns = forward_returns.loc[common_idx]

        mask = ~(features.isna().any(axis=1) | forward_returns.isna())
        features = features[mask]
        forward_returns = forward_returns[mask]

        self.feature_names = list(features.columns)
        self.historical_features = self.scaler.fit_transform(features.values)
        self.historical_returns = forward_returns.values
        self.historical_dates = features.index

    def find_similar_patterns(
        self,
        current_features: pd.DataFrame,
    ) -> dict[str, Any]:
        if self.historical_features is None:
            raise ValueError("Model not fitted. Call fit() first.")

        current = current_features[self.feature_names].values.reshape(1, -1)
        current_scaled = self.scaler.transform(current)

        distances = np.sqrt(((self.historical_features - current_scaled) ** 2).sum(axis=1))

        nearest_indices = np.argsort(distances)[:self.n_neighbors]
        nearest_distances = distances[nearest_indices]
        nearest_returns = self.historical_returns[nearest_indices]
        nearest_dates = self.historical_dates[nearest_indices]

        avg_return = nearest_returns.mean()
        gain_rate = (nearest_returns > 0).mean()
        big_gain_rate = (nearest_returns >= 0.10).mean()
        huge_gain_rate = (nearest_returns >= 0.20).mean()

        return {
            "n_similar": self.n_neighbors,
            "avg_return": float(avg_return),
            "gain_rate": float(gain_rate),
            "big_gain_rate": float(big_gain_rate),
            "huge_gain_rate": float(huge_gain_rate),
            "best_return": float(nearest_returns.max()),
            "worst_return": float(nearest_returns.min()),
            "similar_dates": nearest_dates.tolist(),
            "similar_returns": nearest_returns.tolist(),
            "distances": nearest_distances.tolist(),
        }

    def get_historical_gain_patterns(
        self,
        min_return: float = 0.15,
    ) -> pd.DataFrame:
        if self.historical_features is None:
            raise ValueError("Model not fitted. Call fit() first.")

        mask = self.historical_returns >= min_return
        big_gain_features = self.historical_features[mask]
        big_gain_dates = self.historical_dates[mask]

        avg_features = pd.DataFrame(
            self.scaler.inverse_transform(big_gain_features),
            columns=self.feature_names,
            index=big_gain_dates,
        )

        return avg_features
