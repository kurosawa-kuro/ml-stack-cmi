"""
モデル定義・学習・予測機能
- LightGBMベースラインモデル
- クロスバリデーション対応
- ハイパーパラメータ管理
- モデル保存・読み込み機能
"""

# type: ignore

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Configuration-driven imports
try:
    from .config import get_project_config, should_use_single_algorithm, get_primary_algorithm
    CONFIG_AVAILABLE = True
except ImportError:
    try:
        from config import get_project_config, should_use_single_algorithm, get_primary_algorithm
        CONFIG_AVAILABLE = True
    except ImportError:
        CONFIG_AVAILABLE = False

try:
    from .validation import (
        CVLogger,
        CVStrategy,
        aggregate_cv_scores,
        calculate_accuracy,
        calculate_auc,
        calculate_prediction_distribution,
        check_data_integrity,
    )
except ImportError:
    from validation import (
        CVLogger,
        CVStrategy,
        aggregate_cv_scores,
        calculate_accuracy,
        calculate_auc,
        calculate_prediction_distribution,
        check_data_integrity,
    )

logger = logging.getLogger(__name__)


# Configuration-driven parameter loading
def get_lightgbm_params() -> Dict[str, Any]:
    """Get LightGBM parameters from configuration"""
    if CONFIG_AVAILABLE:
        config = get_project_config()
        return config.get_model_params("lightgbm")
    else:
        # Fallback to default parameters
        return {
            "objective": "multiclass",
            "num_class": 18,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42,
        }

# Legacy parameter dict for backward compatibility
LIGHTGBM_PARAMS = get_lightgbm_params()


class LightGBMModel:
    """LightGBM model wrapper with training and prediction functionality"""

    def __init__(self, params: Optional[Dict[str, Any]] = None, use_pipeline: bool = True):
        """
        Initialize LightGBM model

        Args:
            params: Model hyperparameters (uses defaults if None)
            use_pipeline: Whether to use Pipeline for strict preprocessing separation
        """
        self.params = params or LIGHTGBM_PARAMS.copy()
        self.model: Optional[lgb.LGBMClassifier] = None
        self.pipeline: Optional[Pipeline] = None
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False
        self.use_pipeline = use_pipeline

        # Validate parameters
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate model parameters"""
        required_params = ["objective", "random_state"]
        for param in required_params:
            if param not in self.params:
                raise ValueError(f"Required parameter '{param}' missing")

        # Validate learning rate
        if "learning_rate" in self.params:
            try:
                lr = float(self.params["learning_rate"])  # type: ignore[arg-type]
                if not (0 < lr <= 1):
                    raise ValueError(f"Learning rate must be in (0, 1], got {lr}")
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid learning_rate: {e}")

        # Validate num_leaves
        if "num_leaves" in self.params:
            try:
                num_leaves = int(self.params["num_leaves"])  # type: ignore[arg-type]
                if num_leaves <= 0:
                    raise ValueError(f"num_leaves must be positive, got {num_leaves}")
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid num_leaves: {e}")

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "LightGBMModel":
        """
        Train the model with strict preprocessing separation

        Args:
            X: Training features
            y: Training targets
            feature_names: Optional feature names

        Returns:
            Self for method chaining
        """
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(X, "columns") and not isinstance(X, np.ndarray):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        if self.use_pipeline:
            # Create Pipeline with strict preprocessing separation
            self.pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', lgb.LGBMClassifier(**self.params))
            ])
            self.pipeline.fit(X, y)
            self.model = self.pipeline.named_steps['model']
        else:
            # Direct model training (legacy mode)
            self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(X, y)
        
        self.is_fitted = True

        if self.feature_names is not None:
            logger.info(f"LightGBM model trained with {len(self.feature_names)} features (Pipeline: {self.use_pipeline})")
        else:
            logger.info(f"LightGBM model trained (Pipeline: {self.use_pipeline})")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using Pipeline or direct model

        Args:
            X: Features to predict on

        Returns:
            Binary predictions (0/1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if self.use_pipeline and self.pipeline is not None:
            return np.asarray(self.pipeline.predict(X))
        elif self.model is not None:
            return np.asarray(self.model.predict(X))
        else:
            raise ValueError("Neither pipeline nor model is available for predictions")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using Pipeline or direct model

        Args:
            X: Features to predict on

        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if self.use_pipeline and self.pipeline is not None:
            return np.asarray(self.pipeline.predict_proba(X))
        elif self.model is not None:
            return np.asarray(self.model.predict_proba(X))
        else:
            raise ValueError("Neither pipeline nor model is available for predictions")

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Get feature importance

        Args:
            importance_type: Type of importance ('gain', 'split', etc.)

        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted or self.model is None or self.feature_names is None:
            raise ValueError("Model must be fitted to get feature importance")

        importance_values = self.model.feature_importances_

        importance_df = pd.DataFrame({"feature": self.feature_names, "importance": importance_values}).sort_values(
            "importance", ascending=False
        )

        return importance_df

    def save(self, filepath: str) -> None:
        """
        Save model to file

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_data = {
            "model": self.model,
            "pipeline": self.pipeline if self.use_pipeline else None,
            "params": self.params,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "use_pipeline": self.use_pipeline,
            "timestamp": datetime.now().isoformat(),
        }

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "LightGBMModel":
        """
        Load model from file

        Args:
            filepath: Path to load the model from

        Returns:
            Loaded LightGBMModel instance
        """
        model_data = joblib.load(filepath)

        use_pipeline = model_data.get("use_pipeline", False)
        instance = cls(params=model_data["params"], use_pipeline=use_pipeline)
        instance.model = model_data["model"]
        instance.pipeline = model_data.get("pipeline")
        instance.feature_names = model_data["feature_names"]
        instance.is_fitted = model_data["is_fitted"]

        logger.info(f"Model loaded from {filepath}")
        return instance


class CrossValidationTrainer:
    """Cross-validation trainer for model evaluation"""

    def __init__(self, cv_strategy: Optional[CVStrategy] = None):
        """
        Initialize CV trainer

        Args:
            cv_strategy: CV strategy (uses default if None)
        """
        self.cv_strategy = cv_strategy or CVStrategy()
        self.logger = CVLogger()

    def train_cv(
        self,
        model_class: type,
        X: np.ndarray,
        y: np.ndarray,
        model_params: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform cross-validation training

        Args:
            model_class: Model class to use (e.g., LightGBMModel)
            X: Features
            y: Targets
            model_params: Model parameters
            feature_names: Feature names

        Returns:
            CV results dictionary
        """
        start_time = time.time()

        # Data integrity checks
        integrity_checks = check_data_integrity(X, y)
        if not all(integrity_checks.values()):
            failed_checks = [k for k, v in integrity_checks.items() if not v]
            raise ValueError(f"Data integrity checks failed: {failed_checks}")

        # Initialize results
        fold_scores = []
        fold_auc_scores = []
        oof_predictions = np.zeros(len(y))
        feature_importance_list = []
        models = []

        # Get CV splits
        cv_splits = self.cv_strategy.split(X, y)

        logger.info(f"Starting {len(cv_splits)}-fold cross-validation")

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            fold_start_time = time.time()

            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train model
            model = model_class(params=model_params)
            model.fit(X_train, y_train, feature_names=feature_names)

            # Predictions
            val_pred = model.predict(X_val)
            val_pred_proba = model.predict_proba(X_val)[:, 1]

            # Store OOF predictions
            oof_predictions[val_idx] = val_pred_proba

            # Calculate scores
            fold_accuracy = calculate_accuracy(y_val, val_pred)
            fold_auc = calculate_auc(y_val, val_pred_proba)

            fold_scores.append(fold_accuracy)
            fold_auc_scores.append(fold_auc)

            # Feature importance
            if hasattr(model, "get_feature_importance"):
                feature_importance_list.append(model.get_feature_importance())

            # Store model
            models.append(model)

            fold_time = time.time() - fold_start_time
            logger.info(
                f"Fold {fold_idx + 1}/{len(cv_splits)}: "
                f"Accuracy={fold_accuracy: .4f}, AUC={fold_auc: .4f}, "
                f"Time={fold_time: .1f}s"
            )

        # Aggregate results
        total_time = time.time() - start_time
        aggregated_scores = aggregate_cv_scores(fold_scores)
        aggregated_auc = aggregate_cv_scores(fold_auc_scores)

        # Feature importance aggregation
        feature_importance = None
        if feature_importance_list:
            feature_importance = self._aggregate_feature_importance(feature_importance_list)

        # Prediction distribution
        oof_binary_pred = (oof_predictions > 0.5).astype(int)
        pred_distribution = calculate_prediction_distribution(oof_binary_pred)

        # Create results dictionary
        cv_results = {
            "fold_scores": fold_scores,
            "fold_auc_scores": fold_auc_scores,
            "mean_score": aggregated_scores["mean_score"],
            "std_score": aggregated_scores["std_score"],
            "mean_auc": aggregated_auc["mean_score"],
            "std_auc": aggregated_auc["std_score"],
            "oof_predictions": oof_predictions,
            "oof_binary_predictions": oof_binary_pred,
            "prediction_distribution": pred_distribution,
            "feature_importance": feature_importance,
            "training_time": total_time,
            "models": models,
            "cv_config": self.cv_strategy.get_config(),
            "data_integrity_checks": integrity_checks,
        }

        logger.info(
            f"CV completed: Accuracy={aggregated_scores['mean_score']: .4f}±"
            f"{aggregated_scores['std_score']: .4f}, "
            f"AUC={aggregated_auc['mean_score']: .4f}±"
            f"{aggregated_auc['std_score']: .4f}, "
            f"Time={total_time: .1f}s"
        )

        return cv_results

    def _aggregate_feature_importance(self, importance_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate feature importance across folds

        Args:
            importance_list: List of feature importance DataFrames

        Returns:
            Aggregated feature importance
        """
        if not importance_list:
            return pd.DataFrame()

        # Combine all importance dataframes
        all_importance = pd.concat(importance_list, ignore_index=True)

        # Calculate mean importance for each feature
        aggregated = all_importance.groupby("feature")["importance"].agg(["mean", "std", "count"]).reset_index()

        aggregated.columns = ["feature", "importance_mean", "importance_std", "fold_count"]
        aggregated = aggregated.sort_values("importance_mean", ascending=False)

        return aggregated


def evaluate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive model evaluation metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities

    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {"accuracy": calculate_accuracy(y_true, y_pred), "auc": calculate_auc(y_true, y_pred_proba)}

    return metrics


def create_learning_curve_data(train_scores: List[float], val_scores: List[float]) -> Dict[str, List[float]]:
    """
    Create learning curve data structure

    Args:
        train_scores: Training scores by epoch
        val_scores: Validation scores by epoch

    Returns:
        Learning curve data
    """
    return {"train_scores": train_scores, "val_scores": val_scores, "epochs": list(range(len(train_scores)))}


def save_model_with_metadata(
    model: LightGBMModel, cv_results: Dict[str, Any], filepath: str, metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model with comprehensive metadata

    Args:
        model: Trained model
        cv_results: Cross-validation results
        filepath: Path to save the model
        metadata: Additional metadata
    """
    model_data = {
        "model": model.model,
        "params": model.params,
        "feature_names": model.feature_names,
        "cv_score": cv_results["mean_score"],
        "cv_std": cv_results["std_score"],
        "cv_results": cv_results,
        "timestamp": datetime.now().isoformat(),
        "model_type": "LightGBM",
        "version": "1.0.0",
    }

    if metadata:
        model_data.update(metadata)

    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_data, filepath)
    logger.info(f"Model with metadata saved to {filepath}")


def load_model_with_metadata(filepath: str) -> Tuple[LightGBMModel, Dict[str, Any]]:
    """
    Load model with metadata

    Args:
        filepath: Path to load the model from

    Returns:
        Tuple of (model, metadata)
    """
    data = joblib.load(filepath)

    # Reconstruct model
    model = LightGBMModel(params=data["params"])
    model.model = data["model"]
    model.feature_names = data["feature_names"]
    model.is_fitted = True

    # Extract metadata
    metadata = {k: v for k, v in data.items() if k not in ["model", "params", "feature_names"]}

    logger.info(f"Model with metadata loaded from {filepath}")
    return model, metadata


class OptunaOptimizer:
    """Optuna-based hyperparameter optimization for LightGBM"""

    def __init__(self, n_trials: int = 100, cv_folds: int = 5, random_state: int = 42):
        """
        Initialize Optuna optimizer

        Args:
            n_trials: Number of optimization trials
            cv_folds: Number of CV folds for evaluation
            random_state: Random state for reproducibility
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None

        # Suppress Optuna logging noise
        # Set Optuna logging to INFO level for progress visibility
        optuna.logging.set_verbosity(optuna.logging.INFO)

    def objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """
        Objective function for Optuna optimization

        Args:
            trial: Optuna trial object
            X: Training features
            y: Training targets

        Returns:
            Negative accuracy (for minimization)
        """
        logger.info(f"Trial {trial.number} started...")
        
        # Suggest hyperparameters
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "verbose": -1,
            "random_state": self.random_state,
            # Key parameters to optimize
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            # Additional parameters for fine-tuning
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        }

        try:
            # Create model
            model = lgb.LGBMClassifier(**params)  # type: ignore

            # Perform cross-validation (n_jobs=1 for stability in Optuna)
            scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring="accuracy", n_jobs=1)
            score_mean = scores.mean()
            logger.info(f"Trial {trial.number} completed: Accuracy = {score_mean:.4f}")
            
            # Return negative mean score (Optuna minimizes)
            return -score_mean

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return 0.0  # Return worst score for failed trials

    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run hyperparameter optimization

        Args:
            X: Training features
            y: Training targets

        Returns:
            Optimization results dictionary
        """
        logger.info(f"Starting Optuna optimization with {self.n_trials} trials...")

        # Create study
        self.study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )

        # Optimize with parallel execution (limited to avoid memory issues)
        start_time = time.time()
        self.study.optimize(
            lambda trial: self.objective(trial, X, y), 
            n_trials=self.n_trials, 
            show_progress_bar=True,
            n_jobs=2  # Run 2 trials in parallel (reduced for stability)
        )
        optimization_time = time.time() - start_time

        # Extract best parameters
        if self.study.best_params is not None:
            self.best_params = self.study.best_params.copy()
            if self.best_params is not None:
                self.best_params.update(
                    {
                        "objective": "binary",
                        "metric": "binary_logloss",
                        "boosting_type": "gbdt",
                        "verbose": -1,
                        "random_state": self.random_state,
                    }
                )
        else:
            raise ValueError("Optimization failed to find best parameters")

        # Create results dictionary
        results = {
            "best_params": self.best_params,
            "best_score": -self.study.best_value if self.study.best_value is not None else 0.0,
            "n_trials": len(self.study.trials),
            "optimization_time": optimization_time,
            "study": self.study,
        }

        logger.info("Optimization completed: ")
        logger.info(f"Best score: {results['best_score']: .4f}")
        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Optimization time: {optimization_time: .1f} seconds")

        return results

    def get_feature_importance_analysis(self) -> pd.DataFrame:
        """
        Analyze parameter importance from optimization

        Returns:
            DataFrame with parameter importance
        """
        if self.study is None:
            raise ValueError("Must run optimization first")

        importance = optuna.importance.get_param_importances(self.study)

        importance_df = pd.DataFrame(
            [{"parameter": param, "importance": imp} for param, imp in importance.items()]
        ).sort_values("importance", ascending=False)

        return importance_df


def optimize_lightgbm_hyperparams(
    X: np.ndarray, y: np.ndarray, n_trials: int = 100, cv_folds: int = 5, random_state: int = 42
) -> Dict[str, Any]:
    """
    Convenience function for LightGBM hyperparameter optimization

    Args:
        X: Training features
        y: Training targets
        n_trials: Number of optimization trials
        cv_folds: Number of CV folds
        random_state: Random state

    Returns:
        Optimization results
    """
    optimizer = OptunaOptimizer(n_trials=n_trials, cv_folds=cv_folds, random_state=random_state)
    return optimizer.optimize(X, y)


def create_optimized_model(
    optimization_results: Dict[str, Any], feature_names: Optional[List[str]] = None
) -> LightGBMModel:
    """
    Create LightGBM model with optimized hyperparameters

    Args:
        optimization_results: Results from hyperparameter optimization
        feature_names: Optional feature names

    Returns:
        LightGBMModel with optimized parameters
    """
    best_params = optimization_results["best_params"]
    model = LightGBMModel(params=best_params)

    if feature_names:
        model.feature_names = feature_names

    logger.info(f"Created optimized model with score: {optimization_results['best_score']: .4f}")

    return model
