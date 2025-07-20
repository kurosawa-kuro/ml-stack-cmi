"""
Light Version Training Script
- Enhanced feature engineering
- Default hyperparameters (no Optuna optimization)
- Fast iterations for development
- All improvements but optimized for speed
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.data.gold import load_gold_data
from src.models import CrossValidationTrainer, LightGBMModel, save_model_with_metadata
from src.util.notifications import notify_error, notify_start
from src.util.time_tracker import WorkflowTimer, WorkflowTimeTracker
from src.validation import CVLogger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Ensure output directory exists for logging
Path("outputs/logs").mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("outputs/logs/light_training.log", mode="a")],
)
logger = logging.getLogger(__name__)

# Light version optimized hyperparameters (based on previous runs)
LIGHT_OPTIMIZED_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "verbose": -1,
    "random_state": 42,
    # Optimized values from previous Optuna runs
    "num_leaves": 45,
    "learning_rate": 0.08,
    "max_depth": 8,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}


def create_output_directories():
    """Create necessary output directories"""
    output_dirs = ["outputs/models", "outputs/logs", "outputs/submissions"]

    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {dir_path}")


def initialize_tracking() -> WorkflowTimeTracker:
    """Initialize time tracking"""
    tracker_path = "outputs/logs/light_workflow_times.json"
    tracker = WorkflowTimeTracker(tracker_path)
    logger.info("Light time tracker initialized")
    return tracker


def load_and_prepare_enhanced_data_light() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load and prepare enhanced data for light training (reuse existing tables if available)

    Returns:
        Tuple of (X_train, y_train, X_test, feature_names)
    """
    logger.info("Loading Enhanced Gold level data (light mode)...")

    try:
        # Try to load existing gold data first
        try:
            train_df, test_df = load_gold_data()
            logger.info("Using existing gold data for faster iteration")
        except Exception:
            # If not available, create it
            logger.info("Creating enhanced data layers...")
            from src.data.gold import create_gold_tables
            from src.data.silver import create_silver_tables

            create_silver_tables()
            create_gold_tables()
            train_df, test_df = load_gold_data()

        logger.info(f"Loaded enhanced train: {train_df.shape}, test: {test_df.shape}")

        # Separate features and target
        id_cols = ["id"]
        target_cols = ["Personality", "Personality_encoded", "Personality_encoded_1"]

        # Get feature columns
        feature_cols = [col for col in train_df.columns if col not in id_cols + target_cols]

        # Extract features and target
        X_train = train_df[feature_cols].values
        y_train = train_df["Personality_encoded"].values
        X_test = test_df[feature_cols].values

        logger.info(f"Light Features: {len(feature_cols)}, Samples: {len(X_train)}")

        # Quick data integrity check and cleaning
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)) or np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
            logger.warning("Data has NaN/Inf values, cleaning...")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Ensure clean data
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(
            f"Light target distribution: Extrovert: {np.sum(y_train)}, Introvert: {len(y_train) - np.sum(y_train)}"
        )

        return X_train, y_train, X_test, feature_cols

    except Exception as e:
        logger.error(f"Light data loading failed: {str(e)}")
        raise


def train_light_model(X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]) -> Dict[str, Any]:
    """
    Train light model with pre-optimized hyperparameters

    Args:
        X_train: Training features
        y_train: Training targets
        feature_names: Feature names

    Returns:
        Cross-validation results
    """
    logger.info("Starting Light Enhanced LightGBM model training...")

    # Initialize CV trainer
    cv_trainer = CrossValidationTrainer()

    # Send training start notification
    model_config = {
        "model_type": "Light Enhanced LightGBM",
        "features": len(feature_names),
        "samples": len(X_train),
        "cv_folds": 5,
        "mode": "light_development",
    }

    try:
        notify_start("Light Enhanced LightGBM", model_config)
    except Exception as e:
        logger.warning(f"Failed to send start notification: {str(e)}")

    # Perform cross-validation with pre-optimized parameters
    try:
        cv_results = cv_trainer.train_cv(
            model_class=LightGBMModel,
            X=X_train,
            y=y_train,
            model_params=LIGHT_OPTIMIZED_PARAMS,
            feature_names=feature_names,
        )

        logger.info(f"Light CV Score: {cv_results['mean_score']: .4f} ± {cv_results['std_score']: .4f}")
        logger.info(f"Light CV AUC: {cv_results['mean_auc']: .4f} ± {cv_results['std_auc']: .4f}")
        logger.info(f"Light Training time: {cv_results['training_time']: .1f} seconds")

        return cv_results

    except Exception as e:
        logger.error(f"Light training failed: {str(e)}")
        try:
            notify_error("light_training", str(e))
        except Exception:
            pass
        raise


def save_light_results(cv_results: Dict[str, Any], feature_names: list[str]) -> None:
    """
    Save light training results

    Args:
        cv_results: Cross-validation results
        feature_names: Feature names
    """
    logger.info("Saving light results...")

    # Save the best model (first fold for simplicity)
    best_model = cv_results["models"][0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Model save path
    model_path = f"outputs/models/light_enhanced_lightgbm_{timestamp}.pkl"

    # Additional metadata
    metadata = {
        "model_type": "Light Enhanced LightGBM",
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "training_samples": len(cv_results["oof_predictions"]),
        "prediction_distribution": cv_results["prediction_distribution"],
        "mode": "light_development",
        "hyperparameters": LIGHT_OPTIMIZED_PARAMS,
    }

    # Save model with metadata
    save_model_with_metadata(best_model, cv_results, model_path, metadata)

    # Save light CV log
    cv_logger = CVLogger()
    log_entry = cv_logger.create_log_entry(
        model_type="Light Enhanced LightGBM",
        cv_config=cv_results["cv_config"],
        fold_scores=cv_results["fold_scores"],
        training_time=cv_results["training_time"],
        feature_count=len(feature_names),
        data_samples=len(cv_results["oof_predictions"]),
        prediction_distribution=cv_results["prediction_distribution"],
        feature_importance=(
            cv_results.get("feature_importance", {}).to_dict("records")
            if cv_results.get("feature_importance") is not None
            else None
        ),
    )

    log_path = cv_logger.save_json_log(log_entry, f"light_training_{timestamp}.json")
    logger.info(f"Light results saved: model={model_path}, log={log_path}")


def evaluate_light_success_criteria(cv_results: Dict[str, Any]) -> Dict[str, bool]:
    """
    Evaluate if light training meets development criteria

    Args:
        cv_results: Cross-validation results

    Returns:
        Dictionary of success criteria evaluation
    """
    dev_target_score = 0.975  # Development target (slightly lower than bronze)
    stability_threshold = 0.003
    max_training_time = 60  # 1 minute for light version

    criteria = {
        "dev_target_achieved": cv_results["mean_score"] >= dev_target_score,
        "stable_performance": cv_results["std_score"] <= stability_threshold,
        "fast_training": cv_results["training_time"] <= max_training_time,
        "ready_for_heavy": cv_results["mean_score"] >= 0.973,  # Ready for heavy optimization
    }

    overall_success = all(criteria.values())
    criteria["overall_success"] = overall_success

    return criteria


def main():
    """Main light training workflow"""
    logger.info("=" * 50)
    logger.info("Starting Light Enhanced LightGBM Training")
    logger.info("=" * 50)

    # Initialize
    create_output_directories()
    tracker = initialize_tracking()

    try:
        with WorkflowTimer(tracker, "light_enhanced_training"):
            # Load and prepare enhanced data
            with WorkflowTimer(tracker, "light_data_preparation"):
                X_train, y_train, X_test, feature_names = load_and_prepare_enhanced_data_light()

            # Train light model
            with WorkflowTimer(tracker, "light_model_training"):
                cv_results = train_light_model(X_train, y_train, feature_names)

            # Save results
            with WorkflowTimer(tracker, "save_light_results"):
                save_light_results(cv_results, feature_names)

            # Evaluate success
            success_criteria = evaluate_light_success_criteria(cv_results)
            logger.info(f"Light success criteria: {success_criteria}")

            # Final summary
            logger.info("=" * 50)
            logger.info("Light Training Summary: ")
            logger.info(f"CV Accuracy: {cv_results['mean_score']: .4f} ± {cv_results['std_score']: .4f}")
            logger.info(f"CV AUC: {cv_results['mean_auc']: .4f} ± {cv_results['std_auc']: .4f}")
            logger.info(f"Training Time: {cv_results['training_time']: .1f} seconds")
            logger.info(f"Dev Target (0.975): {'✓' if success_criteria['dev_target_achieved'] else '✗'}")
            logger.info(f"Ready for Heavy: {'✓' if success_criteria['ready_for_heavy'] else '✗'}")
            logger.info("=" * 50)

            if success_criteria["overall_success"]:
                logger.info("🚀 Light version successful - Ready for heavy optimization!")
            else:
                logger.warning("⚠️ Light version needs improvement")

            # Recommendation
            if success_criteria["ready_for_heavy"]:
                logger.info("💡 Recommendation: Run heavy version (train_heavy.py) for bronze medal attempt")
            else:
                logger.info("💡 Recommendation: Improve features/preprocessing before heavy optimization")

    except Exception as e:
        logger.error(f"Light training workflow failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
