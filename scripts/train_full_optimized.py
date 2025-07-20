"""
Enhanced Model Training with Optimizations
- Enhanced feature engineering with interaction terms
- Polynomial features for non-linear relationships
- Optuna hyperparameter optimization
- Improved data preprocessing
- Comprehensive validation
"""

# type: ignore

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.data.gold import load_gold_data
from src.models import CrossValidationTrainer, LightGBMModel, optimize_lightgbm_hyperparams, save_model_with_metadata
from src.util.notifications import notify_complete, notify_error, notify_start
from src.util.time_tracker import WorkflowTimer, WorkflowTimeTracker
from src.validation import CVLogger, check_data_integrity, validate_target_distribution

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Ensure output directory exists for logging
Path("outputs/logs").mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("outputs/logs/enhanced_training.log", mode="a")],
)
logger = logging.getLogger(__name__)


def create_output_directories():
    """Create necessary output directories"""
    output_dirs = ["outputs/models", "outputs/logs", "outputs/submissions", "outputs/optimization"]

    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {dir_path}")


def initialize_tracking() -> WorkflowTimeTracker:
    """Initialize time tracking"""
    tracker_path = "outputs/logs/enhanced_workflow_times.json"
    tracker = WorkflowTimeTracker(tracker_path)
    logger.info("Enhanced time tracker initialized")
    return tracker


def load_and_prepare_enhanced_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load and prepare enhanced data for training

    Returns:
        Tuple of (X_train, y_train, X_test, feature_names)
    """
    logger.info("Loading Enhanced Gold level data...")

    try:
        # Force recreation of silver and gold tables to include new features
        logger.info("Recreating data layers with enhanced features...")
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

        logger.info(f"Enhanced Features: {len(feature_cols)}, Samples: {len(X_train)}")
        logger.info("New feature types found: ")

        # Analyze feature types
        interaction_features = [col for col in feature_cols if "interaction" in col]
        poly_features = [col for col in feature_cols if col.startswith("poly_")]
        scaled_features = [col for col in feature_cols if col.endswith("_scaled")]

        logger.info(f"  - Interaction features: {len(interaction_features)}")
        logger.info(f"  - Polynomial features: {len(poly_features)}")
        logger.info(f"  - Scaled features: {len(scaled_features)}")

        # Data integrity checks
        integrity_checks = check_data_integrity(X_train, y_train)
        logger.info(f"Enhanced data integrity checks: {integrity_checks}")

        if not all(integrity_checks.values()):
            failed_checks = [k for k, v in integrity_checks.items() if not v]
            raise ValueError(f"Enhanced data integrity checks failed: {failed_checks}")

        # Target distribution analysis
        target_dist = validate_target_distribution(y_train)
        logger.info(f"Enhanced target distribution: {target_dist}")

        return X_train, y_train, X_test, feature_cols

    except Exception as e:
        logger.error(f"Enhanced data loading failed: {str(e)}")
        raise


def optimize_hyperparameters(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Optimize hyperparameters using Optuna

    Args:
        X_train: Training features
        y_train: Training targets

    Returns:
        Optimization results
    """
    logger.info("Starting Optuna hyperparameter optimization...")

    try:
        # Run optimization with more trials for better results
        optimization_results = optimize_lightgbm_hyperparams(
            X_train, y_train, n_trials=150, cv_folds=5, random_state=42  # Increased trials for better optimization
        )

        # Save optimization results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"outputs/optimization/optuna_results_{timestamp}.json"

        # Create a serializable version of results
        serializable_results = {
            "best_params": optimization_results["best_params"],
            "best_score": optimization_results["best_score"],
            "n_trials": optimization_results["n_trials"],
            "optimization_time": optimization_results["optimization_time"],
        }

        import json

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Optimization results saved to: {results_path}")
        return optimization_results

    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {str(e)}")
        raise


def train_enhanced_model(
    X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str], optimization_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train enhanced model with optimized hyperparameters

    Args:
        X_train: Training features
        y_train: Training targets
        feature_names: Feature names
        optimization_results: Hyperparameter optimization results

    Returns:
        Cross-validation results
    """
    logger.info("Starting Enhanced LightGBM model training...")

    # Initialize CV trainer
    cv_trainer = CrossValidationTrainer()

    # Create optimized model parameters
    optimized_params = optimization_results["best_params"]

    # Send training start notification
    model_config = {
        "model_type": "Enhanced LightGBM",
        "features": len(feature_names),
        "samples": len(X_train),
        "cv_folds": 5,
        "optimization_score": optimization_results["best_score"],
        "optimized_params": optimized_params,
    }

    try:
        notify_start("Enhanced LightGBM", model_config)
    except Exception as e:
        logger.warning(f"Failed to send start notification: {str(e)}")

    # Perform cross-validation with optimized parameters
    try:
        cv_results = cv_trainer.train_cv(
            model_class=LightGBMModel, X=X_train, y=y_train, model_params=optimized_params, feature_names=feature_names
        )

        logger.info(f"Enhanced CV Score: {cv_results['mean_score']: .4f} ± {cv_results['std_score']: .4f}")
        logger.info(f"Enhanced CV AUC: {cv_results['mean_auc']: .4f} ± {cv_results['std_auc']: .4f}")
        logger.info(f"Enhanced Training time: {cv_results['training_time']: .1f} seconds")

        # Add optimization info to results
        cv_results["optimization_results"] = optimization_results

        return cv_results

    except Exception as e:
        logger.error(f"Enhanced training failed: {str(e)}")
        try:
            notify_error("enhanced_training", str(e))
        except Exception:
            pass
        raise


def save_enhanced_results(cv_results: Dict[str, Any], feature_names: list[str]) -> None:
    """
    Save enhanced training results and models

    Args:
        cv_results: Cross-validation results
        feature_names: Feature names
    """
    logger.info("Saving enhanced results...")

    # Save the best model (first fold for simplicity)
    best_model = cv_results["models"][0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Model save path
    model_path = f"outputs/models/enhanced_lightgbm_{timestamp}.pkl"

    # Additional metadata
    metadata = {
        "model_type": "Enhanced LightGBM",
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "training_samples": len(cv_results["oof_predictions"]),
        "prediction_distribution": cv_results["prediction_distribution"],
        "optimization_results": cv_results.get("optimization_results", {}),
        "enhancement_features": {
            "interaction_features": len([f for f in feature_names if "interaction" in f]),
            "polynomial_features": len([f for f in feature_names if f.startswith("poly_")]),
            "scaled_features": len([f for f in feature_names if f.endswith("_scaled")]),
        },
    }

    # Save model with metadata
    save_model_with_metadata(best_model, cv_results, model_path, metadata)

    # Save CV log
    cv_logger = CVLogger()
    log_entry = cv_logger.create_log_entry(
        model_type="Enhanced LightGBM",
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

    log_path = cv_logger.save_json_log(log_entry, f"enhanced_training_{timestamp}.json")

    # Save fold-wise results as CSV
    fold_results_df = pd.DataFrame(
        {
            "fold": range(len(cv_results["fold_scores"])),
            "accuracy": cv_results["fold_scores"],
            "auc": cv_results["fold_auc_scores"],
        }
    )

    csv_path = cv_logger.save_csv_log(fold_results_df, f"enhanced_fold_results_{timestamp}.csv")

    logger.info(f"Enhanced results saved: model={model_path}, log={log_path}, csv={csv_path}")


def send_enhanced_completion_notification(cv_results: Dict[str, Any]) -> None:
    """
    Send enhanced training completion notification

    Args:
        cv_results: Cross-validation results
    """
    try:
        metrics = {
            "accuracy": cv_results["mean_score"],
            "accuracy_std": cv_results["std_score"],
            "auc": cv_results["mean_auc"],
            "auc_std": cv_results["std_auc"],
            "extrovert_ratio": cv_results["prediction_distribution"]["extrovert_ratio"],
            "optimization_score": cv_results.get("optimization_results", {}).get("best_score", 0.0),
        }

        notify_complete(model_name="Enhanced LightGBM", metrics=metrics, duration=cv_results["training_time"])

        logger.info("Enhanced completion notification sent")

    except Exception as e:
        logger.warning(f"Failed to send enhanced completion notification: {str(e)}")


def evaluate_enhanced_success_criteria(cv_results: Dict[str, Any]) -> Dict[str, bool]:
    """
    Evaluate if enhanced training meets success criteria

    Args:
        cv_results: Cross-validation results

    Returns:
        Dictionary of success criteria evaluation
    """
    target_score = 0.976518  # Bronze medal threshold
    stability_threshold = 0.002
    max_training_time = 600  # 10 minutes for enhanced model

    criteria = {
        "bronze_target_achieved": cv_results["mean_score"] >= target_score,
        "stable_performance": cv_results["std_score"] <= stability_threshold,
        "reasonable_time": cv_results["training_time"] <= max_training_time,
        "data_integrity": all(cv_results["data_integrity_checks"].values()),
        "optimization_successful": cv_results.get("optimization_results", {}).get("best_score", 0.0) > 0.97,
    }

    overall_success = all(criteria.values())
    criteria["overall_success"] = overall_success

    return criteria


def main():
    """Main enhanced training workflow"""
    logger.info("=" * 60)
    logger.info("Starting Enhanced LightGBM Training with Optimizations")
    logger.info("=" * 60)

    # Initialize
    create_output_directories()
    tracker = initialize_tracking()

    try:
        with WorkflowTimer(tracker, "enhanced_baseline_training"):
            # Load and prepare enhanced data
            with WorkflowTimer(tracker, "enhanced_data_preparation"):
                X_train, y_train, X_test, feature_names = load_and_prepare_enhanced_data()

            # Optimize hyperparameters
            with WorkflowTimer(tracker, "hyperparameter_optimization"):
                optimization_results = optimize_hyperparameters(X_train, y_train)

            # Train enhanced model
            with WorkflowTimer(tracker, "enhanced_model_training"):
                cv_results = train_enhanced_model(X_train, y_train, feature_names, optimization_results)

            # Save results
            with WorkflowTimer(tracker, "save_enhanced_results"):
                save_enhanced_results(cv_results, feature_names)

            # Evaluate success
            success_criteria = evaluate_enhanced_success_criteria(cv_results)
            logger.info(f"Enhanced success criteria: {success_criteria}")

            # Send completion notification
            send_enhanced_completion_notification(cv_results)

            # Final summary
            logger.info("=" * 60)
            logger.info("Enhanced Training Summary: ")
            logger.info(f"CV Accuracy: {cv_results['mean_score']: .4f} ± {cv_results['std_score']: .4f}")
            logger.info(f"CV AUC: {cv_results['mean_auc']: .4f} ± {cv_results['std_auc']: .4f}")
            logger.info(f"Training Time: {cv_results['training_time']: .1f} seconds")
            logger.info(f"Bronze Target (0.976518): {'✓' if success_criteria['bronze_target_achieved'] else '✗'}")
            logger.info(f"Optimization Score: {optimization_results['best_score']: .4f}")
            logger.info(f"Feature Count: {len(feature_names)}")
            logger.info("=" * 60)

            if success_criteria["overall_success"]:
                logger.info("🎉 All enhanced success criteria met - Bronze medal threshold achieved!")
            else:
                logger.warning("⚠️ Some enhanced success criteria not met")

            # Check bronze medal achievement specifically
            if success_criteria["bronze_target_achieved"]:
                logger.info("🥉 BRONZE MEDAL THRESHOLD ACHIEVED! 🥉")

    except Exception as e:
        logger.error(f"Enhanced training workflow failed: {str(e)}")
        try:
            notify_error("enhanced_workflow", str(e))
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
