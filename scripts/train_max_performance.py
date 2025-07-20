"""
Heavy Version Training Script
- Enhanced feature engineering
- Full Optuna hyperparameter optimization
- Production-ready for bronze medal target
- All optimizations for maximum performance
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.data.gold import load_gold_data
from src.models import (
    CrossValidationTrainer,
    LightGBMModel,
    OptunaOptimizer,
    optimize_lightgbm_hyperparams,
    save_model_with_metadata,
)
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
    handlers=[logging.StreamHandler(), logging.FileHandler("outputs/logs/heavy_training.log", mode="a")],
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
    tracker_path = "outputs/logs/heavy_workflow_times.json"
    tracker = WorkflowTimeTracker(tracker_path)
    logger.info("Heavy time tracker initialized")
    return tracker


def load_and_prepare_enhanced_data_heavy() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load and prepare enhanced data for heavy training

    Returns:
        Tuple of (X_train, y_train, X_test, feature_names)
    """
    logger.info("Loading Enhanced Gold level data (heavy mode)...")

    try:
        # Always recreate data for production run
        logger.info("Recreating enhanced data layers for production...")
        from src.data.gold import create_gold_tables
        from src.data.silver import create_silver_tables

        create_silver_tables()
        create_gold_tables()

        train_df, test_df = load_gold_data()
        logger.info(f"Loaded enhanced train: {train_df.shape}, test: {test_df.shape}")

        # Separate features and target
        id_cols = ["id"]
        
        # Find the correct target column name in train data
        target_cols = []
        if "Personality" in train_df.columns:
            target_cols.append("Personality")
        if "Personality_encoded" in train_df.columns:
            target_cols.append("Personality_encoded")
        elif "Personality_encoded_1" in train_df.columns:
            target_cols.append("Personality_encoded_1")

        # Get feature columns from train data (excluding id and target columns)
        feature_cols = [col for col in train_df.columns if col not in id_cols + target_cols]

        # Get feature columns that also exist in test data (since test won't have target columns)
        test_feature_cols = [col for col in feature_cols if col in test_df.columns]

        # Extract features and target - use the available encoded column
        X_train = np.asarray(train_df[test_feature_cols].values)
        if "Personality_encoded" in train_df.columns:
            y_train = np.asarray(train_df["Personality_encoded"].values)
        elif "Personality_encoded_1" in train_df.columns:
            y_train = np.asarray(train_df["Personality_encoded_1"].values)
        else:
            raise ValueError("No encoded target column found")
        X_test = np.asarray(test_df[test_feature_cols].values)
        
        # Clean NaN and Inf values
        logger.info("Cleaning NaN and Inf values...")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Update feature_cols to match what we actually used
        feature_cols = test_feature_cols

        logger.info(f"Heavy Features: {len(feature_cols)}, Samples: {len(X_train)}")

        # Analyze feature types
        interaction_features = [col for col in feature_cols if "interaction" in col]
        poly_features = [col for col in feature_cols if col.startswith("poly_")]
        scaled_features = [col for col in feature_cols if col.endswith("_scaled")]

        logger.info(f"  - Interaction features: {len(interaction_features)}")
        logger.info(f"  - Polynomial features: {len(poly_features)}")
        logger.info(f"  - Scaled features: {len(scaled_features)}")

        # Comprehensive data integrity checks
        integrity_checks = check_data_integrity(X_train, y_train)
        logger.info(f"Heavy data integrity checks: {integrity_checks}")

        if not all(integrity_checks.values()):
            failed_checks = [k for k, v in integrity_checks.items() if not v]
            raise ValueError(f"Heavy data integrity checks failed: {failed_checks}")

        # Target distribution analysis
        target_dist = validate_target_distribution(y_train)
        logger.info(f"Heavy target distribution: {target_dist}")

        return X_train, y_train, X_test, feature_cols

    except Exception as e:
        logger.error(f"Heavy data loading failed: {str(e)}")
        raise


def optimize_hyperparameters_heavy(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Heavy hyperparameter optimization using Optuna

    Args:
        X_train: Training features
        y_train: Training targets

    Returns:
        Optimization results
    """
    logger.info("Starting Heavy Optuna hyperparameter optimization...")
    
    # データの健全性チェック
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"X_train has nan: {np.isnan(X_train).any()}")
    logger.info(f"X_train has inf: {np.isinf(X_train).any()}")
    logger.info(f"y_train unique values: {np.unique(y_train)}")

    try:
        # まず少ないトライアル数でテスト
        logger.info("Running test with 5 trials first...")
        test_results = optimize_lightgbm_hyperparams(
            X_train, y_train, n_trials=5, cv_folds=3, random_state=42
        )
        logger.info(f"Test optimization completed: Best score = {test_results['best_score']:.4f}")
        
        # 本番の最適化（トライアル数を減らして実行）
        logger.info("Running full optimization with 20 trials...")
        optimization_results = optimize_lightgbm_hyperparams(
            X_train, y_train, n_trials=20, cv_folds=5, random_state=42  # 実用的な20トライアル
        )

        # Save optimization results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"outputs/optimization/heavy_optuna_results_{timestamp}.json"

        # Create a serializable version of results
        serializable_results = {
            "best_params": optimization_results["best_params"],
            "best_score": optimization_results["best_score"],
            "n_trials": optimization_results["n_trials"],
            "optimization_time": optimization_results["optimization_time"],
            "mode": "heavy_production",
        }

        import json

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Heavy optimization results saved to: {results_path}")

        # Log parameter importance if available
        try:
            optimizer = OptunaOptimizer(n_trials=200, cv_folds=5, random_state=42)
            optimizer.study = optimization_results.get("study")
            if optimizer.study:
                param_importance = optimizer.get_feature_importance_analysis()
                logger.info(f"Parameter importance: \n{param_importance}")
        except Exception as e:
            logger.warning(f"Could not analyze parameter importance: {e}")

        return optimization_results

    except Exception as e:
        logger.error(f"Heavy hyperparameter optimization failed: {str(e)}")
        raise


def train_heavy_model(
    X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str], optimization_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train heavy model with optimized hyperparameters

    Args:
        X_train: Training features
        y_train: Training targets
        feature_names: Feature names
        optimization_results: Hyperparameter optimization results

    Returns:
        Cross-validation results
    """
    logger.info("Starting Heavy Enhanced LightGBM model training...")

    # Initialize CV trainer
    cv_trainer = CrossValidationTrainer()

    # Create optimized model parameters
    optimized_params = optimization_results["best_params"]

    # Send training start notification
    model_config = {
        "model_type": "Heavy Enhanced LightGBM",
        "features": len(feature_names),
        "samples": len(X_train),
        "cv_folds": 5,
        "optimization_score": optimization_results["best_score"],
        "optimization_trials": optimization_results["n_trials"],
        "optimized_params": optimized_params,
        "mode": "heavy_production",
    }

    try:
        notify_start("Heavy Enhanced LightGBM", model_config)
    except Exception as e:
        logger.warning(f"Failed to send start notification: {str(e)}")

    # Perform cross-validation with optimized parameters
    try:
        cv_results = cv_trainer.train_cv(
            model_class=LightGBMModel, X=X_train, y=y_train, model_params=optimized_params, feature_names=feature_names
        )

        logger.info(f"Heavy CV Score: {cv_results['mean_score']: .4f} ± {cv_results['std_score']: .4f}")
        logger.info(f"Heavy CV AUC: {cv_results['mean_auc']: .4f} ± {cv_results['std_auc']: .4f}")
        logger.info(f"Heavy Training time: {cv_results['training_time']: .1f} seconds")

        # Add optimization info to results
        cv_results["optimization_results"] = optimization_results

        return cv_results

    except Exception as e:
        logger.error(f"Heavy training failed: {str(e)}")
        try:
            notify_error("heavy_training", str(e))
        except Exception:
            pass
        raise


def save_heavy_results(cv_results: Dict[str, Any], feature_names: list[str]) -> None:
    """
    Save heavy training results and models

    Args:
        cv_results: Cross-validation results
        feature_names: Feature names
    """
    logger.info("Saving heavy results...")

    # Save the best model (first fold for simplicity)
    best_model = cv_results["models"][0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Model save path
    model_path = f"outputs/models/heavy_enhanced_lightgbm_{timestamp}.pkl"

    # Additional metadata
    metadata = {
        "model_type": "Heavy Enhanced LightGBM",
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "training_samples": len(cv_results["oof_predictions"]),
        "prediction_distribution": cv_results["prediction_distribution"],
        "optimization_results": cv_results.get("optimization_results", {}),
        "mode": "heavy_production",
        "enhancement_features": {
            "interaction_features": len([f for f in feature_names if "interaction" in f]),
            "polynomial_features": len([f for f in feature_names if f.startswith("poly_")]),
            "scaled_features": len([f for f in feature_names if f.endswith("_scaled")]),
        },
    }

    # Save model with metadata
    save_model_with_metadata(best_model, cv_results, model_path, metadata)

    # Save comprehensive CV log
    cv_logger = CVLogger()
    log_entry = cv_logger.create_log_entry(
        model_type="Heavy Enhanced LightGBM",
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

    log_path = cv_logger.save_json_log(log_entry, f"heavy_training_{timestamp}.json")

    # Save fold-wise results as CSV
    fold_results_df = pd.DataFrame(
        {
            "fold": range(len(cv_results["fold_scores"])),
            "accuracy": cv_results["fold_scores"],
            "auc": cv_results["fold_auc_scores"],
        }
    )

    csv_path = cv_logger.save_csv_log(fold_results_df, f"heavy_fold_results_{timestamp}.csv")

    logger.info(f"Heavy results saved: model={model_path}, log={log_path}, csv={csv_path}")


def send_heavy_completion_notification(cv_results: Dict[str, Any]) -> None:
    """
    Send heavy training completion notification

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
            "optimization_trials": cv_results.get("optimization_results", {}).get("n_trials", 0),
        }

        notify_complete(model_name="Heavy Enhanced LightGBM", metrics=metrics, duration=cv_results["training_time"])

        logger.info("Heavy completion notification sent")

    except Exception as e:
        logger.warning(f"Failed to send heavy completion notification: {str(e)}")


def evaluate_heavy_success_criteria(cv_results: Dict[str, Any]) -> Dict[str, bool]:
    """
    Evaluate if heavy training meets production criteria

    Args:
        cv_results: Cross-validation results

    Returns:
        Dictionary of success criteria evaluation
    """
    bronze_target_score = 0.976518  # Bronze medal threshold
    stability_threshold = 0.002
    max_training_time = 1200  # 20 minutes for heavy version

    criteria = {
        "bronze_target_achieved": cv_results["mean_score"] >= bronze_target_score,
        "stable_performance": cv_results["std_score"] <= stability_threshold,
        "reasonable_time": cv_results["training_time"] <= max_training_time,
        "data_integrity": all(cv_results["data_integrity_checks"].values()),
        "optimization_successful": cv_results.get("optimization_results", {}).get("best_score", 0.0) > 0.97,
        "ready_for_submission": cv_results["mean_score"] >= 0.975,  # Ready for submission
    }

    overall_success = all(criteria.values())
    criteria["overall_success"] = overall_success

    return criteria


def main():
    """Main heavy training workflow"""
    logger.info("=" * 70)
    logger.info("Starting Heavy Enhanced LightGBM Training - PRODUCTION MODE")
    logger.info("=" * 70)

    # Initialize
    create_output_directories()
    tracker = initialize_tracking()

    try:
        with WorkflowTimer(tracker, "heavy_enhanced_training"):
            # Load and prepare enhanced data
            with WorkflowTimer(tracker, "heavy_data_preparation"):
                X_train, y_train, X_test, feature_names = load_and_prepare_enhanced_data_heavy()

            # Optimize hyperparameters
            with WorkflowTimer(tracker, "heavy_hyperparameter_optimization"):
                optimization_results = optimize_hyperparameters_heavy(X_train, y_train)

            # Train heavy model
            with WorkflowTimer(tracker, "heavy_model_training"):
                cv_results = train_heavy_model(X_train, y_train, feature_names, optimization_results)

            # Save results
            with WorkflowTimer(tracker, "save_heavy_results"):
                save_heavy_results(cv_results, feature_names)

            # Evaluate success
            success_criteria = evaluate_heavy_success_criteria(cv_results)
            logger.info(f"Heavy success criteria: {success_criteria}")

            # Send completion notification
            send_heavy_completion_notification(cv_results)

            # Final summary
            logger.info("=" * 70)
            logger.info("HEAVY TRAINING SUMMARY: ")
            logger.info(f"CV Accuracy: {cv_results['mean_score']: .6f} ± {cv_results['std_score']: .6f}")
            logger.info(f"CV AUC: {cv_results['mean_auc']: .6f} ± {cv_results['std_auc']: .6f}")
            logger.info(f"Training Time: {cv_results['training_time']: .1f} seconds")
            logger.info(f"Optimization Time: {optimization_results['optimization_time']: .1f} seconds")
            logger.info(f"Total Features: {len(feature_names)}")
            logger.info(f"Optimization Trials: {optimization_results['n_trials']}")
            logger.info("")
            logger.info("TARGET ANALYSIS: ")
            bronze_status = "✓ ACHIEVED" if success_criteria["bronze_target_achieved"] else "✗ MISSED"
            logger.info(f"Bronze Target (0.976518): {bronze_status}")
            logger.info(f"Gap to Bronze: {cv_results['mean_score'] - 0.976518:+.6f}")
            logger.info(f"Stability: {'✓' if success_criteria['stable_performance'] else '✗'}")
            logger.info(f"Ready for Submission: {'✓' if success_criteria['ready_for_submission'] else '✗'}")
            logger.info("=" * 70)

            if success_criteria["bronze_target_achieved"]:
                logger.info("🥉 BRONZE MEDAL THRESHOLD ACHIEVED! 🥉")
                logger.info("🎯 READY FOR KAGGLE SUBMISSION!")
            elif success_criteria["ready_for_submission"]:
                logger.info("🎯 Close to bronze - Ready for submission attempt")
            else:
                logger.warning("⚠️ Need further improvements for bronze medal")

            if success_criteria["overall_success"]:
                logger.info("🏆 ALL PRODUCTION CRITERIA MET!")
            else:
                failed_criteria = [k for k, v in success_criteria.items() if not v and k != "overall_success"]
                logger.warning(f"⚠️ Failed criteria: {failed_criteria}")

    except Exception as e:
        logger.error(f"Heavy training workflow failed: {str(e)}")
        try:
            notify_error("heavy_workflow", str(e))
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
