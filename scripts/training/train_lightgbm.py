#!/usr/bin/env python3
"""
CMI Competition - LightGBM Baseline Training
============================================
Week 1-2: LightGBM baseline model training with GroupKFold validation

Features:
- LightGBM training with GroupKFold (participant-aware)
- Binary + Multiclass prediction strategy
- tsfresh feature integration
- Hyperparameter optimization with Optuna
- Comprehensive evaluation metrics
- Model persistence and submission generation
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import optuna
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def setup_training():
    """Setup training environment"""
    output_dir = Path(__file__).parent.parent / "outputs" / "models" / "lgb_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir, log_dir

def load_ml_ready_data():
    """Load ML-ready data from gold layer"""
    print("📊 Loading ML-ready data...")
    
    try:
        from src.data.gold import get_ml_ready_data
        X_train, y_train, X_test, groups = get_ml_ready_data()
        
        print(f"  ✓ Train features: {X_train.shape}")
        print(f"  ✓ Train labels: {y_train.shape}")
        print(f"  ✓ Test features: {X_test.shape}")
        print(f"  ✓ Groups (participants): {len(np.unique(groups))}")
        
        return X_train, y_train, X_test, groups
        
    except Exception as e:
        print(f"  ✗ Failed to load ML-ready data: {e}")
        print("  Please run workflow-to-gold.py first")
        return None, None, None, None

def evaluate_composite_f1(y_true, y_pred):
    """Calculate composite F1 score (Binary F1 + Macro F1) / 2"""
    # Convert multiclass to binary (0 = no behavior, >0 = behavior present)
    y_true_binary = (y_true > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)
    
    # Binary F1 score
    binary_f1 = f1_score(y_true_binary, y_pred_binary, average='binary')
    
    # Macro F1 score (multiclass)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # Composite score
    composite_f1 = (binary_f1 + macro_f1) / 2
    
    return composite_f1, binary_f1, macro_f1

def train_lgb_with_cv(X_train, y_train, groups, params, output_dir, n_splits=5):
    """Train LightGBM with GroupKFold cross-validation"""
    print(f"\n🚀 Training LightGBM with {n_splits}-fold GroupKFold CV...")
    
    # Initialize GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    
    # Track metrics across folds
    fold_metrics = []
    models = []
    oof_predictions = np.zeros(len(y_train))
    
    # Cross-validation training
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        print(f"\n📊 Training Fold {fold + 1}/{n_splits}")
        
        # Split data
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(100)
            ]
        )
        
        # Predict on validation set
        val_pred = model.predict(X_fold_val, num_iteration=model.best_iteration)
        val_pred_labels = np.argmax(val_pred, axis=1) if len(val_pred.shape) > 1 else np.round(val_pred).astype(int)
        
        # Store out-of-fold predictions
        oof_predictions[val_idx] = val_pred_labels
        
        # Calculate metrics
        composite_f1, binary_f1, macro_f1 = evaluate_composite_f1(y_fold_val, val_pred_labels)
        
        fold_metrics.append({
            'fold': fold + 1,
            'composite_f1': composite_f1,
            'binary_f1': binary_f1,
            'macro_f1': macro_f1,
            'best_iteration': model.best_iteration
        })
        
        print(f"  Fold {fold + 1} - Composite F1: {composite_f1:.4f} (Binary: {binary_f1:.4f}, Macro: {macro_f1:.4f})")
        
        # Save model
        models.append(model)
        model_path = output_dir / f"lgb_fold_{fold + 1}.txt"
        model.save_model(str(model_path))
    
    # Calculate overall CV performance
    overall_composite, overall_binary, overall_macro = evaluate_composite_f1(y_train, oof_predictions)
    
    cv_results = {
        'cv_composite_f1': overall_composite,
        'cv_binary_f1': overall_binary, 
        'cv_macro_f1': overall_macro,
        'cv_std': np.std([m['composite_f1'] for m in fold_metrics]),
        'fold_metrics': fold_metrics,
        'n_splits': n_splits
    }
    
    print(f"\n📈 Overall CV Results:")
    print(f"  Composite F1: {overall_composite:.4f} ± {cv_results['cv_std']:.4f}")
    print(f"  Binary F1: {overall_binary:.4f}")
    print(f"  Macro F1: {overall_macro:.4f}")
    
    # Save CV results
    with open(output_dir / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)
    
    # Save out-of-fold predictions
    oof_df = pd.DataFrame({
        'true_label': y_train,
        'oof_prediction': oof_predictions
    })
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)
    
    return models, cv_results, oof_predictions

def optimize_hyperparameters(X_train, y_train, groups, output_dir, n_trials=50):
    """Optimize LightGBM hyperparameters using Optuna"""
    print(f"\n🔧 Optimizing hyperparameters with {n_trials} trials...")
    
    def objective(trial):
        # Suggest hyperparameters
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y_train)),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': 42,
            
            # Hyperparameters to optimize
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        }
        
        # Quick 3-fold CV for optimization
        gkf = GroupKFold(n_splits=3)
        fold_scores = []
        
        for train_idx, val_idx in gkf.split(X_train, y_train, groups):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
            val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=200,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            val_pred = model.predict(X_fold_val, num_iteration=model.best_iteration)
            val_pred_labels = np.argmax(val_pred, axis=1) if len(val_pred.shape) > 1 else np.round(val_pred).astype(int)
            
            composite_f1, _, _ = evaluate_composite_f1(y_fold_val, val_pred_labels)
            fold_scores.append(composite_f1)
        
        return np.mean(fold_scores)
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"  ✓ Best score: {study.best_value:.4f}")
    print(f"  ✓ Best params: {study.best_params}")
    
    # Save optimization results
    with open(output_dir / "hyperparameter_optimization.json", "w") as f:
        json.dump({
            'best_score': study.best_value,
            'best_params': study.best_params,
            'n_trials': n_trials
        }, f, indent=2)
    
    return study.best_params

def generate_test_predictions(models, X_test, output_dir):
    """Generate test predictions using ensemble of trained models"""
    print("\n🔮 Generating test predictions...")
    
    # Average predictions across all folds
    test_predictions = np.zeros((len(X_test), len(models[0].predict(X_test.head(1)))))
    
    for i, model in enumerate(models):
        pred = model.predict(X_test, num_iteration=model.best_iteration)
        test_predictions += pred
        
    test_predictions /= len(models)
    
    # Convert to class labels
    if len(test_predictions.shape) > 1:
        test_pred_labels = np.argmax(test_predictions, axis=1)
    else:
        test_pred_labels = np.round(test_predictions).astype(int)
    
    # Create submission format
    submission = pd.DataFrame({
        'id': range(len(test_pred_labels)),  # Adjust based on actual test format
        'label': test_pred_labels
    })
    
    submission_path = output_dir / f"submission_lgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    submission.to_csv(submission_path, index=False)
    
    print(f"  ✓ Submission saved to: {submission_path}")
    
    return submission, test_pred_labels

def generate_training_report(cv_results, params, output_dir):
    """Generate comprehensive training report"""
    print("\n📋 Generating training report...")
    
    report_content = f"""
# LightGBM Baseline Training Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Performance
- **Composite F1 Score**: {cv_results['cv_composite_f1']:.4f} ± {cv_results['cv_std']:.4f}
- **Binary F1 Score**: {cv_results['cv_binary_f1']:.4f}
- **Macro F1 Score**: {cv_results['cv_macro_f1']:.4f}

## Cross-Validation Setup
- **Strategy**: GroupKFold (participant-aware)
- **Number of Folds**: {cv_results['n_splits']}
- **Validation Type**: Out-of-fold predictions

## Fold-by-Fold Results
"""
    
    for fold_metric in cv_results['fold_metrics']:
        report_content += f"- **Fold {fold_metric['fold']}**: {fold_metric['composite_f1']:.4f} (Binary: {fold_metric['binary_f1']:.4f}, Macro: {fold_metric['macro_f1']:.4f})\n"
    
    report_content += f"""

## Model Parameters
```json
{json.dumps(params, indent=2)}
```

## Files Generated
- `lgb_fold_*.txt` - Trained LightGBM models for each fold
- `cv_results.json` - Cross-validation metrics
- `oof_predictions.csv` - Out-of-fold predictions for analysis
- `submission_lgb_*.csv` - Test predictions for submission
- `hyperparameter_optimization.json` - Optimization results (if run)

## Next Steps
1. **Error Analysis**: Run `python scripts/workflow-to-evaluate.py` for detailed analysis
2. **Feature Importance**: Run `python scripts/workflow-to-feature-importance.py`
3. **Model Enhancement**: Consider 1D CNN with `python scripts/workflow-to-training-cnn.py`
4. **Submission**: Upload the generated submission file to Kaggle

## Competition Context
- **Target Score**: Bronze medal (LB 0.60+)
- **Current Model**: Baseline LightGBM with tsfresh features
- **Next Phase**: Deep learning integration (Week 2)
"""

    with open(output_dir / "TRAINING_REPORT.md", "w") as f:
        f.write(report_content)
    
    print(f"  ✓ Training report saved to: {output_dir}/TRAINING_REPORT.md")

def main():
    """Main LightGBM training workflow"""
    print("🎯 CMI Competition - LightGBM Baseline Training")
    print("=" * 50)
    
    # Setup
    output_dir, log_dir = setup_training()
    
    # Load data
    X_train, y_train, X_test, groups = load_ml_ready_data()
    if X_train is None:
        print("❌ Failed to load ML-ready data.")
        print("Please run the following in order:")
        print("  1. python scripts/workflow-to-bronze.py")
        print("  2. python scripts/workflow-to-silver.py") 
        print("  3. python scripts/workflow-to-gold.py")
        sys.exit(1)
    
    # Base LightGBM parameters
    base_params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y_train)),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        'learning_rate': 0.1,
        'num_leaves': 100,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
    }
    
    # Option to optimize hyperparameters
    optimize = input("\n🔧 Optimize hyperparameters? (y/N): ").lower().strip() == 'y'
    
    if optimize:
        best_params = optimize_hyperparameters(X_train, y_train, groups, output_dir)
        base_params.update(best_params)
    
    # Train model with cross-validation
    models, cv_results, oof_predictions = train_lgb_with_cv(
        X_train, y_train, groups, base_params, output_dir
    )
    
    # Generate test predictions
    submission, test_pred_labels = generate_test_predictions(models, X_test, output_dir)
    
    # Generate comprehensive report
    generate_training_report(cv_results, base_params, output_dir)
    
    # Final summary
    print("\n" + "=" * 50)
    print("✅ LightGBM baseline training completed!")
    print(f"📊 Final CV Score: {cv_results['cv_composite_f1']:.4f} ± {cv_results['cv_std']:.4f}")
    print(f"📁 Results saved to: {output_dir}")
    
    # Competition context
    target_score = 0.60
    current_score = cv_results['cv_composite_f1']
    
    if current_score >= target_score:
        print(f"🎉 Great! CV score {current_score:.4f} exceeds bronze target {target_score}")
    else:
        gap = target_score - current_score
        print(f"📈 CV score {current_score:.4f} is {gap:.4f} below bronze target {target_score}")
        print("   Consider feature engineering or deep learning approaches")
    
    print("\n🚀 Next steps:")
    print("  1. Review TRAINING_REPORT.md")
    print("  2. python scripts/workflow-to-evaluate.py    # Detailed error analysis")
    print("  3. python scripts/workflow-to-training-cnn.py # Try deep learning")
    print("  4. Submit to Kaggle to validate CV-LB alignment")

if __name__ == "__main__":
    main()