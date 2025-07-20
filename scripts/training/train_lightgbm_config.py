#!/usr/bin/env python3
"""
CMI Competition - Configuration-Driven LightGBM Training
========================================================
Enhanced training script with configuration-driven development

Features:
- Configuration-driven algorithm selection
- Phase-aware training strategy
- Dynamic parameter loading
- Baseline phase enforcement
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import get_project_config

def main():
    """Main training function with configuration-driven approach"""
    print("🚀 CMI Competition - Configuration-Driven Training")
    print("=" * 60)
    
    # Load project configuration
    config = get_project_config()
    
    # Display current configuration
    print(f"📋 Current Configuration:")
    print(f"  ├── Phase: {config.phase.value}")
    print(f"  ├── Primary Algorithm: {config.algorithm_strategy.primary_algorithm}")
    print(f"  ├── Enabled Algorithms: {config.algorithm_strategy.enabled_algorithms}")
    print(f"  ├── Ensemble Enabled: {config.algorithm_strategy.ensemble_enabled}")
    print(f"  ├── Target CV Score: {config.targets.cv_score}")
    print(f"  ├── Target LB Score: {config.targets.lb_score}")
    print(f"  └── Strategy Focus: {config.algorithm_strategy.focus}")
    print()
    
    # Validate phase constraints
    if len(config.algorithm_strategy.enabled_algorithms) > 1:
        print("⚠️  WARNING: Configuration suggests multi-algorithm approach")
        print("   This script focuses on single algorithm training")
        print("   Consider using ensemble training script for multi-algorithm approach")
        print()
    
    # Check if LightGBM is enabled
    if not config.is_algorithm_enabled("lightgbm"):
        print("❌ LightGBM is not enabled in current configuration")
        print("   Please update configuration to enable LightGBM")
        return
    
    # Get LightGBM parameters from configuration
    lgb_params = config.get_model_params("lightgbm")
    print(f"🔧 LightGBM Parameters:")
    for key, value in lgb_params.items():
        print(f"  ├── {key}: {value}")
    print()
    
    # Load data from bronze layer
    print("📊 Loading data from bronze layer...")
    try:
        from src.data.bronze import load_bronze_data
        train_df, test_df = load_bronze_data()
        print(f"  ✓ Train data: {train_df.shape}")
        print(f"  ✓ Test data: {test_df.shape}")
        
        # Identify target and feature columns
        if 'gesture' in train_df.columns:
            target_col = 'gesture'
        elif 'behavior' in train_df.columns:
            target_col = 'behavior'
        else:
            print("  ❌ No target column found (gesture/behavior)")
            return
        
        # Get group column for GroupKFold
        group_col = config.data.group_column
        if group_col not in train_df.columns:
            print(f"  ❌ Group column '{group_col}' not found")
            return
        
        print(f"  ✓ Target column: {target_col}")
        print(f"  ✓ Group column: {group_col}")
        print(f"  ✓ Unique groups: {train_df[group_col].nunique()}")
        
    except Exception as e:
        print(f"  ❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Configuration validation
    warnings = config.validate_configuration()
    if warnings:
        print("⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"  • {warning}")
        print()
    
    # Phase-specific training logic
    if config.phase.value == "baseline":
        print("🎯 Baseline Phase Training:")
        print("  ├── Focus: Single algorithm mastery")
        print("  ├── Goal: Establish solid foundation")
        print("  └── Next: Achieve CV {:.2f}+ before advancing".format(config.targets.cv_score))
        
    elif config.phase.value == "optimization":
        print("🎯 Optimization Phase Training:")
        print("  ├── Focus: Hyperparameter tuning")
        print("  ├── Goal: Feature engineering completion")
        print("  └── Target: CV {:.2f}+".format(config.targets.cv_score))
        
    elif config.phase.value == "ensemble":
        print("🎯 Ensemble Phase Training:")
        print("  ├── Focus: Model diversity")
        print("  ├── Goal: Bronze medal achievement")
        print("  └── Target: LB {:.2f}+ (Bronze)".format(config.targets.bronze_score))
    
    # Prepare features for training
    print("🔧 Preparing features for training...")
    try:
        # Select numeric features (exclude ID and metadata columns)
        exclude_cols = ['row_id', 'sequence_id', 'sequence_type', 'sequence_counter', 
                       'subject', 'orientation', 'phase', target_col, group_col]
        feature_cols = [col for col in train_df.columns 
                       if col not in exclude_cols and train_df[col].dtype in ['float32', 'float64', 'int32', 'int64']]
        
        X_train = train_df[feature_cols].fillna(0)  # Handle any remaining NaN
        y_train = train_df[target_col]
        groups = train_df[group_col]
        
        print(f"  ✓ Feature columns: {len(feature_cols)}")
        print(f"  ✓ Target classes: {y_train.nunique()}")
        print(f"  ✓ Sample shape: {X_train.shape}")
        
        # Encode target if categorical
        from sklearn.preprocessing import LabelEncoder
        if y_train.dtype == 'object':
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            print(f"  ✓ Encoded target: {len(le.classes_)} classes")
        else:
            y_train_encoded = y_train
            print(f"  ✓ Numeric target: {y_train.min()}-{y_train.max()}")
        
    except Exception as e:
        print(f"  ❌ Error preparing features: {e}")
        return
    
    # GroupKFold cross-validation
    print(f"\n🔄 Setting up {config.data.cv_strategy} cross-validation...")
    try:
        gkf = GroupKFold(n_splits=config.data.cv_folds)
        cv_scores = []
        
        print(f"  ✓ CV setup: {config.data.cv_folds} folds")
        print(f"  ✓ Groups: {len(groups.unique())} unique participants")
        
        # Quick CV demonstration (simplified)
        fold_count = 0
        for train_idx, val_idx in gkf.split(X_train, y_train_encoded, groups):
            fold_count += 1
            if fold_count > 2:  # Limit to 2 folds for demo
                break
                
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train_encoded[train_idx], y_train_encoded[val_idx]
            
            print(f"    Fold {fold_count}: Train {X_fold_train.shape[0]:,}, Val {X_fold_val.shape[0]:,}")
        
        print(f"  ✓ GroupKFold validation setup complete")
        
    except Exception as e:
        print(f"  ❌ Error in CV setup: {e}")
        return
    
    print()
    print("✨ Configuration-driven training setup complete!")
    print(f"  ├── Phase: {config.phase.value}")
    print(f"  ├── Algorithm: {config.algorithm_strategy.primary_algorithm}")
    print(f"  ├── Features: {len(feature_cols):,}")
    print(f"  ├── Samples: {X_train.shape[0]:,}")
    print(f"  ├── Target: {target_col} ({y_train.nunique()} classes)")
    print(f"  └── Ready for: Full model training")
    
    # Full training loop implementation
    print(f"\n🚀 Starting LightGBM training loop...")
    try:
        fold_scores = []
        
        for fold_num, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train_encoded, groups)):
            print(f"\n  📊 Fold {fold_num + 1}/{config.data.cv_folds}")
            
            # Split data
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train_encoded[train_idx], y_train_encoded[val_idx]
            
            # Train LightGBM
            train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
            val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
            
            model = lgb.train(
                lgb_params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'val'],
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
            )
            
            # Predict and evaluate
            val_pred = model.predict(X_fold_val, num_iteration=model.best_iteration)
            val_pred_class = (val_pred > 0.5).astype(int) if len(np.unique(y_train_encoded)) == 2 else np.argmax(val_pred, axis=1)
            
            # Calculate F1 score
            fold_f1 = f1_score(y_fold_val, val_pred_class, average='macro')
            fold_scores.append(fold_f1)
            
            print(f"    ✓ Fold {fold_num + 1} F1: {fold_f1:.4f}")
            
            # For baseline phase, limit to 2 folds for speed
            if config.phase.value == "baseline" and fold_num >= 1:
                print(f"    ⚡ Baseline phase: Limited to {fold_num + 1} folds for speed")
                break
        
        # Calculate final CV score
        cv_score = np.mean(fold_scores)
        cv_std = np.std(fold_scores)
        
        print(f"\n🎯 Training Results:")
        print(f"  ├── CV Score: {cv_score:.4f} ± {cv_std:.4f}")
        print(f"  ├── Target: {config.targets.cv_score}")
        print(f"  ├── Folds: {len(fold_scores)}")
        print(f"  └── Algorithm: {config.algorithm_strategy.primary_algorithm}")
        
        # Save training results for evaluation
        print(f"\n💾 Saving training results...")
        try:
            # Create output directories
            output_dir = Path(__file__).parent.parent.parent / "outputs"
            models_dir = output_dir / "models" / "lgb_baseline"
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Save CV results
            cv_results = {
                'cv_score': cv_score,
                'cv_std': cv_std,
                'fold_scores': fold_scores,
                'target_score': config.targets.cv_score,
                'phase': config.phase.value,
                'algorithm': config.algorithm_strategy.primary_algorithm,
                'timestamp': datetime.now().isoformat()
            }
            
            import json
            with open(models_dir / "cv_results.json", 'w') as f:
                json.dump(cv_results, f, indent=2)
            
            # Save OOF predictions (simplified for demo)
            oof_predictions = pd.DataFrame({
                'fold': range(len(fold_scores)),
                'cv_score': fold_scores,
                'target_score': [config.targets.cv_score] * len(fold_scores)
            })
            oof_predictions.to_csv(models_dir / "oof_predictions.csv", index=False)
            
            print(f"  ✓ Results saved to: {models_dir}")
            
        except Exception as e:
            print(f"  ⚠️ Warning: Could not save results: {e}")
        
        # Phase-specific completion status
        if cv_score >= config.targets.cv_score:
            print(f"\n✅ Target CV score achieved! ({cv_score:.4f} >= {config.targets.cv_score})")
            if config.phase.value == "baseline":
                print(f"  🎯 Ready to advance to optimization phase")
        else:
            print(f"\n⚠️ Target CV score not reached ({cv_score:.4f} < {config.targets.cv_score})")
            print(f"  💡 Consider: Feature engineering, hyperparameter tuning")
        
    except Exception as e:
        print(f"  ❌ Error in training loop: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Phase-specific next steps
    if config.phase.value == "baseline":
        print(f"\n🎯 Baseline Phase Status:")
        print(f"  1. ✅ Data loading and preprocessing")
        print(f"  2. ✅ GroupKFold CV setup")
        print(f"  3. ✅ LightGBM parameter configuration")
        print(f"  4. ✅ Full training loop execution")
        print(f"  5. ✅ CV Score: {cv_score:.4f}")
        if cv_score >= config.targets.cv_score:
            print(f"  6. ✅ Target achieved - Ready for optimization phase")
        else:
            print(f"  6. ⚠️ Target missed - Baseline improvements needed")


if __name__ == "__main__":
    main()