#!/usr/bin/env python3
"""Optimize LightGBM with class imbalance handling and hyperparameter tuning."""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Load enhanced data
from src.data.gold import load_gold_data

print("🚀 LightGBM Optimization with Class Imbalance Handling")
print("=" * 60)

df_train, df_test = load_gold_data()

# Add the BFRB features we created
def add_bfrb_features(df):
    """Add BFRB features."""
    tof_cols = [col for col in df.columns if col.startswith('tof_')]
    if tof_cols:
        df['hand_face_proximity'] = df[tof_cols].min(axis=1)
        df['close_contact'] = (df['hand_face_proximity'] < df['hand_face_proximity'].quantile(0.2)).astype(int)
    
    thm_cols = [col for col in df.columns if col.startswith('thm_')]
    if thm_cols:
        df['thermal_contact'] = df[thm_cols].max(axis=1)
        df['warm_contact'] = (df['thermal_contact'] > df['thermal_contact'].quantile(0.8)).astype(int)
    
    acc_cols = [col for col in df.columns if col.startswith('acc_')]
    if 'acc_x' in df.columns and 'acc_y' in df.columns and 'acc_z' in df.columns:
        df['total_acceleration'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        df['high_movement'] = (df['total_acceleration'] > df['total_acceleration'].quantile(0.8)).astype(int)
    elif len(acc_cols) > 0:
        df['total_acceleration'] = df[acc_cols].abs().sum(axis=1)
        df['high_movement'] = (df['total_acceleration'] > df['total_acceleration'].quantile(0.8)).astype(int)
    
    if 'hand_face_proximity' in df.columns and 'total_acceleration' in df.columns:
        df['proximity_movement_interaction'] = df['hand_face_proximity'] * df['total_acceleration']
    
    if 'thermal_contact' in df.columns and 'total_acceleration' in df.columns:
        df['thermal_movement_interaction'] = df['thermal_contact'] * df['total_acceleration']
    
    return df

df_train = add_bfrb_features(df_train)
df_test = add_bfrb_features(df_test)

# Prepare data
exclude_cols = ['participant_id', 'sequence_id', 'timestamp', 'behavior', 'behavior_encoded']
feature_cols = [col for col in df_train.columns if col not in exclude_cols]
X = df_train[feature_cols].fillna(0)
y = df_train['behavior_encoded']
groups = df_train['participant_id']

print(f"Features: {len(feature_cols)}")
print(f"Training samples: {len(X)}")
print(f"Class distribution:")
class_counts = y.value_counts().sort_index()
for i, count in class_counts.items():
    pct = count / len(y) * 100
    print(f"  Class {i}: {count:6} ({pct:.1f}%)")

# Calculate class weights for imbalance handling
classes = np.unique(y)
class_weights = compute_class_weight('balanced', classes=classes, y=y)
class_weight_dict = dict(zip(classes, class_weights))
print(f"\nClass weights: {class_weight_dict}")

# Test different configurations
configs = [
    {
        'name': 'Baseline',
        'params': {
            'n_estimators': 100,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbosity': -1
        }
    },
    {
        'name': 'Balanced',
        'params': {
            'n_estimators': 100,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'class_weight': 'balanced',
            'random_state': 42,
            'verbosity': -1
        }
    },
    {
        'name': 'Balanced + More Trees',
        'params': {
            'n_estimators': 200,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'class_weight': 'balanced',
            'random_state': 42,
            'verbosity': -1
        }
    },
    {
        'name': 'Balanced + Tuned',
        'params': {
            'n_estimators': 300,
            'num_leaves': 50,
            'learning_rate': 0.05,
            'max_depth': 7,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'class_weight': 'balanced',
            'random_state': 42,
            'verbosity': -1
        }
    }
]

# Test each configuration
results = []
gkf = GroupKFold(n_splits=5)

for config in configs:
    print(f"\n🧪 Testing {config['name']}...")
    
    lgb = LGBMClassifier(**config['params'])
    
    # Cross-validation
    cv_scores = []
    fold = 0
    for train_idx, val_idx in gkf.split(X, y, groups):
        fold += 1
        print(f"  Fold {fold}/5", end="")
        
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        lgb.fit(X_train_fold, y_train_fold)
        y_pred_fold = lgb.predict(X_val_fold)
        
        fold_score = f1_score(y_val_fold, y_pred_fold, average='macro')
        cv_scores.append(fold_score)
        print(f" - F1: {fold_score:.4f}")
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    
    results.append({
        'config': config['name'],
        'mean_f1': mean_score,
        'std_f1': std_score,
        'scores': cv_scores
    })
    
    print(f"  Mean F1: {mean_score:.4f} ± {std_score:.4f}")

# Results summary
print("\n📊 Results Summary:")
print("-" * 50)
for result in sorted(results, key=lambda x: x['mean_f1'], reverse=True):
    print(f"{result['config']:20}: {result['mean_f1']:.4f} ± {result['std_f1']:.4f}")

# Best configuration
best_config = max(results, key=lambda x: x['mean_f1'])
print(f"\n🏆 Best Configuration: {best_config['config']}")
print(f"F1 Score: {best_config['mean_f1']:.4f} ± {best_config['std_f1']:.4f}")

# Train final model with best config
best_params = next(c['params'] for c in configs if c['name'] == best_config['config'])
final_model = LGBMClassifier(**best_params)

print(f"\n🎯 Training final model with {best_config['config']} configuration...")
final_model.fit(X, y)

# Feature importance of final model
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 Top 15 Most Important Features:")
print(importance_df.head(15).to_string(index=False, float_format='%.1f'))

# Save results
output_path = 'outputs/reports/model_optimization_results.csv'
pd.DataFrame(results).to_csv(output_path, index=False)
print(f"\n💾 Results saved to {output_path}")

print(f"\n✅ Model optimization completed!")
print(f"Best F1 Score: {best_config['mean_f1']:.4f}")
print(f"Improvement from baseline: {best_config['mean_f1'] - results[0]['mean_f1']:.4f}")