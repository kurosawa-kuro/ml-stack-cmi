#!/usr/bin/env python3
"""Analyze feature importance and performance."""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

# Load data
from src.data.gold import load_gold_data

print("Loading Gold layer data...")
df_train, df_test = load_gold_data()

# Prepare features and target
exclude_cols = ['participant_id', 'series_id', 'timestamp', 'behavior', 'behavior_encoded']
feature_cols = [col for col in df_train.columns if col not in exclude_cols]
X = df_train[feature_cols]
y = df_train['behavior_encoded']
groups = df_train['participant_id']

print(f"\nTraining data shape: {X.shape}")
print(f"Number of classes: {y.nunique()}")
print(f"Class distribution:\n{y.value_counts().sort_index()}")

# Train a simple model to get feature importance
print("\nTraining LightGBM to analyze features...")
lgb = LGBMClassifier(
    n_estimators=100,
    num_leaves=31,
    learning_rate=0.1,
    random_state=42,
    verbosity=-1
)

# Use only first fold for quick analysis
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(X, y, groups))

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]
X_val = X.iloc[val_idx]
y_val = y.iloc[val_idx]

lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])

# Get predictions and score
y_pred = lgb.predict(X_val)
val_score = f1_score(y_val, y_pred, average='macro')
print(f"\nValidation F1 Score: {val_score:.4f}")

# Feature importance analysis
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(importance_df.head(20).to_string(index=False))

# Analyze feature categories
print("\n\nFeature Category Analysis:")
sensor_categories = {
    'imu': ['acc_', 'rot_', 'gyro'],
    'thermal': ['thm_', 'thermal'],
    'tof': ['tof_'],
    'movement': ['movement', 'motion'],
    'proximity': ['proximity', 'close'],
    'statistical': ['mean', 'std', 'min', 'max'],
    'frequency': ['fft', 'power', 'freq']
}

for category, keywords in sensor_categories.items():
    cat_features = [f for f in feature_cols if any(k in f.lower() for k in keywords)]
    cat_importance = importance_df[importance_df['feature'].isin(cat_features)]['importance'].sum()
    print(f"{category:15} : {len(cat_features):3} features, {cat_importance:.1f} total importance")

# Save detailed analysis
importance_df.to_csv('outputs/reports/feature_analysis/feature_importance_detailed.csv', index=False)
print("\n✓ Detailed analysis saved to outputs/reports/feature_analysis/feature_importance_detailed.csv")