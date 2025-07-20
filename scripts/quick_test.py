#!/usr/bin/env python3
"""Quick test script to verify data processing pipeline."""

from src.data.gold import load_gold_data

print("Loading Gold layer data...")
df_train, df_test = load_gold_data()

print(f"\nTrain shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")

# Check for behavior_encoded
print(f"\nChecking for target leakage...")
train_cols = df_train.columns.tolist()
test_cols = df_test.columns.tolist()

leak_cols = [col for col in train_cols if 'behavior' in col.lower() and col != 'participant_id']
print(f"Columns with 'behavior': {leak_cols}")

# Show feature columns
feature_cols = [col for col in train_cols if col not in ['participant_id', 'series_id', 'timestamp', 'behavior']]
print(f"\nNumber of features: {len(feature_cols)}")
print(f"Sample features: {feature_cols[:10]}")

# Check target distribution
if 'behavior' in df_train.columns:
    print(f"\nTarget distribution:")
    print(df_train['behavior'].value_counts())

# Double check feature extraction
from src.data.gold import get_ml_ready_sequences
try:
    X_train, y_train, X_test = get_ml_ready_sequences()
    print(f"\nML-ready shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    
    # Check if behavior_encoded is in features
    if hasattr(X_train, 'columns'):
        behavior_cols_in_features = [col for col in X_train.columns if 'behavior' in col.lower()]
        print(f"\nBehavior columns in features: {behavior_cols_in_features}")
except Exception as e:
    print(f"\nError loading ML sequences: {e}")