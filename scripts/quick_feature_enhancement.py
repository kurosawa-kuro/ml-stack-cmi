#!/usr/bin/env python3
"""Quick feature enhancement focusing on high-impact BFRB features."""

import pandas as pd
import numpy as np

# Load existing gold data
from src.data.gold import load_gold_data

print("Loading current Gold data...")
df_train, df_test = load_gold_data()

# Create quick BFRB-specific features
def add_bfrb_features(df):
    """Add key BFRB features without heavy computation."""
    
    # 1. Hand-to-face proximity (from ToF sensors)
    tof_cols = [col for col in df.columns if col.startswith('tof_')]
    if tof_cols:
        df['hand_face_proximity'] = df[tof_cols].min(axis=1)
        df['close_contact'] = (df['hand_face_proximity'] < df['hand_face_proximity'].quantile(0.2)).astype(int)
    
    # 2. Temperature-based contact detection
    thm_cols = [col for col in df.columns if col.startswith('thm_')]
    if thm_cols:
        df['thermal_contact'] = df[thm_cols].max(axis=1)
        df['warm_contact'] = (df['thermal_contact'] > df['thermal_contact'].quantile(0.8)).astype(int)
    
    # 3. Movement-based features
    acc_cols = [col for col in df.columns if col.startswith('acc_')]
    if 'acc_x' in df.columns and 'acc_y' in df.columns and 'acc_z' in df.columns:
        df['total_acceleration'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        df['high_movement'] = (df['total_acceleration'] > df['total_acceleration'].quantile(0.8)).astype(int)
    elif len(acc_cols) > 0:
        # Use existing acceleration features if available
        df['total_acceleration'] = df[acc_cols].abs().sum(axis=1)
        df['high_movement'] = (df['total_acceleration'] > df['total_acceleration'].quantile(0.8)).astype(int)
    
    # 4. Sensor combinations
    if 'hand_face_proximity' in df.columns and 'total_acceleration' in df.columns:
        df['proximity_movement_interaction'] = df['hand_face_proximity'] * df['total_acceleration']
    
    if 'thermal_contact' in df.columns and 'total_acceleration' in df.columns:
        df['thermal_movement_interaction'] = df['thermal_contact'] * df['total_acceleration']
    
    return df

print("Adding BFRB-specific features...")
df_train = add_bfrb_features(df_train)
df_test = add_bfrb_features(df_test)

print(f"Enhanced data shapes - Train: {df_train.shape}, Test: {df_test.shape}")

# Quick feature importance check
exclude_cols = ['participant_id', 'sequence_id', 'timestamp', 'behavior', 'behavior_encoded']
feature_cols = [col for col in df_train.columns if col not in exclude_cols]
X = df_train[feature_cols]
y = df_train['behavior_encoded']

print(f"Total features: {len(feature_cols)}")

# Show new features
new_features = [col for col in feature_cols if col in ['hand_face_proximity', 'close_contact', 'thermal_contact', 'warm_contact', 'total_acceleration', 'high_movement', 'proximity_movement_interaction', 'thermal_movement_interaction']]
print(f"New BFRB features: {new_features}")

# Quick model test
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training quick model for validation...")
lgb = LGBMClassifier(n_estimators=50, random_state=42, verbosity=-1)
lgb.fit(X_train, y_train)

y_pred = lgb.predict(X_val)
score = f1_score(y_val, y_pred, average='macro')
print(f"\nQuick validation F1 score: {score:.4f}")

# Show importance of new features
if new_features:
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': lgb.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nImportance of new BFRB features:")
    for feat in new_features:
        if feat in importance_df['feature'].values:
            imp = importance_df[importance_df['feature'] == feat]['importance'].iloc[0]
            rank = importance_df[importance_df['feature'] == feat].index[0] + 1
            print(f"  {feat}: {imp:.1f} (rank {rank}/{len(feature_cols)})")

print("\n✅ Quick feature enhancement completed!")