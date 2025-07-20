#!/usr/bin/env python3
"""Test two-stage prediction: Binary detection + Multi-class classification."""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load enhanced data
from src.data.gold import load_gold_data

print("🎯 Two-Stage Prediction Approach")
print("=" * 40)

df_train, df_test = load_gold_data()

# Add BFRB features
def add_bfrb_features(df):
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

# Prepare data
exclude_cols = ['participant_id', 'sequence_id', 'timestamp', 'behavior', 'behavior_encoded']
feature_cols = [col for col in df_train.columns if col not in exclude_cols]
X = df_train[feature_cols].fillna(0)
y = df_train['behavior_encoded']
groups = df_train['participant_id']

# Create binary target: 'Performs gesture' (class 2) vs others
# Class mapping: 0=Hand at target, 1=Moves hand to target, 2=Performs gesture, 3=Relaxes and moves
y_binary = (y == 2).astype(int)  # 1 if 'Performs gesture', 0 otherwise

print(f"Features: {len(feature_cols)}")
print(f"Training samples: {len(X)}")

print(f"\nBinary task distribution:")
binary_counts = y_binary.value_counts()
for i, count in binary_counts.items():
    label = 'Performs gesture' if i == 1 else 'Other behaviors'
    pct = count / len(y_binary) * 100
    print(f"  {label}: {count:6} ({pct:.1f}%)")

print(f"\nMulti-class distribution:")
class_counts = y.value_counts().sort_index()
class_names = ['Hand at target', 'Moves hand to target', 'Performs gesture', 'Relaxes and moves']
for i, count in class_counts.items():
    pct = count / len(y) * 100
    print(f"  {class_names[i]}: {count:6} ({pct:.1f}%)")

# Test approaches
gkf = GroupKFold(n_splits=5)
approaches = []

# Approach 1: Direct multi-class classification
print(f"\n🔄 Approach 1: Direct Multi-class Classification")
direct_scores = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]
    
    model = LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)
    model.fit(X_train_fold, y_train_fold)
    y_pred_fold = model.predict(X_val_fold)
    
    score = f1_score(y_val_fold, y_pred_fold, average='macro')
    direct_scores.append(score)
    print(f"  Fold {fold+1}: {score:.4f}")

direct_mean = np.mean(direct_scores)
print(f"  Mean F1: {direct_mean:.4f} ± {np.std(direct_scores):.4f}")

# Approach 2: Two-stage prediction
print(f"\n🔄 Approach 2: Two-stage (Binary + Multi-class)")
two_stage_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    y_binary_train_fold = y_binary.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]
    y_binary_val_fold = y_binary.iloc[val_idx]
    
    # Stage 1: Binary classification
    binary_model = LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)
    binary_model.fit(X_train_fold, y_binary_train_fold)
    binary_pred = binary_model.predict(X_val_fold)
    
    # Stage 2: Multi-class classification for detected gestures
    multiclass_model = LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)
    multiclass_model.fit(X_train_fold, y_train_fold)
    multiclass_pred = multiclass_model.predict(X_val_fold)
    
    # Combine predictions: use multiclass only where binary predicts gesture
    final_pred = np.where(binary_pred == 1, multiclass_pred, 
                         np.where(multiclass_pred == 2, 1, multiclass_pred))  # Convert gesture class to default
    
    score = f1_score(y_val_fold, final_pred, average='macro')
    two_stage_scores.append(score)
    print(f"  Fold {fold+1}: {score:.4f}")

two_stage_mean = np.mean(two_stage_scores)
print(f"  Mean F1: {two_stage_mean:.4f} ± {np.std(two_stage_scores):.4f}")

# Approach 3: Confidence-based two-stage
print(f"\n🔄 Approach 3: Confidence-based Two-stage")
confidence_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]
    
    # Train multi-class model
    model = LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)
    model.fit(X_train_fold, y_train_fold)
    
    # Get prediction probabilities
    pred_proba = model.predict_proba(X_val_fold)
    pred_confidence = np.max(pred_proba, axis=1)
    
    # Use high-confidence predictions, fallback for low-confidence
    threshold = 0.5
    high_conf_mask = pred_confidence > threshold
    
    final_pred = model.predict(X_val_fold)
    # For low confidence, bias towards most common class (class 2)
    final_pred[~high_conf_mask] = 2
    
    score = f1_score(y_val_fold, final_pred, average='macro')
    confidence_scores.append(score)
    print(f"  Fold {fold+1}: {score:.4f} (high_conf: {high_conf_mask.sum()}/{len(high_conf_mask)})")

confidence_mean = np.mean(confidence_scores)
print(f"  Mean F1: {confidence_mean:.4f} ± {np.std(confidence_scores):.4f}")

# Results summary
print(f"\n📊 Results Summary:")
print(f"{'Approach':25} {'F1 Score':>10} {'Std':>8} {'Improvement':>12}")
print("-" * 60)
print(f"{'Direct Multi-class':25} {direct_mean:>10.4f} {np.std(direct_scores):>8.4f} {'baseline':>12}")
print(f"{'Two-stage':25} {two_stage_mean:>10.4f} {np.std(two_stage_scores):>8.4f} {two_stage_mean-direct_mean:>+12.4f}")
print(f"{'Confidence-based':25} {confidence_mean:>10.4f} {np.std(confidence_scores):>8.4f} {confidence_mean-direct_mean:>+12.4f}")

# Best approach
best_score = max(direct_mean, two_stage_mean, confidence_mean)
if best_score == direct_mean:
    best_approach = "Direct Multi-class"
elif best_score == two_stage_mean:
    best_approach = "Two-stage"
else:
    best_approach = "Confidence-based"

print(f"\n🏆 Best Approach: {best_approach}")
print(f"Best F1 Score: {best_score:.4f}")

print(f"\n✅ Two-stage prediction analysis completed!")
print(f"Recommendation: Use {'Direct Multi-class' if best_score == direct_mean else best_approach} approach")