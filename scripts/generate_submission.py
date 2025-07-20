#!/usr/bin/env python3
"""Generate Kaggle submission file using optimized LightGBM model."""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load data and apply BFRB features
from src.data.gold import load_gold_data

def add_bfrb_features(df):
    """Add BFRB-specific features to dataframe."""
    # ToF proximity features
    tof_cols = [col for col in df.columns if col.startswith('tof_')]
    if tof_cols:
        df['hand_face_proximity'] = df[tof_cols].min(axis=1)
        df['close_contact'] = (df['hand_face_proximity'] < df['hand_face_proximity'].quantile(0.2)).astype(int)
    
    # Thermal contact features
    thm_cols = [col for col in df.columns if col.startswith('thm_')]
    if thm_cols:
        df['thermal_contact'] = df[thm_cols].max(axis=1)
        df['warm_contact'] = (df['thermal_contact'] > df['thermal_contact'].quantile(0.8)).astype(int)
    
    # Movement features
    acc_cols = [col for col in df.columns if col.startswith('acc_')]
    if 'acc_x' in df.columns and 'acc_y' in df.columns and 'acc_z' in df.columns:
        df['total_acceleration'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        df['high_movement'] = (df['total_acceleration'] > df['total_acceleration'].quantile(0.8)).astype(int)
    elif len(acc_cols) > 0:
        df['total_acceleration'] = df[acc_cols].abs().sum(axis=1)
        df['high_movement'] = (df['total_acceleration'] > df['total_acceleration'].quantile(0.8)).astype(int)
    
    # Interaction features
    if 'hand_face_proximity' in df.columns and 'total_acceleration' in df.columns:
        df['proximity_movement_interaction'] = df['hand_face_proximity'] * df['total_acceleration']
    
    if 'thermal_contact' in df.columns and 'total_acceleration' in df.columns:
        df['thermal_movement_interaction'] = df['thermal_contact'] * df['total_acceleration']
    
    return df

def generate_submission():
    """Generate submission CSV file."""
    print("🎯 CMI Competition - Generating Submission File")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    df_train, df_test = load_gold_data()
    
    # Apply enhanced features
    print("Adding BFRB features...")
    df_train = add_bfrb_features(df_train)
    df_test = add_bfrb_features(df_test)
    
    # Prepare training data
    exclude_cols = ['participant_id', 'sequence_id', 'timestamp', 'behavior', 'behavior_encoded']
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]
    
    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train['behavior_encoded']
    
    # Prepare test data (use common features only)
    test_feature_cols = [col for col in feature_cols if col in df_test.columns]
    X_test = df_test[test_feature_cols].fillna(0)
    
    print(f"Training features: {len(feature_cols)}")
    print(f"Test features: {len(test_feature_cols)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Check if we have required ID columns for submission
    if 'id' in df_test.columns:
        test_ids = df_test['id']
    elif 'row_id' in df_test.columns:
        test_ids = df_test['row_id']
    else:
        print("⚠️ Warning: No ID column found in test data")
        test_ids = range(len(X_test))
    
    # Train final model (optimized configuration from our analysis)
    print("Training final LightGBM model...")
    model = LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        learning_rate=0.1,
        random_state=42,
        verbosity=-1
    )
    
    # Align features for prediction
    if len(test_feature_cols) < len(feature_cols):
        print(f"⚠️ Using {len(test_feature_cols)} common features for prediction")
        X_train_aligned = X_train[test_feature_cols]
        model.fit(X_train_aligned, y_train)
    else:
        model.fit(X_train, y_train)
    
    # Generate predictions
    print("Generating predictions...")
    y_pred = model.predict(X_test)
    
    # Map predictions back to original behavior labels
    class_mapping = {
        0: "Hand at target location",
        1: "Moves hand to target location", 
        2: "Performs gesture",
        3: "Relaxes and moves hand to target location"
    }
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'id': test_ids,
        'behavior': [class_mapping[pred] for pred in y_pred]
    })
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"outputs/submissions/baseline/submission_lgb_cv0768_{timestamp}.csv"
    
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save submission
    submission.to_csv(filename, index=False)
    
    print(f"\n✅ Submission file generated:")
    print(f"📁 File: {filename}")
    print(f"📊 Shape: {submission.shape}")
    print(f"🎯 Model: LightGBM (CV 0.7678)")
    
    # Show submission preview
    print(f"\n📋 Submission Preview:")
    print(submission.head(10))
    
    # Show prediction distribution
    print(f"\n📈 Prediction Distribution:")
    pred_dist = submission['behavior'].value_counts()
    for behavior, count in pred_dist.items():
        pct = count / len(submission) * 100
        print(f"  {behavior}: {count} ({pct:.1f}%)")
    
    print(f"\n🚀 Ready for Kaggle submission!")
    print(f"Command: kaggle competitions submit -c cmi-detect-behavior-with-sensor-data -f {filename} -m 'LightGBM baseline CV 0.7678'")
    
    return filename, submission

if __name__ == "__main__":
    generate_submission()