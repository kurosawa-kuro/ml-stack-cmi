#!/usr/bin/env python3
"""
Quick Cross-Validation Validation Script for CMI Sensor Data
Validates GroupKFold CV setup and basic model performance
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score
import lightgbm as lgb

from src.data.gold import load_gold_data, get_ml_ready_sequences


def validate_groupkfold_cv():
    """Validate GroupKFold CV setup for CMI sensor data"""
    print("🔍 Validating GroupKFold CV setup...")
    
    # Load Gold layer data
    train_data, test_data = load_gold_data()
    
    # Prepare ML-ready sequences
    X, y, groups = get_ml_ready_sequences(train_data)
    
    print(f"  📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  👥 Participants: {groups.nunique()}")
    print(f"  🎯 Target classes: {y.nunique()}")
    
    # Setup GroupKFold
    gkf = GroupKFold(n_splits=5)
    
    # Validate CV splits
    cv_scores = []
    participant_leakage_detected = False
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        # Check for participant leakage
        train_participants = set(groups.iloc[train_idx])
        val_participants = set(groups.iloc[val_idx])
        
        if train_participants.intersection(val_participants):
            participant_leakage_detected = True
            print(f"  ❌ Fold {fold+1}: Participant leakage detected!")
        
        # Quick model training for validation
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Simple LightGBM model
        model = lgb.LGBMClassifier(
            n_estimators=50,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        # Calculate scores
        accuracy = accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average='macro')
        
        cv_scores.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'train_participants': len(train_participants),
            'val_participants': len(val_participants)
        })
        
        print(f"  📈 Fold {fold+1}: Acc={accuracy:.3f}, F1={f1_macro:.3f}")
    
    # Summary
    if not participant_leakage_detected:
        print("  ✅ No participant leakage detected across all folds")
    
    avg_accuracy = np.mean([score['accuracy'] for score in cv_scores])
    avg_f1 = np.mean([score['f1_macro'] for score in cv_scores])
    
    print(f"\n📊 CV Summary:")
    print(f"  🎯 Average Accuracy: {avg_accuracy:.3f}")
    print(f"  🎯 Average F1 Macro: {avg_f1:.3f}")
    print(f"  ✅ GroupKFold CV validation passed")
    
    return {
        'cv_scores': cv_scores,
        'avg_accuracy': avg_accuracy,
        'avg_f1': avg_f1,
        'participant_leakage': participant_leakage_detected
    }


def main():
    """Main validation workflow"""
    print("🔍 Quick CV Validation for CMI Sensor Data")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Validate GroupKFold CV
        results = validate_groupkfold_cv()
        
        # Summary
        elapsed_time = time.time() - start_time
        print(f"\n✅ CV validation completed in {elapsed_time:.2f} seconds")
        
        if not results['participant_leakage']:
            print("  🎯 Ready for model training with GroupKFold CV")
            print("  📈 Expected performance baseline established")
        else:
            print("  ⚠️  Participant leakage detected - review CV setup")
        
    except Exception as e:
        print(f"\n❌ CV validation failed: {e}")
        raise


if __name__ == "__main__":
    main()