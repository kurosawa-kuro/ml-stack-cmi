#!/usr/bin/env python
"""
Quick CV test for enhanced Silver features
Bronze medal target smoke test
"""

import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.bronze import load_data, create_bronze_tables
from src.data.bronze import load_bronze_data
from src.data.silver import EnhancedSilverPreprocessor
from src.models import LightGBMModel

warnings.filterwarnings("ignore")


def run_quick_cv_with_enhanced_features(sample_ratio: float = 0.1, folds: int = 3, random_state: int = 42) -> float:
    """
    Quick CV test with enhanced silver features
    
    Args:
        sample_ratio: Fraction of data to use for speed
        folds: Number of CV folds
        random_state: Random seed
    
    Returns:
        Mean CV accuracy score
    """
    print(f"🚀 Quick CV Test - Enhanced Silver Features")
    print(f"   Sample: {sample_ratio*100:.1f}%, Folds: {folds}")
    
    # Load bronze data (preprocessed)
    try:
        train_data, test_data = load_bronze_data()
    except Exception:
        print("   Creating bronze tables...")
        create_bronze_tables()
        train_data, test_data = load_bronze_data()
    
    # Sample for speed
    if sample_ratio < 1.0:
        n_samples = int(len(train_data) * sample_ratio)
        train_data = train_data.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
        print(f"   Sampled: {len(train_data)} rows")
    
    # Prepare features and target
    X = train_data.drop(['id', 'Personality'], axis=1, errors='ignore')
    y = (train_data['Personality'] == 'Extrovert').astype(int)
    
    print(f"   Base features: {X.shape[1]}")
    
    # Apply enhanced silver features
    start_time = time.time()
    preprocessor = EnhancedSilverPreprocessor()
    preprocessor.fit(X, y)
    X_enhanced = preprocessor.transform(X)
    feature_time = time.time() - start_time
    
    print(f"   Enhanced features: {X_enhanced.shape[1]} (+{X_enhanced.shape[1] - X.shape[1]})")
    print(f"   Feature engineering: {feature_time:.2f}s")
    
    # Quick CV
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    cv_scores = []
    
    # Optimized LightGBM params for speed
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 100,  # Reduced for speed
        "num_leaves": 31,
        "learning_rate": 0.1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "random_state": random_state,
        "verbosity": -1,
        "n_jobs": -1,
    }
    
    print(f"   Running {folds}-fold CV...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_enhanced, y)):
        fold_start = time.time()
        
        X_train_fold = X_enhanced.iloc[train_idx]
        X_val_fold = X_enhanced.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train model
        model = LightGBMModel(params=lgb_params)
        model.fit(X_train_fold, y_train_fold)
        
        # Predict and score
        y_pred = model.predict(X_val_fold)
        score = accuracy_score(y_val_fold, y_pred)
        cv_scores.append(score)
        
        fold_time = time.time() - fold_start
        print(f"     Fold {fold+1}: {score:.6f} ({fold_time:.2f}s)")
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    
    print(f"   📊 CV Result: {mean_score:.6f} ± {std_score:.6f}")
    print(f"   🎯 Bronze target: 0.976518 (Gap: {0.976518 - mean_score:+.6f})")
    
    return mean_score


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick CV test for enhanced silver features")
    parser.add_argument("--sample", type=float, default=0.1, help="Sample ratio (default: 0.1)")
    parser.add_argument("--folds", type=int, default=3, help="Number of CV folds (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    score = run_quick_cv_with_enhanced_features(
        sample_ratio=args.sample,
        folds=args.folds,
        random_state=args.seed
    )
    
    # Output just the score for subprocess parsing
    if len(sys.argv) == 1:  # No arguments, just print score
        print(f"{score:.6f}")
    
    return score


if __name__ == "__main__":
    main()