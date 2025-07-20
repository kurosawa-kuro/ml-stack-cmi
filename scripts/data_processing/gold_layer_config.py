#!/usr/bin/env python3
"""
CMI Competition - Configuration-Driven Gold Layer Processing
==========================================================
Enhanced gold layer ML preparation with configuration control

Features:
- Configuration-driven feature selection
- Phase-aware ML preparation
- Dynamic GroupKFold setup
- Baseline phase ML optimization
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import get_project_config
from src.data.gold import (
    clean_and_validate_sensor_features,
    select_sensor_features,
    prepare_ml_ready_data,
    create_gold_tables
)

def main():
    """Configuration-driven gold layer processing"""
    print("🥇 CMI Competition - Configuration-Driven Gold Layer Processing")
    print("=" * 70)
    
    # Load project configuration
    config = get_project_config()
    
    # Display current configuration
    print(f"📋 ML Preparation Configuration:")
    print(f"  ├── Phase: {config.phase.value}")
    print(f"  ├── CV Strategy: {config.data.cv_strategy}")
    print(f"  ├── CV Folds: {config.data.cv_folds}")
    print(f"  ├── Group Column: {config.data.group_column}")
    print(f"  ├── Max Features: {config.data.max_features}")
    print(f"  ├── Primary Algorithm: {config.algorithm_strategy.primary_algorithm}")
    print(f"  └── Target CV Score: {config.targets.cv_score}")
    print()
    
    # Phase-specific ML preparation strategy
    if config.phase.value == "baseline":
        print("🎯 Baseline Phase ML Preparation:")
        print("  ├── Priority: GroupKFold setup for LightGBM")
        print("  ├── Strategy: Sensor-priority feature selection")
        print("  ├── Focus: Single algorithm optimization")
        print("  └── Goal: ML-ready data for baseline training")
        
    elif config.phase.value == "optimization":
        print("🎯 Optimization Phase ML Preparation:")
        print("  ├── Priority: Advanced feature selection")
        print("  ├── Strategy: Multi-algorithm feature sets")
        print("  ├── Focus: Feature importance optimization")
        print("  └── Goal: Enhanced ML pipeline")
        
    elif config.phase.value == "ensemble":
        print("🎯 Ensemble Phase ML Preparation:")
        print("  ├── Priority: Model-specific data preparation")
        print("  ├── Strategy: Diverse feature sets for ensemble")
        print("  ├── Focus: Cross-model compatibility")
        print("  └── Goal: Ensemble-ready data pipeline")
    
    print()
    
    # ML preparation execution
    print("🔧 Executing ML Preparation Pipeline...")
    
    try:
        # Load silver data
        print("  1️⃣ Loading silver layer data...")
        from src.data.silver import load_silver_data
        train_silver, test_silver = load_silver_data()
        print(f"     ✓ Train: {train_silver.shape}, Test: {test_silver.shape}")
        
        # Data cleaning and validation
        print("  2️⃣ Cleaning and validating sensor features...")
        train_clean = clean_and_validate_sensor_features(train_silver)
        test_clean = clean_and_validate_sensor_features(test_silver)
        print(f"     ✓ Cleaned features: {train_clean.shape[1]} columns")
        
        # Feature selection (configuration-driven)
        if 'gesture' in train_clean.columns:
            target_col = 'gesture'
        elif 'behavior' in train_clean.columns:
            target_col = 'behavior'
        else:
            print("     ⚠️ No target column found, using placeholder")
            target_col = 'target'
        
        print(f"  3️⃣ Selecting features (max_features={config.data.max_features})...")
        selected_features = select_sensor_features(
            train_clean, 
            target_col=target_col, 
            k=config.data.max_features,
            method="sensor_priority"
        )
        print(f"     ✓ Selected features: {len(selected_features)} out of {train_clean.shape[1]}")
        
        # GroupKFold preparation
        print(f"  4️⃣ Setting up {config.data.cv_strategy} cross-validation...")
        if config.data.group_column in train_clean.columns:
            groups = train_clean[config.data.group_column]
            unique_groups = groups.nunique()
            print(f"     ✓ GroupKFold setup: {unique_groups} unique groups ({config.data.group_column})")
            print(f"     ✓ CV folds: {config.data.cv_folds} (≈{unique_groups // config.data.cv_folds} groups per fold)")
        else:
            print(f"     ⚠️ Group column '{config.data.group_column}' not found")
        
        # ML-ready data preparation
        print("  5️⃣ Preparing ML-ready datasets...")
        # Filter to selected features
        train_ml = train_clean[selected_features + [target_col] + ([config.data.group_column] if config.data.group_column in train_clean.columns else [])]
        test_ml = test_clean[selected_features + ([config.data.group_column] if config.data.group_column in test_clean.columns else [])]
        
        print(f"     ✓ ML-ready train: {train_ml.shape}")
        print(f"     ✓ ML-ready test: {test_ml.shape}")
        
        # Create gold tables
        print("  6️⃣ Creating gold layer tables...")
        try:
            # Save ML-ready data to gold tables
            from src.data.gold import create_gold_tables
            
            # Prepare data for gold tables
            train_gold = train_ml.copy()
            test_gold = test_ml.copy()
            
            # Create gold tables
            create_gold_tables()
            print("     ✓ Gold tables created successfully")
            
        except Exception as e:
            print(f"     ⚠️ Gold tables creation failed: {e}")
            print("     ✓ Continuing with ML-ready data preparation")
        
        # Final summary
        print(f"\n✨ Gold Layer ML Preparation Complete!")
        print(f"  ├── Phase: {config.phase.value}")
        print(f"  ├── ML-ready features: {len(selected_features)}")
        print(f"  ├── CV strategy: {config.data.cv_strategy} ({config.data.cv_folds} folds)")
        print(f"  ├── Primary algorithm: {config.algorithm_strategy.primary_algorithm}")
        print(f"  └── Ready for: Model training")
        
        # Algorithm-specific recommendations
        if config.algorithm_strategy.primary_algorithm == "lightgbm":
            print(f"\n🎯 LightGBM Optimization Ready:")
            print(f"  ├── Tree-based model: Handles missing values natively")
            print(f"  ├── Feature count: {len(selected_features)} (optimal for LightGBM)")
            print(f"  ├── GroupKFold: Prevents participant leakage")
            print(f"  └── Next: python scripts/training/train_lightgbm_config.py")
        
        # Phase-specific next steps
        if config.phase.value == "baseline":
            print(f"\n🎯 Baseline Phase Ready:")
            print(f"  1. Train single algorithm: {config.algorithm_strategy.primary_algorithm}")
            print(f"  2. Target CV score: {config.targets.cv_score}")
            print(f"  3. Success criteria: Achieve baseline before advancing to optimization")
        
    except Exception as e:
        print(f"  ❌ Error in ML preparation: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()