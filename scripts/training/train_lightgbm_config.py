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
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import get_project_config, should_use_single_algorithm, get_primary_algorithm

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
    if not should_use_single_algorithm():
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
    
    # Load data based on configuration
    print("📊 Loading data...")
    try:
        # This would be replaced with actual data loading from gold layer
        # For now, demonstrate configuration usage
        print(f"  ├── Data source: {config.data.source_type}")
        print(f"  ├── CV strategy: {config.data.cv_strategy}")
        print(f"  ├── CV folds: {config.data.cv_folds}")
        print(f"  └── Group column: {config.data.group_column}")
        
        # Placeholder for actual data loading
        print("  ⚠️  Data loading not implemented - this is a configuration demo")
        
    except Exception as e:
        print(f"  ❌ Error loading data: {e}")
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
    
    print()
    print("✨ Configuration-driven training setup complete!")
    print("   Ready for implementation with phase-appropriate strategy")


if __name__ == "__main__":
    main()