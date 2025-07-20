#!/usr/bin/env python3
"""
CMI Competition - Configuration-Driven Bronze Layer Processing
============================================================
Enhanced bronze layer processing with configuration-driven development

Features:
- Configuration-driven data source management
- Phase-aware processing strategy
- Dynamic parameter loading from config
- Baseline phase optimization
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
from src.data.bronze import load_data, create_bronze_tables, validate_data_quality_cmi

def main():
    """Configuration-driven bronze layer processing"""
    print("🥉 CMI Competition - Configuration-Driven Bronze Layer Processing")
    print("=" * 70)
    
    # Load project configuration
    config = get_project_config()
    
    # Display current configuration
    print(f"📋 Current Configuration:")
    print(f"  ├── Phase: {config.phase.value}")
    print(f"  ├── Data Source: {config.data.source_type}")
    print(f"  ├── Database Path: {config.data.source_path}")
    print(f"  ├── CV Strategy: {config.data.cv_strategy}")
    print(f"  ├── CV Folds: {config.data.cv_folds}")
    print(f"  └── Group Column: {config.data.group_column}")
    print()
    
    # Configuration validation
    warnings = config.validate_configuration()
    if warnings:
        print("⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"  • {warning}")
        print()
    
    # Phase-specific processing messages
    if config.phase.value == "baseline":
        print("🎯 Baseline Phase Bronze Processing:")
        print("  ├── Focus: Data quality assurance and standardization")
        print("  ├── Priority: Participant grouping for GroupKFold")
        print("  └── Goal: Prepare clean foundation for LightGBM training")
        
    elif config.phase.value == "optimization":
        print("🎯 Optimization Phase Bronze Processing:")
        print("  ├── Focus: Enhanced data quality with optimized preprocessing")
        print("  ├── Priority: Memory optimization for larger feature sets")
        print("  └── Goal: Support advanced feature engineering")
        
    elif config.phase.value == "ensemble":
        print("🎯 Ensemble Phase Bronze Processing:")
        print("  ├── Focus: Multi-algorithm data preparation")
        print("  ├── Priority: Consistent preprocessing across models")
        print("  └── Goal: Support diverse model requirements")
    
    print()
    
    # Load raw data with configuration
    print("📊 Loading raw CMI sensor data...")
    try:
        train_df, test_df = load_data()
        print(f"  ✓ Train data: {train_df.shape}")
        print(f"  ✓ Test data: {test_df.shape}")
        
        # Data quality validation
        print("\n🔍 Data Quality Validation...")
        validation_results = validate_data_quality_cmi(train_df)
        
        print(f"  ✓ Quality metrics:")
        quality_metrics = validation_results.get('quality_metrics', {})
        for metric, value in quality_metrics.items():
            print(f"    • {metric}: {value}")
        
    except Exception as e:
        print(f"  ❌ Error loading data: {e}")
        return
    
    # Create bronze tables with configuration
    print("\n🏗️  Creating bronze layer tables...")
    try:
        create_bronze_tables()
        print("  ✓ Bronze tables created successfully")
        
        # Configuration-driven summary
        print(f"\n✨ Bronze Layer Processing Complete!")
        print(f"  ├── Phase: {config.phase.value}")
        print(f"  ├── Data processed: {train_df.shape[0] + test_df.shape[0]:,} rows")
        print(f"  ├── Participants: {train_df['subject'].nunique() if 'subject' in train_df.columns else 'N/A'}")
        print(f"  └── Ready for: Silver layer feature engineering")
        
        # Next steps based on phase
        if config.phase.value == "baseline":
            print(f"\n🎯 Next Steps (Baseline Phase):")
            print(f"  1. Run silver layer: python scripts/data_processing/silver_layer_config.py")
            print(f"  2. Target CV score: {config.targets.cv_score}")
            print(f"  3. Focus: Single algorithm mastery")
        
    except Exception as e:
        print(f"  ❌ Error creating bronze tables: {e}")
        return

if __name__ == "__main__":
    main()