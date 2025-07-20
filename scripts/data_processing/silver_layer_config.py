#!/usr/bin/env python3
"""
CMI Competition - Configuration-Driven Silver Layer Processing
============================================================
Enhanced silver layer feature engineering with configuration control

Features:
- Configuration-driven feature engineering
- Phase-aware processing strategy
- Dynamic tsfresh/FFT control
- Baseline phase feature optimization
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
from src.data.silver import (
    extract_time_series_features,
    extract_frequency_domain_features,
    extract_tsfresh_features,
    extract_bfrb_specific_features,
    extract_advanced_statistical_features,
    extract_cross_sensor_features,
    extract_temporal_pattern_features,
    create_silver_tables,
    load_silver_data
)

def main():
    """Configuration-driven silver layer processing"""
    print("🥈 CMI Competition - Configuration-Driven Silver Layer Processing")
    print("=" * 70)
    
    # Load project configuration
    config = get_project_config()
    
    # Display current configuration
    print(f"📋 Feature Engineering Configuration:")
    print(f"  ├── Phase: {config.phase.value}")
    print(f"  ├── tsfresh Enabled: {config.data.tsfresh_enabled}")
    print(f"  ├── FFT Enabled: {config.data.fft_enabled}")
    print(f"  ├── Multimodal Fusion: {config.data.multimodal_fusion}")
    print(f"  ├── Max Features: {config.data.max_features}")
    print(f"  └── Target CV Score: {config.targets.cv_score}")
    print()
    
    # Phase-specific feature engineering strategy
    if config.phase.value == "baseline":
        print("🎯 Baseline Phase Feature Engineering:")
        print("  ├── Priority: Essential statistical features")
        print("  ├── Strategy: IMU magnitude, rolling statistics")
        print("  ├── tsfresh: Comprehensive parameters")
        print("  └── Goal: Solid feature foundation for LightGBM")
        
    elif config.phase.value == "optimization":
        print("🎯 Optimization Phase Feature Engineering:")
        print("  ├── Priority: Advanced frequency domain features")
        print("  ├── Strategy: FFT spectral analysis + enhanced tsfresh")
        print("  ├── Focus: Feature selection optimization")
        print("  └── Goal: Maximize predictive power")
        
    elif config.phase.value == "ensemble":
        print("🎯 Ensemble Phase Feature Engineering:")
        print("  ├── Priority: Diverse feature sets for different models")
        print("  ├── Strategy: Full multimodal fusion pipeline")
        print("  ├── Focus: Model-specific feature engineering")
        print("  └── Goal: Support ensemble diversity")
    
    print()
    
    # Feature engineering execution
    print("🔧 Executing Feature Engineering Pipeline...")
    
    try:
        # Load bronze data
        print("  1️⃣ Loading bronze layer data...")
        from src.data.bronze import load_bronze_data
        train_bronze, test_bronze = load_bronze_data()
        print(f"     ✓ Train: {train_bronze.shape}, Test: {test_bronze.shape}")
        
        # Enhanced time-series features (always enabled)
        print("  2️⃣ Extracting enhanced time-series features...")
        train_features = extract_time_series_features(train_bronze)
        test_features = extract_time_series_features(test_bronze)
        print(f"     ✓ Added basic statistical features: {train_features.shape[1] - train_bronze.shape[1]} new columns")
        
        # BFRB-specific features (NEW - High impact)
        print("  3️⃣ Extracting BFRB-specific features...")
        train_features = extract_bfrb_specific_features(train_features)
        test_features = extract_bfrb_specific_features(test_features)
        print(f"     ✓ Added BFRB-specific features: {train_features.shape[1] - train_bronze.shape[1]} total columns")
        
        # Advanced statistical features (NEW - Enhanced)
        print("  4️⃣ Extracting advanced statistical features...")
        train_features = extract_advanced_statistical_features(train_features)
        test_features = extract_advanced_statistical_features(test_features)
        print(f"     ✓ Added advanced statistical features: {train_features.shape[1] - train_bronze.shape[1]} total columns")
        
        # Cross-sensor features (NEW - Multimodal fusion)
        print("  5️⃣ Extracting cross-sensor features...")
        train_features = extract_cross_sensor_features(train_features)
        test_features = extract_cross_sensor_features(test_features)
        print(f"     ✓ Added cross-sensor features: {train_features.shape[1] - train_bronze.shape[1]} total columns")
        
        # Temporal pattern features (NEW - Time-series patterns)
        print("  6️⃣ Extracting temporal pattern features...")
        train_features = extract_temporal_pattern_features(train_features)
        test_features = extract_temporal_pattern_features(test_features)
        print(f"     ✓ Added temporal pattern features: {train_features.shape[1] - train_bronze.shape[1]} total columns")
        
        # FFT features (configuration-controlled)
        if config.data.fft_enabled:
            print("  7️⃣ Extracting frequency domain features (FFT)...")
            train_features = extract_frequency_domain_features(train_features)
            test_features = extract_frequency_domain_features(test_features)
            print(f"     ✓ Added FFT features: enabled by configuration")
        else:
            print("  7️⃣ Skipping FFT features (disabled by configuration)")
        
        # tsfresh features (configuration-controlled)
        if config.data.tsfresh_enabled:
            print("  8️⃣ Extracting tsfresh comprehensive features...")
            print("     ⏳ This may take several minutes...")
            train_features = extract_tsfresh_features(train_features, max_features=config.data.max_features)
            test_features = extract_tsfresh_features(test_features, max_features=config.data.max_features)
            print(f"     ✓ Added tsfresh features: max_features={config.data.max_features}")
        else:
            print("  8️⃣ Skipping tsfresh features (disabled by configuration)")
        
        # Create silver tables
        print("  9️⃣ Creating silver layer tables...")
        # This would call create_silver_tables() if implemented
        print("     ✓ Silver tables creation (placeholder)")
        
        # Final summary
        print(f"\n✨ Silver Layer Feature Engineering Complete!")
        print(f"  ├── Phase: {config.phase.value}")
        print(f"  ├── Train features: {train_features.shape}")
        print(f"  ├── Test features: {test_features.shape}")
        print(f"  ├── Total features: {train_features.shape[1]:,}")
        print(f"  └── Ready for: Gold layer ML preparation")
        
        # Phase-specific next steps
        if config.phase.value == "baseline":
            print(f"\n🎯 Next Steps (Baseline Phase):")
            print(f"  1. Run gold layer: python scripts/data_processing/gold_layer_config.py")
            print(f"  2. Train LightGBM: python scripts/training/train_lightgbm_config.py")
            print(f"  3. Target: CV {config.targets.cv_score}+ with single algorithm")
        
    except Exception as e:
        print(f"  ❌ Error in feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()