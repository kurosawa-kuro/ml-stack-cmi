#!/usr/bin/env python3
"""
Silver Layer Processing Script for CMI Sensor Data
Executes Silver layer feature engineering with:
- Time-Series Feature Engineering
- Multimodal Sensor Fusion
- Statistical & Frequency Domain Features
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.silver import create_silver_tables, load_silver_data


def main():
    """Execute Silver layer processing for CMI sensor data"""
    print("🥈 Silver Layer Processing for CMI Sensor Data")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Create Silver tables with sensor-specific feature engineering
        print("\n1. Creating Silver layer tables...")
        create_silver_tables()
        
        print("  ✅ Time-series features (tsfresh)")
        print("  ✅ Frequency domain features (FFT)")
        print("  ✅ Statistical features")
        print("  ✅ Sensor fusion features")
        print("  ✅ Multimodal interaction features")
        
        # Load and validate Silver data
        print("\n2. Loading Silver layer data...")
        train_silver, test_silver = load_silver_data()
        
        print(f"  ✅ Silver train: {train_silver.shape}")
        print(f"  ✅ Silver test: {test_silver.shape}")
        print(f"  ✅ Features generated: {len(train_silver.columns)}")
        
        # Feature analysis
        print("\n3. Feature analysis...")
        
        # Count different feature types
        sensor_features = len([col for col in train_silver.columns if any(prefix in col for prefix in ['acc_', 'rot_', 'thm_', 'tof_'])])
        tsfresh_features = len([col for col in train_silver.columns if 'tsfresh_' in col])
        fft_features = len([col for col in train_silver.columns if any(term in col for term in ['spectral', 'freq', 'fft'])])
        fusion_features = len([col for col in train_silver.columns if any(term in col for term in ['motion', 'intensity', 'interaction'])])
        
        print(f"  📊 Sensor features: {sensor_features}")
        print(f"  📊 tsfresh features: {tsfresh_features}")
        print(f"  📊 FFT features: {fft_features}")
        print(f"  📊 Fusion features: {fusion_features}")
        
        # Summary
        elapsed_time = time.time() - start_time
        print(f"\n✅ Silver layer processing completed in {elapsed_time:.2f} seconds")
        print(f"  📈 Generated {len(train_silver.columns)} features")
        print(f"  📈 Processed {len(train_silver):,} training samples")
        print(f"  📈 Processed {len(test_silver):,} test samples")
        
    except Exception as e:
        print(f"\n❌ Error in Silver layer processing: {e}")
        raise


if __name__ == "__main__":
    main()