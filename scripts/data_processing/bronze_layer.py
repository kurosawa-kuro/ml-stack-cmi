#!/usr/bin/env python3
"""
Bronze Layer Data Processing Script for CMI Sensor Data
Executes bronze layer data cleaning, normalization, and quality assurance
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.bronze import (
    create_bronze_tables,
    load_data,
    validate_data_quality_cmi,
    SENSOR_COLUMNS,
    DB_PATH
)


def main():
    """Main bronze layer processing workflow"""
    print("=" * 60)
    print("CMI SENSOR DATA BRONZE LAYER PROCESSING")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Step 1: Load and validate raw data
        print("\n1. Loading raw CMI sensor data...")
        train_raw, test_raw = load_data()
        
        print(f" Train data: {train_raw.shape}")
        print(f" Test data: {test_raw.shape}")
        print(f" Database: {DB_PATH}")
        
        # Step 2: Validate raw data quality
        print("\n2. Validating raw data quality...")
        train_validation = validate_data_quality_cmi(train_raw)
        
        print(f" Total samples: {train_validation['schema_validation'].get('total_samples', 'N/A')}")
        print(f" Unique IDs: {train_validation['schema_validation'].get('unique_ids', 'N/A')}")
        print(f" Numeric features: {train_validation['quality_metrics']['numeric_features_count']}")
        print(f" Categorical features: {train_validation['quality_metrics']['categorical_features_count']}")
        print(f" Total rows: {train_validation['quality_metrics']['total_rows']:,}")
        
        # Step 3: Check data structure
        print("\n3. Analyzing data structure...")
        
        # Check sensor features
        acc_cols = [col for col in train_raw.columns if col in SENSOR_COLUMNS['accelerometer']]
        gyro_cols = [col for col in train_raw.columns if col in SENSOR_COLUMNS['gyroscope']]
        thermal_cols = [col for col in train_raw.columns if col in SENSOR_COLUMNS['thermopile']]
        tof_cols = [col for col in train_raw.columns if col in SENSOR_COLUMNS['tof_sensors']]
        
        print(f" Accelerometer features: {len(acc_cols)} ({acc_cols[:3] if acc_cols else 'None'})")
        print(f" Gyroscope features: {len(gyro_cols)} ({gyro_cols[:3] if gyro_cols else 'None'})")
        print(f" Thermopile features: {len(thermal_cols)} ({thermal_cols[:3] if thermal_cols else 'None'})")
        print(f" ToF sensor features: {len(tof_cols)} ({tof_cols[:3] if tof_cols else 'None'})")
        
        # Check metadata
        metadata_cols = [col for col in train_raw.columns if col in ['subject', 'sequence_id', 'behavior', 'gesture']]
        print(f" Metadata features: {len(metadata_cols)} ({metadata_cols})")
        
        # Step 4: Create bronze tables
        print("\n4. Creating bronze layer tables...")
        create_bronze_tables()
        
        # Step 5: Validate bronze data
        print("\n5. Validating bronze layer data...")
        from src.data.bronze import load_bronze_data
        train_bronze, test_bronze = load_bronze_data()
        
        bronze_validation = validate_data_quality_cmi(train_bronze)
        print(f" Bronze train: {train_bronze.shape}")
        print(f" Bronze test: {test_bronze.shape}")
        print(f" Bronze features: {len(train_bronze.columns)}")
        
        # Step 6: Summary
        elapsed_time = time.time() - start_time
        print(f"\n✅ Bronze layer processing completed in {elapsed_time:.2f} seconds")
        print(f" Processed {len(train_raw):,} training samples")
        print(f" Processed {len(test_raw):,} test samples")
        print(f" Generated {len(train_bronze.columns)} features")
        
    except Exception as e:
        print(f"\n❌ Error in bronze layer processing: {e}")
        raise


if __name__ == "__main__":
    main()