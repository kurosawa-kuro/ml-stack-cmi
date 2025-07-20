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
    print("CMI BRONZE LAYER DATA PROCESSING")
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
        
        print(f" Unique participants: {train_validation['schema_validation'].get('unique_subjects', 'N/A')}")
        print(f" Unique sequences: {train_validation['schema_validation'].get('unique_sequences', 'N/A')}")
        print(f" Sensor columns: {train_validation['quality_metrics']['sensor_columns_count']}")
        print(f" Total rows: {train_validation['quality_metrics']['total_rows']:,}")
        
        # Step 3: Check sensor data structure
        print("\n3. Analyzing sensor data structure...")
        
        # Check accelerometer data
        acc_cols = [col for col in train_raw.columns if col in SENSOR_COLUMNS['accelerometer']]
        print(f" Accelerometer channels: {len(acc_cols)} ({acc_cols})")
        
        # Check gyroscope data  
        gyro_cols = [col for col in train_raw.columns if col in SENSOR_COLUMNS['gyroscope']]
        print(f" Gyroscope channels: {len(gyro_cols)} ({gyro_cols})")
        
        # Check thermopile data
        thm_cols = [col for col in train_raw.columns if col in SENSOR_COLUMNS['thermopile']]
        print(f" Thermopile channels: {len(thm_cols)} ({thm_cols})")
        
        # Check ToF data
        tof_cols = [col for col in train_raw.columns if col.startswith('tof_')]
        print(f" ToF sensor channels: {len(tof_cols)}")
        
        # Analyze ToF missing data patterns
        if tof_cols:
            tof_missing_rate = (train_raw[tof_cols[:5]] == -1.0).mean().mean()  # Sample first 5 ToF channels
            print(f" ToF missing rate (sample): {tof_missing_rate:.2%}")
        
        # Step 4: Create bronze tables
        print("\n4. Creating bronze layer tables...")
        create_bronze_tables()
        print(" Bronze tables created successfully")
        
        # Step 5: Validate bronze layer results
        print("\n5. Validating bronze layer output...")
        
        # Import after bronze tables are created
        from src.data.bronze import load_bronze_data
        
        train_bronze, test_bronze = load_bronze_data()
        train_bronze_validation = validate_data_quality_cmi(train_bronze)
        
        print(f" Bronze train shape: {train_bronze.shape}")
        print(f" Bronze test shape: {test_bronze.shape}")
        print(f" Features added: {train_bronze.shape[1] - train_raw.shape[1]}")
        
        # Check for participant grouping
        if 'participant_id' in train_bronze.columns:
            print(f" Participant groups created: {train_bronze['participant_id'].nunique()}")
        
        # Check for sequence features
        sequence_features = [col for col in train_bronze.columns if 'sequence' in col.lower()]
        if sequence_features:
            print(f" Sequence features: {len(sequence_features)}")
        
        # Check for missing flags
        missing_flags = [col for col in train_bronze.columns if col.endswith('_missing')]
        if missing_flags:
            print(f" Missing value flags: {len(missing_flags)}")
        
        # Step 6: Quality assurance summary
        print("\n6. Quality assurance summary...")
        
        # Data type validation
        sensor_type_validation = sum(train_bronze_validation['type_validation'].values())
        total_sensors = len(train_bronze_validation['type_validation'])
        
        if total_sensors > 0:
            print(f" Sensor type validation: {sensor_type_validation}/{total_sensors} passed")
        
        # Range validation
        range_validations = []
        for sensor_results in train_bronze_validation['range_validation'].values():
            if isinstance(sensor_results, dict):
                range_validations.extend(sensor_results.values())
        
        if range_validations:
            passed_range = sum(range_validations)
            print(f" Range validation: {passed_range}/{len(range_validations)} passed")
        
        # Memory usage
        memory_mb = train_bronze.memory_usage(deep=True).sum() / 1024 / 1024
        print(f" Memory usage: {memory_mb:.1f} MB")
        
        # Processing time
        elapsed_time = time.time() - start_time
        print(f" Processing time: {elapsed_time:.1f} seconds")
        
        print("\n" + "=" * 60)
        print("BRONZE LAYER PROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        # Return success summary
        return {
            'status': 'success',
            'train_shape': train_bronze.shape,
            'test_shape': test_bronze.shape,
            'participants': train_bronze_validation['schema_validation'].get('unique_subjects', 0),
            'features_added': train_bronze.shape[1] - train_raw.shape[1],
            'processing_time': elapsed_time
        }
        
    except Exception as e:
        print(f"\nL Error in bronze layer processing: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'error',
            'error': str(e),
            'processing_time': time.time() - start_time
        }


if __name__ == "__main__":
    result = main()
    
    # Exit with appropriate code
    if result['status'] == 'success':
        print(f"\n Bronze layer processing completed in {result['processing_time']:.1f}s")
        sys.exit(0)
    else:
        print(f"\nL Bronze layer processing failed: {result['error']}")
        sys.exit(1)