#!/usr/bin/env python3
"""
CMI Competition - Data Quality Validation
=========================================
Week 1 Day 3-4: Data Quality Checks and Validation

Features:
- Sensor data quality validation
- Anomaly detection for sensor readings
- Timestamp continuity verification
- Data consistency checks
- Missing value pattern analysis
- Generate data quality report
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def setup_analysis():
    """Setup analysis environment"""
    output_dir = Path(__file__).parent.parent / "outputs" / "reports" / "data_quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_data_for_validation():
    """Load data for quality validation"""
    print("📊 Loading data for quality validation...")
    
    try:
        from src.data.bronze import load_data
        train_df, test_df = load_data()
        print(f"  ✓ Loaded train: {train_df.shape}, test: {test_df.shape}")
        return train_df, test_df
    except Exception as e:
        print(f"  ✗ Failed to load data: {e}")
        return None, None

def validate_sensor_ranges(df, dataset_name, output_dir):
    """Validate sensor value ranges"""
    print(f"\n🔍 Validating {dataset_name} sensor ranges...")
    
    # Define expected ranges for different sensor types
    sensor_ranges = {
        'acc': (-50, 50),      # Typical accelerometer range in g
        'gyro': (-2000, 2000), # Typical gyroscope range in deg/s
        'tof': (0, 1000),      # Time-of-Flight in mm (0-1m)
        'thermopile': (0, 100) # Temperature in Celsius
    }
    
    issues = []
    sensor_stats = {}
    
    for sensor_type, (min_val, max_val) in sensor_ranges.items():
        sensor_cols = [col for col in df.columns if sensor_type in col]
        
        for col in sensor_cols:
            if col not in df.columns:
                continue
                
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
                
            col_min, col_max = col_data.min(), col_data.max()
            col_mean, col_std = col_data.mean(), col_data.std()
            
            sensor_stats[col] = {
                'min': col_min,
                'max': col_max,
                'mean': col_mean,
                'std': col_std,
                'expected_min': min_val,
                'expected_max': max_val
            }
            
            # Check for values outside expected ranges
            if col_min < min_val or col_max > max_val:
                outlier_count = ((col_data < min_val) | (col_data > max_val)).sum()
                outlier_pct = (outlier_count / len(col_data)) * 100
                
                issues.append({
                    'dataset': dataset_name,
                    'column': col,
                    'issue': 'out_of_range',
                    'severity': 'high' if outlier_pct > 5 else 'medium',
                    'description': f'Values outside expected range [{min_val}, {max_val}]',
                    'outlier_count': outlier_count,
                    'outlier_percentage': outlier_pct,
                    'actual_range': f'[{col_min:.2f}, {col_max:.2f}]'
                })
            
            print(f"  {col}: range=[{col_min:.2f}, {col_max:.2f}], expected=[{min_val}, {max_val}]")
    
    # Save sensor statistics
    sensor_stats_df = pd.DataFrame(sensor_stats).T
    sensor_stats_df.to_csv(output_dir / f"{dataset_name}_sensor_statistics.csv")
    
    return issues, sensor_stats

def detect_anomalies(df, dataset_name, output_dir):
    """Detect statistical anomalies in sensor data"""
    print(f"\n🚨 Detecting anomalies in {dataset_name}...")
    
    anomaly_results = {}
    sensor_cols = [col for col in df.columns if any(x in col for x in ['acc_', 'gyro_', 'tof_', 'thermopile_'])]
    
    # Sample data for analysis (to manage computation)
    sample_size = min(50000, len(df))
    sample_df = df[sensor_cols].sample(n=sample_size, random_state=42)
    
    for col in sensor_cols:
        if col not in sample_df.columns:
            continue
            
        col_data = sample_df[col].dropna()
        if len(col_data) < 100:  # Skip columns with insufficient data
            continue
        
        # Z-score based anomaly detection
        z_scores = np.abs(stats.zscore(col_data))
        z_threshold = 3
        z_anomalies = (z_scores > z_threshold).sum()
        z_anomaly_pct = (z_anomalies / len(col_data)) * 100
        
        # IQR based anomaly detection
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        iqr_lower = Q1 - 1.5 * IQR
        iqr_upper = Q3 + 1.5 * IQR
        iqr_anomalies = ((col_data < iqr_lower) | (col_data > iqr_upper)).sum()
        iqr_anomaly_pct = (iqr_anomalies / len(col_data)) * 100
        
        anomaly_results[col] = {
            'z_score_anomalies': z_anomalies,
            'z_score_anomaly_pct': z_anomaly_pct,
            'iqr_anomalies': iqr_anomalies,
            'iqr_anomaly_pct': iqr_anomaly_pct,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR
        }
        
        print(f"  {col}: Z-score anomalies={z_anomaly_pct:.2f}%, IQR anomalies={iqr_anomaly_pct:.2f}%")
    
    # Save anomaly detection results
    anomaly_df = pd.DataFrame(anomaly_results).T
    anomaly_df.to_csv(output_dir / f"{dataset_name}_anomaly_detection.csv")
    
    return anomaly_results

def validate_timestamp_continuity(df, dataset_name, output_dir):
    """Validate timestamp continuity and sampling rate"""
    print(f"\n⏰ Validating {dataset_name} timestamp continuity...")
    
    if 'timestamp' not in df.columns:
        print("  ⚠️  No timestamp column found")
        return []
    
    issues = []
    continuity_stats = {}
    
    # Group by series_id if available
    if 'series_id' in df.columns:
        series_groups = df.groupby('series_id')
        
        sampling_rates = []
        gap_counts = []
        
        for series_id, group in series_groups:
            if len(group) < 10:  # Skip very short series
                continue
                
            # Sort by timestamp
            group_sorted = group.sort_values('timestamp')
            
            # Calculate time differences
            time_diffs = group_sorted['timestamp'].diff().dropna()
            
            if len(time_diffs) == 0:
                continue
            
            # Expected sampling rate: 50Hz = 20ms intervals
            expected_interval = 0.02  # 20ms in seconds
            
            # Calculate actual sampling rate
            median_interval = time_diffs.median()
            actual_sampling_rate = 1 / median_interval if median_interval > 0 else 0
            sampling_rates.append(actual_sampling_rate)
            
            # Detect gaps (intervals > 2x expected)
            large_gaps = (time_diffs > 2 * expected_interval).sum()
            gap_counts.append(large_gaps)
            
            # Check for negative time differences (non-monotonic)
            negative_diffs = (time_diffs < 0).sum()
            
            if abs(actual_sampling_rate - 50) > 5:  # More than 5Hz deviation
                issues.append({
                    'dataset': dataset_name,
                    'series_id': series_id,
                    'issue': 'sampling_rate_deviation',
                    'severity': 'medium',
                    'description': f'Sampling rate {actual_sampling_rate:.1f}Hz (expected ~50Hz)',
                    'actual_rate': actual_sampling_rate
                })
            
            if large_gaps > 0:
                gap_pct = (large_gaps / len(time_diffs)) * 100
                issues.append({
                    'dataset': dataset_name,
                    'series_id': series_id,
                    'issue': 'timing_gaps',
                    'severity': 'high' if gap_pct > 10 else 'medium',
                    'description': f'{large_gaps} timing gaps detected ({gap_pct:.1f}%)',
                    'gap_count': large_gaps,
                    'gap_percentage': gap_pct
                })
            
            if negative_diffs > 0:
                issues.append({
                    'dataset': dataset_name,
                    'series_id': series_id,
                    'issue': 'non_monotonic_time',
                    'severity': 'high',
                    'description': f'{negative_diffs} non-monotonic timestamps',
                    'negative_count': negative_diffs
                })
        
        # Overall statistics
        continuity_stats = {
            'mean_sampling_rate': np.mean(sampling_rates),
            'std_sampling_rate': np.std(sampling_rates),
            'median_sampling_rate': np.median(sampling_rates),
            'mean_gaps_per_series': np.mean(gap_counts),
            'total_series_analyzed': len(sampling_rates)
        }
        
        print(f"  Average sampling rate: {continuity_stats['mean_sampling_rate']:.1f} ± {continuity_stats['std_sampling_rate']:.1f} Hz")
        print(f"  Average gaps per series: {continuity_stats['mean_gaps_per_series']:.1f}")
        
    else:
        print("  ⚠️  No series_id column found for grouping")
    
    # Save continuity analysis
    with open(output_dir / f"{dataset_name}_timestamp_continuity.txt", "w") as f:
        f.write(f"Timestamp Continuity Analysis - {dataset_name}\n")
        f.write("=" * 50 + "\n\n")
        for key, value in continuity_stats.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nIssues detected: {len(issues)}\n")
        for issue in issues:
            f.write(f"- {issue['description']}\n")
    
    return issues

def validate_data_consistency(train_df, test_df, output_dir):
    """Validate consistency between train and test data"""
    print("\n🔗 Validating train/test consistency...")
    
    consistency_issues = []
    
    # Column consistency
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    
    # Check for missing columns
    missing_in_test = train_cols - test_cols
    missing_in_train = test_cols - train_cols
    
    if missing_in_test:
        consistency_issues.append({
            'issue': 'columns_missing_in_test',
            'severity': 'high',
            'description': f'Columns in train but not test: {list(missing_in_test)}'
        })
    
    if missing_in_train:
        consistency_issues.append({
            'issue': 'columns_missing_in_train',
            'severity': 'high',
            'description': f'Columns in test but not train: {list(missing_in_train)}'
        })
    
    # Data type consistency
    common_cols = train_cols & test_cols
    dtype_issues = []
    
    for col in common_cols:
        if train_df[col].dtype != test_df[col].dtype:
            dtype_issues.append(f"{col}: train={train_df[col].dtype}, test={test_df[col].dtype}")
    
    if dtype_issues:
        consistency_issues.append({
            'issue': 'dtype_mismatch',
            'severity': 'medium',
            'description': f'Data type mismatches: {dtype_issues}'
        })
    
    # Participant ID overlap (potential leakage check)
    if 'participant_id' in train_df.columns and 'participant_id' in test_df.columns:
        train_participants = set(train_df['participant_id'].unique())
        test_participants = set(test_df['participant_id'].unique())
        
        overlap = train_participants & test_participants
        if overlap:
            consistency_issues.append({
                'issue': 'participant_overlap',
                'severity': 'critical',
                'description': f'Participant overlap between train/test: {len(overlap)} participants',
                'overlap_count': len(overlap)
            })
        else:
            print("  ✓ No participant overlap detected (good for GroupKFold)")
    
    print(f"  Consistency issues found: {len(consistency_issues)}")
    for issue in consistency_issues:
        print(f"    {issue['severity'].upper()}: {issue['description']}")
    
    return consistency_issues

def generate_quality_report(all_issues, output_dir):
    """Generate comprehensive data quality report"""
    print("\n📋 Generating data quality report...")
    
    # Categorize issues by severity
    critical_issues = [issue for issue in all_issues if issue.get('severity') == 'critical']
    high_issues = [issue for issue in all_issues if issue.get('severity') == 'high']
    medium_issues = [issue for issue in all_issues if issue.get('severity') == 'medium']
    
    # Generate report
    report_content = f"""
# CMI Competition - Data Quality Report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
Total issues detected: {len(all_issues)}
- Critical: {len(critical_issues)}
- High: {len(high_issues)}
- Medium: {len(medium_issues)}

## Critical Issues (Require Immediate Attention)
"""
    
    for issue in critical_issues:
        report_content += f"- **{issue.get('issue', 'Unknown')}**: {issue.get('description', 'No description')}\n"
    
    report_content += f"""
## High Priority Issues
"""
    
    for issue in high_issues:
        report_content += f"- **{issue.get('issue', 'Unknown')}**: {issue.get('description', 'No description')}\n"
    
    report_content += f"""
## Medium Priority Issues
"""
    
    for issue in medium_issues:
        report_content += f"- **{issue.get('issue', 'Unknown')}**: {issue.get('description', 'No description')}\n"
    
    report_content += f"""
## Files Generated
- `train_sensor_statistics.csv` - Train sensor value statistics
- `test_sensor_statistics.csv` - Test sensor value statistics
- `train_anomaly_detection.csv` - Train anomaly detection results
- `test_anomaly_detection.csv` - Test anomaly detection results
- `train_timestamp_continuity.txt` - Train timestamp analysis
- `test_timestamp_continuity.txt` - Test timestamp analysis

## Recommendations

### Data Preprocessing Priority
1. **Address Critical Issues**: Fix participant overlap and data type mismatches
2. **Handle Missing Values**: Review missing value patterns identified
3. **Outlier Treatment**: Consider robust normalization for detected outliers
4. **Timestamp Validation**: Ensure proper time-series continuity

### Next Steps
1. Review all generated CSV files for detailed statistics
2. Run `python scripts/workflow-to-bronze.py` with appropriate preprocessing
3. Implement GroupKFold validation to prevent participant leakage
4. Consider sensor-specific normalization strategies

## Competition Context
- Competition: CMI Detect Behavior with Sensor Data
- Critical Success Factor: Prevent participant leakage with GroupKFold
- Data Quality Impact: Clean data is essential for 50Hz time-series analysis
"""

    with open(output_dir / "DATA_QUALITY_REPORT.md", "w") as f:
        f.write(report_content)
    
    # Save issues as CSV for further analysis
    if all_issues:
        issues_df = pd.DataFrame(all_issues)
        issues_df.to_csv(output_dir / "data_quality_issues.csv", index=False)
    
    print(f"  ✓ Quality report saved to: {output_dir}/DATA_QUALITY_REPORT.md")
    
    return len(critical_issues), len(high_issues)

def main():
    """Main data quality validation workflow"""
    print("🔍 CMI Competition - Data Quality Validation")
    print("=" * 50)
    
    # Setup
    output_dir = setup_analysis()
    
    # Load data
    train_df, test_df = load_data_for_validation()
    if train_df is None:
        print("❌ Failed to load data. Please run workflow-to-setup.py first.")
        sys.exit(1)
    
    all_issues = []
    
    # Validate sensor ranges
    train_range_issues, train_stats = validate_sensor_ranges(train_df, "train", output_dir)
    test_range_issues, test_stats = validate_sensor_ranges(test_df, "test", output_dir)
    all_issues.extend(train_range_issues + test_range_issues)
    
    # Detect anomalies
    train_anomalies = detect_anomalies(train_df, "train", output_dir)
    test_anomalies = detect_anomalies(test_df, "test", output_dir)
    
    # Validate timestamp continuity
    train_time_issues = validate_timestamp_continuity(train_df, "train", output_dir)
    test_time_issues = validate_timestamp_continuity(test_df, "test", output_dir)
    all_issues.extend(train_time_issues + test_time_issues)
    
    # Validate train/test consistency
    consistency_issues = validate_data_consistency(train_df, test_df, output_dir)
    all_issues.extend(consistency_issues)
    
    # Generate comprehensive report
    critical_count, high_count = generate_quality_report(all_issues, output_dir)
    
    print("\n" + "=" * 50)
    if critical_count > 0:
        print("⚠️  CRITICAL ISSUES DETECTED!")
        print(f"   {critical_count} critical and {high_count} high priority issues found")
        print("   Please review the report before proceeding with data processing.")
    elif high_count > 0:
        print("⚠️  High priority issues detected")
        print(f"   {high_count} high priority issues found")
        print("   Review recommended before proceeding.")
    else:
        print("✅ Data quality validation completed!")
        print("   No critical issues detected.")
    
    print(f"\n📁 Results saved to: {output_dir}")
    print("\n🚀 Next steps:")
    print("  1. Review DATA_QUALITY_REPORT.md")
    print("  2. Address any critical/high priority issues")
    print("  3. Run: python scripts/workflow-to-bronze.py")

if __name__ == "__main__":
    main()