
# CMI Competition - Data Quality Report

Generated on: 2025-07-20 20:17:20

## Summary
Total issues detected: 1
- Critical: 0
- High: 1
- Medium: 0

## Critical Issues (Require Immediate Attention)

## High Priority Issues
- **columns_missing_in_test**: Columns in train but not test: ['Personality']

## Medium Priority Issues

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
