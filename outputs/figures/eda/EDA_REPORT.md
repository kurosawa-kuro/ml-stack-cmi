
# CMI Competition - Exploratory Data Analysis Report

Generated on: 2025-07-21 01:15:07

## Summary
This report contains comprehensive exploratory data analysis for the CMI Detect Behavior with Sensor Data competition.

## Files Generated
- `data_structure_summary.txt` - Basic data structure information
- `missing_values_report.csv` - Missing value analysis
- `missing_values_analysis.png` - Missing value visualization
- `label_distribution_summary.csv` - Target label analysis
- `label_distribution.png` - Label distribution plots
- `*_statistics.csv` - Sensor statistics by group
- `*_distributions.png` - Sensor distribution plots
- `participant_statistics.csv` - Participant-level analysis
- `participant_analysis.png` - Participant pattern visualization
- `time_series_sample_*.html` - Interactive time-series plots
- `time_series_sample.png` - Sample time-series visualization

## Key Findings
Please review the generated files for detailed insights:

1. **Data Quality**: Check missing_values_analysis.png for data completeness
2. **Class Balance**: Review label_distribution.png for imbalance issues
3. **Sensor Patterns**: Examine sensor distribution plots for anomalies
4. **Participant Variation**: Check participant_analysis.png for data leakage risks

## Next Steps
1. Run `python scripts/workflow-to-data-check.py` for data quality validation
2. Proceed with `python scripts/workflow-to-bronze.py` for data preprocessing
3. Review CLAUDE.md for detailed implementation guidance

## Competition Context
- Competition: CMI Detect Behavior with Sensor Data
- Target: Bronze medal (LB 0.60+)
- Timeline: 5-week implementation plan
- Key Challenge: Multimodal sensor fusion with participant leakage prevention
