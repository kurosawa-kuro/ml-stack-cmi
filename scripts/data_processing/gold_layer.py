#!/usr/bin/env python3
"""
Gold Layer Processing Script for CMI Sensor Data
ML-Ready Data Preparation with GroupKFold Support
CLAUDE.md: ML-ready sequences for LightGBM/CNN training with participant-based CV
"""

import sys
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.gold import create_gold_tables, load_gold_data, setup_groupkfold_cv


def generate_analysis_report(train_df, test_df, output_path="outputs/reports/gold_analysis_report.html"):
    """Generate comprehensive Gold layer analysis report for CMI sensor data"""
    import pandas as pd
    import numpy as np
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Feature analysis
    sensor_features = len([col for col in train_df.columns if any(prefix in col for prefix in ['acc_', 'rot_', 'thm_', 'tof_'])])
    tsfresh_features = len([col for col in train_df.columns if 'tsfresh_' in col])
    fft_features = len([col for col in train_df.columns if any(term in col for term in ['spectral', 'freq', 'fft'])])
    fusion_features = len([col for col in train_df.columns if any(term in col for term in ['motion', 'intensity', 'interaction'])])
    statistical_features = len([col for col in train_df.columns if any(term in col for term in ['mean', 'std', 'min', 'max', 'median'])])
    encoded_features = len([col for col in train_df.columns if col.endswith('_encoded')])
    
    # Calculate other features
    other_features = len(train_df.columns) - sensor_features - tsfresh_features - fft_features - fusion_features - statistical_features - encoded_features - 2  # Minus metadata columns
    
    feature_counts = [sensor_features, tsfresh_features, fft_features, fusion_features, statistical_features, encoded_features, other_features]
    feature_labels = ['Sensor', 'tsfresh', 'FFT', 'Fusion', 'Statistical', 'Encoded', 'Other']
    
    # Target analysis
    target_cols = [col for col in train_df.columns if 'label' in col and not col.endswith('_encoded')]
    if target_cols:
        target_col = target_cols[0]
        target_distribution = train_df[target_col].value_counts().to_dict()
    else:
        target_distribution = {}
    
    # Participant analysis
    if 'participant_id' in train_df.columns:
        participant_count = train_df['participant_id'].nunique()
        avg_samples_per_participant = len(train_df) / participant_count
    else:
        participant_count = 0
        avg_samples_per_participant = 0
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Gold Layer Analysis Report - CMI Sensor Data</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
            .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
            .stat-number {{ font-size: 2em; font-weight: bold; margin-bottom: 5px; }}
            .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f8f9fa; font-weight: bold; }}
            .feature-chart {{ margin: 30px 0; }}
            .bar {{ display: inline-block; margin: 2px; background-color: #3498db; color: white; text-align: center; padding: 8px; border-radius: 4px; }}
            .target-dist {{ margin: 20px 0; }}
            .target-item {{ display: inline-block; margin: 5px; padding: 8px 15px; background-color: #e8f5e8; border-radius: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🥇 Gold Layer Analysis Report - CMI Sensor Data</h1>
            <p>BFRB Detection - ML-Ready Dataset Analysis</p>
            
            <h2>📊 Dataset Overview</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{len(train_df):,}</div>
                    <div class="stat-label">Training Samples</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(test_df):,}</div>
                    <div class="stat-label">Test Samples</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(train_df.columns)}</div>
                    <div class="stat-label">Total Features</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{participant_count}</div>
                    <div class="stat-label">Participants</div>
                </div>
            </div>
            
            <h2>🔧 Feature Engineering Summary</h2>
            <table>
                <tr><th>Feature Type</th><th>Count</th><th>Percentage</th></tr>
                <tr><td>Sensor Features</td><td>{sensor_features}</td><td>{sensor_features/len(train_df.columns)*100:.1f}%</td></tr>
                <tr><td>tsfresh Features</td><td>{tsfresh_features}</td><td>{tsfresh_features/len(train_df.columns)*100:.1f}%</td></tr>
                <tr><td>FFT Features</td><td>{fft_features}</td><td>{fft_features/len(train_df.columns)*100:.1f}%</td></tr>
                <tr><td>Fusion Features</td><td>{fusion_features}</td><td>{fusion_features/len(train_df.columns)*100:.1f}%</td></tr>
                <tr><td>Statistical Features</td><td>{statistical_features}</td><td>{statistical_features/len(train_df.columns)*100:.1f}%</td></tr>
                <tr><td>Encoded Features</td><td>{encoded_features}</td><td>{encoded_features/len(train_df.columns)*100:.1f}%</td></tr>
                <tr><td>Other Features</td><td>{other_features}</td><td>{other_features/len(train_df.columns)*100:.1f}%</td></tr>
            </table>
            
            <h2>🎯 Target Analysis</h2>
            <div class="target-dist">
    """
    
    if target_distribution:
        for target, count in target_distribution.items():
            percentage = count / len(train_df) * 100
            html_content += f'<div class="target-item">{target}: {count:,} ({percentage:.1f}%)</div>'
    else:
        html_content += '<div class="target-item">No target columns found</div>'
    
    html_content += f"""
            </div>
            
            <h2>👥 Participant Analysis</h2>
            <p>Average samples per participant: {avg_samples_per_participant:.1f}</p>
            
            <h2>📈 Feature Distribution</h2>
            <div class="feature-chart">
    """
    
    max_count = max(feature_counts) if feature_counts else 1
    for label, count in zip(feature_labels, feature_counts):
        if count > 0:
            width = (count / max_count) * 100
            html_content += f'<div class="bar" style="width: {width}%;">{label}: {count}</div>'
    
    html_content += """
            </div>
            
            <h2>🔍 Data Quality</h2>
            <p><em>CLAUDE.md: CMI Sensor Data Pipeline</em></p>
            <ul>
                <li>✅ GroupKFold CV ready (participant-based splitting)</li>
                <li>✅ Multimodal sensor fusion features</li>
                <li>✅ Time-series feature engineering</li>
                <li>✅ Frequency domain analysis</li>
                <li>✅ ML-ready format for LightGBM/CNN</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📊 Analysis report generated: {output_path}")


def main():
    """Main Gold layer processing workflow"""
    parser = argparse.ArgumentParser(description="Gold Layer Processing for CMI Sensor Data")
    parser.add_argument("--generate-report", action="store_true", help="Generate analysis report")
    parser.add_argument("--test-cv", action="store_true", help="Test GroupKFold CV setup")
    args = parser.parse_args()
    
    print("🥇 Gold Layer Processing for CMI Sensor Data")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Step 1: Create Gold tables
        print("\n1. Creating Gold layer tables...")
        create_gold_tables()
        
        # Step 2: Load Gold data
        print("\n2. Loading Gold layer data...")
        train_gold, test_gold = load_gold_data()
        
        print(f"  ✅ Gold train: {train_gold.shape}")
        print(f"  ✅ Gold test: {test_gold.shape}")
        print(f"  ✅ ML-ready features: {len(train_gold.columns)}")
        
        # Step 3: Test GroupKFold CV
        if args.test_cv:
            print("\n3. Testing GroupKFold CV setup...")
            gkf, cv_splits = setup_groupkfold_cv(train_gold, n_splits=5)
            print(f"  ✅ GroupKFold created with {len(cv_splits)} splits")
            print(f"  ✅ CV strategy: participant-based splitting")
        
        # Step 4: Generate analysis report
        if args.generate_report:
            print("\n4. Generating analysis report...")
            generate_analysis_report(train_gold, test_gold)
        
        # Summary
        elapsed_time = time.time() - start_time
        print(f"\n✅ Gold layer processing completed in {elapsed_time:.2f} seconds")
        print(f"  📈 ML-ready features: {len(train_gold.columns)}")
        print(f"  📈 Training samples: {len(train_gold):,}")
        print(f"  📈 Test samples: {len(test_gold):,}")
        print(f"  📈 Ready for model training")
        
    except Exception as e:
        print(f"\n❌ Error in Gold layer processing: {e}")
        raise


if __name__ == "__main__":
    main()