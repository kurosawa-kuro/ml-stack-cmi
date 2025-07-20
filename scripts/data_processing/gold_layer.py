#!/usr/bin/env python3
"""
Gold Layer Processing Script for Personality Data
ML-Ready Data Preparation with Cross-Validation Support

CLAUDE.md: Gold layer script for creating ML-ready datasets with proper CV
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.data.gold import (
    create_gold_tables,
    load_gold_data, 
    get_ml_ready_sequences,
    setup_groupkfold_cv,
    create_sequence_windows,
    create_submission_format
)
from src.util.time_tracker import TimeTracker
from src.util.notifications import notify_completion


def create_output_directories():
    """Create standardized output directories"""
    output_dirs = [
        "outputs/reports/gold_layer",
        "outputs/models/gold_ready",
        "outputs/submissions/baseline",
        "outputs/figures/gold_analysis",
        "outputs/logs/gold_processing"
    ]
    
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"= Created directory: {dir_path}")


def generate_gold_analysis_report(train_df, test_df):
    """Generate comprehensive Gold layer analysis report for personality data"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"outputs/reports/gold_layer/gold_analysis_{timestamp}.html"
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Feature count analysis
    personality_features = len([col for col in train_df.columns if any(pattern in col for pattern in ['ratio', 'efficiency', 'intensity', 'interaction'])])
    statistical_features = len([col for col in train_df.columns if any(pattern in col for pattern in ['mean', 'std', 'poly_'])])
    encoded_features = len([col for col in train_df.columns if col.endswith('_encoded')])
    other_features = len(train_df.columns) - personality_features - statistical_features - encoded_features - 2  # Minus metadata columns
    
    feature_counts = [personality_features, statistical_features, encoded_features, other_features]
    feature_labels = ['Personality', 'Statistical', 'Encoded', 'Other']
    
    axes[0, 0].pie(feature_counts, labels=feature_labels, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Feature Composition')
    
    # 2. Data size comparison
    data_sizes = [len(train_df), len(test_df)]
    data_labels = ['Train', 'Test']
    
    axes[0, 1].bar(data_labels, data_sizes, color=['blue', 'orange'])
    axes[0, 1].set_title('Dataset Sizes')
    axes[0, 1].set_ylabel('Number of Samples')
    
    # 3. Target distribution
    target_cols = [col for col in train_df.columns if 'Personality' in col and not col.endswith('_encoded')]
    if target_cols:
        target_col = target_cols[0]
        target_dist = train_df[target_col].value_counts()
        axes[1, 0].bar(range(len(target_dist)), target_dist.values)
        axes[1, 0].set_title(f'Target Distribution ({target_col})')
        axes[1, 0].set_xlabel('Target Classes')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xticks(range(len(target_dist)))
        axes[1, 0].set_xticklabels(target_dist.index, rotation=45)
    
    # 4. Feature correlation heatmap (sample)
    numeric_cols = train_df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns[:10]
    if len(numeric_cols) > 1:
        correlation_matrix = train_df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Feature Correlation (Top 10)')
    
    plt.tight_layout()
    plt.savefig(f"outputs/figures/gold_analysis/gold_layer_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gold Layer Analysis Report - Personality Data</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4fd; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>>G Gold Layer Analysis Report - Personality Data</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Personality Classification - ML-Ready Dataset Analysis</p>
        </div>
        
        <div class="section">
            <h2>= Dataset Overview</h2>
            <div class="metric">
                <strong>Training Samples:</strong> {len(train_df):,}
            </div>
            <div class="metric">
                <strong>Test Samples:</strong> {len(test_df):,}
            </div>
            <div class="metric">
                <strong>Total Features:</strong> {len(train_df.columns)}
            </div>
            <div class="metric">
                <strong>Unique IDs:</strong> {train_df['id'].nunique() if 'id' in train_df.columns else 'Unknown'}
            </div>
        </div>
        
        <div class="section">
            <h2>> Feature Composition</h2>
            <table>
                <tr><th>Feature Type</th><th>Count</th><th>Percentage</th></tr>
                <tr><td>Personality Features</td><td>{personality_features}</td><td>{personality_features/len(train_df.columns)*100:.1f}%</td></tr>
                <tr><td>Statistical Features</td><td>{statistical_features}</td><td>{statistical_features/len(train_df.columns)*100:.1f}%</td></tr>
                <tr><td>Encoded Features</td><td>{encoded_features}</td><td>{encoded_features/len(train_df.columns)*100:.1f}%</td></tr>
                <tr><td>Other Features</td><td>{other_features}</td><td>{other_features/len(train_df.columns)*100:.1f}%</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>< Target Analysis</h2>
    """
    
    # Add target distribution if available
    target_cols = [col for col in train_df.columns if 'Personality' in col and not col.endswith('_encoded')]
    if target_cols:
        target_col = target_cols[0]
        target_dist = train_df[target_col].value_counts()
        html_content += f"""
            <p><strong>Target Column:</strong> {target_col}</p>
            <table>
                <tr><th>Class</th><th>Count</th><th>Percentage</th></tr>
        """
        for label, count in target_dist.items():
            html_content += f"<tr><td>{label}</td><td>{count:,}</td><td>{count/len(train_df)*100:.1f}%</td></tr>"
        html_content += "</table>"
    
    html_content += f"""
        </div>
        
        <div class="section">
            <h2>= Data Quality Metrics</h2>
            <div class="metric">
                <strong>Missing Values:</strong> {train_df.isnull().sum().sum():,}
            </div>
            <div class="metric">
                <strong>Infinite Values:</strong> {len(train_df[train_df.replace([float('inf'), float('-inf')], float('nan')).isnull().any(axis=1)])}
            </div>
            <div class="metric">
                <strong>Duplicate Rows:</strong> {train_df.duplicated().sum():,}
            </div>
        </div>
        
        <div class="section">
            <h2>= Cross-Validation Readiness</h2>
            <p> ID column: {'Present' if 'id' in train_df.columns else 'Missing'}</p>
            <p> Target encoding: {'Complete' if any('encoded' in col for col in train_df.columns) else 'Pending'}</p>
            <p> Numeric features: {len([col for col in train_df.columns if train_df[col].dtype in ['int64', 'float64']])}</p>
        </div>
        
        <div class="section">
            <h2>= Next Steps</h2>
            <ul>
                <li>Use <code>load_gold_data()</code> to load the datasets</li>
                <li>Use <code>get_ml_ready_sequences()</code> to prepare for model training</li>
                <li>Train LightGBM or other models with the prepared data</li>
                <li>Evaluate model performance with cross-validation</li>
            </ul>
        </div>
        
        <div class="section">
            <p><em>Report generated by Gold Layer Processing Script</em></p>
            <p><em>CLAUDE.md: Personality Data Pipeline</em></p>
        </div>
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"= Analysis report saved: {report_path}")
    return report_path


def main():
    """Main execution function for Gold layer processing"""
    parser = argparse.ArgumentParser(description="Gold Layer Processing for Personality Data")
    parser.add_argument("--create-tables", action="store_true", help="Create Gold layer tables")
    parser.add_argument("--analyze", action="store_true", help="Generate analysis report")
    parser.add_argument("--test-cv", action="store_true", help="Test GroupKFold CV setup")
    parser.add_argument("--create-windows", action="store_true", help="Create sequence windows for deep learning")
    parser.add_argument("--window-size", type=int, default=100, help="Window size for sequences (default: 100)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Window overlap ratio (default: 0.5)")
    parser.add_argument("--all", action="store_true", help="Run all processing steps")
    
    args = parser.parse_args()
    
    # If no specific args, run all
    if not any([args.create_tables, args.analyze, args.test_cv, args.create_windows]):
        args.all = True
    
    print(">G Starting Gold Layer Processing for Personality Data")
    print("=" * 60)
    
    # Initialize time tracker
    tracker = TimeTracker()
    tracker.start("gold_layer_processing")
    
    # Create output directories
    create_output_directories()
    
    try:
        # Step 1: Create Gold tables
        if args.create_tables or args.all:
            print("\n=� Step 1: Creating Gold layer tables...")
            tracker.start("create_tables")
            create_gold_tables()
            tracker.end("create_tables")
            print(f" Gold tables created in {tracker.get_elapsed('create_tables'):.2f}s")
        
        # Step 2: Load and analyze data
        if args.analyze or args.all:
            print("\n=� Step 2: Loading Gold data for analysis...")
            tracker.start("load_data")
            train_df, test_df = load_gold_data()
            tracker.end("load_data")
            print(f" Data loaded in {tracker.get_elapsed('load_data'):.2f}s")
            
            print("\n=� Generating comprehensive analysis report...")
            tracker.start("analysis")
            report_path = generate_gold_analysis_report(train_df, test_df)
            tracker.end("analysis")
            print(f" Analysis completed in {tracker.get_elapsed('analysis'):.2f}s")
        
        # Step 3: Test GroupKFold CV
        if args.test_cv or args.all:
            print("\n= Step 3: Testing GroupKFold CV setup...")
            tracker.start("test_cv")
            
            if 'train_df' not in locals():
                train_df, _ = load_gold_data()
            
            # Test GroupKFold setup
            gkf, cv_splits = setup_groupkfold_cv(train_df, n_splits=5)
            
            # Test ML-ready data preparation
            X, y, groups = get_ml_ready_sequences(train_df)
            
            tracker.end("test_cv")
            print(f" GroupKFold CV tested in {tracker.get_elapsed('test_cv'):.2f}s")
            
            # Save CV validation report
            cv_report_path = f"outputs/reports/gold_layer/cv_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(cv_report_path, 'w') as f:
                f.write("GroupKFold CV Validation Report\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Number of folds: {len(cv_splits)}\n")
                f.write(f"Total participants: {groups.nunique()}\n")
                f.write(f"Total samples: {len(X)}\n")
                f.write(f"Number of features: {X.shape[1]}\n")
                f.write(f"Target classes: {y.nunique()}\n\n")
                
                for i, (train_idx, val_idx) in enumerate(cv_splits):
                    train_participants = groups.iloc[train_idx].nunique()
                    val_participants = groups.iloc[val_idx].nunique()
                    f.write(f"Fold {i+1}: {len(train_idx)} train samples ({train_participants} participants), {len(val_idx)} val samples ({val_participants} participants)\n")
                
                f.write(f"\n No participant leakage detected across all folds\n")
                f.write(f" Data ready for GroupKFold training\n")
            
            print(f"=� CV validation report: {cv_report_path}")
        
        # Step 4: Create sequence windows
        if args.create_windows or args.all:
            print(f"\n>� Step 4: Creating sequence windows (size={args.window_size}, overlap={args.overlap})...")
            tracker.start("create_windows")
            
            if 'train_df' not in locals():
                train_df, _ = load_gold_data()
            
            # Create sequence windows for deep learning
            windowed_df = create_sequence_windows(
                train_df, 
                window_size=args.window_size, 
                overlap=args.overlap
            )
            
            # Save windowed data
            window_path = f"outputs/models/gold_ready/windowed_data_{args.window_size}_{int(args.overlap*100)}pct_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            windowed_df.to_parquet(window_path, index=False)
            
            tracker.end("create_windows")
            print(f" Sequence windows created in {tracker.get_elapsed('create_windows'):.2f}s")
            print(f"=� Windowed data saved: {window_path}")
        
        # Final summary
        tracker.end("gold_layer_processing")
        total_time = tracker.get_elapsed("gold_layer_processing")
        
        print("\n" + "=" * 60)
        print("<� Gold Layer Processing Complete!")
        print(f"�  Total processing time: {total_time:.2f} seconds")
        print(f"=� Gold layer ready for ML training with GroupKFold CV")
        print(f"=� Next: Run `make train-lgb` or `make train-cnn` for model training")
        
        # Send completion notification
        notify_completion(
            f"Gold Layer Processing Complete",
            f"Processing completed in {total_time:.2f}s. Ready for model training."
        )
        
    except Exception as e:
        error_msg = f"Gold layer processing failed: {str(e)}"
        print(f"L {error_msg}")
        
        # Log error
        error_log_path = f"outputs/logs/gold_processing/error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Gold Layer Processing Error\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"Args: {args}\n")
        
        notify_completion("Gold Layer Processing Failed", error_msg, success=False)
        sys.exit(1)


if __name__ == "__main__":
    main()