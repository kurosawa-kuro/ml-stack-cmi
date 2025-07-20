#!/usr/bin/env python3
"""
CMI Competition - Comprehensive Exploratory Data Analysis
=========================================================
Week 1 Day 1-2: Data Understanding and EDA

Features:
- Sensor data distribution analysis
- Missing value pattern analysis
- Label distribution and class imbalance
- Time-series visualization
- Participant-level analysis
- Generate comprehensive EDA report
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def setup_plotting():
    """Configure plotting settings"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create output directory - fixed to use project root
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "figures" / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_competition_data():
    """Load competition data for EDA"""
    print("📊 Loading competition data...")
    
    try:
        from src.data.bronze import load_data
        train_df, test_df = load_data()
        print(f"  ✓ Train data: {train_df.shape}")
        print(f"  ✓ Test data: {test_df.shape}")
        return train_df, test_df
    except Exception as e:
        print(f"  ✗ Failed to load data: {e}")
        return None, None

def analyze_data_structure(train_df, test_df, output_dir):
    """Analyze basic data structure"""
    print("\n🔍 Analyzing data structure...")
    
    # Basic info
    print(f"Train dataset shape: {train_df.shape}")
    print(f"Test dataset shape: {test_df.shape}")
    
    # Column analysis
    sensor_columns = [col for col in train_df.columns if any(x in col for x in ['acc_', 'gyro_', 'tof_', 'thermopile_'])]
    print(f"Sensor columns ({len(sensor_columns)}): {sensor_columns}")
    
    # Data types
    print("\nData types:")
    print(train_df.dtypes.value_counts())
    
    # Memory usage
    train_memory = train_df.memory_usage(deep=True).sum() / 1024**2
    test_memory = test_df.memory_usage(deep=True).sum() / 1024**2
    print(f"\nMemory usage:")
    print(f"  Train: {train_memory:.2f} MB")
    print(f"  Test: {test_memory:.2f} MB")
    
    # Save summary
    with open(output_dir / "data_structure_summary.txt", "w") as f:
        f.write(f"CMI Competition - Data Structure Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Train shape: {train_df.shape}\n")
        f.write(f"Test shape: {test_df.shape}\n")
        f.write(f"Sensor columns: {len(sensor_columns)}\n")
        f.write(f"Train memory: {train_memory:.2f} MB\n")
        f.write(f"Test memory: {test_memory:.2f} MB\n\n")
        f.write("Column details:\n")
        f.write(train_df.dtypes.to_string())

def analyze_missing_values(train_df, test_df, output_dir):
    """Analyze missing value patterns"""
    print("\n🕳️  Analyzing missing values...")
    
    # Missing value analysis
    train_missing = train_df.isnull().sum()
    test_missing = test_df.isnull().sum()
    
    missing_info = pd.DataFrame({
        'train_missing': train_missing,
        'train_missing_pct': (train_missing / len(train_df)) * 100,
        'test_missing': test_missing,
        'test_missing_pct': (test_missing / len(test_df)) * 100
    })
    
    # Filter columns with missing values
    missing_cols = missing_info[missing_info['train_missing'] > 0].index.tolist()
    
    if missing_cols:
        print(f"Columns with missing values: {missing_cols}")
        print(missing_info[missing_info['train_missing'] > 0])
        
        # Visualize missing patterns
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Train missing values
        train_missing_pct = missing_info['train_missing_pct']
        train_missing_pct = train_missing_pct[train_missing_pct > 0]
        
        if len(train_missing_pct) > 0:
            train_missing_pct.plot(kind='bar', ax=axes[0], title='Train Missing Values (%)')
            axes[0].set_ylabel('Missing Percentage')
            axes[0].tick_params(axis='x', rotation=45)
        
        # Test missing values  
        test_missing_pct = missing_info['test_missing_pct']
        test_missing_pct = test_missing_pct[test_missing_pct > 0]
        
        if len(test_missing_pct) > 0:
            test_missing_pct.plot(kind='bar', ax=axes[1], title='Test Missing Values (%)')
            axes[1].set_ylabel('Missing Percentage')
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "missing_values_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    else:
        print("  ✓ No missing values found!")
    
    # Save missing value report
    missing_info.to_csv(output_dir / "missing_values_report.csv")

def analyze_label_distribution(train_df, output_dir):
    """Analyze target label distribution"""
    print("\n🎯 Analyzing label distribution...")
    
    if 'label' not in train_df.columns:
        print("  ⚠️  No 'label' column found")
        return
    
    # Label distribution
    label_counts = train_df['label'].value_counts().sort_index()
    label_pct = train_df['label'].value_counts(normalize=True).sort_index() * 100
    
    print("Label distribution:")
    for label, count in label_counts.items():
        pct = label_pct[label]
        print(f"  Class {label}: {count:,} samples ({pct:.2f}%)")
    
    # Visualize label distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Count plot
    label_counts.plot(kind='bar', ax=axes[0], title='Label Distribution (Counts)')
    axes[0].set_ylabel('Count')
    axes[0].set_xlabel('Label')
    
    # Percentage plot
    label_pct.plot(kind='bar', ax=axes[1], title='Label Distribution (%)')
    axes[1].set_ylabel('Percentage')
    axes[1].set_xlabel('Label')
    
    plt.tight_layout()
    plt.savefig(output_dir / "label_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate class imbalance metrics
    max_class = label_counts.max()
    min_class = label_counts.min()
    imbalance_ratio = max_class / min_class
    
    print(f"\nClass imbalance analysis:")
    print(f"  Most frequent class: {max_class:,} samples")
    print(f"  Least frequent class: {min_class:,} samples") 
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # Save label analysis
    label_summary = pd.DataFrame({
        'count': label_counts,
        'percentage': label_pct,
        'imbalance_ratio': max_class / label_counts
    })
    label_summary.to_csv(output_dir / "label_distribution_summary.csv")

def analyze_sensor_distributions(train_df, output_dir):
    """Analyze sensor data distributions"""
    print("\n📡 Analyzing sensor distributions...")
    
    # Identify sensor columns
    sensor_groups = {
        'IMU_acc': [col for col in train_df.columns if 'acc_' in col],
        'IMU_gyro': [col for col in train_df.columns if 'gyro_' in col],
        'ToF': [col for col in train_df.columns if 'tof_' in col],
        'Thermopile': [col for col in train_df.columns if 'thermopile_' in col]
    }
    
    # Sample data for visualization (to manage memory)
    sample_size = min(10000, len(train_df))
    sample_df = train_df.sample(n=sample_size, random_state=42)
    
    for group_name, columns in sensor_groups.items():
        if not columns:
            continue
            
        print(f"\nAnalyzing {group_name} sensors: {columns}")
        
        # Statistical summary
        sensor_stats = sample_df[columns].describe()
        print(sensor_stats)
        
        # Distribution plots
        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3  # 3 columns per row
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(columns):
            row = i // 3
            col_idx = i % 3
            
            if n_rows > 1:
                ax = axes[row, col_idx]
            else:
                ax = axes[col_idx]
                
            sample_df[col].hist(bins=50, ax=ax, alpha=0.7)
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
        
        # Hide empty subplots
        total_plots = n_rows * 3
        for i in range(len(columns), total_plots):
            row = i // 3
            col_idx = i % 3
            if n_rows > 1:
                axes[row, col_idx].set_visible(False)
            else:
                axes[col_idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{group_name.lower()}_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save statistics
        sensor_stats.to_csv(output_dir / f"{group_name.lower()}_statistics.csv")

def analyze_participant_patterns(train_df, output_dir):
    """Analyze participant-level patterns"""
    print("\n👥 Analyzing participant patterns...")
    
    if 'participant_id' not in train_df.columns:
        print("  ⚠️  No 'participant_id' column found")
        return
    
    # Participant statistics
    participant_stats = train_df.groupby('participant_id').agg({
        'series_id': 'nunique',
        'label': ['count', lambda x: x.value_counts().to_dict()]
    }).round(2)
    
    # Flatten column names
    participant_stats.columns = ['n_sessions', 'n_samples', 'label_distribution']
    
    print(f"Number of participants: {train_df['participant_id'].nunique()}")
    print(f"Sessions per participant:")
    print(participant_stats['n_sessions'].describe())
    
    # Visualize participant data distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Sessions per participant
    participant_stats['n_sessions'].hist(bins=20, ax=axes[0,0])
    axes[0,0].set_title('Sessions per Participant')
    axes[0,0].set_xlabel('Number of Sessions')
    axes[0,0].set_ylabel('Number of Participants')
    
    # Samples per participant
    participant_stats['n_samples'].hist(bins=20, ax=axes[0,1])
    axes[0,1].set_title('Samples per Participant')
    axes[0,1].set_xlabel('Number of Samples')
    axes[0,1].set_ylabel('Number of Participants')
    
    # Data volume by participant (top 20)
    top_participants = participant_stats.nlargest(20, 'n_samples')
    top_participants['n_samples'].plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Top 20 Participants by Sample Count')
    axes[1,0].set_xlabel('Participant ID')
    axes[1,0].set_ylabel('Sample Count')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Session distribution by participant (top 20)
    top_participants['n_sessions'].plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title('Top 20 Participants by Session Count')
    axes[1,1].set_xlabel('Participant ID')
    axes[1,1].set_ylabel('Session Count')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "participant_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save participant statistics
    participant_stats.to_csv(output_dir / "participant_statistics.csv")

def create_sensor_time_series_sample(train_df, output_dir):
    """Create sample time-series visualizations"""
    print("\n⏱️  Creating time-series sample visualizations...")
    
    # Sample a few series for visualization
    if 'series_id' in train_df.columns:
        sample_series = train_df['series_id'].unique()[:5]  # First 5 series
        
        sensor_cols = [col for col in train_df.columns if any(x in col for x in ['acc_', 'gyro_', 'tof_', 'thermopile_'])]
        
        for series_id in sample_series:
            series_data = train_df[train_df['series_id'] == series_id].head(1000)  # First 1000 samples
            
            if len(series_data) < 50:  # Skip very short series
                continue
                
            # Create subplot for each sensor group
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=['Accelerometer', 'Gyroscope', 'Time-of-Flight', 'Thermopile'],
                vertical_spacing=0.08
            )
            
            # Accelerometer
            acc_cols = [col for col in sensor_cols if 'acc_' in col]
            for col in acc_cols:
                if col in series_data.columns:
                    fig.add_trace(
                        go.Scatter(y=series_data[col], name=col, line=dict(width=1)),
                        row=1, col=1
                    )
            
            # Gyroscope
            gyro_cols = [col for col in sensor_cols if 'gyro_' in col]
            for col in gyro_cols:
                if col in series_data.columns:
                    fig.add_trace(
                        go.Scatter(y=series_data[col], name=col, line=dict(width=1)),
                        row=2, col=1
                    )
            
            # Time-of-Flight
            tof_cols = [col for col in sensor_cols if 'tof_' in col]
            for col in tof_cols:
                if col in series_data.columns:
                    fig.add_trace(
                        go.Scatter(y=series_data[col], name=col, line=dict(width=1)),
                        row=3, col=1
                    )
            
            # Thermopile
            thermo_cols = [col for col in sensor_cols if 'thermopile_' in col]
            for col in thermo_cols:
                if col in series_data.columns:
                    fig.add_trace(
                        go.Scatter(y=series_data[col], name=col, line=dict(width=1)),
                        row=4, col=1
                    )
            
            fig.update_layout(
                height=800,
                title_text=f"Sensor Time Series - Series {series_id}",
                showlegend=True
            )
            
            fig.write_html(output_dir / f"time_series_sample_{series_id}.html")
            
            # Save just the first one as PNG for quick viewing
            if series_id == sample_series[0]:
                fig.write_image(output_dir / "time_series_sample.png", width=1200, height=800)

def generate_eda_report(output_dir):
    """Generate comprehensive EDA report"""
    print("\n📋 Generating EDA report...")
    
    report_content = f"""
# CMI Competition - Exploratory Data Analysis Report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

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
"""

    with open(output_dir / "EDA_REPORT.md", "w") as f:
        f.write(report_content)
    
    print(f"  ✓ EDA report saved to: {output_dir}/EDA_REPORT.md")

def main():
    """Main EDA workflow"""
    print("📊 CMI Competition - Comprehensive EDA")
    print("=" * 50)
    
    # Setup
    output_dir = setup_plotting()
    
    # Load data
    train_df, test_df = load_competition_data()
    if train_df is None:
        print("❌ Failed to load data. Please run workflow-to-setup.py first.")
        sys.exit(1)
    
    # Run analyses
    analyze_data_structure(train_df, test_df, output_dir)
    analyze_missing_values(train_df, test_df, output_dir)
    analyze_label_distribution(train_df, output_dir)
    analyze_sensor_distributions(train_df, output_dir)
    analyze_participant_patterns(train_df, output_dir)
    create_sensor_time_series_sample(train_df, output_dir)
    
    # Generate report
    generate_eda_report(output_dir)
    
    print("\n" + "=" * 50)
    print("✅ EDA completed successfully!")
    print(f"📁 Results saved to: {output_dir}")
    print("\n🚀 Next steps:")
    print("  1. Review generated plots and reports")
    print("  2. Run: python scripts/workflow-to-data-check.py")
    print("  3. Proceed with bronze layer: python scripts/workflow-to-bronze.py")

if __name__ == "__main__":
    main()