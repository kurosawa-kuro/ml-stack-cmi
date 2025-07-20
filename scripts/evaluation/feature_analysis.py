#!/usr/bin/env python3
"""
CMI Competition - Feature Importance Analysis
=============================================
Week 3: Comprehensive feature importance and selection analysis

Features:
- Model-specific feature importance (LightGBM, CNN)
- Permutation importance analysis
- Feature correlation analysis
- Sensor-wise importance ranking
- Feature selection recommendations
- Visualization and reporting
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def setup_feature_analysis():
    """Setup feature analysis environment"""
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "reports" / "feature_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir, plots_dir

def load_trained_models_and_data():
    """Load trained models and feature data"""
    print("📊 Loading trained models and feature data...")
    
    models_dir = Path(__file__).parent.parent.parent / "outputs" / "models"
    
    # Load LightGBM models
    lgb_models = []
    lgb_dir = models_dir / "lgb_baseline"
    
    if lgb_dir.exists():
        for fold in range(1, 6):  # Assume 5 folds
            model_path = lgb_dir / f"lgb_fold_{fold}.txt"
            if model_path.exists():
                model = lgb.Booster(model_file=str(model_path))
                lgb_models.append(model)
        
        if lgb_models:
            print(f"  ✓ Loaded {len(lgb_models)} LightGBM models")
    
    # Load feature data
    try:
        from src.data.gold import get_ml_ready_data
        X_train, y_train, X_test, groups = get_ml_ready_data()
        
        print(f"  ✓ Feature data: {X_train.shape}")
        print(f"  ✓ Feature names: {len(X_train.columns)}")
        
        return lgb_models, X_train, y_train, X_test, groups
        
    except Exception as e:
        print(f"  ✗ Failed to load feature data: {e}")
        return lgb_models, None, None, None, None

def analyze_lgb_feature_importance(lgb_models, feature_names, plots_dir):
    """Analyze LightGBM feature importance"""
    print("\n🌟 Analyzing LightGBM feature importance...")
    
    if not lgb_models:
        print("  ⚠️  No LightGBM models found")
        return None
    
    # Collect feature importance from all folds
    importance_types = ['gain', 'split']
    all_importance = {imp_type: [] for imp_type in importance_types}
    
    for model in lgb_models:
        for imp_type in importance_types:
            importance = model.feature_importance(importance_type=imp_type)
            all_importance[imp_type].append(importance)
    
    # Average importance across folds
    avg_importance = {}
    std_importance = {}
    
    for imp_type in importance_types:
        importance_matrix = np.array(all_importance[imp_type])
        avg_importance[imp_type] = np.mean(importance_matrix, axis=0)
        std_importance[imp_type] = np.std(importance_matrix, axis=0)
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'gain_importance': avg_importance['gain'],
        'gain_std': std_importance['gain'],
        'split_importance': avg_importance['split'],
        'split_std': std_importance['split']
    })
    
    # Sort by gain importance
    importance_df = importance_df.sort_values('gain_importance', ascending=False)
    
    # Add feature categories
    importance_df['sensor_type'] = importance_df['feature'].apply(categorize_feature)
    
    print(f"  Top 10 features by gain importance:")
    for i, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']}: {row['gain_importance']:.2f} ± {row['gain_std']:.2f}")
    
    # Visualize feature importance
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Top 30 features by gain
    top_features = importance_df.head(30)
    
    axes[0,0].barh(range(len(top_features)), top_features['gain_importance'])
    axes[0,0].set_yticks(range(len(top_features)))
    axes[0,0].set_yticklabels(top_features['feature'], fontsize=8)
    axes[0,0].set_xlabel('Gain Importance')
    axes[0,0].set_title('Top 30 Features - Gain Importance')
    axes[0,0].invert_yaxis()
    
    # Feature importance by sensor type
    sensor_importance = importance_df.groupby('sensor_type')['gain_importance'].sum().sort_values(ascending=False)
    
    axes[0,1].bar(sensor_importance.index, sensor_importance.values)
    axes[0,1].set_xlabel('Sensor Type')
    axes[0,1].set_ylabel('Total Gain Importance')
    axes[0,1].set_title('Feature Importance by Sensor Type')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Split vs Gain importance correlation
    axes[1,0].scatter(importance_df['split_importance'], importance_df['gain_importance'], alpha=0.6)
    axes[1,0].set_xlabel('Split Importance')
    axes[1,0].set_ylabel('Gain Importance')
    axes[1,0].set_title('Split vs Gain Importance')
    
    # Feature importance distribution
    axes[1,1].hist(importance_df['gain_importance'], bins=50, alpha=0.7)
    axes[1,1].set_xlabel('Gain Importance')
    axes[1,1].set_ylabel('Number of Features')
    axes[1,1].set_title('Feature Importance Distribution')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "lgb_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_df

def categorize_feature(feature_name):
    """Categorize feature by sensor type"""
    feature_lower = feature_name.lower()
    
    if any(x in feature_lower for x in ['acc_x', 'acc_y', 'acc_z']):
        return 'accelerometer'
    elif any(x in feature_lower for x in ['gyro_x', 'gyro_y', 'gyro_z']):
        return 'gyroscope'
    elif 'tof_' in feature_lower:
        return 'time_of_flight'
    elif 'thermopile_' in feature_lower:
        return 'thermopile'
    elif any(x in feature_lower for x in ['mean', 'std', 'max', 'min']):
        return 'statistical'
    elif any(x in feature_lower for x in ['fft', 'freq', 'spectral']):
        return 'frequency'
    elif 'tsfresh' in feature_lower:
        return 'tsfresh'
    else:
        return 'other'

def analyze_permutation_importance(X_train, y_train, groups, plots_dir):
    """Analyze permutation importance using Random Forest"""
    print("\n🔀 Analyzing permutation importance...")
    
    # Sample data for faster computation
    sample_size = min(10000, len(X_train))
    sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
    
    X_sample = X_train.iloc[sample_indices]
    y_sample = y_train.iloc[sample_indices]
    
    # Train Random Forest for permutation importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_sample, y_sample)
    
    # Calculate permutation importance
    perm_importance = permutation_importance(
        rf, X_sample, y_sample, 
        n_repeats=10, 
        random_state=42,
        n_jobs=-1
    )
    
    # Create permutation importance DataFrame
    perm_df = pd.DataFrame({
        'feature': X_train.columns,
        'perm_importance_mean': perm_importance.importances_mean,
        'perm_importance_std': perm_importance.importances_std,
        'sensor_type': [categorize_feature(f) for f in X_train.columns]
    })
    
    perm_df = perm_df.sort_values('perm_importance_mean', ascending=False)
    
    print(f"  Top 10 features by permutation importance:")
    for i, row in perm_df.head(10).iterrows():
        print(f"    {row['feature']}: {row['perm_importance_mean']:.4f} ± {row['perm_importance_std']:.4f}")
    
    # Visualize permutation importance
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Top 20 features
    top_perm = perm_df.head(20)
    
    axes[0].barh(range(len(top_perm)), top_perm['perm_importance_mean'], 
                xerr=top_perm['perm_importance_std'])
    axes[0].set_yticks(range(len(top_perm)))
    axes[0].set_yticklabels(top_perm['feature'], fontsize=10)
    axes[0].set_xlabel('Permutation Importance')
    axes[0].set_title('Top 20 Features - Permutation Importance')
    axes[0].invert_yaxis()
    
    # Permutation importance by sensor type
    sensor_perm = perm_df.groupby('sensor_type')['perm_importance_mean'].sum().sort_values(ascending=False)
    
    axes[1].bar(sensor_perm.index, sensor_perm.values)
    axes[1].set_xlabel('Sensor Type')
    axes[1].set_ylabel('Total Permutation Importance')
    axes[1].set_title('Permutation Importance by Sensor Type')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "permutation_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return perm_df

def analyze_feature_correlations(X_train, plots_dir):
    """Analyze feature correlations and multicollinearity"""
    print("\n🔗 Analyzing feature correlations...")
    
    # Sample features for correlation analysis (too many features would be unreadable)
    if len(X_train.columns) > 100:
        # Select top features from various categories
        feature_sample = []
        for sensor_type in ['accelerometer', 'gyroscope', 'time_of_flight', 'thermopile']:
            sensor_features = [f for f in X_train.columns if categorize_feature(f) == sensor_type]
            feature_sample.extend(sensor_features[:10])  # Top 10 from each sensor
        
        X_corr = X_train[feature_sample[:50]]  # Limit to 50 features
    else:
        X_corr = X_train
    
    # Calculate correlation matrix
    corr_matrix = X_corr.corr()
    
    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.9:  # High correlation threshold
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    print(f"  Found {len(high_corr_pairs)} highly correlated feature pairs (>0.9)")
    
    # Visualize correlation matrix
    plt.figure(figsize=(15, 12))
    
    # Use hierarchical clustering to order features
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    
    distance_matrix = 1 - abs(corr_matrix)
    linkage_matrix = linkage(squareform(distance_matrix), method='average')
    dendro = dendrogram(linkage_matrix, labels=corr_matrix.columns, no_plot=True)
    feature_order = dendro['leaves']
    
    # Reorder correlation matrix
    ordered_corr = corr_matrix.iloc[feature_order, feature_order]
    
    sns.heatmap(ordered_corr, cmap='RdBu_r', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Matrix (Hierarchically Clustered)')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save high correlation pairs
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_df.to_csv(plots_dir.parent / "high_correlation_pairs.csv", index=False)
    
    return corr_matrix, high_corr_pairs

def compare_importance_methods(lgb_importance_df, perm_importance_df, plots_dir):
    """Compare different feature importance methods"""
    print("\n⚖️  Comparing importance methods...")
    
    if lgb_importance_df is None or perm_importance_df is None:
        print("  ⚠️  Cannot compare - missing importance data")
        return None
    
    # Merge importance DataFrames
    merged_df = pd.merge(
        lgb_importance_df[['feature', 'gain_importance', 'sensor_type']], 
        perm_importance_df[['feature', 'perm_importance_mean']], 
        on='feature',
        how='inner'
    )
    
    # Normalize importances for comparison
    merged_df['gain_importance_norm'] = merged_df['gain_importance'] / merged_df['gain_importance'].max()
    merged_df['perm_importance_norm'] = merged_df['perm_importance_mean'] / merged_df['perm_importance_mean'].max()
    
    # Calculate correlation between methods
    importance_corr = merged_df['gain_importance_norm'].corr(merged_df['perm_importance_norm'])
    
    print(f"  Correlation between LightGBM and Permutation importance: {importance_corr:.3f}")
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Scatter plot of importance methods
    axes[0].scatter(merged_df['gain_importance_norm'], merged_df['perm_importance_norm'], alpha=0.6)
    axes[0].set_xlabel('LightGBM Gain Importance (Normalized)')
    axes[0].set_ylabel('Permutation Importance (Normalized)')
    axes[0].set_title(f'Importance Method Comparison (r={importance_corr:.3f})')
    
    # Add diagonal line
    max_val = max(merged_df['gain_importance_norm'].max(), merged_df['perm_importance_norm'].max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
    
    # Top features by each method
    top_lgb = set(merged_df.nlargest(20, 'gain_importance_norm')['feature'])
    top_perm = set(merged_df.nlargest(20, 'perm_importance_norm')['feature'])
    
    overlap = len(top_lgb & top_perm)
    lgb_only = len(top_lgb - top_perm)
    perm_only = len(top_perm - top_lgb)
    
    # Venn diagram data
    categories = ['LGB Only', 'Overlap', 'Perm Only']
    sizes = [lgb_only, overlap, perm_only]
    
    axes[1].pie(sizes, labels=categories, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Top 20 Features Overlap Between Methods')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "importance_method_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create combined importance ranking
    merged_df['combined_importance'] = (merged_df['gain_importance_norm'] + merged_df['perm_importance_norm']) / 2
    merged_df = merged_df.sort_values('combined_importance', ascending=False)
    
    return merged_df

def generate_feature_selection_recommendations(combined_importance_df, high_corr_pairs, plots_dir):
    """Generate feature selection recommendations"""
    print("\n💡 Generating feature selection recommendations...")
    
    recommendations = {
        'top_features': [],
        'redundant_features': [],
        'sensor_recommendations': {},
        'selection_strategy': []
    }
    
    if combined_importance_df is not None:
        # Top features to keep
        top_n = min(50, len(combined_importance_df))
        top_features = combined_importance_df.head(top_n)['feature'].tolist()
        recommendations['top_features'] = top_features
        
        # Analyze by sensor type
        sensor_analysis = combined_importance_df.groupby('sensor_type').agg({
            'combined_importance': ['count', 'mean', 'sum']
        }).round(4)
        
        for sensor_type in sensor_analysis.index:
            count = sensor_analysis.loc[sensor_type, ('combined_importance', 'count')]
            mean_importance = sensor_analysis.loc[sensor_type, ('combined_importance', 'mean')]
            total_importance = sensor_analysis.loc[sensor_type, ('combined_importance', 'sum')]
            
            recommendations['sensor_recommendations'][sensor_type] = {
                'feature_count': int(count),
                'mean_importance': float(mean_importance),
                'total_importance': float(total_importance),
                'recommendation': 'high_priority' if mean_importance > 0.1 else 'medium_priority' if mean_importance > 0.05 else 'low_priority'
            }
    
    # Identify redundant features from high correlations
    if high_corr_pairs:
        redundant_candidates = set()
        for pair in high_corr_pairs:
            # Keep the feature with higher importance (if available)
            if combined_importance_df is not None:
                feat1_importance = combined_importance_df[combined_importance_df['feature'] == pair['feature1']]['combined_importance'].iloc[0] if not combined_importance_df[combined_importance_df['feature'] == pair['feature1']].empty else 0
                feat2_importance = combined_importance_df[combined_importance_df['feature'] == pair['feature2']]['combined_importance'].iloc[0] if not combined_importance_df[combined_importance_df['feature'] == pair['feature2']].empty else 0
                
                if feat1_importance > feat2_importance:
                    redundant_candidates.add(pair['feature2'])
                else:
                    redundant_candidates.add(pair['feature1'])
            else:
                redundant_candidates.add(pair['feature2'])  # Default: remove second feature
        
        recommendations['redundant_features'] = list(redundant_candidates)
    
    # Generate selection strategies
    recommendations['selection_strategy'] = [
        f"Keep top {len(recommendations['top_features'])} features by combined importance",
        f"Remove {len(recommendations['redundant_features'])} highly correlated features",
        "Focus on high-priority sensor types for feature engineering",
        "Consider dimensionality reduction for remaining features"
    ]
    
    print(f"  ✓ Top features to keep: {len(recommendations['top_features'])}")
    print(f"  ✓ Redundant features to remove: {len(recommendations['redundant_features'])}")
    print(f"  ✓ Sensor type analysis: {len(recommendations['sensor_recommendations'])} types")
    
    # Save recommendations
    with open(plots_dir.parent / "feature_selection_recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=2)
    
    return recommendations

def generate_feature_analysis_report(output_dir):
    """Generate comprehensive feature analysis report"""
    print("\n📋 Generating feature analysis report...")
    
    report_content = f"""
# Feature Importance Analysis Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Analysis Overview
This report provides comprehensive analysis of feature importance across different methods and sensors.

## Key Findings

### Feature Importance Methods
1. **LightGBM Gain Importance**: Based on information gain from tree splits
2. **Permutation Importance**: Based on performance drop when features are shuffled
3. **Combined Ranking**: Normalized average of both methods

### Sensor Analysis
Review the sensor-wise importance analysis to understand which sensor modalities contribute most to behavior detection.

## Files Generated
- `lgb_feature_importance.png` - LightGBM feature importance visualization
- `permutation_importance.png` - Permutation importance analysis
- `feature_correlations.png` - Feature correlation heatmap
- `importance_method_comparison.png` - Comparison between importance methods
- `high_correlation_pairs.csv` - Highly correlated feature pairs
- `feature_selection_recommendations.json` - Automated feature selection suggestions

## Recommendations

### Feature Selection Strategy
1. **Remove Redundant Features**: Eliminate highly correlated features (>0.9 correlation)
2. **Focus on Top Features**: Prioritize features with high combined importance scores
3. **Sensor-Specific Analysis**: Consider sensor-wise feature engineering based on importance rankings
4. **Dimensionality Reduction**: Apply PCA or similar techniques to remaining feature sets

### Model Improvement
1. **Feature Engineering**: Create new features based on important sensor combinations
2. **Interaction Features**: Explore interactions between high-importance features from different sensors
3. **Temporal Features**: Consider time-lagged versions of important features
4. **Ensemble Strategy**: Use different feature subsets for different models in ensemble

## Competition Context
- **Target**: Bronze medal (LB 0.60+)
- **Feature Strategy**: Balance between informativeness and computational efficiency
- **Next Phase**: Apply feature selection to improve model performance
"""

    with open(output_dir / "FEATURE_ANALYSIS_REPORT.md", "w") as f:
        f.write(report_content)
    
    print(f"  ✓ Feature analysis report saved to: {output_dir}/FEATURE_ANALYSIS_REPORT.md")

def main():
    """Main feature importance analysis workflow"""
    print("🌟 CMI Competition - Feature Importance Analysis")
    print("=" * 50)
    
    # Setup
    output_dir, plots_dir = setup_feature_analysis()
    
    # Load models and data
    lgb_models, X_train, y_train, X_test, groups = load_trained_models_and_data()
    
    if X_train is None:
        print("❌ Failed to load feature data.")
        print("Please ensure you have trained models and data:")
        print("  1. python scripts/workflow-to-training-lgb.py")
        print("  2. Ensure gold layer data is available")
        sys.exit(1)
    
    feature_names = X_train.columns.tolist()
    
    # Analyze LightGBM feature importance
    lgb_importance_df = None
    if lgb_models:
        lgb_importance_df = analyze_lgb_feature_importance(lgb_models, feature_names, plots_dir)
        
        # Save LightGBM importance
        lgb_importance_df.to_csv(output_dir / "lgb_feature_importance.csv", index=False)
    
    # Analyze permutation importance
    perm_importance_df = analyze_permutation_importance(X_train, y_train, groups, plots_dir)
    perm_importance_df.to_csv(output_dir / "permutation_importance.csv", index=False)
    
    # Analyze feature correlations
    corr_matrix, high_corr_pairs = analyze_feature_correlations(X_train, plots_dir)
    
    # Compare importance methods
    combined_importance_df = compare_importance_methods(lgb_importance_df, perm_importance_df, plots_dir)
    if combined_importance_df is not None:
        combined_importance_df.to_csv(output_dir / "combined_feature_importance.csv", index=False)
    
    # Generate feature selection recommendations
    recommendations = generate_feature_selection_recommendations(
        combined_importance_df, high_corr_pairs, plots_dir
    )
    
    # Generate comprehensive report
    generate_feature_analysis_report(output_dir)
    
    print("\n" + "=" * 50)
    print("✅ Feature importance analysis completed!")
    print(f"📁 Results saved to: {output_dir}")
    
    # Summary insights
    if combined_importance_df is not None:
        top_sensor = combined_importance_df.groupby('sensor_type')['combined_importance'].sum().idxmax()
        print(f"🏆 Most important sensor type: {top_sensor}")
        
        print(f"📊 Feature insights:")
        print(f"  - Total features analyzed: {len(feature_names)}")
        print(f"  - Recommended to keep: {len(recommendations['top_features'])}")
        print(f"  - Recommended to remove: {len(recommendations['redundant_features'])}")
    
    print("\n🚀 Next steps:")
    print("  1. Review FEATURE_ANALYSIS_REPORT.md")
    print("  2. Apply feature selection based on recommendations")
    print("  3. python scripts/workflow-to-training-ensemble.py  # Create ensemble")
    print("  4. Retrain models with selected features")

if __name__ == "__main__":
    main()