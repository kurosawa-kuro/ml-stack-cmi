#!/usr/bin/env python3
"""
CMI Competition - Model Evaluation and Error Analysis
=====================================================
Week 2: Comprehensive model evaluation and error analysis

Features:
- Detailed confusion matrix analysis
- Per-participant performance evaluation
- Misclassification pattern analysis
- Feature importance analysis
- Model comparison framework
- Performance visualization and reporting
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    precision_recall_curve, roc_curve, auc
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def setup_evaluation():
    """Setup evaluation environment"""
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "reports" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir, plots_dir

def load_model_predictions():
    """Load model predictions and true labels"""
    print("📊 Loading model predictions...")
    
    # Look for available model results
    models_dir = Path(__file__).parent.parent.parent / "outputs" / "models"
    
    available_models = []
    predictions_data = {}
    
    # Check for LightGBM results
    lgb_dir = models_dir / "lgb_baseline"
    if (lgb_dir / "oof_predictions.csv").exists():
        oof_df = pd.read_csv(lgb_dir / "oof_predictions.csv")
        cv_results = json.load(open(lgb_dir / "cv_results.json"))
        
        predictions_data['lgb_baseline'] = {
            'oof_predictions': oof_df,
            'cv_results': cv_results,
            'model_type': 'LightGBM'
        }
        available_models.append('lgb_baseline')
        print(f"  ✓ Found LightGBM baseline results")
    
    # Check for CNN results (if available)
    cnn_dir = models_dir / "cnn_1d"
    if (cnn_dir / "oof_predictions.csv").exists():
        oof_df = pd.read_csv(cnn_dir / "oof_predictions.csv")
        
        predictions_data['cnn_1d'] = {
            'oof_predictions': oof_df,
            'model_type': '1D CNN'
        }
        available_models.append('cnn_1d')
        print(f"  ✓ Found 1D CNN results")
    
    if not available_models:
        print("  ⚠️  No model predictions found. Please train a model first.")
        return None, None
    
    print(f"  Found {len(available_models)} model(s): {available_models}")
    return predictions_data, available_models

def evaluate_composite_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Binary classification metrics (behavior vs no behavior)
    y_true_binary = (y_true > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)
    
    binary_f1 = f1_score(y_true_binary, y_pred_binary, average='binary')
    
    # Multiclass metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Composite score (competition metric)
    composite_f1 = (binary_f1 + macro_f1) / 2
    
    # Per-class F1 scores
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    
    metrics = {
        'composite_f1': composite_f1,
        'binary_f1': binary_f1,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'per_class_f1': per_class_f1.tolist()
    }
    
    return metrics

def analyze_confusion_matrix(y_true, y_pred, model_name, plots_dir):
    """Analyze and visualize confusion matrix"""
    print(f"\n🔍 Analyzing confusion matrix for {model_name}...")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create subplot with both raw and normalized matrices
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Raw confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'{model_name} - Raw Confusion Matrix')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    # Normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', ax=axes[1])
    axes[1].set_title(f'{model_name} - Normalized Confusion Matrix')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(plots_dir / f"{model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate per-class metrics from confusion matrix
    class_metrics = []
    n_classes = cm.shape[0]
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics.append({
            'class': i,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': cm[i, :].sum()
        })
    
    return cm, cm_normalized, class_metrics

def analyze_per_participant_performance(oof_df, plots_dir):
    """Analyze performance by participant (if participant info available)"""
    print("\n👥 Analyzing per-participant performance...")
    
    # Try to load participant information
    try:
        from src.data.gold import get_ml_ready_data
        _, _, _, groups = get_ml_ready_data()
        
        # Add participant info to predictions
        oof_df['participant_id'] = groups
        
        # Calculate per-participant metrics
        participant_metrics = []
        
        for participant in oof_df['participant_id'].unique():
            participant_data = oof_df[oof_df['participant_id'] == participant]
            
            if len(participant_data) < 10:  # Skip participants with very few samples
                continue
            
            metrics = evaluate_composite_metrics(
                participant_data['true_label'], 
                participant_data['oof_prediction']
            )
            
            metrics['participant_id'] = participant
            metrics['sample_count'] = len(participant_data)
            participant_metrics.append(metrics)
        
        participant_df = pd.DataFrame(participant_metrics)
        
        # Visualize per-participant performance
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Composite F1 distribution
        participant_df['composite_f1'].hist(bins=20, ax=axes[0,0])
        axes[0,0].set_title('Distribution of Composite F1 by Participant')
        axes[0,0].set_xlabel('Composite F1 Score')
        axes[0,0].set_ylabel('Number of Participants')
        
        # Binary vs Macro F1
        axes[0,1].scatter(participant_df['binary_f1'], participant_df['macro_f1'], alpha=0.6)
        axes[0,1].set_xlabel('Binary F1')
        axes[0,1].set_ylabel('Macro F1')
        axes[0,1].set_title('Binary F1 vs Macro F1 by Participant')
        
        # Sample count vs performance
        axes[1,0].scatter(participant_df['sample_count'], participant_df['composite_f1'], alpha=0.6)
        axes[1,0].set_xlabel('Sample Count')
        axes[1,0].set_ylabel('Composite F1')
        axes[1,0].set_title('Sample Count vs Performance')
        
        # Top/bottom performers
        top_performers = participant_df.nlargest(10, 'composite_f1')['composite_f1']
        bottom_performers = participant_df.nsmallest(10, 'composite_f1')['composite_f1']
        
        axes[1,1].bar(range(10), top_performers, alpha=0.7, label='Top 10')
        axes[1,1].bar(range(10, 20), bottom_performers, alpha=0.7, label='Bottom 10')
        axes[1,1].set_title('Top 10 vs Bottom 10 Participants')
        axes[1,1].set_xlabel('Participant Rank')
        axes[1,1].set_ylabel('Composite F1')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / "per_participant_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save participant metrics
        participant_df.to_csv(plots_dir.parent / "per_participant_metrics.csv", index=False)
        
        print(f"  ✓ Analyzed {len(participant_df)} participants")
        print(f"  Best participant F1: {participant_df['composite_f1'].max():.4f}")
        print(f"  Worst participant F1: {participant_df['composite_f1'].min():.4f}")
        print(f"  Mean participant F1: {participant_df['composite_f1'].mean():.4f}")
        
        return participant_df
        
    except Exception as e:
        print(f"  ⚠️  Could not load participant information: {e}")
        return None

def identify_misclassification_patterns(oof_df, plots_dir):
    """Identify and analyze misclassification patterns"""
    print("\n🕵️ Identifying misclassification patterns...")
    
    y_true = oof_df['true_label']
    y_pred = oof_df['oof_prediction']
    
    # Find misclassified samples
    misclassified = oof_df[y_true != y_pred].copy()
    correctly_classified = oof_df[y_true == y_pred].copy()
    
    misclassification_rate = len(misclassified) / len(oof_df)
    
    print(f"  Overall misclassification rate: {misclassification_rate:.3f}")
    print(f"  Misclassified samples: {len(misclassified)}")
    print(f"  Correctly classified: {len(correctly_classified)}")
    
    # Analyze misclassification by true class
    misclass_by_true = misclassified.groupby('true_label').size()
    total_by_true = oof_df.groupby('true_label').size()
    misclass_rate_by_true = misclass_by_true / total_by_true
    
    print("\nMisclassification rate by true class:")
    for true_class in sorted(oof_df['true_label'].unique()):
        rate = misclass_rate_by_true.get(true_class, 0)
        print(f"  Class {true_class}: {rate:.3f}")
    
    # Analyze common misclassification pairs
    misclass_pairs = misclassified.groupby(['true_label', 'oof_prediction']).size()
    top_misclass_pairs = misclass_pairs.nlargest(10)
    
    print("\nTop 10 misclassification patterns:")
    for (true_class, pred_class), count in top_misclass_pairs.items():
        print(f"  {true_class} → {pred_class}: {count} samples")
    
    # Visualize misclassification patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Misclassification rate by class
    misclass_rate_by_true.plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Misclassification Rate by True Class')
    axes[0,0].set_xlabel('True Class')
    axes[0,0].set_ylabel('Misclassification Rate')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Distribution of true vs predicted for misclassified
    axes[0,1].scatter(misclassified['true_label'], misclassified['oof_prediction'], alpha=0.6)
    axes[0,1].set_xlabel('True Label')
    axes[0,1].set_ylabel('Predicted Label')
    axes[0,1].set_title('Misclassified Samples')
    
    # Correct vs incorrect prediction distribution
    class_counts = pd.DataFrame({
        'correct': correctly_classified['true_label'].value_counts(),
        'incorrect': misclassified['true_label'].value_counts()
    }).fillna(0)
    
    class_counts.plot(kind='bar', ax=axes[1,0], stacked=True)
    axes[1,0].set_title('Correct vs Incorrect Predictions by Class')
    axes[1,0].set_xlabel('Class')
    axes[1,0].set_ylabel('Count')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Misclassification heatmap
    misclass_matrix = pd.crosstab(misclassified['true_label'], misclassified['oof_prediction'])
    sns.heatmap(misclass_matrix, annot=True, fmt='d', cmap='Reds', ax=axes[1,1])
    axes[1,1].set_title('Misclassification Heatmap')
    axes[1,1].set_xlabel('Predicted Label')
    axes[1,1].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "misclassification_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed misclassification analysis
    misclass_analysis = {
        'overall_misclass_rate': misclassification_rate,
        'misclass_by_true_class': misclass_rate_by_true.to_dict(),
        'top_misclass_pairs': {f"{true}->{pred}": count for (true, pred), count in top_misclass_pairs.items()}
    }
    
    with open(plots_dir.parent / "misclassification_analysis.json", "w") as f:
        json.dump(misclass_analysis, f, indent=2)
    
    return misclass_analysis

def compare_models(predictions_data, available_models, plots_dir):
    """Compare multiple models if available"""
    if len(available_models) < 2:
        print("\n📊 Single model detected - skipping model comparison")
        return None
    
    print(f"\n📊 Comparing {len(available_models)} models...")
    
    model_comparison = []
    
    for model_name in available_models:
        oof_df = predictions_data[model_name]['oof_predictions']
        metrics = evaluate_composite_metrics(oof_df['true_label'], oof_df['oof_prediction'])
        
        metrics['model_name'] = model_name
        metrics['model_type'] = predictions_data[model_name]['model_type']
        model_comparison.append(metrics)
    
    comparison_df = pd.DataFrame(model_comparison)
    
    # Visualize model comparison
    metrics_to_plot = ['composite_f1', 'binary_f1', 'macro_f1', 'micro_f1', 'weighted_f1']
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 4))
    
    for i, metric in enumerate(metrics_to_plot):
        comparison_df.plot(x='model_name', y=metric, kind='bar', ax=axes[i])
        axes[i].set_title(f'{metric.upper()}')
        axes[i].set_xlabel('Model')
        axes[i].set_ylabel('Score')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comparison results
    comparison_df.to_csv(plots_dir.parent / "model_comparison.csv", index=False)
    
    print("Model comparison:")
    for _, row in comparison_df.iterrows():
        print(f"  {row['model_name']}: {row['composite_f1']:.4f}")
    
    return comparison_df

def generate_evaluation_report(predictions_data, available_models, output_dir):
    """Generate comprehensive evaluation report"""
    print("\n📋 Generating evaluation report...")
    
    report_content = f"""
# Model Evaluation Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Models Evaluated
"""
    
    for model_name in available_models:
        model_info = predictions_data[model_name]
        oof_df = model_info['oof_predictions']
        metrics = evaluate_composite_metrics(oof_df['true_label'], oof_df['oof_prediction'])
        
        report_content += f"""
### {model_name} ({model_info['model_type']})
- **Composite F1**: {metrics['composite_f1']:.4f}
- **Binary F1**: {metrics['binary_f1']:.4f}  
- **Macro F1**: {metrics['macro_f1']:.4f}
- **Micro F1**: {metrics['micro_f1']:.4f}
- **Weighted F1**: {metrics['weighted_f1']:.4f}
"""
    
    report_content += f"""

## Key Findings

### Performance Summary
- Best performing model: {'TBD based on comparison'}
- Competition target (Bronze): 0.60+ Composite F1
- Current best score: {'TBD'}

### Error Analysis
Please review the generated analysis files:
- `confusion_matrix.png` - Detailed confusion matrix analysis
- `misclassification_analysis.png` - Misclassification patterns
- `per_participant_analysis.png` - Participant-level performance (if available)

## Files Generated
- `model_comparison.csv` - Quantitative model comparison
- `per_participant_metrics.csv` - Per-participant performance analysis
- `misclassification_analysis.json` - Detailed error patterns
- Various visualization plots in `plots/` directory

## Recommendations

### Model Improvement
1. **Address Weak Classes**: Focus on classes with low F1 scores
2. **Reduce Misclassification**: Target common misclassification patterns
3. **Participant Variation**: Consider participant-specific features or normalization

### Next Steps
1. **Feature Engineering**: Based on error analysis insights
2. **Data Augmentation**: For underperforming behavior classes
3. **Ensemble Methods**: Combine multiple model approaches
4. **Hyperparameter Tuning**: Fine-tune based on error patterns

## Competition Context
- **Target**: Bronze medal (LB 0.60+)
- **Current Status**: {'On track' if max(metrics['composite_f1'] for model in available_models) >= 0.55 else 'Needs improvement'}
- **Phase**: Week 2 - Model refinement and deep learning integration
"""

    with open(output_dir / "EVALUATION_REPORT.md", "w") as f:
        f.write(report_content)
    
    print(f"  ✓ Evaluation report saved to: {output_dir}/EVALUATION_REPORT.md")

def main():
    """Main evaluation workflow"""
    print("📊 CMI Competition - Model Evaluation & Error Analysis")
    print("=" * 50)
    
    # Setup
    output_dir, plots_dir = setup_evaluation()
    
    # Load model predictions
    predictions_data, available_models = load_model_predictions()
    if predictions_data is None:
        print("❌ No model predictions found.")
        print("Please train a model first:")
        print("  python scripts/workflow-to-training-lgb.py")
        sys.exit(1)
    
    # Evaluate each model
    for model_name in available_models:
        print(f"\n🔍 Evaluating {model_name}...")
        
        oof_df = predictions_data[model_name]['oof_predictions']
        
        # Calculate comprehensive metrics
        metrics = evaluate_composite_metrics(oof_df['true_label'], oof_df['oof_prediction'])
        print(f"  Composite F1: {metrics['composite_f1']:.4f}")
        print(f"  Binary F1: {metrics['binary_f1']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        
        # Analyze confusion matrix
        cm, cm_norm, class_metrics = analyze_confusion_matrix(
            oof_df['true_label'], oof_df['oof_prediction'], model_name, plots_dir
        )
        
        # Identify misclassification patterns
        misclass_analysis = identify_misclassification_patterns(oof_df, plots_dir)
    
    # Per-participant analysis (for any model)
    main_oof_df = predictions_data[available_models[0]]['oof_predictions']
    participant_df = analyze_per_participant_performance(main_oof_df, plots_dir)
    
    # Compare models if multiple available
    comparison_df = compare_models(predictions_data, available_models, plots_dir)
    
    # Generate comprehensive report
    generate_evaluation_report(predictions_data, available_models, output_dir)
    
    print("\n" + "=" * 50)
    print("✅ Model evaluation completed!")
    print(f"📁 Results saved to: {output_dir}")
    
    # Determine best model
    best_model = available_models[0]  # Default to first model
    if len(available_models) > 1:
        # Find best model based on composite F1
        best_score = 0
        for model_name in available_models:
            oof_df = predictions_data[model_name]['oof_predictions']
            metrics = evaluate_composite_metrics(oof_df['true_label'], oof_df['oof_prediction'])
            if metrics['composite_f1'] > best_score:
                best_score = metrics['composite_f1']
                best_model = model_name
        
        print(f"🏆 Best model: {best_model} (F1: {best_score:.4f})")
    
    print("\n🚀 Next steps:")
    print("  1. Review EVALUATION_REPORT.md and generated plots")
    print("  2. python scripts/workflow-to-feature-importance.py  # Feature analysis")
    print("  3. python scripts/workflow-to-training-cnn.py        # Try deep learning")
    print("  4. Address identified weaknesses in error patterns")

if __name__ == "__main__":
    main()