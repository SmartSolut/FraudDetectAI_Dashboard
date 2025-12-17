"""
Evaluation Metrics & Threshold Analysis - 3.6 & 3.7
===================================================
Detailed analysis of metrics and threshold/cost analysis

Includes:
- 3.6: Detailed analysis of each metric
- 3.7: Threshold Sweep and Cost Analysis
"""

import pandas as pd
import numpy as np
import argparse
import os
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# Set style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


# ============================================================================
# 3.6 Evaluation Metrics - Detailed Analysis
# ============================================================================

def calculate_all_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate all evaluation metrics
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predictions (0 or 1)
    y_pred_proba : array-like
        Fraud probability (0-1)
    
    Returns:
    --------
    dict : All metrics
    """
    # Basic metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Advanced metrics
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'accuracy': accuracy,
        'specificity': specificity,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def detailed_metric_analysis(y_true, y_pred, y_pred_proba, model_name="Model"):
    """
    Detailed analysis of each metric
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predictions
    y_pred_proba : array-like
        Probabilities
    model_name : str
        Model name
    
    Returns:
    --------
    dict : Detailed analysis of each metric
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    analysis = {}
    
    # 1. Precision
    analysis['precision'] = {
        'value': precision,
        'meaning': 'Proportion of positive predictions that are correct',
        'formula': f'TP / (TP + FP) = {tp} / ({tp} + {fp})',
        'interpretation': f'Out of 100 alerts, {precision*100:.1f} are correct',
        'strength': 'Excellent' if precision > 0.9 else 'Good' if precision > 0.7 else 'Weak',
        'recommendation': 'Excellent - No false alerts' if precision > 0.9 else 
                         'Good - Few false alerts' if precision > 0.7 else 
                         'Weak - Many false alerts'
    }
    
    # 2. Recall
    analysis['recall'] = {
        'value': recall,
        'meaning': 'Proportion of actual fraud cases detected',
        'formula': f'TP / (TP + FN) = {tp} / ({tp} + {fn})',
        'interpretation': f'Model detects {recall*100:.1f}% of fraud cases',
        'strength': 'Excellent' if recall > 0.9 else 'Good' if recall > 0.7 else 'Weak',
        'recommendation': 'Excellent - Detects most fraud' if recall > 0.9 else 
                         'Good - Detects large portion of fraud' if recall > 0.7 else 
                         'Weak - Misses much fraud'
    }
    
    # 3. F1-Score
    analysis['f1_score'] = {
        'value': f1,
        'meaning': 'Harmonic mean of Precision and Recall',
        'formula': f'2 * (Precision * Recall) / (Precision + Recall)',
        'interpretation': f'Balance between precision and recall: {f1*100:.1f}%',
        'strength': 'Excellent' if f1 > 0.9 else 'Good' if f1 > 0.7 else 'Weak',
        'recommendation': 'Excellent - Good balance' if f1 > 0.9 else 
                         'Good - Acceptable balance' if f1 > 0.7 else 
                         'Weak - Needs improvement'
    }
    
    # 4. ROC-AUC
    analysis['roc_auc'] = {
        'value': roc_auc,
        'meaning': 'Model ability to distinguish between fraud and normal',
        'interpretation': f'Model distinguishes correctly {roc_auc*100:.1f}% of the time',
        'strength': 'Excellent' if roc_auc > 0.95 else 'Good' if roc_auc > 0.85 else 'Weak',
        'recommendation': 'Excellent - Strong discrimination' if roc_auc > 0.95 else 
                         'Good - Acceptable discrimination' if roc_auc > 0.85 else 
                         'Weak - Needs improvement'
    }
    
    # 5. PR-AUC (Most important for imbalanced data)
    analysis['pr_auc'] = {
        'value': pr_auc,
        'meaning': 'Performance metric for imbalanced data',
        'interpretation': f'Excellent fraud detection performance: {pr_auc*100:.1f}%',
        'strength': 'Excellent' if pr_auc > 0.9 else 'Good' if pr_auc > 0.7 else 'Weak',
        'note': 'This is the most important metric for imbalanced data',
        'recommendation': 'Excellent - High performance' if pr_auc > 0.9 else 
                         'Good - Acceptable performance' if pr_auc > 0.7 else 
                         'Weak - Needs improvement'
    }
    
    return analysis


def compare_models(models_dict, X_test, y_test):
    """
    Comprehensive comparison between models
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary containing models {name: model}
    X_test : DataFrame
        Test data
    y_test : Series
        True labels
    
    Returns:
    --------
    comparison_df : DataFrame
        Comparison table
    best_models : dict
        Best model for each metric
    """
    comparison = []
    
    for model_name, model in models_dict.items():
        print(f"[INFO] Evaluating {model_name}...")
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_all_metrics(y_test, y_pred, y_pred_proba)
        metrics['model'] = model_name
        
        comparison.append(metrics)
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison)
    
    # Find best model for each metric
    best_models = {}
    for metric in ['precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']:
        if metric in comparison_df.columns:
            best_idx = comparison_df[metric].idxmax()
            best_models[metric] = {
                'model': comparison_df.loc[best_idx, 'model'],
                'value': comparison_df.loc[best_idx, metric]
            }
    
    return comparison_df, best_models


def generate_evaluation_report(models_dict, X_test, y_test, output_dir):
    """
    Generate detailed evaluation report
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of models
    X_test : DataFrame
        Test data
    y_test : Series
        True labels
    output_dir : str
        Output directory
    """
    print("\n" + "="*70)
    print("Detailed Evaluation Report - 3.6")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare models
    comparison_df, best_models = compare_models(models_dict, X_test, y_test)
    
    # Save comparison table
    comparison_path = os.path.join(output_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n[OK] Comparison table saved to: {comparison_path}")
    
    # Detailed analysis for each model
    detailed_reports = {}
    
    for model_name, model in models_dict.items():
        print(f"\n[INFO] Detailed analysis for {model_name}...")
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Detailed analysis
        analysis = detailed_metric_analysis(y_test, y_pred, y_pred_proba, model_name)
        detailed_reports[model_name] = analysis
        
        # Print analysis
        print(f"\n{'='*70}")
        print(f"Analysis for {model_name}")
        print(f"{'='*70}")
        for metric, info in analysis.items():
            print(f"\n{metric.upper()}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
    
    # Save detailed reports
    import json
    reports_path = os.path.join(output_dir, 'detailed_analysis.json')
    with open(reports_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_reports, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Detailed reports saved to: {reports_path}")
    
    # Print best models
    print("\n" + "="*70)
    print("Best model for each metric:")
    print("="*70)
    for metric, info in best_models.items():
        print(f"  {metric}: {info['model']} ({info['value']:.4f})")
    
    return comparison_df, detailed_reports, best_models


# ============================================================================
# 3.7 Threshold and Cost Analysis - Threshold and Cost Analysis
# ============================================================================

def threshold_sweep(model, X_test, y_test, thresholds=None):
    """
    Test model performance on different thresholds
    
    Parameters:
    -----------
    model : classifier
        Trained model
    X_test : DataFrame
        Test data
    y_test : Series
        True labels
    thresholds : array-like, optional
        List of thresholds (default: 0.1 to 0.9 with step 0.05)
    
    Returns:
    --------
    DataFrame : Threshold sweep results
    """
    if thresholds is None:
        # Thresholds from 0.1 to 0.9 with step 0.05
        thresholds = np.arange(0.1, 0.95, 0.05)
    
    print(f"\n[INFO] Testing {len(thresholds)} thresholds...")
    
    # Get probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    results = []
    
    for threshold in thresholds:
        # Predict based on threshold
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge cases
            if cm.shape == (1, 1):
                if y_pred.sum() == 0:
                    tn, fp, fn, tp = len(y_test) - y_test.sum(), 0, y_test.sum(), 0
                else:
                    tn, fp, fn, tp = 0, 0, 0, len(y_test)
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        })
    
    return pd.DataFrame(results)


def calculate_cost_analysis(threshold_results, cost_fp, cost_fn):
    """
    Calculate total cost for each threshold
    
    Parameters:
    -----------
    threshold_results : DataFrame
        Threshold sweep results
    cost_fp : float
        Cost of False Positive (false alert)
    cost_fn : float
        Cost of False Negative (missed fraud)
    
    Returns:
    --------
    DataFrame : Results with cost
    float : Optimal threshold
    float : Optimal cost
    """
    # Calculate total cost
    threshold_results = threshold_results.copy()
    threshold_results['cost_fp'] = threshold_results['false_positives'] * cost_fp
    threshold_results['cost_fn'] = threshold_results['false_negatives'] * cost_fn
    threshold_results['total_cost'] = threshold_results['cost_fp'] + threshold_results['cost_fn']
    
    # Find optimal threshold (lowest cost)
    optimal_idx = threshold_results['total_cost'].idxmin()
    optimal_threshold = threshold_results.loc[optimal_idx, 'threshold']
    optimal_cost = threshold_results.loc[optimal_idx, 'total_cost']
    
    return threshold_results, optimal_threshold, optimal_cost


def plot_threshold_analysis(threshold_results, output_dir):
    """
    Plot curve showing trade-off between Precision and Recall
    
    Parameters:
    -----------
    threshold_results : DataFrame
        Threshold sweep results
    output_dir : str
        Output directory
    """
    print("\n[INFO] Plotting threshold analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Precision vs Recall
    axes[0, 0].plot(threshold_results['recall'], 
                    threshold_results['precision'], 
                    marker='o', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Recall', fontsize=12)
    axes[0, 0].set_ylabel('Precision', fontsize=12)
    axes[0, 0].set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. F1-Score vs Threshold
    axes[0, 1].plot(threshold_results['threshold'], 
                    threshold_results['f1_score'], 
                    marker='o', linewidth=2, color='green', markersize=4)
    axes[0, 1].set_xlabel('Threshold', fontsize=12)
    axes[0, 1].set_ylabel('F1-Score', fontsize=12)
    axes[0, 1].set_title('F1-Score vs Threshold', fontsize=14, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. False Positives vs False Negatives
    axes[1, 0].plot(threshold_results['threshold'], 
                    threshold_results['false_positives'], 
                    label='False Positives', linewidth=2, marker='o', markersize=4)
    axes[1, 0].plot(threshold_results['threshold'], 
                    threshold_results['false_negatives'], 
                    label='False Negatives', linewidth=2, marker='s', markersize=4)
    axes[1, 0].set_xlabel('Threshold', fontsize=12)
    axes[1, 0].set_ylabel('Count', fontsize=12)
    axes[1, 0].set_title('False Positives vs False Negatives', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Precision & Recall vs Threshold
    axes[1, 1].plot(threshold_results['threshold'], 
                    threshold_results['precision'], 
                    label='Precision', linewidth=2, marker='o', markersize=4)
    axes[1, 1].plot(threshold_results['threshold'], 
                    threshold_results['recall'], 
                    label='Recall', linewidth=2, marker='s', markersize=4)
    axes[1, 1].set_xlabel('Threshold', fontsize=12)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('Precision & Recall vs Threshold', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'threshold_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Plot saved to: {output_path}")
    plt.close()


def plot_cost_analysis(cost_results, cost_fp, cost_fn, output_dir):
    """
    Plot cost analysis curve
    
    Parameters:
    -----------
    cost_results : DataFrame
        Cost analysis results
    cost_fp : float
        Cost of False Positive
    cost_fn : float
        Cost of False Negative
    output_dir : str
        Output directory
    """
    print("\n[INFO] Plotting cost analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. Total Cost vs Threshold
    axes[0].plot(cost_results['threshold'], 
                 cost_results['total_cost'], 
                 marker='o', linewidth=2, color='red', markersize=4)
    
    # Identify optimal threshold
    optimal_idx = cost_results['total_cost'].idxmin()
    optimal_threshold = cost_results.loc[optimal_idx, 'threshold']
    optimal_cost = cost_results.loc[optimal_idx, 'total_cost']
    
    axes[0].axvline(x=optimal_threshold, 
                     color='green', linestyle='--', linewidth=2, 
                     label=f'Optimal Threshold: {optimal_threshold:.3f}')
    axes[0].set_xlabel('Threshold', fontsize=12)
    axes[0].set_ylabel('Total Cost ($)', fontsize=12)
    axes[0].set_title(f'Total Cost vs Threshold\n(FP Cost: ${cost_fp}, FN Cost: ${cost_fn})', 
                      fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 2. FP Cost vs FN Cost
    axes[1].plot(cost_results['threshold'], 
                 cost_results['cost_fp'], 
                 label='False Positive Cost', linewidth=2, marker='o', markersize=4)
    axes[1].plot(cost_results['threshold'], 
                 cost_results['cost_fn'], 
                 label='False Negative Cost', linewidth=2, marker='s', markersize=4)
    axes[1].set_xlabel('Threshold', fontsize=12)
    axes[1].set_ylabel('Cost ($)', fontsize=12)
    axes[1].set_title('Cost Breakdown', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'cost_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Plot saved to: {output_path}")
    plt.close()


def generate_threshold_recommendation(threshold_results, cost_results, cost_fp, cost_fn):
    """
    Generate comprehensive recommendation report
    
    Parameters:
    -----------
    threshold_results : DataFrame
        Threshold sweep results
    cost_results : DataFrame
        Cost analysis results
    cost_fp : float
        Cost of False Positive
    cost_fn : float
        Cost of False Negative
    
    Returns:
    --------
    dict : Recommendations
    """
    # 1. Best threshold based on cost
    optimal_cost_idx = cost_results['total_cost'].idxmin()
    optimal_cost_threshold = cost_results.loc[optimal_cost_idx, 'threshold']
    
    # 2. Best threshold based on F1-Score
    optimal_f1_idx = threshold_results['f1_score'].idxmax()
    optimal_f1_threshold = threshold_results.loc[optimal_f1_idx, 'threshold']
    
    # 3. Best threshold based on Recall (to detect most fraud)
    optimal_recall_idx = threshold_results['recall'].idxmax()
    optimal_recall_threshold = threshold_results.loc[optimal_recall_idx, 'threshold']
    
    # 4. Best threshold based on Precision (to minimize False Positives)
    optimal_precision_idx = threshold_results['precision'].idxmax()
    optimal_precision_threshold = threshold_results.loc[optimal_precision_idx, 'threshold']
    
    recommendation = {
        'optimal_cost': {
            'threshold': float(optimal_cost_threshold),
            'precision': float(cost_results.loc[optimal_cost_idx, 'precision']),
            'recall': float(cost_results.loc[optimal_cost_idx, 'recall']),
            'f1_score': float(cost_results.loc[optimal_cost_idx, 'f1_score']),
            'total_cost': float(cost_results.loc[optimal_cost_idx, 'total_cost']),
            'reason': f'Lowest total cost (${cost_results.loc[optimal_cost_idx, "total_cost"]:,.2f})'
        },
        'optimal_f1': {
            'threshold': float(optimal_f1_threshold),
            'precision': float(threshold_results.loc[optimal_f1_idx, 'precision']),
            'recall': float(threshold_results.loc[optimal_f1_idx, 'recall']),
            'f1_score': float(threshold_results.loc[optimal_f1_idx, 'f1_score']),
            'reason': 'Best balance between Precision and Recall'
        },
        'optimal_recall': {
            'threshold': float(optimal_recall_threshold),
            'precision': float(threshold_results.loc[optimal_recall_idx, 'precision']),
            'recall': float(threshold_results.loc[optimal_recall_idx, 'recall']),
            'reason': 'Detects maximum number of fraud cases'
        },
        'optimal_precision': {
            'threshold': float(optimal_precision_threshold),
            'precision': float(threshold_results.loc[optimal_precision_idx, 'precision']),
            'recall': float(threshold_results.loc[optimal_precision_idx, 'recall']),
            'reason': 'Minimum number of False Positives'
        }
    }
    
    return recommendation


def generate_threshold_report(threshold_results, cost_results, recommendation, output_dir):
    """
    Generate comprehensive threshold report
    
    Parameters:
    -----------
    threshold_results : DataFrame
        Threshold sweep results
    cost_results : DataFrame
        Cost analysis results
    recommendation : dict
        Recommendations
    output_dir : str
        Output directory
    """
    print("\n" + "="*70)
    print("Optimal Threshold Recommendations - 3.7")
    print("="*70)
    
    # Save threshold results
    threshold_path = os.path.join(output_dir, 'threshold_sweep_results.csv')
    threshold_results.to_csv(threshold_path, index=False)
    print(f"\n[OK] Threshold sweep results saved to: {threshold_path}")
    
    # Save cost results
    cost_path = os.path.join(output_dir, 'cost_analysis_results.csv')
    cost_results.to_csv(cost_path, index=False)
    print(f"[OK] Cost analysis results saved to: {cost_path}")
    
    # Print recommendations
    for criterion, info in recommendation.items():
        print(f"\n{criterion.replace('_', ' ').title()}:")
        print(f"  Threshold: {info['threshold']:.3f}")
        if 'precision' in info:
            print(f"  Precision: {info['precision']:.4f}")
        if 'recall' in info:
            print(f"  Recall: {info['recall']:.4f}")
        if 'f1_score' in info:
            print(f"  F1-Score: {info['f1_score']:.4f}")
        if 'total_cost' in info:
            print(f"  Total Cost: ${info['total_cost']:,.2f}")
        print(f"  Reason: {info['reason']}")
    
    # Final recommendation
    print("\n" + "="*70)
    print("Final Recommendation:")
    print("="*70)
    optimal = recommendation['optimal_cost']
    print(f"Use threshold {optimal['threshold']:.3f}")
    print(f"Because it provides lowest total cost: ${optimal['total_cost']:,.2f}")
    print(f"With Precision: {optimal['precision']:.4f}")
    print(f"And Recall: {optimal['recall']:.4f}")
    
    # Save recommendations
    import json
    recommendation_path = os.path.join(output_dir, 'threshold_recommendations.json')
    with open(recommendation_path, 'w', encoding='utf-8') as f:
        json.dump(recommendation, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Recommendations saved to: {recommendation_path}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main program"""
    
    # Default values
    DEFAULT_INFILE = 'data/processed/paysim_features.parquet'
    DEFAULT_MODELDIR = 'models'
    DEFAULT_OUTPUT_DIR = 'models/evaluation_reports'
    DEFAULT_COST_FP = 50  # Cost of False Positive
    DEFAULT_COST_FN = 5000  # Cost of False Negative
    
    parser = argparse.ArgumentParser(
        description='Evaluation Metrics & Threshold Analysis - 3.6 & 3.7',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # استخدام القيم الافتراضية
  python src/models/evaluation_and_threshold_analysis.py
  
  # تحديد مسارات مخصصة
  python src/models/evaluation_and_threshold_analysis.py \\
      --infile data/processed/paysim_features.parquet \\
      --modeldir models \\
      --output evaluation_reports \\
      --cost-fp 50 \\
      --cost-fn 5000
        """
    )
    
    parser.add_argument(
        '--infile',
        default=DEFAULT_INFILE,
        help=f'Input parquet file with features. Default: {DEFAULT_INFILE}'
    )
    
    parser.add_argument(
        '--modeldir',
        default=DEFAULT_MODELDIR,
        help=f'Directory containing trained models. Default: {DEFAULT_MODELDIR}'
    )
    
    parser.add_argument(
        '--output',
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for reports. Default: {DEFAULT_OUTPUT_DIR}'
    )
    
    parser.add_argument(
        '--cost-fp',
        type=float,
        default=DEFAULT_COST_FP,
        help=f'Cost of False Positive (default: {DEFAULT_COST_FP})'
    )
    
    parser.add_argument(
        '--cost-fn',
        type=float,
        default=DEFAULT_COST_FN,
        help=f'Cost of False Negative (default: {DEFAULT_COST_FN})'
    )
    
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip detailed evaluation (3.6)'
    )
    
    parser.add_argument(
        '--skip-threshold',
        action='store_true',
        help='Skip threshold analysis (3.7)'
    )
    
    args = parser.parse_args()
    
    # Display configuration
    print("\n" + "="*70)
    print("Evaluation Metrics & Threshold Analysis - 3.6 & 3.7")
    print("="*70)
    print(f"Input file:  {args.infile}")
    print(f"Models dir:  {args.modeldir}")
    print(f"Output dir:  {args.output}")
    print(f"Cost FP:     ${args.cost_fp}")
    print(f"Cost FN:     ${args.cost_fn}")
    print("="*70 + "\n")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    figures_dir = os.path.join(args.output, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load data
    print("[INFO] Loading data...")
    df = pd.read_parquet(args.infile)
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']
    
    # Split data (if not already split)
    print("[INFO] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Training set: {X_train.shape[0]:,} samples")
    print(f"  Test set:     {X_test.shape[0]:,} samples")
    
    # Load models
    print("\n[INFO] Loading models...")
    models_dict = {}
    
    model_files = {
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgboost_model.pkl',
        'Logistic Regression': 'logistic_regression_model.pkl'
    }
    
    for model_name, model_file in model_files.items():
        model_path = os.path.join(args.modeldir, model_file)
        if os.path.exists(model_path):
            try:
                models_dict[model_name] = joblib.load(model_path)
                print(f"  ✅ {model_name} loaded")
            except Exception as e:
                print(f"  ❌ Error loading {model_name}: {e}")
        else:
            print(f"  ⚠️ {model_name} not found: {model_path}")
    
    if not models_dict:
        print("\n❌ No models loaded. Make sure to run model_development.py first")
        return
    
    # ========================================================================
    # 3.6 Evaluation Metrics
    # ========================================================================
    if not args.skip_evaluation:
        comparison_df, detailed_reports, best_models = generate_evaluation_report(
            models_dict, X_test, y_test, args.output
        )
    else:
        print("\n[INFO] Skipping detailed metrics analysis (3.6)")
    
    # ========================================================================
    # 3.7 Threshold and Cost Analysis
    # ========================================================================
    if not args.skip_threshold:
        # Use best model (usually Random Forest)
        best_model_name = 'Random Forest'
        if best_model_name not in models_dict:
            best_model_name = list(models_dict.keys())[0]
        
        best_model = models_dict[best_model_name]
        
        print(f"\n[INFO] Using {best_model_name} for threshold analysis...")
        
        # Threshold Sweep
        threshold_results = threshold_sweep(best_model, X_test, y_test)
        
        # Cost Analysis
        cost_results, optimal_threshold, optimal_cost = calculate_cost_analysis(
            threshold_results, args.cost_fp, args.cost_fn
        )
        
        # Generate Recommendation
        recommendation = generate_threshold_recommendation(
            threshold_results, cost_results, args.cost_fp, args.cost_fn
        )
        
        # Plot Analysis
        plot_threshold_analysis(threshold_results, figures_dir)
        plot_cost_analysis(cost_results, args.cost_fp, args.cost_fn, figures_dir)
        
        # Generate Report
        generate_threshold_report(threshold_results, cost_results, recommendation, args.output)
        
        print(f"\n✅ Threshold and cost analysis completed!")
        print(f"   Optimal threshold: {optimal_threshold:.3f}")
        print(f"   Optimal cost: ${optimal_cost:,.2f}")
    else:
        print("\n[INFO] Skipping threshold and cost analysis (3.7)")
    
    print("\n" + "="*70)
    print("✅ All analyses completed successfully!")
    print(f"📁 Results saved to: {args.output}")
    print("="*70)


if __name__ == '__main__':
    main()

