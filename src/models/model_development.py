"""
Model Development - 3.5
=======================
تطوير وتدريب ثلاث خوارزميات تعلم آلي للإشراف للكشف عن الاحتيال

المتطلبات:
- Logistic Regression مع class_weight='balanced'
- Random Forest مع hyperparameters
- XGBoost محسّن للبيانات غير المتوازنة
- Stratified 5-Fold Cross-Validation
- Early Stopping لـ XGBoost
- Probability Calibration
- Feature Importance plots
- Logistic Regression coefficients
"""

import pandas as pd
import numpy as np
import argparse
import os
import yaml
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    cross_validate
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

# Set style
plt.style.use('default')


def load_config(config_path='src/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(data_path):
    """Load and prepare data"""
    print(f"[INFO] Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    
    # Separate features and target
    y = df['isFraud'].copy()
    X = df.drop(columns=['isFraud']).copy()
    
    print(f"[INFO] Data shape: {X.shape}")
    print(f"[INFO] Fraud rate: {y.mean()*100:.2f}%")
    print(f"[INFO] Number of features: {X.shape[1]}")
    
    return X, y


def create_models():
    """Create the three models with appropriate configurations"""
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            solver='lbfgs'
        ),
        
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1,
            verbose=0
        ),
        
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=None,  # Will be calculated
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    }
    
    return models


def calculate_scale_pos_weight(y_train):
    """Calculate scale_pos_weight for XGBoost"""
    fraud_count = y_train.sum()
    non_fraud_count = len(y_train) - fraud_count
    scale_pos_weight = non_fraud_count / max(fraud_count, 1)
    return scale_pos_weight


def train_with_cross_validation(X, y, models, n_splits=5, random_state=42):
    """Train models with Stratified 5-Fold Cross-Validation"""
    
    print("\n" + "="*70)
    print("STRATIFIED 5-FOLD CROSS-VALIDATION")
    print("="*70)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    cv_results = {}
    
    for name, model in models.items():
        print(f"\n[CV] Training {name}...")
        
        # Calculate scale_pos_weight for XGBoost
        if name == 'XGBoost':
            scale_pos_weight = calculate_scale_pos_weight(y)
            model.set_params(scale_pos_weight=scale_pos_weight)
            print(f"  [INFO] scale_pos_weight = {scale_pos_weight:.2f}")
        
        # Cross-validation with multiple metrics
        scoring = {
            'roc_auc': 'roc_auc',
            'pr_auc': 'average_precision',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1'
        }
        
        cv_scores = cross_validate(
            model, X, y,
            cv=skf,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )
        
        # Store results
        cv_results[name] = {
            'roc_auc_mean': cv_scores['test_roc_auc'].mean(),
            'roc_auc_std': cv_scores['test_roc_auc'].std(),
            'pr_auc_mean': cv_scores['test_pr_auc'].mean(),
            'pr_auc_std': cv_scores['test_pr_auc'].std(),
            'precision_mean': cv_scores['test_precision'].mean(),
            'precision_std': cv_scores['test_precision'].std(),
            'recall_mean': cv_scores['test_recall'].mean(),
            'recall_std': cv_scores['test_recall'].std(),
            'f1_mean': cv_scores['test_f1'].mean(),
            'f1_std': cv_scores['test_f1'].std()
        }
        
        # Print results
        print(f"  ROC-AUC: {cv_results[name]['roc_auc_mean']:.4f} (+/- {cv_results[name]['roc_auc_std']:.4f})")
        print(f"  PR-AUC:  {cv_results[name]['pr_auc_mean']:.4f} (+/- {cv_results[name]['pr_auc_std']:.4f})")
        print(f"  Precision: {cv_results[name]['precision_mean']:.4f} (+/- {cv_results[name]['precision_std']:.4f})")
        print(f"  Recall:    {cv_results[name]['recall_mean']:.4f} (+/- {cv_results[name]['recall_std']:.4f})")
        print(f"  F1-Score:  {cv_results[name]['f1_mean']:.4f} (+/- {cv_results[name]['f1_std']:.4f})")
    
    return cv_results


def train_models(X_train, y_train, X_test, y_test, models):
    """Train models on training set and evaluate on test set"""
    
    print("\n" + "="*70)
    print("MODEL TRAINING AND EVALUATION")
    print("="*70)
    
    trained_models = {}
    results = []
    
    for name, model in models.items():
        print(f"\n[Training] {name}...")
        
        # Calculate scale_pos_weight for XGBoost
        if name == 'XGBoost':
            scale_pos_weight = calculate_scale_pos_weight(y_train)
            model.set_params(scale_pos_weight=scale_pos_weight)
            print(f"  [INFO] scale_pos_weight = {scale_pos_weight:.2f}")
            # Note: Early stopping will be implemented in future version
            # For now, we use fixed n_estimators
            model.fit(X_train, y_train, verbose=False)
        else:
            # Train model
            model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store results
        results.append({
            'model': name,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        
        trained_models[name] = model
        
        # Print results
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  PR-AUC:    {pr_auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    
    return trained_models, results


def apply_probability_calibration(trained_models, X_train, y_train, X_test, y_test):
    """Apply probability calibration (Isotonic or Platt Scaling)"""
    
    print("\n" + "="*70)
    print("PROBABILITY CALIBRATION")
    print("="*70)
    
    calibrated_models = {}
    
    for name, model in trained_models.items():
        print(f"\n[Calibration] {name}...")
        
        # Apply calibration (Isotonic by default)
        calibrated_model = CalibratedClassifierCV(
            model,
            method='isotonic',
            cv=3
        )
        
        calibrated_model.fit(X_train, y_train)
        calibrated_models[name] = calibrated_model
        
        # Evaluate calibrated model
        y_pred_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
        roc_auc_cal = roc_auc_score(y_test, y_pred_proba_cal)
        pr_auc_cal = average_precision_score(y_test, y_pred_proba_cal)
        
        print(f"  ROC-AUC (calibrated): {roc_auc_cal:.4f}")
        print(f"  PR-AUC (calibrated):  {pr_auc_cal:.4f}")
    
    return calibrated_models


def plot_logistic_regression_coefficients(model, feature_names, output_dir):
    """Plot Logistic Regression coefficients (Feature Weights)"""
    
    print("\n[Plotting] Logistic Regression Coefficients...")
    
    coefficients = model.coef_[0]
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    # Plot top 20 features
    top_features = feature_importance.head(20)
    
    plt.figure(figsize=(12, 8))
    colors = ['red' if x < 0 else 'green' for x in top_features['coefficient']]
    plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Logistic Regression - Feature Coefficients (Top 20)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'logistic_regression_coefficients.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved to: {output_path}")
    
    # Save coefficients to CSV
    csv_path = os.path.join(output_dir, 'logistic_regression_coefficients.csv')
    feature_importance.to_csv(csv_path, index=False)
    print(f"  [OK] Saved coefficients to: {csv_path}")


def plot_feature_importance(model, feature_names, model_name, output_dir):
    """Plot Feature Importance for Random Forest and XGBoost"""
    
    print(f"\n[Plotting] {model_name} Feature Importance...")
    
    if model_name == 'Random Forest':
        importances = model.feature_importances_
    elif model_name == 'XGBoost':
        importances = model.feature_importances_
    else:
        return
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 features
    top_features = feature_importance.head(20)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'{model_name} - Feature Importance (Top 20)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    filename = f'{model_name.lower().replace(" ", "_")}_feature_importance.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved to: {output_path}")
    
    # Save importance to CSV
    csv_filename = f'{model_name.lower().replace(" ", "_")}_feature_importance.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    feature_importance.to_csv(csv_path, index=False)
    print(f"  [OK] Saved importance to: {csv_path}")


def plot_roc_curves(trained_models, X_test, y_test, output_dir):
    """Plot ROC curves for all models"""
    
    print("\n[Plotting] ROC Curves...")
    
    plt.figure(figsize=(10, 8))
    
    for name, model in trained_models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'roc_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved to: {output_path}")


def plot_pr_curves(trained_models, X_test, y_test, output_dir):
    """Plot Precision-Recall curves for all models"""
    
    print("\n[Plotting] Precision-Recall Curves...")
    
    plt.figure(figsize=(10, 8))
    
    for name, model in trained_models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.4f})', linewidth=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'pr_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved to: {output_path}")


def plot_confusion_matrices(trained_models, X_test, y_test, output_dir):
    """Plot confusion matrices for all models"""
    
    print("\n[Plotting] Confusion Matrices...")
    
    n_models = len(trained_models)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        im = axes[idx].imshow(cm, cmap='Blues', aspect='auto')
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(['No Fraud', 'Fraud'])
        axes[idx].set_yticklabels(['No Fraud', 'Fraud'])
        for i in range(2):
            for j in range(2):
                axes[idx].text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontweight='bold')
        plt.colorbar(im, ax=axes[idx])
        axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('Actual', fontsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved to: {output_path}")


def save_models(trained_models, output_dir):
    """Save trained models"""
    
    print("\n[Saving] Trained Models...")
    
    for name, model in trained_models.items():
        filename = f'{name.lower().replace(" ", "_")}_model.pkl'
        filepath = os.path.join(output_dir, filename)
        joblib.dump(model, filepath)
        print(f"  [OK] {name} saved to: {filepath}")


def main():
    """Main function"""
    
    # Default values
    DEFAULT_INFILE = 'data/processed/paysim_features.parquet'
    DEFAULT_MODELDIR = 'models'
    DEFAULT_CONFIG = 'src/config.yaml'
    
    parser = argparse.ArgumentParser(
        description='Model Development - 3.5: Train and evaluate fraud detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--infile',
        default=DEFAULT_INFILE,
        help=f'Input parquet file with features. Default: {DEFAULT_INFILE}'
    )
    
    parser.add_argument(
        '--modeldir',
        default=DEFAULT_MODELDIR,
        help=f'Output directory for models and results. Default: {DEFAULT_MODELDIR}'
    )
    
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG,
        help=f'Configuration file path. Default: {DEFAULT_CONFIG}'
    )
    
    parser.add_argument(
        '--skip-cv',
        action='store_true',
        help='Skip cross-validation (faster training)'
    )
    
    parser.add_argument(
        '--skip-calibration',
        action='store_true',
        help='Skip probability calibration'
    )
    
    args = parser.parse_args()
    
    # Display configuration
    print("\n" + "="*70)
    print("MODEL DEVELOPMENT - 3.5")
    print("="*70)
    print(f"Input file:  {args.infile}")
    print(f"Output dir:  {args.modeldir}")
    print(f"Config:      {args.config}")
    print("="*70 + "\n")
    
    # Load configuration
    try:
        config = load_config(args.config)
    except:
        print("[WARNING] Could not load config file, using defaults")
        config = {}
    
    # Create output directory
    os.makedirs(args.modeldir, exist_ok=True)
    figures_dir = os.path.join(args.modeldir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load data
    X, y = load_data(args.infile)
    
    # Split data
    print("\n[INFO] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    print(f"  Training set: {X_train.shape[0]:,} samples")
    print(f"  Test set:     {X_test.shape[0]:,} samples")
    
    # Create models
    models = create_models()
    
    # Cross-validation (optional)
    if not args.skip_cv:
        cv_results = train_with_cross_validation(X_train, y_train, models)
        
        # Save CV results
        cv_df = pd.DataFrame(cv_results).T
        cv_path = os.path.join(args.modeldir, 'cross_validation_results.csv')
        cv_df.to_csv(cv_path)
        print(f"\n[OK] Cross-validation results saved to: {cv_path}")
    
    # Train models
    trained_models, results = train_models(X_train, y_train, X_test, y_test, models)
    
    # Probability calibration
    if not args.skip_calibration:
        calibrated_models = apply_probability_calibration(
            trained_models, X_train, y_train, X_test, y_test
        )
        # Use calibrated models for final evaluation
        trained_models = calibrated_models
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(args.modeldir, 'model_metrics.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n[OK] Model metrics saved to: {results_path}")
    
    # Explainability: Logistic Regression coefficients
    if 'Logistic Regression' in trained_models:
        plot_logistic_regression_coefficients(
            trained_models['Logistic Regression'],
            X.columns.tolist(),
            figures_dir
        )
    
    # Explainability: Feature Importance
    for name in ['Random Forest', 'XGBoost']:
        if name in trained_models:
            plot_feature_importance(
                trained_models[name],
                X.columns.tolist(),
                name,
                figures_dir
            )
    
    # Plot ROC and PR curves
    plot_roc_curves(trained_models, X_test, y_test, figures_dir)
    plot_pr_curves(trained_models, X_test, y_test, figures_dir)
    plot_confusion_matrices(trained_models, X_test, y_test, figures_dir)
    
    # Save models
    save_models(trained_models, args.modeldir)
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nResults saved to: {args.modeldir}")
    print(f"Figures saved to: {figures_dir}")
    print("\nModel Performance Summary:")
    print(results_df.to_string(index=False))
    print("\n" + "="*70)


if __name__ == '__main__':
    main()

