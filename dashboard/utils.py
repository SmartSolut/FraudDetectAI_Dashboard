"""
Dashboard Utilities
===================
Helper functions for the Dashboard
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Resolve project root (one level above dashboard/)
BASE_DIR = Path(__file__).resolve().parent.parent


def load_model(model_path=None):
    """Load trained model"""
    if model_path is None:
        model_path = BASE_DIR / 'models' / 'random_forest_model.pkl'
    else:
        model_path = Path(model_path)
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def load_model_by_name(model_name):
    """Load model by name"""
    model_files = {
        'Random Forest': BASE_DIR / 'models' / 'random_forest_model.pkl',
        'XGBoost': BASE_DIR / 'models' / 'xgboost_model.pkl',
        'Logistic Regression': BASE_DIR / 'models' / 'logistic_regression_model.pkl'
    }
    
    if model_name in model_files:
        return load_model(model_files[model_name])
    return None


def get_available_models():
    """Get list of available models"""
    available = []
    model_files = {
        'Random Forest': BASE_DIR / 'models' / 'random_forest_model.pkl',
        'XGBoost': BASE_DIR / 'models' / 'xgboost_model.pkl',
        'Logistic Regression': BASE_DIR / 'models' / 'logistic_regression_model.pkl'
    }
    
    for name, path in model_files.items():
        if path.exists():
            available.append(name)
    
    return available


def load_feature_importance(path=None):
    """Load feature importance"""
    if path is None:
        path = BASE_DIR / 'models' / 'figures' / 'random_forest_feature_importance.csv'
    else:
        path = Path(path)
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Error loading feature importance: {e}")
        return None


def build_features_from_transaction(transaction_data):
    """
    Build features from a single transaction
    
    Parameters:
    -----------
    transaction_data : dict
        Dictionary containing transaction data
    
    Returns:
    --------
    pd.DataFrame : Features ready for prediction (23 columns, same order as training data)
    """
    # Convert to DataFrame
    df = pd.DataFrame([transaction_data])
    
    # Ensure all required base columns exist
    required_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                     'oldbalanceDest', 'newbalanceDest', 'deltaOrg', 'deltaDest']
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    # Calculate deltaOrg and deltaDest if not provided
    if 'deltaOrg' not in df.columns or df['deltaOrg'].iloc[0] == 0:
        expected_orig = df['oldbalanceOrg'] - df['amount']
        df['deltaOrg'] = df['newbalanceOrig'] - expected_orig
    
    if 'deltaDest' not in df.columns or df['deltaDest'].iloc[0] == 0:
        expected_dest = df['oldbalanceDest'] + df['amount']
        df['deltaDest'] = df['newbalanceDest'] - expected_dest
    
    # Time features
    if 'step' in df.columns:
        df['hour_of_day'] = (df['step'] % 24).astype(int)
        df['day_of_month'] = ((df['step'] // 24) % 30).astype(int)
        df['is_weekend'] = ((df['step'] // 24) % 7).isin([5, 6]).astype(int)
    else:
        df['hour_of_day'] = 0
        df['day_of_month'] = 0
        df['is_weekend'] = 0
    
    # Transaction features
    df['amount_to_oldbalance_ratio'] = np.where(
        df['oldbalanceOrg'] > 0,
        df['amount'] / df['oldbalanceOrg'],
        0.0
    )
    
    df['account_emptied'] = (
        (df['oldbalanceOrg'] > 0) & 
        (df['newbalanceOrig'] == 0)
    ).astype(int)
    
    df['dest_was_empty'] = (df['oldbalanceDest'] == 0).astype(int)
    
    # Error features
    expected_orig = df['oldbalanceOrg'] - df['amount']
    df['error_balance_orig'] = np.abs(expected_orig - df['newbalanceOrig'])
    
    expected_dest = df['oldbalanceDest'] + df['amount']
    df['error_balance_dest'] = np.abs(expected_dest - df['newbalanceDest'])
    
    df['has_balance_error'] = (
        (df['error_balance_orig'] > 0) | 
        (df['error_balance_dest'] > 0)
    ).astype(int)
    
    # Categorical encoding - One-Hot Encoding for type
    type_columns = ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
    
    if 'type' in df.columns:
        # Get the transaction type
        trans_type = df['type'].iloc[0] if len(df) > 0 else None
        
        # Set all type columns to 0 first
        for col in type_columns:
            df[col] = 0
        
        # Set the correct type column to 1
        if trans_type:
            type_col = f'type_{trans_type}'
            if type_col in type_columns:
                df[type_col] = 1
    else:
        # If type not provided, set all to 0 (default to PAYMENT)
        for col in type_columns:
            df[col] = 0
        df['type_PAYMENT'] = 1  # Default
    
    # Remove target columns if they exist (but keep isFlaggedFraud if model expects it)
    if 'isFraud' in df.columns:
        df = df.drop(columns=['isFraud'])
    
    # Note: isFlaggedFraud might be expected by the model, so we'll add it as 0
    # Don't drop it yet - we'll handle it in predict_fraud
    
    # Ensure correct column order (same as training data)
    expected_columns = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud', 'deltaOrg', 'deltaDest',
        'hour_of_day', 'day_of_month', 'is_weekend',
        'amount_to_oldbalance_ratio', 'account_emptied', 'dest_was_empty',
        'error_balance_orig', 'error_balance_dest', 'has_balance_error',
        'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
    ]
    
    # Add isFlaggedFraud if not present (set to 0 for prediction)
    if 'isFlaggedFraud' not in df.columns:
        df['isFlaggedFraud'] = 0
    
    # Reorder columns to match training data
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df[expected_columns]
    
    return df


def predict_fraud(model, transaction_data, threshold=0.2):
    """
    Predict fraud probability for a transaction
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained model
    transaction_data : dict or pd.DataFrame
        Transaction data
    threshold : float
        Decision threshold
    
    Returns:
    --------
    dict : Prediction results
    """
    if model is None:
        return None
    
    # Build features
    if isinstance(transaction_data, dict):
        features_df = build_features_from_transaction(transaction_data)
    else:
        features_df = transaction_data.copy()
    
    # Predict
    try:
        # Get model's expected features
        if hasattr(model, 'feature_names_in_'):
            model_features = list(model.feature_names_in_)
            
            # Remove isFraud if it exists (it's a target, not a feature)
            if 'isFraud' in model_features:
                model_features.remove('isFraud')
            
            # Keep isFlaggedFraud if model expects it (set to 0 for prediction)
            # Note: Some models were trained with isFlaggedFraud as a feature
            
            # Ensure all model features are in features_df
            for feat in model_features:
                if feat not in features_df.columns:
                    features_df[feat] = 0
            
            # Reorder to match model's expected order
            available_features = [f for f in model_features if f in features_df.columns]
            features_df = features_df[available_features]
            
            # If model still expects more features, add them as 0
            missing_features = [f for f in model_features if f not in features_df.columns]
            for feat in missing_features:
                features_df[feat] = 0
            
            # Final reorder to match model exactly
            features_df = features_df[model_features]
        
        probability = model.predict_proba(features_df)[0, 1]
        prediction = 1 if probability >= threshold else 0
        
        # Get feature importance
        feature_names = features_df.columns.tolist()
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        else:
            # Fallback for models without feature_importances_ (e.g., LogisticRegression)
            feature_importance = np.zeros(len(feature_names))
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance,
            'value': features_df.iloc[0].values
        }).sort_values('importance', ascending=False)
        
        top_features = importance_df.head(10).to_dict('records')
        
        return {
            'probability': float(probability),
            'prediction': int(prediction),
            'risk_level': get_risk_level(probability),
            'threshold': threshold,
            'top_features': top_features
        }
    except Exception as e:
        import traceback
        print(f"Error in prediction: {e}")
        print(traceback.format_exc())
        return None


def get_risk_level(probability):
    """Get risk level from probability"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.5:
        return "Medium"
    elif probability < 0.7:
        return "High"
    else:
        return "Critical"


def calculate_cost(fp_count, fn_count, cost_fp=50, cost_fn=5000):
    """Calculate total cost"""
    return (fp_count * cost_fp) + (fn_count * cost_fn)


def load_model_metrics(path='models/model_metrics.csv'):
    """Load model metrics"""
    try:
        df = pd.read_csv(path)
        return df
    except:
        return None


def load_threshold_recommendations(path='models/evaluation_reports/threshold_recommendations.json'):
    """Load threshold recommendations"""
    try:
        import json
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

