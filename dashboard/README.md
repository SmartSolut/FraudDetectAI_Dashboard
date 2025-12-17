# Fraud Detection Dashboard

## Overview

Interactive Streamlit dashboard for fraud detection using the trained Random Forest model.

## Features

### 1. Overview
- Total transactions count
- Fraud rate statistics
- Recent alerts
- Transaction type distribution

### 2. Model Performance
- Model comparison (Random Forest, XGBoost, Logistic Regression)
- Performance metrics (ROC-AUC, PR-AUC, Precision, Recall, F1-Score)
- Feature importance visualization
- Best model recommendation

### 3. Threshold & Cost Analysis
- Interactive threshold control
- Real-time cost calculation
- Precision vs Recall visualization
- Cost breakdown analysis
- Optimal threshold recommendations

### 4. Risk Scoring
- Risk probability for transactions
- Filtering by risk level
- Export functionality
- Risk distribution charts

### 5. Transaction Analysis
- Single transaction analysis
- Real-time fraud prediction
- Top contributing features
- Risk level assessment

### 6. Operational Health
- Model status monitoring
- Performance metrics
- Retraining recommendations
- Data quality checks

## Installation

```bash
pip install streamlit pandas numpy plotly scikit-learn joblib
```

## Usage

```bash
streamlit run dashboard/app.py
```

## Configuration

The dashboard uses:
- **Model:** Random Forest (best performing model)
- **Default Threshold:** 0.2 (optimal cost threshold)
- **Cost FP:** $50 (configurable in sidebar)
- **Cost FN:** $5000 (configurable in sidebar)

## File Structure

```
dashboard/
├── __init__.py
├── app.py              # Main Streamlit application
├── utils.py            # Helper functions
└── README.md           # This file
```

## Requirements

- Python 3.7+
- streamlit
- pandas
- numpy
- plotly
- scikit-learn
- joblib

## Model Files Required

- `models/random_forest_model.pkl` - Trained model
- `models/model_metrics.csv` - Model performance metrics
- `models/figures/random_forest_feature_importance.csv` - Feature importance
- `models/evaluation_reports/threshold_sweep_results.csv` - Threshold analysis
- `models/evaluation_reports/cost_analysis_results.csv` - Cost analysis
- `models/evaluation_reports/threshold_recommendations.json` - Recommendations
- `data/processed/paysim_features.parquet` - Data for analysis

## Notes

- Make sure to run `model_development.py` and `evaluation_and_threshold_analysis.py` first
- The dashboard uses cached data for better performance
- All visualizations are interactive using Plotly




