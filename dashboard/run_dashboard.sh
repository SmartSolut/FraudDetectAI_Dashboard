#!/bin/bash
cd "$(dirname "$0")/.."
echo "========================================"
echo "Fraud Detection Dashboard"
echo "========================================"
echo ""
echo "Starting Streamlit Dashboard..."
echo ""
streamlit run dashboard/app.py

