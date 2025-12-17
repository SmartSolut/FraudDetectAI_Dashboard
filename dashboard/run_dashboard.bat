@echo off
cd /d %~dp0..
echo ========================================
echo Fraud Detection Dashboard
echo ========================================
echo.
echo Starting Streamlit Dashboard...
echo.
echo If browser doesn't open automatically, go to:
echo http://localhost:8501
echo.
streamlit run dashboard/app.py
pause

