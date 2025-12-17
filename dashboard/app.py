"""
Fraud Detection Dashboard - Professional Version
==================================================
Modern dashboard with Arabic/English support
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path
import os

# Import utilities
import sys
from pathlib import Path
dashboard_path = Path(__file__).parent
BASE_DIR = dashboard_path.parent
sys.path.insert(0, str(dashboard_path))

from utils import (
    load_model, load_feature_importance, build_features_from_transaction,
    predict_fraud, get_risk_level, calculate_cost,
    load_model_metrics, load_threshold_recommendations
)

# ============================================================================
# Language Support
# ============================================================================
LANGUAGES = {
    "en": {
        "title": "🔍 Fraud Detection Dashboard",
        "nav_home": "🏠 Home",
        "nav_check": "🔍 Check Transaction",
        "nav_view": "📊 View All",
        "nav_performance": "📈 Performance",
        "nav_how": "💡 How It Works",
        "enter_details": "Enter Transaction Details",
        "analyze": "🔍 Analyze Transaction",
        "fraud_detected": "🚨 FRAUD DETECTED!",
        "normal_transaction": "✅ NORMAL TRANSACTION",
        "fraud_probability": "Fraud Probability",
        "risk_level": "Risk Level",
        "prediction": "Prediction",
        "why_result": "Why This Result? - Top Contributing Features",
        "total_transactions": "Total Transactions",
        "fraud_rate": "Fraud Rate",
        "fraud_cases": "Fraud Cases",
        "total_amount": "Total Amount",
        "recent_alerts": "🚨 Recent Fraud Alerts",
        "no_fraud": "✅ No fraud detected",
        "model_settings": "⚙️ Model Settings",
        "select_model": "Select Model",
        "threshold": "Decision Threshold",
        "cost_settings": "💰 Cost Settings",
        "fp_cost": "False Positive Cost ($)",
        "fn_cost": "False Negative Cost ($)",
        "recommended": "💡 Recommended",
        "step": "Step (Time)",
        "type": "Transaction Type",
        "amount": "Amount ($)",
        "old_balance_sender": "Original Balance (Sender)",
        "new_balance_sender": "New Balance (Sender)",
        "old_balance_receiver": "Original Balance (Receiver)",
        "new_balance_receiver": "New Balance (Receiver)",
        "delta_org": "Delta Org",
        "delta_dest": "Delta Dest",
        "analyzing": "🔍 Analyzing transaction...",
        "should_block": "⚠️ This transaction is highly suspicious and should be blocked or reviewed immediately!",
        "appears_legitimate": "✅ This transaction appears to be legitimate.",
        "increases_risk": "🔴 Increases",
        "decreases_risk": "🟢 Decreases",
        "fraud_risk": "fraud risk",
        "importance": "Importance",
        "value": "Value",
        "filter_transactions": "🔍 Filter Transactions by Risk",
        "risk_filter": "🎚️ Risk Filter",
        "min_risk": "Minimum Risk",
        "max_risk": "Maximum Risk",
        "display_options": "📊 Display Options",
        "limit": "🔢 Limit",
        "num_transactions": "Number of Transactions",
        "showing": "Showing",
        "of": "of",
        "transactions": "transactions",
        "export_csv": "📥 Export to CSV",
        "risk_distribution": "📊 Risk Distribution",
        "fraud_vs_normal": "Fraud vs Normal",
        "best_model": "🏆 Best Model",
        "model_comparison": "📊 Model Comparison",
        "feature_importance": "🔍 Feature Importance",
        "num_features": "Number of features",
        "how_works": "💡 How Fraud Detection Works",
        "how_works_desc": "The system uses Random Forest to analyze transactions and detect fraud automatically.",
        "step1": "Collect Transaction Data",
        "step1_desc": "Amount, transaction type, old and new balances, time",
        "step2": "Build Features (24 features)",
        "step2_desc": "Time features, amount features, account features, error features",
        "step3": "Model Prediction",
        "step3_desc": "Model analyzes all features and gives fraud probability (0% - 100%)",
        "step4": "Interpret Result",
        "step4_desc": "Top 10 contributing features, risk level, recommendation (block/review/allow)",
        "key_features": "🎯 Key Features for Fraud Detection",
        "fraud_patterns": "📝 Fraud Pattern Examples",
        "pattern1": "Account Emptied",
        "pattern1_desc": "Transaction drains all balance → account_emptied = 1",
        "pattern2": "Large Amount Ratio",
        "pattern2_desc": "Amount > 80% of balance → amount_to_oldbalance_ratio > 0.8",
        "pattern3": "Suspicious Type",
        "pattern3_desc": "CASH_OUT or TRANSFER → type_CASH_OUT = 1",
        "model_loaded": "✅ Model: Loaded",
        "model_not_loaded": "❌ Model: Not Loaded",
        "data_loaded": "✅ Data:",
        "data_not_loaded": "❌ Data: Not Loaded",
        "quick_actions": "🚀 Quick Actions",
        "check_transaction": "🔍 Check Transaction",
        "view_all": "📊 View All Transactions",
        "learn_how": "💡 Learn How It Works",
        "transaction_types": "📈 Transaction Type Distribution",
        "error_analyzing": "❌ Error analyzing transaction",
        "model_not_available": "❌ Model not available. Please check model file path.",
        "error": "❌ Error",
        "tip": "💡 Tip",
        "make_sure": "Make sure model features match data columns",
        "model_features_not_available": "⚠️ Model features not available",
        "model_or_data_not_loaded": "⚠️ Model or data not loaded",
        "model_metrics_not_found": "⚠️ Model metrics not found. Please run model_development.py first.",
        "settings": "⚙️ Settings",
        "settings_desc": "Settings are controlled from the sidebar. Adjust threshold and costs there.",
        "footer": "🔍 Fraud Detection Dashboard | Powered by Random Forest AI",
        "accuracy": "Accuracy: 95% | PR-AUC: 94.95%"
    },
    "ar": {
        "title": "🔍 لوحة اكتشاف الاحتيال",
        "nav_home": "🏠 الرئيسية",
        "nav_check": "🔍 فحص معاملة",
        "nav_view": "📊 عرض الكل",
        "nav_performance": "📈 الأداء",
        "nav_how": "💡 كيف يعمل",
        "enter_details": "أدخل تفاصيل المعاملة",
        "analyze": "🔍 تحليل المعاملة",
        "fraud_detected": "🚨 تم اكتشاف احتيال!",
        "normal_transaction": "✅ معاملة عادية",
        "fraud_probability": "احتمالية الاحتيال",
        "risk_level": "مستوى المخاطرة",
        "prediction": "التنبؤ",
        "why_result": "لماذا هذه النتيجة؟ - أهم الخصائص المساهمة",
        "total_transactions": "إجمالي المعاملات",
        "fraud_rate": "معدل الاحتيال",
        "fraud_cases": "حالات الاحتيال",
        "total_amount": "إجمالي المبلغ",
        "recent_alerts": "🚨 تنبيهات الاحتيال الأخيرة",
        "no_fraud": "✅ لم يتم اكتشاف احتيال",
        "model_settings": "⚙️ إعدادات النموذج",
        "select_model": "اختر النموذج",
        "threshold": "عتبة القرار",
        "cost_settings": "💰 إعدادات التكلفة",
        "fp_cost": "تكلفة الإيجابي الكاذب ($)",
        "fn_cost": "تكلفة السلبي الكاذب ($)",
        "recommended": "💡 موصى به",
        "step": "الخطوة (الوقت)",
        "type": "نوع المعاملة",
        "amount": "المبلغ ($)",
        "old_balance_sender": "الرصيد الأصلي (المرسل)",
        "new_balance_sender": "الرصيد الجديد (المرسل)",
        "old_balance_receiver": "الرصيد الأصلي (المستقبل)",
        "new_balance_receiver": "الرصيد الجديد (المستقبل)",
        "delta_org": "دلتا المرسل",
        "delta_dest": "دلتا المستقبل",
        "analyzing": "🔍 جاري تحليل المعاملة...",
        "should_block": "⚠️ هذه المعاملة مشبوهة للغاية ويجب حظرها أو مراجعتها فوراً!",
        "appears_legitimate": "✅ هذه المعاملة تبدو شرعية.",
        "increases_risk": "🔴 يزيد",
        "decreases_risk": "🟢 يقلل",
        "fraud_risk": "مخاطرة الاحتيال",
        "importance": "الأهمية",
        "value": "القيمة",
        "filter_transactions": "🔍 فلترة المعاملات حسب المخاطرة",
        "risk_filter": "🎚️ فلتر المخاطرة",
        "min_risk": "الحد الأدنى للمخاطرة",
        "max_risk": "الحد الأقصى للمخاطرة",
        "display_options": "📊 خيارات العرض",
        "limit": "🔢 الحد",
        "num_transactions": "عدد المعاملات",
        "showing": "عرض",
        "of": "من",
        "transactions": "معاملة",
        "export_csv": "📥 تصدير إلى CSV",
        "risk_distribution": "📊 توزيع المخاطرة",
        "fraud_vs_normal": "الاحتيال مقابل العادي",
        "best_model": "🏆 أفضل نموذج",
        "model_comparison": "📊 مقارنة النماذج",
        "feature_importance": "🔍 أهمية الخصائص",
        "num_features": "عدد الخصائص",
        "how_works": "💡 كيف يعمل اكتشاف الاحتيال",
        "how_works_desc": "يستخدم النظام Random Forest لتحليل المعاملات واكتشاف الاحتيال تلقائياً.",
        "step1": "جمع بيانات المعاملة",
        "step1_desc": "المبلغ، نوع المعاملة، الرصيد القديم والجديد، الوقت",
        "step2": "بناء الخصائص (24 خاصية)",
        "step2_desc": "خصائص زمنية، خصائص المبلغ، خصائص الحساب، خصائص الأخطاء",
        "step3": "تنبؤ النموذج",
        "step3_desc": "يحلل النموذج جميع الخصائص ويعطي احتمالية الاحتيال (0% - 100%)",
        "step4": "تفسير النتيجة",
        "step4_desc": "أهم 10 خصائص مساهمة، مستوى المخاطرة، توصية (حظر/مراجعة/السماح)",
        "key_features": "🎯 أهم الخصائص لاكتشاف الاحتيال",
        "fraud_patterns": "📝 أمثلة على أنماط الاحتيال",
        "pattern1": "إفراغ الحساب",
        "pattern1_desc": "المعاملة تستنزف كل الرصيد → account_emptied = 1",
        "pattern2": "نسبة مبلغ كبيرة",
        "pattern2_desc": "المبلغ > 80% من الرصيد → amount_to_oldbalance_ratio > 0.8",
        "pattern3": "نوع مشبوه",
        "pattern3_desc": "CASH_OUT أو TRANSFER → type_CASH_OUT = 1",
        "model_loaded": "✅ النموذج: محمل",
        "model_not_loaded": "❌ النموذج: غير محمل",
        "data_loaded": "✅ البيانات:",
        "data_not_loaded": "❌ البيانات: غير محملة",
        "quick_actions": "🚀 إجراءات سريعة",
        "check_transaction": "🔍 فحص معاملة",
        "view_all": "📊 عرض جميع المعاملات",
        "learn_how": "💡 تعلم كيف يعمل",
        "transaction_types": "📈 توزيع أنواع المعاملات",
        "error_analyzing": "❌ خطأ في تحليل المعاملة",
        "model_not_available": "❌ النموذج غير متاح. يرجى التحقق من مسار ملف النموذج.",
        "error": "❌ خطأ",
        "tip": "💡 نصيحة",
        "make_sure": "تأكد من تطابق خصائص النموذج مع أعمدة البيانات",
        "model_features_not_available": "⚠️ خصائص النموذج غير متاحة",
        "model_or_data_not_loaded": "⚠️ النموذج أو البيانات غير محملة",
        "model_metrics_not_found": "⚠️ مقاييس النموذج غير موجودة. يرجى تشغيل model_development.py أولاً.",
        "settings": "⚙️ الإعدادات",
        "settings_desc": "يتم التحكم في الإعدادات من الشريط الجانبي. اضبط العتبة والتكاليف هناك.",
        "footer": "🔍 لوحة اكتشاف الاحتيال | مدعومة بـ Random Forest AI",
        "accuracy": "الدقة: 95% | PR-AUC: 94.95%",
        "transaction_info": "معلومات المعاملة",
        "account_balances": "أرصدة الحسابات",
        "current": "الحالي",
        "analyzing_transactions": "🔍 جاري تحليل المعاملات...",
        "analyzing": "جاري التحليل...",
        "fraud": "احتيال",
        "normal": "عادي",
        "contribution": "المساهمة",
        "to_fraud_risk": "في مخاطرة الاحتيال",
        "top_features_title": "أهم 10 خصائص مساهمة",
        "most_important": "أهم الخصائص",
        "compare_models": "🔬 مقارنة النماذج",
        "model_testing": "🧪 اختبار النماذج",
        "settings_page": "⚙️ الإعدادات",
        "upload_file": "📤 رفع ملف معاملات",
        "manual_entry": "✍️ إدخال يدوي",
        "filter_search": "🔍 فلترة وبحث",
        "export_reports": "📥 تصدير التقارير",
        "test_model": "اختبار النموذج",
        "model_comparison": "مقارنة النماذج",
        "test_transaction": "اختبار معاملة",
        "select_test_model": "اختر النموذج للاختبار",
        "test_results": "نتائج الاختبار",
        "model_info": "معلومات النموذج",
        "no_model_selected": "لم يتم اختيار نموذج",
        "test_with_sample": "اختبار ببيانات عينة",
        "upload_csv": "رفع ملف CSV",
        "file_uploaded": "تم رفع الملف بنجاح",
        "analyze_file": "تحليل الملف",
        "search_transactions": "بحث في المعاملات",
        "filter_by": "فلترة حسب",
        "transaction_type_filter": "نوع المعاملة",
        "amount_range": "نطاق المبلغ",
        "risk_level_filter": "مستوى المخاطرة",
        "date_range": "نطاق التاريخ",
        "search_results": "نتائج البحث",
        "export_data": "تصدير البيانات",
        "export_report": "تصدير التقرير",
        "generate_report": "إنشاء تقرير"
    }
}

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="🔍 Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"  # Expanded on desktop, responsive on mobile
)

# ============================================================================
# Custom CSS - Professional Design with RTL Support
# ============================================================================
def get_css(lang, theme_settings=None):
    """Get CSS based on language and theme settings"""
    # Default theme settings
    if theme_settings is None:
        theme_settings = {
            'primary_color': '#2563eb',
            'bg_color': '#f3f4f6',
            'theme': 'professional_blue',
            'font_size': 'medium',
            'card_style': 'modern'
        }
    
    primary = theme_settings.get('primary_color', '#2563eb')
    bg = theme_settings.get('bg_color', '#f3f4f6')
    theme = theme_settings.get('theme', 'professional_blue')
    font_size = theme_settings.get('font_size', 'medium')
    card_style = theme_settings.get('card_style', 'modern')
    
    # Theme presets
    theme_colors = {
        'professional_blue': {'primary': '#2563eb', 'primary_dark': '#1d4ed8', 'bg': '#f3f4f6', 'card': '#ffffff', 'text': '#111827'},
        'dark': {'primary': '#3b82f6', 'primary_dark': '#2563eb', 'bg': '#111827', 'card': '#1f2937', 'text': '#f9fafb'},
        'light': {'primary': '#3b82f6', 'primary_dark': '#2563eb', 'bg': '#ffffff', 'card': '#ffffff', 'text': '#111827'},
        'ocean': {'primary': '#0d9488', 'primary_dark': '#0f766e', 'bg': '#f0fdfa', 'card': '#ffffff', 'text': '#134e4a'},
        'sunset': {'primary': '#ea580c', 'primary_dark': '#c2410c', 'bg': '#fff7ed', 'card': '#ffffff', 'text': '#7c2d12'},
        'purple': {'primary': '#7c3aed', 'primary_dark': '#6d28d9', 'bg': '#faf5ff', 'card': '#ffffff', 'text': '#581c87'},
        'minimal': {'primary': '#4b5563', 'primary_dark': '#374151', 'bg': '#f9fafb', 'card': '#ffffff', 'text': '#111827'}
    }
    
    colors = theme_colors.get(theme, theme_colors['professional_blue'])
    # Override with custom colors if set
    if primary != '#2563eb':
        colors['primary'] = primary
        colors['primary_dark'] = primary
    if bg != '#f3f4f6':
        colors['bg'] = bg
    
    # Font size multipliers
    font_multipliers = {'small': 0.85, 'medium': 1.0, 'large': 1.15}
    fm = font_multipliers.get(font_size, 1.0)
    
    # Card style variations
    card_styles = {
        'modern': f"background: linear-gradient(135deg, {colors['primary']} 0%, {colors['primary_dark']} 100%); box-shadow: 0 2px 8px {colors['primary']}40;",
        'flat': f"background: {colors['primary']}; box-shadow: none;",
        'neumorphism': f"background: {colors['card']}; box-shadow: 5px 5px 10px #d1d5db, -5px -5px 10px #ffffff; border: none;",
        'glass': f"background: {colors['primary']}dd; backdrop-filter: blur(10px); box-shadow: 0 4px 15px rgba(0,0,0,0.1);",
        'bordered': f"background: {colors['card']}; border: 2px solid {colors['primary']}; box-shadow: none;"
    }
    metric_card_style = card_styles.get(card_style, card_styles['modern'])
    
    # For bordered and neumorphism, text should be dark
    metric_text_color = '#ffffff' if card_style in ['modern', 'flat', 'glass'] else colors['text']
    
    rtl_style = """
    /* RTL Support */
    body, .main, .block-container, [data-testid="stAppViewContainer"] {
        direction: rtl !important;
        text-align: right !important;
    }
    
    .stSelectbox, .stNumberInput, .stSlider, .stTextInput {
        direction: rtl !important;
    }
    
    .metric-card, .info-card, .feature-item, .step-box {
        text-align: right !important;
    }
    
    .fraud-box, .normal-box {
        text-align: center !important;
    }
    """ if lang == 'ar' else ""
    
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
    
    :root {{
        --primary: {colors['primary']};
        --primary-dark: {colors['primary_dark']};
        --success: #16a34a;
        --danger: #dc2626;
        --warning: #ca8a04;
        --card: {colors['card']};
        --text: {colors['text']};
        --text-light: #ffffff;
        --border: #d1d5db;
        --bg: {colors['bg']};
        --font-mult: {fm};
    }}
    
    * {{ font-family: 'Tajawal', 'Times New Roman', serif; }}
    
    /* Background */
    [data-testid="stAppViewContainer"] {{ background: {colors['bg']}; }}
    
    .main .block-container {{
        background: var(--card);
        padding: 0.75rem 1rem 0 1rem !important;
        padding-bottom: 0 !important;
        max-width: 1200px;
        margin: 0 auto;
    }}
    
    /* Responsive Design for Mobile */
    @media screen and (max-width: 768px) {{
        .main .block-container {{
            padding: 0.5rem 0.75rem 0 0.75rem !important;
            max-width: 100% !important;
        }}
        
        /* Sidebar adjustments */
        [data-testid="stSidebar"] {{
            min-width: 200px !important;
            max-width: 250px !important;
        }}
        
        /* Metric cards - stack vertically on mobile */
        [data-testid="stHorizontalBlock"] {{
            flex-direction: column !important;
        }}
        
        .metric-card {{
            margin-bottom: 0.5rem !important;
            width: 100% !important;
        }}
        
        /* Header adjustments */
        .main-header {{
            font-size: calc(1.2rem * var(--font-mult)) !important;
            padding: 0.3rem 0 !important;
        }}
        
        /* Alert cards - full width */
        .alert-card {{
            flex-direction: column !important;
            text-align: center !important;
            padding: 0.75rem !important;
        }}
        
        /* Feature items - full width */
        .feature-item {{
            padding: 0.5rem !important;
            font-size: 0.85rem !important;
        }}
        
        /* Step boxes - full width */
        .step-box {{
            padding: 0.5rem !important;
            margin: 0.25rem 0 !important;
        }}
        
        /* Fraud/Normal boxes - full width */
        .fraud-box, .normal-box {{
            padding: 0.75rem !important;
            margin: 0.5rem 0 !important;
        }}
        
        /* Sidebar header - smaller on mobile */
        .sidebar-header h1 {{
            font-size: calc(1.3rem * var(--font-mult)) !important;
        }}
        
        .sidebar-header h2 {{
            font-size: calc(0.75rem * var(--font-mult)) !important;
        }}
        
        /* Info cards - full width */
        .info-card {{
            padding: 0.5rem !important;
            margin: 0.4rem 0 !important;
        }}
        
        /* Buttons - full width on mobile */
        .stButton > button {{
            width: 100% !important;
        }}
        
        /* Charts - responsive */
        .js-plotly-plot {{
            width: 100% !important;
            height: auto !important;
        }}
        
        /* Tables - scrollable on mobile */
        [data-testid="stDataFrame"] {{
            overflow-x: auto !important;
        }}
        
        /* Columns - stack on mobile */
        [data-testid="column"] {{
            width: 100% !important;
            margin-bottom: 0.5rem !important;
        }}
    }}
    
    /* Extra small devices (phones) */
    @media screen and (max-width: 480px) {{
        .main .block-container {{
            padding: 0.4rem 0.5rem 0 0.5rem !important;
        }}
        
        .metric-card h3 {{
            font-size: calc(1rem * var(--font-mult)) !important;
        }}
        
        .metric-card p {{
            font-size: calc(0.7rem * var(--font-mult)) !important;
        }}
        
        .main-header {{
            font-size: calc(1rem * var(--font-mult)) !important;
        }}
        
        .sidebar-header h1 {{
            font-size: calc(1.1rem * var(--font-mult)) !important;
        }}
        
        /* Hide sidebar on very small screens (optional) */
        [data-testid="stSidebar"] {{
            min-width: 180px !important;
        }}
    }}
    
    /* Remove ALL bottom space */
    .main {{ padding-bottom: 0 !important; margin-bottom: 0 !important; }}
    [data-testid="stVerticalBlock"] {{ padding-bottom: 0 !important; margin-bottom: 0 !important; }}
    [data-testid="stBottom"] {{ display: none !important; }}
    .block-container {{ padding-bottom: 0 !important; margin-bottom: 0 !important; }}
    
    #MainMenu, footer, header {{ visibility: hidden; display: none !important; }}
    
    /* Sidebar - Dynamic Theme (Desktop) */
    [data-testid="stSidebar"] {{ 
        background: var(--card); 
        border-right: 1px solid var(--border);
    }}
    [data-testid="stSidebar"] > div {{ padding: 0.75rem !important; }}
    [data-testid="stSidebar"] hr {{ margin: 0.5rem 0; }}
    
    /* CRITICAL: Sidebar ALWAYS visible - NEVER hide it */
    [data-testid="stSidebar"] {{
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        min-width: 21rem !important;
        max-width: 21rem !important;
    }}
    
    .sidebar-header {{
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        padding: 0.75rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 0.75rem;
    }}
    .sidebar-header * {{ color: #ffffff !important; }}
    .sidebar-header h1 {{ font-size: calc(1.75rem * var(--font-mult)); margin: 0; text-shadow: 1px 1px 3px rgba(0,0,0,0.3); }}
    .sidebar-header h2 {{ font-size: calc(0.85rem * var(--font-mult)); margin: 0.25rem 0 0 0; font-weight: 600; text-shadow: 1px 1px 2px rgba(0,0,0,0.2); }}
    
    .sidebar-section {{
        background: var(--bg);
        padding: 0.4rem 0.6rem;
        border-radius: 6px;
        margin: 0.4rem 0;
        border-left: 3px solid var(--primary);
    }}
    .sidebar-section * {{ color: var(--text) !important; }}
    .sidebar-section h3 {{ font-size: calc(0.8rem * var(--font-mult)); font-weight: 700; color: var(--primary) !important; margin: 0; }}
    
    /* Radio - Dynamic */
    .stRadio > div {{ gap: 2px !important; }}
    .stRadio label {{ 
        padding: 0.4rem 0.6rem !important;
        border-radius: 6px !important;
        font-size: calc(0.9rem * var(--font-mult)) !important;
        font-weight: 500 !important;
        border: 1px solid var(--border) !important;
        background: var(--card) !important;
        margin: 2px 0 !important;
        color: var(--text) !important;
    }}
    .stRadio label:hover {{ background: var(--bg) !important; border-color: var(--primary) !important; }}
    
    /* Main header */
    .main-header {{
        font-size: calc(1.5rem * var(--font-mult));
        font-weight: 700;
        color: var(--primary);
        text-align: center;
        padding: 0.5rem 0;
        margin-bottom: 0.75rem;
    }}
    
    /* Metric Cards - Dynamic Theme */
    .metric-card {{
        {metric_card_style}
        padding: 0.6rem 0.5rem;
        border-radius: 10px;
        text-align: center;
    }}
    .metric-card * {{ color: {metric_text_color} !important; }}
    .metric-card .icon {{ font-size: calc(1.5rem * var(--font-mult)); display: block; margin-bottom: 0.15rem; }}
    .metric-card h3 {{ font-size: calc(1.2rem * var(--font-mult)); font-weight: 700; margin: 0.1rem 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.2); }}
    .metric-card p {{ font-size: calc(0.75rem * var(--font-mult)); margin: 0; font-weight: 500; opacity: 0.95; }}
    
    /* Info card - Dynamic Theme */
    .info-card {{
        background: var(--card);
        padding: 0.6rem;
        border-radius: 6px;
        border: 1px solid var(--border);
        margin: 0.5rem 0;
    }}
    .info-card * {{ color: var(--text) !important; }}
    .info-card h3 {{ color: var(--primary) !important; font-size: calc(0.95rem * var(--font-mult)); margin-bottom: 0.3rem; font-weight: 700; }}
    
    /* Fraud Alert Cards - Compact */
    .alert-card {{
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        margin: 0.35rem 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }}
    .alert-card .alert-icon {{ font-size: 1.5rem; }}
    .alert-card .alert-content {{ flex: 1; }}
    .alert-card .alert-title {{ font-size: 0.8rem; font-weight: 700; margin: 0 0 0.15rem 0; }}
    .alert-card .alert-details {{ font-size: 0.75rem; margin: 0; opacity: 0.9; }}
    .alert-card .alert-badge {{
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.65rem;
        font-weight: 700;
    }}
    
    /* Critical - Dark Red */
    .alert-critical {{
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        border-left: 5px solid #7f1d1d;
    }}
    .alert-critical * {{ color: #ffffff !important; }}
    .alert-critical .alert-badge {{ background: #7f1d1d; }}
    
    /* High - Orange */
    .alert-high {{
        background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%);
        border-left: 5px solid #7c2d12;
    }}
    .alert-high * {{ color: #ffffff !important; }}
    .alert-high .alert-badge {{ background: #7c2d12; }}
    
    /* Medium - Yellow */
    .alert-medium {{
        background: linear-gradient(135deg, #fbbf24 0%, #d97706 100%);
        border-left: 5px solid #92400e;
    }}
    .alert-medium * {{ color: #1f2937 !important; }}
    .alert-medium .alert-badge {{ background: #92400e; color: #ffffff !important; }}
    
    /* Low - Light Orange */
    .alert-low {{
        background: linear-gradient(135deg, #fed7aa 0%, #fdba74 100%);
        border-left: 5px solid #c2410c;
    }}
    .alert-low * {{ color: #7c2d12 !important; }}
    .alert-low .alert-badge {{ background: #c2410c; color: #ffffff !important; }}
    
    /* Old fraud-box for compatibility */
    .fraud-box {{
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 3px 10px rgba(220,38,38,0.3);
    }}
    .fraud-box * {{ color: #ffffff !important; }}
    .fraud-box h2 {{ font-size: 1.1rem; margin: 0 0 0.35rem 0; font-weight: 700; }}
    .fraud-box p {{ font-size: 0.9rem; font-weight: 500; }}
    
    /* Normal box - Green text on light green background */
    .normal-box {{
        background: #f0fdf4;
        border: 2px solid #16a34a;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
        margin: 0.5rem 0;
    }}
    .normal-box * {{ color: #166534 !important; }}
    .normal-box h2 {{ font-size: 1.1rem; margin: 0 0 0.3rem 0; font-weight: 700; }}
    .normal-box p {{ font-size: 0.9rem; font-weight: 500; }}
    
    /* Feature item - Dark text on light gray background */
    .feature-item {{
        background: #f3f4f6;
        padding: 0.4rem 0.6rem;
        margin: 0.3rem 0;
        border-radius: 6px;
        border-left: 3px solid #2563eb;
        font-size: 0.9rem;
    }}
    .feature-item * {{ color: #111827 !important; }}
    
    /* Warning box - Dark yellow/brown text on light yellow background */
    .warning-box {{
        background: #fefce8;
        border-left: 3px solid #ca8a04;
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }}
    .warning-box * {{ color: #854d0e !important; }}
    
    /* Success box - Dark green text on light green background */
    .success-box {{
        background: #f0fdf4;
        border-left: 3px solid #16a34a;
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }}
    .success-box * {{ color: #166534 !important; }}
    
    /* Step box - Dynamic Theme */
    .step-box {{
        background: var(--card);
        padding: 0.6rem;
        border-radius: 6px;
        border-left: 3px solid var(--primary);
        margin: 0.3rem 0;
        border: 1px solid var(--border);
    }}
    .step-box * {{ color: var(--text) !important; }}
    .step-box h3 {{ color: var(--primary) !important; font-size: calc(0.95rem * var(--font-mult)); margin-bottom: 0.3rem; font-weight: 700; }}
    
    /* Buttons - Dynamic Theme */
    .stButton > button {{
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 700 !important;
        font-size: calc(0.95rem * var(--font-mult)) !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3) !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2) !important;
        transition: all 0.2s ease !important;
    }}
    .stButton > button * {{ color: #ffffff !important; }}
    .stButton > button:hover {{
        background: var(--primary-dark) !important;
        transform: translateY(-1px) !important;
    }}
    
    /* Inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox select {{
        border-radius: 6px !important;
        border: 1px solid var(--border) !important;
        padding: 0.45rem !important;
        font-size: 0.95rem !important;
        color: var(--text) !important;
    }}
    
    /* Status boxes */
    .stSuccess, .stInfo, .stWarning, .stError {{
        border-radius: 6px !important;
        padding: 0.5rem 0.75rem !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }}
    
    /* Text sizes - Clear and readable */
    .stMarkdown {{ margin-bottom: 0.3rem !important; }}
    hr {{ margin: 0.5rem 0 !important; }}
    h1 {{ font-size: 1.25rem !important; font-weight: 700 !important; margin-bottom: 0.5rem !important; color: var(--text) !important; }}
    h2 {{ font-size: 1.1rem !important; font-weight: 700 !important; margin-bottom: 0.4rem !important; color: var(--text) !important; }}
    h3 {{ font-size: 1rem !important; font-weight: 600 !important; margin-bottom: 0.35rem !important; color: var(--text) !important; }}
    p, li {{ font-size: 0.95rem !important; line-height: 1.5 !important; color: var(--text) !important; }}
    
    /* Expander */
    .streamlit-expanderHeader {{ font-size: 0.95rem !important; font-weight: 600 !important; padding: 0.5rem !important; }}
    
    /* Columns */
    [data-testid="column"] {{ padding: 0 0.25rem !important; }}
    
    /* Gaps - Minimal */
    .element-container {{ margin-bottom: 0.25rem !important; }}
    [data-testid="stVerticalBlock"] > div {{ gap: 0.25rem !important; }}
    
    /* Charts */
    .js-plotly-plot {{ border-radius: 8px; }}
    
    /* Tables */
    .stDataFrame {{ font-size: 0.85rem !important; }}
    
    /* Better label visibility */
    label {{ font-size: 0.9rem !important; font-weight: 500 !important; color: var(--text) !important; }}
    
    /* ============================================
       RESPONSIVE DESIGN FOR MOBILE DEVICES
       ============================================ */
    
    /* Tablet and below (max-width: 768px) - MOBILE ONLY */
    @media screen and (max-width: 768px) {{
        /* Main container - full width when sidebar is collapsed */
        .main .block-container {{
            padding: 0.5rem 0.75rem 0 0.75rem !important;
            max-width: 100% !important;
            margin-left: 0 !important;
        }}
        
        /* Sidebar - smaller on mobile but ALWAYS visible - NEVER hide */
        [data-testid="stSidebar"] {{
            min-width: 180px !important;
            max-width: 200px !important;
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
        }}
        
        /* Ensure sidebar never gets hidden on mobile */
        [data-testid="stSidebar"][aria-expanded="false"] {{
            min-width: 180px !important;
            max-width: 200px !important;
            display: block !important;
            visibility: visible !important;
        }}
        
        /* Sidebar content - compact on mobile */
        [data-testid="stSidebar"] > div {{
            padding: 0.5rem !important;
        }}
        
        /* Sidebar header - smaller on mobile */
        [data-testid="stSidebar"] .sidebar-header {{
            padding: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }}
        
        [data-testid="stSidebar"] .sidebar-header h1 {{
            font-size: 1.2rem !important;
        }}
        
        [data-testid="stSidebar"] .sidebar-header h2 {{
            font-size: 0.75rem !important;
        }}
        
        /* Sidebar sections - compact */
        [data-testid="stSidebar"] .sidebar-section {{
            padding: 0.3rem 0.5rem !important;
            margin: 0.3rem 0 !important;
        }}
        
        [data-testid="stSidebar"] .sidebar-section h3 {{
            font-size: 0.75rem !important;
        }}
        
        /* Radio buttons - smaller on mobile */
        [data-testid="stSidebar"] .stRadio label {{
            padding: 0.3rem 0.5rem !important;
            font-size: 0.8rem !important;
            margin: 1px 0 !important;
        }}
        
        /* Main content - adjust margin for smaller sidebar */
        .main .block-container {{
            margin-left: 0 !important;
            padding-left: 0.5rem !important;
        }}
        
        /* Ensure main content doesn't overlap */
        [data-testid="stAppViewContainer"] > div:first-child {{
            margin-left: 0 !important;
        }}
        
        /* Columns - stack vertically on mobile */
        [data-testid="stHorizontalBlock"] {{
            flex-direction: column !important;
        }}
        
        [data-testid="column"] {{
            width: 100% !important;
            margin-bottom: 0.5rem !important;
        }}
        
        /* Metric cards - full width */
        .metric-card {{
            margin-bottom: 0.5rem !important;
            width: 100% !important;
            padding: 0.5rem 0.4rem !important;
        }}
        
        .metric-card h3 {{
            font-size: calc(1rem * var(--font-mult)) !important;
        }}
        
        .metric-card p {{
            font-size: calc(0.7rem * var(--font-mult)) !important;
        }}
        
        /* Header adjustments */
        .main-header {{
            font-size: calc(1.2rem * var(--font-mult)) !important;
            padding: 0.3rem 0 !important;
        }}
        
        /* Hero banner - responsive */
        .hero-banner {{
            flex-direction: column !important;
            padding: 0.75rem !important;
            gap: 0.5rem !important;
        }}
        
        .hero-banner h1 {{
            font-size: calc(1.1rem * var(--font-mult)) !important;
        }}
        
        .hero-banner p {{
            font-size: calc(0.8rem * var(--font-mult)) !important;
        }}
        
        /* Alert cards - full width, stacked */
        .alert-card {{
            flex-direction: column !important;
            text-align: center !important;
            padding: 0.75rem !important;
            margin: 0.5rem 0 !important;
        }}
        
        /* Feature cards - full width */
        .feature-card {{
            margin-bottom: 0.5rem !important;
            width: 100% !important;
        }}
        
        /* Feature items - full width */
        .feature-item {{
            padding: 0.5rem !important;
            font-size: 0.85rem !important;
            margin: 0.25rem 0 !important;
        }}
        
        /* Step boxes - full width */
        .step-box {{
            padding: 0.5rem !important;
            margin: 0.25rem 0 !important;
        }}
        
        /* Fraud/Normal boxes - full width */
        .fraud-box, .normal-box {{
            padding: 0.75rem !important;
            margin: 0.5rem 0 !important;
        }}
        
        .fraud-box h2, .normal-box h2 {{
            font-size: 1rem !important;
        }}
        
        .fraud-box p, .normal-box p {{
            font-size: 0.85rem !important;
        }}
        
        /* Sidebar header - smaller on mobile */
        .sidebar-header {{
            padding: 0.6rem !important;
        }}
        
        .sidebar-header h1 {{
            font-size: calc(1.3rem * var(--font-mult)) !important;
        }}
        
        .sidebar-header h2 {{
            font-size: calc(0.75rem * var(--font-mult)) !important;
        }}
        
        /* Info cards - full width */
        .info-card {{
            padding: 0.5rem !important;
            margin: 0.4rem 0 !important;
        }}
        
        /* Buttons - full width on mobile */
        .stButton > button {{
            width: 100% !important;
            padding: 0.6rem 1rem !important;
            font-size: calc(0.9rem * var(--font-mult)) !important;
        }}
        
        /* Charts - responsive */
        .js-plotly-plot {{
            width: 100% !important;
            height: auto !important;
        }}
        
        /* Tables - scrollable on mobile */
        [data-testid="stDataFrame"] {{
            overflow-x: auto !important;
            font-size: 0.75rem !important;
        }}
        
        /* Text sizes - smaller on mobile */
        h1 {{ font-size: 1.1rem !important; }}
        h2 {{ font-size: 1rem !important; }}
        h3 {{ font-size: 0.95rem !important; }}
        p, li {{ font-size: 0.9rem !important; }}
        
        /* Inputs - full width */
        .stTextInput, .stNumberInput, .stSelectbox, .stSlider {{
            width: 100% !important;
        }}
        
        /* Status boxes - full width */
        .stSuccess, .stInfo, .stWarning, .stError {{
            padding: 0.4rem 0.6rem !important;
            font-size: 0.85rem !important;
        }}
    }}
    
    /* Extra small devices (phones, max-width: 480px) */
    @media screen and (max-width: 480px) {{
        .main .block-container {{
            padding: 0.4rem 0.5rem 0 0.5rem !important;
        }}
        
        /* Sidebar - smaller on very small phones */
        [data-testid="stSidebar"] {{
            min-width: 160px !important;
            max-width: 180px !important;
        }}
        
        /* Metric cards - even smaller */
        .metric-card {{
            padding: 0.4rem 0.3rem !important;
        }}
        
        .metric-card h3 {{
            font-size: calc(0.9rem * var(--font-mult)) !important;
        }}
        
        .metric-card p {{
            font-size: calc(0.65rem * var(--font-mult)) !important;
        }}
        
        .metric-card .icon {{
            font-size: calc(1.2rem * var(--font-mult)) !important;
        }}
        
        /* Header - smaller */
        .main-header {{
            font-size: calc(1rem * var(--font-mult)) !important;
        }}
        
        /* Hero banner - compact */
        .hero-banner {{
            padding: 0.6rem !important;
        }}
        
        .hero-banner h1 {{
            font-size: calc(1rem * var(--font-mult)) !important;
        }}
        
        .hero-banner p {{
            font-size: calc(0.75rem * var(--font-mult)) !important;
        }}
        
        .hero-banner .hero-icon {{
            font-size: calc(1.5rem * var(--font-mult)) !important;
        }}
        
        /* Sidebar header - very small */
        .sidebar-header h1 {{
            font-size: calc(1.1rem * var(--font-mult)) !important;
        }}
        
        .sidebar-header h2 {{
            font-size: calc(0.7rem * var(--font-mult)) !important;
        }}
        
        /* Buttons - compact */
        .stButton > button {{
            padding: 0.5rem 0.75rem !important;
            font-size: calc(0.85rem * var(--font-mult)) !important;
        }}
        
        /* Text - smaller */
        h1 {{ font-size: 1rem !important; }}
        h2 {{ font-size: 0.95rem !important; }}
        h3 {{ font-size: 0.9rem !important; }}
        p, li {{ font-size: 0.85rem !important; }}
        
        /* Tables - very small */
        [data-testid="stDataFrame"] {{
            font-size: 0.7rem !important;
        }}
        
        /* Feature items - compact */
        .feature-item {{
            padding: 0.4rem !important;
            font-size: 0.8rem !important;
        }}
        
        /* Step boxes - compact */
        .step-box {{
            padding: 0.4rem !important;
        }}
        
        /* Info cards - compact */
        .info-card {{
            padding: 0.4rem !important;
        }}
    }}
    
    /* Landscape orientation on mobile */
    @media screen and (max-width: 768px) and (orientation: landscape) {{
        .main .block-container {{
            padding: 0.4rem 0.6rem 0 0.6rem !important;
        }}
        
        [data-testid="stSidebar"] {{
            min-width: 180px !important;
        }}
    }}
    
    /* Hero Banner - Dynamic Theme */
    .hero-banner {{
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        text-align: center;
        box-shadow: 0 3px 12px rgba(0,0,0,0.15);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
    }}
    .hero-banner * {{ color: #ffffff !important; }}
    .hero-banner h1 {{ font-size: calc(1.25rem * var(--font-mult)) !important; margin: 0 !important; text-shadow: 1px 1px 2px rgba(0,0,0,0.2); }}
    .hero-banner p {{ font-size: calc(0.85rem * var(--font-mult)) !important; margin: 0 !important; opacity: 0.9; }}
    .hero-banner .hero-icon {{ font-size: calc(2rem * var(--font-mult)); }}
    .hero-banner .hero-text {{ text-align: left; }}
    
    /* Feature Cards - Dynamic Theme */
    .feature-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.6rem 0.5rem;
        text-align: center;
        transition: all 0.2s ease;
    }}
    .feature-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 3px 8px rgba(0,0,0,0.1);
        border-color: var(--primary);
    }}
    .feature-card * {{ color: var(--text) !important; }}
    .feature-card .feature-icon {{ font-size: calc(1.5rem * var(--font-mult)); display: block; margin-bottom: 0.2rem; }}
    .feature-card h3 {{ font-size: calc(0.85rem * var(--font-mult)) !important; color: var(--primary) !important; margin: 0.2rem 0 !important; font-weight: 700 !important; }}
    .feature-card p {{ font-size: 0.7rem !important; margin: 0 !important; color: #6b7280 !important; line-height: 1.3 !important; }}
    
    /* Stats Row */
    .stats-row {{
        display: flex;
        gap: 0.4rem;
        margin: 0.5rem 0;
    }}
    .stat-item {{
        flex: 1;
        background: #f8fafc;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
        border: 1px solid #e5e7eb;
    }}
    .stat-item * {{ color: #111827 !important; }}
    .stat-item .stat-icon {{ font-size: 1.25rem; }}
    .stat-item .stat-value {{ font-size: 1rem; font-weight: 700; color: #1e40af !important; }}
    .stat-item .stat-label {{ font-size: 0.7rem; color: #6b7280 !important; }}
    
    /* System Info Box - Compact */
    .system-info {{
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #7dd3fc;
        border-radius: 8px;
        padding: 0.6rem;
        margin: 0.4rem 0;
    }}
    .system-info * {{ color: #0c4a6e !important; }}
    .system-info h3 {{ color: #0369a1 !important; font-size: 0.85rem !important; margin-bottom: 0.3rem !important; }}
    .system-info p {{ font-size: 0.75rem !important; line-height: 1.4 !important; margin: 0 !important; }}
    
    /* Custom Footer - Dynamic Theme */
    .custom-footer {{
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
        padding: 0.6rem 1rem;
        margin: 0;
        text-align: center;
        border-top: 3px solid var(--primary);
    }}
    .custom-footer * {{ color: #ffffff !important; }}
    .custom-footer .footer-main {{ 
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
    }}
    .custom-footer .footer-logo {{
        font-size: calc(1.25rem * var(--font-mult));
    }}
    .custom-footer .footer-title {{ 
        font-size: calc(0.9rem * var(--font-mult)); 
        font-weight: 700; 
        margin: 0;
    }}
    .custom-footer .footer-divider {{
        width: 1px;
        height: 20px;
        background: rgba(255,255,255,0.3);
    }}
    .custom-footer .footer-stats {{
        display: flex;
        gap: 0.5rem;
        align-items: center;
        margin: 0;
    }}
    .custom-footer .footer-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        background: rgba(255,255,255,0.15);
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: calc(0.7rem * var(--font-mult));
        font-weight: 600;
        backdrop-filter: blur(5px);
    }}
    .custom-footer .footer-badge.highlight {{
        background: rgba(255,255,255,0.25);
        border: 1px solid rgba(255,255,255,0.3);
    }}
    
    {rtl_style}
    </style>
    """

# ============================================================================
# Cache Functions
# ============================================================================
@st.cache_resource
def load_cached_model():
    """Load model with caching"""
    return load_model()


@st.cache_data
def load_cached_data():
    """Load data with caching - REMOVE isFlaggedFraud"""
    try:
        # Load full data file (1M rows)
        full_path = BASE_DIR / 'data' / 'processed' / 'paysim_features.parquet'
        
        if full_path.exists():
            df = pd.read_parquet(full_path)
        else:
            return None
        
        # Remove target columns that are not features
        columns_to_remove = ['isFlaggedFraud', 'nameOrig', 'nameDest']
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
        return df
    except Exception as e:
        return None


@st.cache_data
def load_cached_metrics():
    """Load metrics with caching"""
    return load_model_metrics()


@st.cache_data
def load_cached_recommendations():
    """Load recommendations with caching"""
    return load_threshold_recommendations()


# ============================================================================
# Initialize Session State
# ============================================================================
if 'model' not in st.session_state:
    st.session_state.model = load_cached_model()

if 'data' not in st.session_state:
    st.session_state.data = load_cached_data()

if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.2

if 'cost_fp' not in st.session_state:
    st.session_state.cost_fp = 50

if 'cost_fn' not in st.session_state:
    st.session_state.cost_fn = 5000

if 'language' not in st.session_state:
    st.session_state.language = 'en'

if 'quick_nav' not in st.session_state:
    st.session_state.quick_nav = None

# Apply CSS based on language and theme settings
theme_settings = {
    'primary_color': st.session_state.get('primary_color', '#2563eb'),
    'bg_color': st.session_state.get('bg_color', '#f3f4f6'),
    'theme': st.session_state.get('theme', 'professional_blue'),
    'font_size': st.session_state.get('font_size', 'medium'),
    'card_style': st.session_state.get('card_style', 'modern')
}
st.markdown(get_css(st.session_state.language, theme_settings), unsafe_allow_html=True)

# Removed custom hamburger menu - using Streamlit's default


# ============================================================================
# Language Helper
# ============================================================================
def t(key):
    """Get translated text"""
    lang = st.session_state.language
    return LANGUAGES[lang].get(key, key)


# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    # Header - Professional Design
    st.markdown(f"""
    <div class="sidebar-header">
        <h1>🔍</h1>
        <h2>{t('title')}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Language Selector
    st.markdown(f"""
    <div class="sidebar-section">
        <h3>🌐 Language / اللغة</h3>
    </div>
    """, unsafe_allow_html=True)
    
    language = st.radio(
        "",
        ["English", "العربية"],
        index=0 if st.session_state.language == 'en' else 1,
        label_visibility="collapsed",
        horizontal=True
    )
    new_lang = 'en' if language == "English" else 'ar'
    if new_lang != st.session_state.language:
        st.session_state.language = new_lang
        st.rerun()
    
    st.markdown("---")
    
    # Navigation Section
    st.markdown(f"""
    <div class="sidebar-section">
        <h3>📍 Navigation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Page options
    page_options = [
        t("nav_home"),
        t("nav_check"),
        t("nav_view"),
        t("nav_performance"),
        t("compare_models"),
        t("model_testing"),
        t("nav_how"),
        t("settings_page")
    ]
    
    # Check for quick navigation
    default_index = 0
    if st.session_state.quick_nav and st.session_state.quick_nav in page_options:
        default_index = page_options.index(st.session_state.quick_nav)
        st.session_state.quick_nav = None  # Reset after use
    
    page = st.radio(
        "",
        page_options,
        index=default_index,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Model Settings Section
    st.markdown(f"""
    <div class="sidebar-section">
        <h3>⚙️ {t('model_settings')}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Get available models
    from utils import get_available_models
    available_models = get_available_models()
    
    if not available_models:
        st.warning("⚠️ No models found")
        model_name = "Random Forest"
    else:
        # Get current model from session state
        current_model = st.session_state.get('selected_model', available_models[0])
        if current_model not in available_models:
            current_model = available_models[0]
        
        model_name = st.selectbox(
            t("select_model"),
            available_models,
            index=available_models.index(current_model) if current_model in available_models else 0,
            help="Select model for predictions"
        )
        st.session_state.selected_model = model_name
    
    # Load selected model
    if model_name != st.session_state.get('current_model_name'):
        from utils import load_model_by_name
        st.session_state.model = load_model_by_name(model_name)
        st.session_state.current_model_name = model_name
        if st.session_state.model is not None:
            st.success(f"✅ {model_name} loaded")
    
    st.markdown("---")
    
    # Threshold Control Section
    st.markdown(f"""
    <div class="sidebar-section">
        <h3>🎚️ {t('threshold')}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    threshold = st.slider(
        t("threshold"),
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.threshold,
        step=0.01,
        help="Transactions with probability above this threshold are flagged as fraud"
    )
    st.session_state.threshold = threshold
    
    st.info(f"📊 Current: {threshold:.2f}")
    
    # Load recommendations
    recommendations = load_cached_recommendations()
    if recommendations:
        st.markdown(f"""
        <div style="background: #f0fdf4; padding: 0.75rem; border-radius: 0.5rem; border-left: 3px solid #10b981; margin-top: 0.5rem;">
            <strong style="color: #059669; font-size: 0.75rem;">💡 {t('recommended')}:</strong><br>
            <span style="font-size: 0.75rem;">• Cost: {recommendations['optimal_cost']['threshold']:.2f}</span><br>
            <span style="font-size: 0.75rem;">• F1: {recommendations['optimal_f1']['threshold']:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Cost Settings Section
    st.markdown(f"""
    <div class="sidebar-section">
        <h3>💰 {t('cost_settings')}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    cost_fp = st.number_input(
        t("fp_cost"),
        min_value=0,
        value=st.session_state.cost_fp,
        step=10,
        help="Cost of False Positive (customer friction)"
    )
    st.session_state.cost_fp = cost_fp
    
    cost_fn = st.number_input(
        t("fn_cost"),
        min_value=0,
        value=st.session_state.cost_fn,
        step=100,
        help="Cost of False Negative (missed fraud)"
    )
    st.session_state.cost_fn = cost_fn


# ============================================================================
# Main Content
# ============================================================================

# ============================================================================
# Page 1: Home
# ============================================================================
if page == t("nav_home"):
    # Hero Banner - Compact
    if st.session_state.language == 'ar':
        hero_title = "نظام كشف الاحتيال الذكي"
        hero_desc = "حماية معاملاتك المالية بالذكاء الاصطناعي"
    else:
        hero_title = "Smart Fraud Detection System"
        hero_desc = "Protecting your transactions with AI"
    
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-icon">🛡️</div>
        <div class="hero-text">
            <h1>{hero_title}</h1>
            <p>{hero_desc}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # System Status Row
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.model is not None:
            st.success("✅ " + t("model_loaded"))
        else:
            st.error("❌ " + t("model_not_loaded"))
    with col2:
        if st.session_state.data is not None:
            st.success(f"✅ {t('data_loaded')} ({len(st.session_state.data):,})")
        else:
            st.error("❌ " + t("data_not_loaded"))
    with col3:
        st.info(f"🎚️ {t('threshold')}: {threshold:.2f}")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Key Metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total = len(df)
            st.markdown(f"""
            <div class="metric-card">
                <div class="icon">📊</div>
                <h3>{total:,}</h3>
                <p>{t('total_transactions')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            fraud_rate = df['isFraud'].mean() * 100 if 'isFraud' in df.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="icon">⚠️</div>
                <h3>{fraud_rate:.2f}%</h3>
                <p>{t('fraud_rate')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            fraud_count = df['isFraud'].sum() if 'isFraud' in df.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="icon">🚨</div>
                <h3>{fraud_count:,}</h3>
                <p>{t('fraud_cases')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            total_amount = df['amount'].sum() if 'amount' in df.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="icon">💰</div>
                <h3>${total_amount/1e9:.1f}B</h3>
                <p>{t('total_amount')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # System Features Section
        st.markdown("---")
        if st.session_state.language == 'ar':
            st.markdown("### 🌟 مميزات النظام")
        else:
            st.markdown("### 🌟 System Features")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.language == 'ar':
                st.markdown("""
                <div class="feature-card">
                    <span class="feature-icon">🤖</span>
                    <h3>ذكاء اصطناعي</h3>
                    <p>نماذج ML متقدمة للكشف الدقيق</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="feature-card">
                    <span class="feature-icon">🤖</span>
                    <h3>AI Powered</h3>
                    <p>Advanced ML models for accurate detection</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if st.session_state.language == 'ar':
                st.markdown("""
                <div class="feature-card">
                    <span class="feature-icon">⚡</span>
                    <h3>سريع وفوري</h3>
                    <p>تحليل المعاملات في ثوانٍ</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="feature-card">
                    <span class="feature-icon">⚡</span>
                    <h3>Real-time</h3>
                    <p>Analyze transactions in seconds</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if st.session_state.language == 'ar':
                st.markdown("""
                <div class="feature-card">
                    <span class="feature-icon">📈</span>
                    <h3>دقة عالية</h3>
                    <p>نسبة دقة تتجاوز 95%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="feature-card">
                    <span class="feature-icon">📈</span>
                    <h3>High Accuracy</h3>
                    <p>Over 95% detection accuracy</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            if st.session_state.language == 'ar':
                st.markdown("""
                <div class="feature-card">
                    <span class="feature-icon">🔍</span>
                    <h3>تحليل شامل</h3>
                    <p>فحص 20+ خاصية لكل معاملة</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="feature-card">
                    <span class="feature-icon">🔍</span>
                    <h3>Deep Analysis</h3>
                    <p>Analyzing 20+ features per transaction</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Quick Actions
        st.markdown("---")
        st.subheader(t("quick_actions"))
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔍 " + t("check_transaction"), use_container_width=True, type="primary", key="quick_check"):
                st.session_state.quick_nav = t("nav_check")
                st.rerun()
        
        with col2:
            if st.button("📊 " + t("view_all"), use_container_width=True, key="quick_view"):
                st.session_state.quick_nav = t("nav_view")
                st.rerun()
        
        with col3:
            if st.button("🔬 " + t("compare_models"), use_container_width=True, key="quick_compare"):
                st.session_state.quick_nav = t("compare_models")
                st.rerun()
        
        with col4:
            if st.button("💡 " + t("learn_how"), use_container_width=True, key="quick_learn"):
                st.session_state.quick_nav = t("nav_how")
                st.rerun()
        
        # How System Works Section
        st.markdown("---")
        if st.session_state.language == 'ar':
            st.markdown("### ⚙️ كيف يعمل النظام")
        else:
            st.markdown("### ⚙️ How The System Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.language == 'ar':
                st.markdown("""
                <div class="system-info">
                    <h3>1️⃣ جمع البيانات</h3>
                    <p>📥 استقبال بيانات المعاملة<br>
                    💳 نوع المعاملة والمبلغ<br>
                    👤 معلومات الحسابات</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="system-info">
                    <h3>1️⃣ Data Collection</h3>
                    <p>📥 Receive transaction data<br>
                    💳 Transaction type & amount<br>
                    👤 Account information</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if st.session_state.language == 'ar':
                st.markdown("""
                <div class="system-info">
                    <h3>2️⃣ تحليل الخصائص</h3>
                    <p>🔬 استخراج 20+ خاصية<br>
                    📊 تحليل أنماط السلوك<br>
                    🧮 حساب نسب الخطر</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="system-info">
                    <h3>2️⃣ Feature Analysis</h3>
                    <p>🔬 Extract 20+ features<br>
                    📊 Analyze behavior patterns<br>
                    🧮 Calculate risk ratios</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if st.session_state.language == 'ar':
                st.markdown("""
                <div class="system-info">
                    <h3>3️⃣ اتخاذ القرار</h3>
                    <p>🤖 تطبيق نموذج ML<br>
                    📈 حساب احتمالية الاحتيال<br>
                    ✅ إصدار التنبيه أو الموافقة</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="system-info">
                    <h3>3️⃣ Decision Making</h3>
                    <p>🤖 Apply ML model<br>
                    📈 Calculate fraud probability<br>
                    ✅ Issue alert or approve</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recent Alerts
        st.subheader(t("recent_alerts"))
        
        if st.session_state.model is not None and len(df) > 0:
            sample_size = min(5000, len(df))
            sample_df = df.sample(n=sample_size, random_state=42).copy()
            
            model_features = st.session_state.model.feature_names_in_ if hasattr(st.session_state.model, 'feature_names_in_') else None
            
            if model_features is not None:
                X_sample = sample_df[[col for col in model_features if col in sample_df.columns]].copy()
                
                for feat in model_features:
                    if feat not in X_sample.columns:
                        X_sample[feat] = 0
                
                X_sample = X_sample[model_features]
                
                try:
                    probabilities = st.session_state.model.predict_proba(X_sample)[:, 1]
                    sample_df['fraud_probability'] = probabilities
                    sample_df['prediction'] = (probabilities >= threshold).astype(int)
                    
                    fraud_alerts = sample_df[sample_df['prediction'] == 1].sort_values('fraud_probability', ascending=False).head(5)
                    
                    if len(fraud_alerts) > 0:
                        for idx, row in fraud_alerts.iterrows():
                            prob = row['fraud_probability'] * 100
                            amount = row['amount']
                            risk = get_risk_level(row['fraud_probability'])
                            
                            # Determine alert class based on probability
                            if prob >= 80:
                                alert_class = "alert-critical"
                                risk_text = "🔴 Critical" if st.session_state.language == 'en' else "🔴 حرج"
                            elif prob >= 50:
                                alert_class = "alert-high"
                                risk_text = "🟠 High" if st.session_state.language == 'en' else "🟠 عالي"
                            elif prob >= 30:
                                alert_class = "alert-medium"
                                risk_text = "🟡 Medium" if st.session_state.language == 'en' else "🟡 متوسط"
                            else:
                                alert_class = "alert-low"
                                risk_text = "🟢 Low" if st.session_state.language == 'en' else "🟢 منخفض"
                            
                            if st.session_state.language == 'ar':
                                st.markdown(f"""
                                <div class="alert-card {alert_class}">
                                    <div class="alert-icon">🚨</div>
                                    <div class="alert-content">
                                        <div class="alert-title">تم اكتشاف احتيال محتمل!</div>
                                        <div class="alert-details">
                                            احتمالية: <strong>{prob:.1f}%</strong> &nbsp;|&nbsp; 
                                            المبلغ: <strong>${amount:,.2f}</strong>
                                        </div>
                                    </div>
                                    <div class="alert-badge">{risk_text}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="alert-card {alert_class}">
                                    <div class="alert-icon">🚨</div>
                                    <div class="alert-content">
                                        <div class="alert-title">Potential Fraud Detected!</div>
                                        <div class="alert-details">
                                            Probability: <strong>{prob:.1f}%</strong> &nbsp;|&nbsp; 
                                            Amount: <strong>${amount:,.2f}</strong>
                                        </div>
                                    </div>
                                    <div class="alert-badge">{risk_text}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.success(t("no_fraud"))
                        
                except Exception as e:
                    st.error(f"{t('error')}: {str(e)}")
            else:
                st.warning(t("model_features_not_available"))
        else:
            st.warning(t("model_or_data_not_loaded"))
        
        st.markdown("---")
        
        # Transaction Type Distribution with Chart Type Selector
        col_title, col_chart_type = st.columns([3, 1])
        with col_title:
            st.subheader(t("transaction_types"))
        with col_chart_type:
            if st.session_state.language == 'ar':
                chart_options = {
                    "🥧 دائري": "pie",
                    "🍩 حلقي": "donut",
                    "📊 أعمدة": "bar_v",
                    "📈 أفقي": "bar_h",
                    "🗺️ خريطة": "treemap",
                    "🔻 قمع": "funnel"
                }
            else:
                chart_options = {
                    "🥧 Pie": "pie",
                    "🍩 Donut": "donut",
                    "📊 Bar": "bar_v",
                    "📈 Horizontal": "bar_h",
                    "🗺️ Treemap": "treemap",
                    "🔻 Funnel": "funnel"
                }
            
            selected_chart = st.selectbox(
                "📊" if st.session_state.language == 'en' else "📊",
                list(chart_options.keys()),
                label_visibility="collapsed"
            )
            chart_type = chart_options[selected_chart]
        
        if 'type_CASH_OUT' in df.columns:
            type_counts = {
                'CASH_OUT': df['type_CASH_OUT'].sum(),
                'PAYMENT': df['type_PAYMENT'].sum() if 'type_PAYMENT' in df.columns else 0,
                'CASH_IN': df['type_CASH_IN'].sum() if 'type_CASH_IN' in df.columns else 0,
                'TRANSFER': df['type_TRANSFER'].sum() if 'type_TRANSFER' in df.columns else 0,
                'DEBIT': df['type_DEBIT'].sum() if 'type_DEBIT' in df.columns else 0
            }
            
            colors = ['#2563eb', '#7c3aed', '#ec4899', '#06b6d4', '#10b981']
            
            # Create chart based on selection
            if chart_type == "pie":
                fig = px.pie(
                    values=list(type_counts.values()),
                    names=list(type_counts.keys()),
                    color_discrete_sequence=colors,
                    hole=0
                )
            elif chart_type == "donut":
                fig = px.pie(
                    values=list(type_counts.values()),
                    names=list(type_counts.keys()),
                    color_discrete_sequence=colors,
                    hole=0.5
                )
                fig.add_annotation(
                    text=f"<b>{sum(type_counts.values()):,}</b><br>Total",
                    x=0.5, y=0.5, font_size=14, showarrow=False
                )
            elif chart_type == "bar_v":
                fig = px.bar(
                    x=list(type_counts.keys()),
                    y=list(type_counts.values()),
                    color=list(type_counts.keys()),
                    color_discrete_sequence=colors
                )
                fig.update_layout(showlegend=False)
            elif chart_type == "bar_h":
                fig = px.bar(
                    y=list(type_counts.keys()),
                    x=list(type_counts.values()),
                    color=list(type_counts.keys()),
                    color_discrete_sequence=colors,
                    orientation='h'
                )
                fig.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
            elif chart_type == "treemap":
                fig = px.treemap(
                    names=list(type_counts.keys()),
                    parents=[""] * len(type_counts),
                    values=list(type_counts.values()),
                    color=list(type_counts.values()),
                    color_continuous_scale=['#dbeafe', '#2563eb', '#1e40af']
                )
            elif chart_type == "funnel":
                sorted_counts = dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True))
                fig = px.funnel(
                    y=list(sorted_counts.keys()),
                    x=list(sorted_counts.values()),
                    color=list(sorted_counts.keys()),
                    color_discrete_sequence=colors
                )
            
            fig.update_layout(
                height=350,
                margin=dict(t=30, b=30, l=30, r=30),
                font=dict(size=12, family='Tajawal, Times New Roman'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# ============================================================================
# Page 2: Check Transaction
# ============================================================================
elif page == t("nav_check"):
    st.header(t("nav_check"))
    
    st.markdown(f"""
    <div class="info-card">
        <h3>📝 {t('enter_details')}</h3>
        <p>{t('enter_details')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Transaction Input Form
    with st.form("transaction_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Transaction Info")
            step = st.number_input(t("step"), min_value=0, value=100, step=1)
            transaction_type = st.selectbox(
                t("type"),
                ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
            )
            amount = st.number_input(
                t("amount"),
                min_value=0.0,
                value=1000.0,
                step=100.0,
                format="%.2f"
            )
        
        with col2:
            st.subheader("🏦 Account Balances")
            oldbalance_org = st.number_input(
                t("old_balance_sender"),
                min_value=0.0,
                value=5000.0,
                step=100.0,
                format="%.2f"
            )
            newbalance_orig = st.number_input(
                t("new_balance_sender"),
                min_value=0.0,
                value=4000.0,
                step=100.0,
                format="%.2f"
            )
            oldbalance_dest = st.number_input(
                t("old_balance_receiver"),
                min_value=0.0,
                value=0.0,
                step=100.0,
                format="%.2f"
            )
            newbalance_dest = st.number_input(
                t("new_balance_receiver"),
                min_value=0.0,
                value=1000.0,
                step=100.0,
                format="%.2f"
            )
        
        submitted = st.form_submit_button(t("analyze"), type="primary", use_container_width=True)
    
    if submitted:
        if st.session_state.model is not None:
            try:
                transaction_data = {
                    'step': int(step),
                    'type': transaction_type,
                    'amount': float(amount),
                    'oldbalanceOrg': float(oldbalance_org),
                    'newbalanceOrig': float(newbalance_orig),
                    'oldbalanceDest': float(oldbalance_dest),
                    'newbalanceDest': float(newbalance_dest),
                    'deltaOrg': 0.0,
                    'deltaDest': 0.0
                }
                
                with st.spinner(t("analyzing")):
                    result = predict_fraud(st.session_state.model, transaction_data, threshold)
                
                if result is None:
                    st.error(f"❌ {t('error_analyzing')}: النتيجة فارغة. تحقق من النموذج والبيانات.")
                elif result:
                    # Result is valid, proceed with display
                    st.markdown("---")
                    
                    prob = result['probability'] * 100
                    is_fraud = result['prediction'] == 1
                    
                    if is_fraud:
                        st.markdown(f"""
                        <div class="fraud-box">
                            <h1 style="font-size: 4rem; margin: 0;">🚨</h1>
                            <h2 style="margin: 0.5rem 0;">{t('fraud_detected')}</h2>
                            <p style="font-size: 2rem; margin: 0.5rem 0;">
                                {t('fraud_probability')}: <strong>{prob:.1f}%</strong>
                            </p>
                            <p style="font-size: 1.2rem; margin: 0;">
                                {t('risk_level')}: <strong>{result['risk_level']}</strong>
                            </p>
                            <p style="margin-top: 1rem;">
                                {t('should_block')}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="normal-box">
                            <h1 style="font-size: 4rem; margin: 0;">✅</h1>
                            <h2 style="margin: 0.5rem 0;">{t('normal_transaction')}</h2>
                            <p style="font-size: 2rem; margin: 0.5rem 0;">
                                {t('fraud_probability')}: <strong>{prob:.1f}%</strong>
                            </p>
                            <p style="font-size: 1.2rem; margin: 0;">
                                {t('risk_level')}: <strong>{result['risk_level']}</strong>
                            </p>
                            <p style="margin-top: 1rem;">
                                {t('appears_legitimate')}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(t("fraud_probability"), f"{prob:.2f}%")
                    with col2:
                        st.metric(t("risk_level"), result['risk_level'])
                    with col3:
                        st.metric(t("threshold"), f"{threshold:.2f}")
                    with col4:
                        st.metric(t("prediction"), f"🚨 {t('fraud')}" if is_fraud else f"✅ {t('normal')}")
                    
                    st.markdown("---")
                    
                    # Top Features
                    st.subheader(t("why_result"))
                    
                    if result['top_features']:
                        features_df = pd.DataFrame(result['top_features'])
                        features_df['contribution'] = features_df['importance'] * features_df['value']
                        features_df = features_df.sort_values('contribution', ascending=False, key=abs)
                        
                        for idx, row in features_df.head(10).iterrows():
                            feat_name = row['feature']
                            importance = row['importance']
                            value = row['value']
                            contribution = row['contribution']
                            
                            risk_direction = t("increases_risk") if contribution > 0 else t("decreases_risk")
                            
                            st.markdown(f"""
                            <div class="feature-item">
                                <strong>#{idx+1}. {feat_name}</strong><br>
                                {risk_direction} {t('fraud_risk')} | {t('importance')}: {importance:.4f} | {t('value')}: {value:.2f}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        fig = px.bar(
                            features_df.head(10),
                            x='contribution',
                            y='feature',
                            orientation='h',
                            title=t("top_features_title"),
                            labels={'contribution': f"{t('contribution')} {t('to_fraud_risk')}", 'feature': 'Feature'},
                            color='contribution',
                            color_continuous_scale='RdBu',
                            color_continuous_midpoint=0
                        )
                        fig.update_layout(
                            yaxis={'categoryorder': 'total ascending', 'gridcolor': 'rgba(0,0,0,0.1)'},
                            title_font_size=20,
                            title_font_color='#1e293b',
                            font=dict(size=13, family='Inter'),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
                    else:
                        st.warning("⚠️ لا توجد خصائص متاحة للعرض")
            except Exception as e:
                st.error(f"❌ {t('error')}: {str(e)}")
                st.info(f"💡 {t('tip')}: {t('make_sure')}")
                import traceback
                with st.expander("تفاصيل الخطأ"):
                    st.code(traceback.format_exc())
        else:
            st.error(t("model_not_available"))
            st.info("💡 تأكد من وجود ملف النموذج: models/random_forest_model.pkl")


# ============================================================================
# Page 3: View All
# ============================================================================
elif page == t("nav_view"):
    st.header(t("nav_view"))
    
    if st.session_state.data is not None and st.session_state.model is not None:
        df = st.session_state.data
        
        st.markdown(f"""
        <div class="info-card">
            <h3>{t('filter_transactions')}</h3>
            <p>Filter and search transactions by various criteria.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Filter & Search Section
        st.subheader("🔍 " + t("filter_search"))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.subheader(t("risk_filter"))
            min_risk = st.slider(t("min_risk"), 0.0, 1.0, 0.0, 0.01)
            max_risk = st.slider(t("max_risk"), 0.0, 1.0, 1.0, 0.01)
        
        with col2:
            st.subheader(t("transaction_type_filter"))
            type_filter = st.multiselect(
                "",
                ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
                default=[],
                label_visibility="collapsed"
            )
        
        with col3:
            st.subheader(t("amount_range"))
            min_amount = st.number_input("Min Amount", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
            max_amount = st.number_input("Max Amount", min_value=0.0, value=float(df['amount'].max()) if 'amount' in df.columns else 1000000.0, step=10000.0, format="%.2f")
        
        with col4:
            st.subheader(t("limit"))
            limit = st.number_input(t("num_transactions"), min_value=10, max_value=1000, value=100, step=10)
        
        st.markdown("---")
        
        # Sample and predict
        sample_size = min(limit * 10, len(df))
        sample_df = df.sample(n=sample_size, random_state=42).copy()
        
        model_features = st.session_state.model.feature_names_in_ if hasattr(st.session_state.model, 'feature_names_in_') else None
        
        if model_features is not None:
            X_sample = sample_df[[col for col in model_features if col in sample_df.columns]].copy()
            
            for feat in model_features:
                if feat not in X_sample.columns:
                    X_sample[feat] = 0
            
            X_sample = X_sample[model_features]
            
            try:
                with st.spinner(t("analyzing_transactions")):
                    probabilities = st.session_state.model.predict_proba(X_sample)[:, 1]
                    sample_df['fraud_probability'] = probabilities
                    sample_df['risk_level'] = sample_df['fraud_probability'].apply(get_risk_level)
                    sample_df['prediction'] = (probabilities >= threshold).astype(int)
                
                filtered_df = sample_df[
                    (sample_df['fraud_probability'] >= min_risk) &
                    (sample_df['fraud_probability'] <= max_risk)
                ].sort_values('fraud_probability', ascending=False).head(limit)
                
                st.markdown(f"### {t('showing')} {len(filtered_df)} {t('of')} {len(sample_df)} {t('transactions')}")
                
                display_df = filtered_df[['amount', 'fraud_probability', 'risk_level', 'prediction']].copy()
                display_df['fraud_probability'] = (display_df['fraud_probability'] * 100).round(2)
                display_df['prediction'] = display_df['prediction'].apply(lambda x: f"🚨 {t('fraud')}" if x == 1 else f"✅ {t('normal')}")
                display_df.columns = [t('amount'), t('fraud_probability'), t('risk_level'), t('prediction')]
                
                st.dataframe(display_df, use_container_width=True, height=400)
                
                csv = filtered_df[['amount', 'fraud_probability', 'risk_level', 'prediction']].to_csv(index=False)
                st.download_button(
                    t("export_csv"),
                    csv,
                    "fraud_risk_scores.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                st.markdown("---")
                
                # Charts with Type Selector
                col_title1, col_chart1, col_title2, col_chart2 = st.columns([2, 1, 2, 1])
                
                with col_title1:
                    st.markdown(f"**{t('risk_distribution')}**")
                with col_chart1:
                    if st.session_state.language == 'ar':
                        chart_opts1 = {"🥧 دائري": "pie", "🍩 حلقي": "donut", "📊 أعمدة": "bar"}
                    else:
                        chart_opts1 = {"🥧 Pie": "pie", "🍩 Donut": "donut", "📊 Bar": "bar"}
                    sel_chart1 = st.selectbox("📊", list(chart_opts1.keys()), label_visibility="collapsed", key="risk_chart")
                
                with col_title2:
                    st.markdown(f"**{t('fraud_vs_normal')}**")
                with col_chart2:
                    if st.session_state.language == 'ar':
                        chart_opts2 = {"📊 أعمدة": "bar", "🥧 دائري": "pie", "📈 أفقي": "bar_h"}
                    else:
                        chart_opts2 = {"📊 Bar": "bar", "🥧 Pie": "pie", "📈 Horizontal": "bar_h"}
                    sel_chart2 = st.selectbox("📊", list(chart_opts2.keys()), label_visibility="collapsed", key="fraud_chart")
                
                risk_counts = filtered_df['risk_level'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    colors1 = ['#dc2626', '#f97316', '#eab308', '#22c55e']
                    chart_t1 = chart_opts1[sel_chart1]
                    
                    if chart_t1 == "pie":
                        fig = px.pie(values=risk_counts.values, names=risk_counts.index, color_discrete_sequence=colors1)
                    elif chart_t1 == "donut":
                        fig = px.pie(values=risk_counts.values, names=risk_counts.index, color_discrete_sequence=colors1, hole=0.5)
                    else:
                        fig = px.bar(x=risk_counts.index, y=risk_counts.values, color=risk_counts.index, color_discrete_sequence=colors1)
                        fig.update_layout(showlegend=False)
                    
                    fig.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20), 
                                      font=dict(size=11), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                with col2:
                    fraud_count = filtered_df['prediction'].sum()
                    normal_count = len(filtered_df) - fraud_count
                    chart_t2 = chart_opts2[sel_chart2]
                    
                    if chart_t2 == "pie":
                        fig = px.pie(values=[normal_count, fraud_count], names=['Normal', 'Fraud'],
                                     color_discrete_sequence=['#22c55e', '#ef4444'])
                    elif chart_t2 == "bar_h":
                        fig = px.bar(y=['Normal', 'Fraud'], x=[normal_count, fraud_count],
                                     color=['Normal', 'Fraud'], color_discrete_map={'Normal': '#22c55e', 'Fraud': '#ef4444'},
                                     orientation='h')
                        fig.update_layout(showlegend=False)
                    else:
                        fig = px.bar(x=['Normal', 'Fraud'], y=[normal_count, fraud_count],
                                     color=['Normal', 'Fraud'], color_discrete_map={'Normal': '#22c55e', 'Fraud': '#ef4444'})
                        fig.update_layout(showlegend=False)
                    
                    fig.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20),
                                      font=dict(size=11), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
                
            except Exception as e:
                st.error(f"{t('error')}: {str(e)}")
        else:
            st.warning(t("model_features_not_available"))
    else:
        st.warning(t("model_or_data_not_loaded"))


# ============================================================================
# Page 4: Performance
# ============================================================================
elif page == t("nav_performance"):
    st.header(t("nav_performance"))
    
    metrics_df = load_cached_metrics()
    
    if metrics_df is not None:
        best_model = metrics_df.loc[metrics_df['pr_auc'].idxmax(), 'model']
        best_pr_auc = metrics_df.loc[metrics_df['pr_auc'].idxmax(), 'pr_auc']
        
        st.markdown(f"""
        <div class="success-box">
            <h3>{t('best_model')}: {best_model}</h3>
            <p>PR-AUC: <strong>{best_pr_auc:.4f}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader(t("model_comparison"))
        st.dataframe(metrics_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                metrics_df,
                x='model',
                y='roc_auc',
                title='ROC-AUC Comparison',
                color='roc_auc',
                color_continuous_scale='Blues',
                height=400
            )
            fig.update_layout(
                title_font_size=18,
                title_font_color='#1e293b',
                font=dict(size=13, family='Inter'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        with col2:
            fig = px.bar(
                metrics_df,
                x='model',
                y='pr_auc',
                title='PR-AUC Comparison',
                color='pr_auc',
                color_continuous_scale='Greens',
                height=400
            )
            fig.update_layout(
                title_font_size=18,
                title_font_color='#1e293b',
                font=dict(size=13, family='Inter'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        st.subheader(t("feature_importance"))
        feature_importance_df = load_feature_importance()
        
        if feature_importance_df is not None:
            top_n = st.slider(t("num_features"), 5, 20, 10)
            top_features = feature_importance_df.nlargest(top_n, 'importance')
            
            fig = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title=f'Top {top_n} {t("most_important")}',
                color='importance',
                color_continuous_scale='Viridis',
                height=450
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending', 'gridcolor': 'rgba(0,0,0,0.1)'},
                title_font_size=20,
                title_font_color='#1e293b',
                font=dict(size=13, family='Inter'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='rgba(0,0,0,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
    else:
        st.warning(t("model_metrics_not_found"))


# ============================================================================
# Page 5: Compare Models
# ============================================================================
elif page == t("compare_models"):
    st.header("🔬 " + t("model_comparison"))
    
    metrics_df = load_cached_metrics()
    
    if metrics_df is not None:
        st.markdown(f"""
        <div class="info-card">
            <h3>📊 {t('model_comparison')}</h3>
            <p>Compare performance of all trained models side by side.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Comparison Table
        st.subheader("📋 " + t("model_comparison"))
        display_df = metrics_df.copy()
        display_df = display_df.round(4)
        st.dataframe(
            display_df,
            use_container_width=True,
            height=300
        )
        
        st.markdown("---")
        
        # Visual Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                metrics_df,
                x='model',
                y='roc_auc',
                title='ROC-AUC Comparison',
                color='roc_auc',
                color_continuous_scale='Blues',
                height=400
            )
            fig.update_layout(
                title_font_size=18,
                title_font_color='#1e293b',
                font=dict(size=13, family='Inter'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        with col2:
            fig = px.bar(
                metrics_df,
                x='model',
                y='pr_auc',
                title='PR-AUC Comparison',
                color='pr_auc',
                color_continuous_scale='Greens',
                height=400
            )
            fig.update_layout(
                title_font_size=18,
                title_font_color='#1e293b',
                font=dict(size=13, family='Inter'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        # Detailed Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.bar(
                metrics_df,
                x='model',
                y='precision',
                title='Precision',
                color='precision',
                color_continuous_scale='Purples',
                height=350
            )
            fig.update_layout(
                title_font_size=16,
                title_font_color='#1e293b',
                font=dict(size=12, family='Inter'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        with col2:
            fig = px.bar(
                metrics_df,
                x='model',
                y='recall',
                title='Recall',
                color='recall',
                color_continuous_scale='Oranges',
                height=350
            )
            fig.update_layout(
                title_font_size=16,
                title_font_color='#1e293b',
                font=dict(size=12, family='Inter'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        with col3:
            fig = px.bar(
                metrics_df,
                x='model',
                y='f1_score',
                title='F1-Score',
                color='f1_score',
                color_continuous_scale='Reds',
                height=350
            )
            fig.update_layout(
                title_font_size=16,
                title_font_color='#1e293b',
                font=dict(size=12, family='Inter'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        # Best Model Summary
        st.markdown("---")
        best_model = metrics_df.loc[metrics_df['pr_auc'].idxmax(), 'model']
        best_pr_auc = metrics_df.loc[metrics_df['pr_auc'].idxmax(), 'pr_auc']
        
        st.markdown(f"""
        <div class="success-box">
            <h3>🏆 Best Model: {best_model}</h3>
            <p>PR-AUC: <strong>{best_pr_auc:.4f}</strong> (Highest)</p>
            <p>This model is recommended for production use.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(t("model_metrics_not_found"))


# ============================================================================
# Page 6: Model Testing
# ============================================================================
elif page == t("model_testing"):
    st.header("🧪 " + t("model_testing"))
    
    st.markdown(f"""
    <div class="info-card">
        <h3>🧪 {t('model_testing')}</h3>
        <p>Test different models with the same transaction to compare predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Selection for Testing
    from utils import get_available_models
    available_models = get_available_models()
    
    if available_models:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(t("select_test_model"))
            test_model_name = st.selectbox(
                "",
                available_models,
                index=0,
                label_visibility="collapsed"
            )
        
        with col2:
            st.subheader("📊 " + t("test_with_sample"))
            use_sample = st.checkbox("Use sample transaction", value=True)
        
        st.markdown("---")
        
        # Transaction Input
        if use_sample:
            st.info("💡 Using sample transaction data")
            sample_data = {
                'step': 100,
                'type': 'CASH_OUT',
                'amount': 50000,
                'oldbalanceOrg': 60000,
                'newbalanceOrig': 10000,
                'oldbalanceDest': 0,
                'newbalanceDest': 50000,
                'deltaOrg': 0,
                'deltaDest': 0
            }
        else:
            st.subheader("📝 " + t("test_transaction"))
            col1, col2 = st.columns(2)
            
            with col1:
                step = st.number_input(t("step"), min_value=0, value=100, step=1)
                transaction_type = st.selectbox(
                    t("type"),
                    ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
                )
                amount = st.number_input(t("amount"), min_value=0.0, value=1000.0, step=100.0, format="%.2f")
            
            with col2:
                oldbalance_org = st.number_input(t("old_balance_sender"), min_value=0.0, value=5000.0, step=100.0, format="%.2f")
                newbalance_orig = st.number_input(t("new_balance_sender"), min_value=0.0, value=4000.0, step=100.0, format="%.2f")
                oldbalance_dest = st.number_input(t("old_balance_receiver"), min_value=0.0, value=0.0, step=100.0, format="%.2f")
                newbalance_dest = st.number_input(t("new_balance_receiver"), min_value=0.0, value=1000.0, step=100.0, format="%.2f")
            
            sample_data = {
                'step': int(step),
                'type': transaction_type,
                'amount': float(amount),
                'oldbalanceOrg': float(oldbalance_org),
                'newbalanceOrig': float(newbalance_orig),
                'oldbalanceDest': float(oldbalance_dest),
                'newbalanceDest': float(newbalance_dest),
                'deltaOrg': 0.0,
                'deltaDest': 0.0
            }
        
        # Test Button
        if st.button("🔍 " + t("test_model"), type="primary", use_container_width=True):
            from utils import load_model_by_name, predict_fraud
            
            test_model = load_model_by_name(test_model_name)
            
            if test_model is not None:
                with st.spinner(f"Testing {test_model_name}..."):
                    result = predict_fraud(test_model, sample_data, threshold)
                
                if result:
                    st.markdown("---")
                    st.subheader("📊 " + t("test_results"))
                    
                    prob = result['probability'] * 100
                    is_fraud = result['prediction'] == 1
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Model", test_model_name)
                    with col2:
                        st.metric(t("fraud_probability"), f"{prob:.2f}%")
                    with col3:
                        st.metric(t("risk_level"), result['risk_level'])
                    with col4:
                        st.metric(t("prediction"), "🚨 FRAUD" if is_fraud else "✅ NORMAL")
                    
                    # Display result box
                    if is_fraud:
                        st.markdown(f"""
                        <div class="fraud-box">
                            <h2>{t('fraud_detected')}</h2>
                            <p style="font-size: 1.5rem;">{t('fraud_probability')}: <strong>{prob:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="normal-box">
                            <h2>{t('normal_transaction')}</h2>
                            <p style="font-size: 1.5rem;">{t('fraud_probability')}: <strong>{prob:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Model Info
                    st.markdown("---")
                    st.subheader("ℹ️ " + t("model_info"))
                    
                    if hasattr(test_model, 'feature_names_in_'):
                        st.info(f"Features: {len(test_model.feature_names_in_)}")
                    if hasattr(test_model, 'n_estimators'):
                        st.info(f"Estimators: {test_model.n_estimators}")
                else:
                    st.error("❌ Test failed")
            else:
                st.error(f"❌ Could not load {test_model_name}")
    else:
        st.warning("⚠️ No models available for testing")


# ============================================================================
# Page 7: Settings - Full Implementation
# ============================================================================
elif page == t("settings_page"):
    if st.session_state.language == 'ar':
        st.markdown("""
        <div class="hero-banner">
            <div class="hero-icon">⚙️</div>
            <div class="hero-text">
                <h1>إعدادات النظام</h1>
                <p>تخصيص وإدارة إعدادات لوحة التحكم</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="hero-banner">
            <div class="hero-icon">⚙️</div>
            <div class="hero-text">
                <h1>System Settings</h1>
                <p>Customize and manage dashboard settings</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Settings Tabs
    if st.session_state.language == 'ar':
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎚️ العتبة والتكاليف", "🤖 النماذج", "📊 البيانات", "🎨 المظهر", "📥 التصدير"])
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎚️ Threshold & Costs", "🤖 Models", "📊 Data", "🎨 Appearance", "📥 Export"])
    
    # Tab 1: Threshold & Cost Settings
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎚️ Decision Threshold" if st.session_state.language == 'en' else "#### 🎚️ عتبة القرار")
            
            new_threshold = st.slider(
                "Threshold" if st.session_state.language == 'en' else "العتبة",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.threshold,
                step=0.01,
                help="Transactions above this probability are flagged as fraud"
            )
            
            if new_threshold != st.session_state.threshold:
                st.session_state.threshold = new_threshold
                st.success(f"✅ Threshold updated to {new_threshold:.2f}")
            
            # Threshold presets
            st.markdown("**Quick Presets:**" if st.session_state.language == 'en' else "**إعدادات سريعة:**")
            preset_col1, preset_col2, preset_col3 = st.columns(3)
            with preset_col1:
                if st.button("Conservative (0.3)", use_container_width=True, key="preset1"):
                    st.session_state.threshold = 0.3
                    st.rerun()
            with preset_col2:
                if st.button("Balanced (0.5)", use_container_width=True, key="preset2"):
                    st.session_state.threshold = 0.5
                    st.rerun()
            with preset_col3:
                if st.button("Aggressive (0.2)", use_container_width=True, key="preset3"):
                    st.session_state.threshold = 0.2
                    st.rerun()
        
        with col2:
            st.markdown("#### 💰 Cost Settings" if st.session_state.language == 'en' else "#### 💰 إعدادات التكلفة")
            
            new_cost_fp = st.number_input(
                "False Positive Cost ($)" if st.session_state.language == 'en' else "تكلفة الإيجابي الكاذب ($)",
                min_value=0,
                max_value=10000,
                value=st.session_state.cost_fp,
                step=10,
                help="Cost of incorrectly flagging a legitimate transaction"
            )
            
            new_cost_fn = st.number_input(
                "False Negative Cost ($)" if st.session_state.language == 'en' else "تكلفة السلبي الكاذب ($)",
                min_value=0,
                max_value=100000,
                value=st.session_state.cost_fn,
                step=100,
                help="Cost of missing a fraudulent transaction"
            )
            
            if new_cost_fp != st.session_state.cost_fp:
                st.session_state.cost_fp = new_cost_fp
            if new_cost_fn != st.session_state.cost_fn:
                st.session_state.cost_fn = new_cost_fn
            
            # Cost ratio display
            if new_cost_fp > 0:
                ratio = new_cost_fn / new_cost_fp
                st.info(f"📊 FN/FP Ratio: {ratio:.1f}x" if st.session_state.language == 'en' else f"📊 نسبة FN/FP: {ratio:.1f}x")
    
    # Tab 2: Model Settings
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🤖 Current Model" if st.session_state.language == 'en' else "#### 🤖 النموذج الحالي")
            
            from utils import get_available_models, load_model_by_name
            available_models = get_available_models()
            
            if available_models:
                current_model = st.session_state.get('current_model_name', available_models[0])
                
                selected_model = st.selectbox(
                    "Select Model" if st.session_state.language == 'en' else "اختر النموذج",
                    available_models,
                    index=available_models.index(current_model) if current_model in available_models else 0,
                    key="settings_model_select"
                )
                
                if selected_model != current_model:
                    with st.spinner("Loading model..." if st.session_state.language == 'en' else "جاري تحميل النموذج..."):
                        st.session_state.model = load_model_by_name(selected_model)
                        st.session_state.current_model_name = selected_model
                        st.success(f"✅ Model changed to {selected_model}")
                
                # Model info
                if st.session_state.model is not None:
                    st.markdown("**Model Info:**" if st.session_state.language == 'en' else "**معلومات النموذج:**")
                    model_type = type(st.session_state.model).__name__
                    st.info(f"Type: {model_type}")
                    if hasattr(st.session_state.model, 'n_estimators'):
                        st.info(f"Estimators: {st.session_state.model.n_estimators}")
                    if hasattr(st.session_state.model, 'feature_names_in_'):
                        st.info(f"Features: {len(st.session_state.model.feature_names_in_)}")
            else:
                st.error("No models found!" if st.session_state.language == 'en' else "لا توجد نماذج!")
        
        with col2:
            st.markdown("#### 📈 Model Performance" if st.session_state.language == 'en' else "#### 📈 أداء النموذج")
            
            metrics = load_cached_metrics()
            if metrics is not None and not metrics.empty:
                for _, row in metrics.iterrows():
                    if row.get('model') == st.session_state.get('current_model_name'):
                        st.metric("ROC-AUC", f"{row.get('roc_auc', 0)*100:.1f}%")
                        st.metric("PR-AUC", f"{row.get('pr_auc', 0)*100:.1f}%")
                        st.metric("F1-Score", f"{row.get('f1_score', 0)*100:.1f}%")
                        break
            else:
                st.warning("Metrics not available" if st.session_state.language == 'en' else "المقاييس غير متاحة")
    
    # Tab 3: Data Settings
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Current Data" if st.session_state.language == 'en' else "#### 📊 البيانات الحالية")
            
            if st.session_state.data is not None:
                df = st.session_state.data
                st.success(f"✅ {len(df):,} transactions loaded")
                st.info(f"Columns: {len(df.columns)}")
                st.info(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                if 'isFraud' in df.columns:
                    fraud_rate = df['isFraud'].mean() * 100
                    st.info(f"Fraud Rate: {fraud_rate:.2f}%")
            else:
                st.warning("No data loaded" if st.session_state.language == 'en' else "لا توجد بيانات")
            
            # Reload data button
            if st.button("🔄 Reload Data", use_container_width=True, key="reload_data"):
                st.cache_data.clear()
                st.session_state.data = load_cached_data()
                st.success("Data reloaded!")
                st.rerun()
        
        with col2:
            st.markdown("#### 📤 Upload Custom Data" if st.session_state.language == 'en' else "#### 📤 رفع بيانات مخصصة")
            
            uploaded_file = st.file_uploader(
                "Upload CSV/Parquet" if st.session_state.language == 'en' else "رفع CSV/Parquet",
                type=['csv', 'parquet'],
                key="data_upload"
            )
            
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        new_data = pd.read_csv(uploaded_file)
                    else:
                        new_data = pd.read_parquet(uploaded_file)
                    
                    st.success(f"✅ Loaded {len(new_data):,} rows")
                    st.dataframe(new_data.head(), use_container_width=True)
                    
                    if st.button("Use This Data", use_container_width=True, key="use_uploaded"):
                        st.session_state.data = new_data
                        st.success("Data updated!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Tab 4: Appearance Settings - Professional
    with tab4:
        # Initialize theme settings
        if 'theme' not in st.session_state:
            st.session_state.theme = 'professional_blue'
        if 'primary_color' not in st.session_state:
            st.session_state.primary_color = '#2563eb'
        if 'bg_color' not in st.session_state:
            st.session_state.bg_color = '#f3f4f6'
        if 'card_style' not in st.session_state:
            st.session_state.card_style = 'modern'
        if 'font_size' not in st.session_state:
            st.session_state.font_size = 'medium'
        if 'animations' not in st.session_state:
            st.session_state.animations = True
        if 'default_chart' not in st.session_state:
            st.session_state.default_chart = 'pie'
        
        # Row 1: Language & Theme
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🌐 Language" if st.session_state.language == 'en' else "#### 🌐 اللغة")
            
            lang_option = st.radio(
                "Select Language" if st.session_state.language == 'en' else "اختر اللغة",
                ["English", "العربية"],
                index=0 if st.session_state.language == 'en' else 1,
                key="settings_lang",
                horizontal=True
            )
            
            new_lang = 'en' if lang_option == "English" else 'ar'
            if new_lang != st.session_state.language:
                st.session_state.language = new_lang
                st.rerun()
        
        with col2:
            st.markdown("#### 🎨 Theme" if st.session_state.language == 'en' else "#### 🎨 السمة")
            
            themes = {
                'Professional Blue': 'professional_blue',
                'Dark Mode': 'dark',
                'Light Clean': 'light',
                'Ocean Green': 'ocean',
                'Sunset Orange': 'sunset',
                'Royal Purple': 'purple',
                'Minimal Gray': 'minimal'
            }
            
            if st.session_state.language == 'ar':
                themes = {
                    'أزرق احترافي': 'professional_blue',
                    'الوضع الداكن': 'dark',
                    'فاتح نظيف': 'light',
                    'أخضر محيطي': 'ocean',
                    'برتقالي غروب': 'sunset',
                    'بنفسجي ملكي': 'purple',
                    'رمادي بسيط': 'minimal'
                }
            
            selected_theme = st.selectbox(
                "Select Theme" if st.session_state.language == 'en' else "اختر السمة",
                list(themes.keys()),
                key="theme_select"
            )
            st.session_state.theme = themes[selected_theme]
        
        st.markdown("---")
        
        # Row 2: Colors
        st.markdown("#### 🎨 Colors" if st.session_state.language == 'en' else "#### 🎨 الألوان")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            primary_color = st.color_picker(
                "Primary Color" if st.session_state.language == 'en' else "اللون الأساسي",
                st.session_state.primary_color,
                key="primary_color_picker"
            )
            if primary_color != st.session_state.primary_color:
                st.session_state.primary_color = primary_color
        
        with col2:
            bg_color = st.color_picker(
                "Background" if st.session_state.language == 'en' else "الخلفية",
                st.session_state.bg_color,
                key="bg_color_picker"
            )
            if bg_color != st.session_state.bg_color:
                st.session_state.bg_color = bg_color
        
        with col3:
            # Preset color schemes
            color_presets = {
                '🔵 Blue': ('#2563eb', '#f3f4f6'),
                '🟢 Green': ('#16a34a', '#f0fdf4'),
                '🟣 Purple': ('#7c3aed', '#faf5ff'),
                '🟠 Orange': ('#ea580c', '#fff7ed'),
                '⚫ Dark': ('#1f2937', '#111827'),
                '⚪ Light': ('#3b82f6', '#ffffff')
            }
            
            preset = st.selectbox(
                "Presets" if st.session_state.language == 'en' else "إعدادات مسبقة",
                list(color_presets.keys()),
                key="color_preset"
            )
            
            if st.button("Apply" if st.session_state.language == 'en' else "تطبيق", key="apply_preset"):
                st.session_state.primary_color, st.session_state.bg_color = color_presets[preset]
                st.rerun()
        
        with col4:
            if st.button("🔄 Reset Colors" if st.session_state.language == 'en' else "🔄 إعادة ضبط", key="reset_colors", use_container_width=True):
                st.session_state.primary_color = '#2563eb'
                st.session_state.bg_color = '#f3f4f6'
                st.rerun()
        
        st.markdown("---")
        
        # Row 3: Typography & Cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔤 Typography" if st.session_state.language == 'en' else "#### 🔤 الخطوط")
            
            font_sizes = {
                'Small': 'small',
                'Medium': 'medium',
                'Large': 'large'
            }
            if st.session_state.language == 'ar':
                font_sizes = {'صغير': 'small', 'متوسط': 'medium', 'كبير': 'large'}
            
            font_size = st.radio(
                "Font Size" if st.session_state.language == 'en' else "حجم الخط",
                list(font_sizes.keys()),
                index=1,
                key="font_size_radio",
                horizontal=True
            )
            st.session_state.font_size = font_sizes[font_size]
        
        with col2:
            st.markdown("#### 🃏 Card Style" if st.session_state.language == 'en' else "#### 🃏 نمط البطاقات")
            
            card_styles = {
                'Modern (Gradient)': 'modern',
                'Flat (Solid)': 'flat',
                'Neumorphism': 'neumorphism',
                'Glass Effect': 'glass',
                'Bordered': 'bordered'
            }
            if st.session_state.language == 'ar':
                card_styles = {
                    'حديث (تدرج)': 'modern',
                    'مسطح': 'flat',
                    'نيومورفيزم': 'neumorphism',
                    'زجاجي': 'glass',
                    'بحدود': 'bordered'
                }
            
            card_style = st.selectbox(
                "Style" if st.session_state.language == 'en' else "النمط",
                list(card_styles.keys()),
                key="card_style_select"
            )
            st.session_state.card_style = card_styles[card_style]
        
        st.markdown("---")
        
        # Row 4: Charts & Animations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Charts" if st.session_state.language == 'en' else "#### 📊 الرسوم البيانية")
            
            chart_types = {'Pie': 'pie', 'Donut': 'donut', 'Bar': 'bar', 'Horizontal': 'horizontal'}
            if st.session_state.language == 'ar':
                chart_types = {'دائري': 'pie', 'حلقي': 'donut', 'شريطي': 'bar', 'أفقي': 'horizontal'}
            
            default_chart = st.selectbox(
                "Default Chart Type" if st.session_state.language == 'en' else "نوع الرسم الافتراضي",
                list(chart_types.keys()),
                key="default_chart_select"
            )
            st.session_state.default_chart = chart_types[default_chart]
        
        with col2:
            st.markdown("#### ✨ Effects" if st.session_state.language == 'en' else "#### ✨ التأثيرات")
            
            animations = st.toggle(
                "Enable Animations" if st.session_state.language == 'en' else "تفعيل الحركات",
                value=st.session_state.animations,
                key="animations_toggle"
            )
            st.session_state.animations = animations
            
            shadows = st.toggle(
                "Enable Shadows" if st.session_state.language == 'en' else "تفعيل الظلال",
                value=True,
                key="shadows_toggle"
            )
        
        st.markdown("---")
        
        # Preview Section
        st.markdown("#### 👁️ Preview" if st.session_state.language == 'en' else "#### 👁️ معاينة")
        
        preview_style = f"""
        <div style="
            background: {st.session_state.bg_color};
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        ">
            <div style="
                background: linear-gradient(135deg, {st.session_state.primary_color} 0%, {st.session_state.primary_color}dd 100%);
                color: white;
                padding: 0.75rem;
                border-radius: 8px;
                text-align: center;
                margin-bottom: 0.5rem;
                box-shadow: 0 2px 8px {st.session_state.primary_color}40;
            ">
                <div style="font-size: 1.5rem;">📊</div>
                <div style="font-size: 1.25rem; font-weight: 700;">1,234</div>
                <div style="font-size: 0.85rem;">Sample Card</div>
            </div>
            <div style="
                background: white;
                padding: 0.5rem;
                border-radius: 6px;
                border: 1px solid #e5e7eb;
                font-size: 0.9rem;
            ">
                ✅ This is how your cards will look with the selected theme.
            </div>
        </div>
        """
        st.markdown(preview_style, unsafe_allow_html=True)
        
        # Apply Theme Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎨 Apply Theme Changes" if st.session_state.language == 'en' else "🎨 تطبيق التغييرات", use_container_width=True, type="primary", key="apply_theme"):
                st.success("✅ Theme applied successfully!" if st.session_state.language == 'en' else "✅ تم تطبيق السمة بنجاح!")
                st.rerun()
    
    # Tab 5: Export Settings
    with tab5:
        st.markdown("#### 📥 Export Data & Reports" if st.session_state.language == 'en' else "#### 📥 تصدير البيانات والتقارير")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Metrics**")
            metrics_df = load_cached_metrics()
            if metrics_df is not None and not metrics_df.empty:
                csv_metrics = metrics_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Metrics CSV",
                    csv_metrics,
                    "model_metrics.csv",
                    "text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No metrics available")
        
        with col2:
            st.markdown("**📈 Recommendations**")
            recommendations = load_cached_recommendations()
            if recommendations:
                import json
                json_str = json.dumps(recommendations, indent=2, ensure_ascii=False)
                st.download_button(
                    "📥 Download Recommendations",
                    json_str,
                    "threshold_recommendations.json",
                    "application/json",
                    use_container_width=True
                )
            else:
                st.warning("No recommendations available")
        
        with col3:
            st.markdown("**📋 Current Data**")
            if st.session_state.data is not None:
                csv_data = st.session_state.data.head(1000).to_csv(index=False)
                st.download_button(
                    "📥 Download Data Sample",
                    csv_data,
                    "data_sample.csv",
                    "text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No data available")
        
        st.markdown("---")
        
        # Reset Options
        st.markdown("#### 🔄 Reset Options" if st.session_state.language == 'en' else "#### 🔄 خيارات إعادة الضبط")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reset Threshold", use_container_width=True, key="reset_threshold"):
                st.session_state.threshold = 0.2
                st.success("Threshold reset to 0.2")
                st.rerun()
        
        with col2:
            if st.button("🔄 Clear Cache", use_container_width=True, key="clear_cache"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Cache cleared!")
                st.rerun()
        
        with col3:
            if st.button("🔄 Reset All", use_container_width=True, type="primary", key="reset_all"):
                st.session_state.threshold = 0.2
                st.session_state.cost_fp = 50
                st.session_state.cost_fn = 5000
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("All settings reset!")
                st.rerun()


# ============================================================================
# Page 8: How It Works
# ============================================================================
elif page == t("nav_how"):
    st.header(t("how_works"))
    
    st.markdown(f"""
    <div class="info-card">
        <h3>{t('how_works')}</h3>
        <p>{t('how_works_desc')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    steps = [
        {"num": "1️⃣", "title": t("step1"), "desc": t("step1_desc")},
        {"num": "2️⃣", "title": t("step2"), "desc": t("step2_desc")},
        {"num": "3️⃣", "title": t("step3"), "desc": t("step3_desc")},
        {"num": "4️⃣", "title": t("step4"), "desc": t("step4_desc")}
    ]
    
    for step in steps:
        st.markdown(f"""
        <div class="step-box">
            <h3>{step['num']} {step['title']}</h3>
            <p>{step['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader(t("key_features"))
    feature_importance_df = load_feature_importance()
    if feature_importance_df is not None:
        top_features = feature_importance_df.nlargest(10, 'importance')
        
        for idx, row in top_features.iterrows():
            st.markdown(f"""
            <div class="feature-item">
                <strong>#{idx+1}. {row['feature']}</strong> - {t('importance')}: {row['importance']:.4f}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader(t("fraud_patterns"))
    
    patterns = [
        {"title": t("pattern1"), "desc": t("pattern1_desc"), "risk": "🔴 Critical"},
        {"title": t("pattern2"), "desc": t("pattern2_desc"), "risk": "🟠 High"},
        {"title": t("pattern3"), "desc": t("pattern3_desc"), "risk": "🟡 Medium"}
    ]
    
    for pattern in patterns:
        st.markdown(f"""
        <div class="warning-box">
            <strong>{pattern['title']}</strong><br>
            {pattern['desc']}<br>
            <strong>{t('risk_level')}:</strong> {pattern['risk']}
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# Footer - Stylish & Compact
# ============================================================================
if st.session_state.language == 'ar':
    footer_title = "لوحة اكتشاف الاحتيال"
else:
    footer_title = "Fraud Detection Dashboard"

st.markdown(f"""
<div class="custom-footer">
    <div class="footer-main">
        <span class="footer-logo">🛡️</span>
        <span class="footer-title">{footer_title}</span>
        <span class="footer-divider"></span>
        <div class="footer-stats">
            <span class="footer-badge highlight">🤖 Random Forest AI</span>
            <span class="footer-badge">✓ 95%</span>
            <span class="footer-badge">✓ PR-AUC 94.95%</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
