"""
Transfer System
===============
Bank transfer system with fraud detection
"""

import streamlit as st
import pandas as pd
from datetime import datetime
try:
    from database import (
        get_account, get_all_accounts, create_transfer, get_all_transfers,
        update_transfer_status, execute_transfer, create_notification,
        get_statistics
    )
    from utils import (
        build_features_from_transaction, predict_fraud, get_risk_level,
        load_model_by_name
    )
    from auth import require_auth, get_username
except ImportError:
    from .database import (
        get_account, get_all_accounts, create_transfer, get_all_transfers,
        update_transfer_status, execute_transfer, create_notification,
        get_statistics
    )
    from .utils import (
        build_features_from_transaction, predict_fraud, get_risk_level,
        load_model_by_name
    )
    from .auth import require_auth, get_username

# Fraud detection threshold
FRAUD_THRESHOLD = 70.0  # If fraud probability > 70%, require admin approval


def show_transfer_page():
    """Show transfer system page"""
    require_auth()
    
    st.title("🏦 نظام الحوالات المصرفية")
    st.markdown("---")
    
    # Statistics
    stats = get_statistics()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("إجمالي التحويلات", stats["total_transfers"])
    with col2:
        st.metric("قيد الانتظار", stats["pending_transfers"], delta=None)
    with col3:
        st.metric("مكتملة", stats["completed_transfers"])
    with col4:
        st.metric("المبلغ الإجمالي", f"${stats['total_amount']:,.2f}")
    
    st.markdown("---")
    
    # Transfer Form
    st.subheader("📝 إجراء حوالة جديدة")
    
    with st.form("transfer_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            from_account = st.text_input(
                "من حساب (رقم الحساب)",
                placeholder="مثال: 1234567890",
                help="أدخل رقم الحساب المرسل"
            )
            amount = st.number_input(
                "المبلغ ($)",
                min_value=0.01,
                value=1000.0,
                step=100.0,
                format="%.2f"
            )
        
        with col2:
            to_account = st.text_input(
                "إلى حساب (رقم الحساب)",
                placeholder="مثال: 0987654321",
                help="أدخل رقم الحساب المستلم"
            )
            transfer_type = st.selectbox(
                "نوع التحويل",
                ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"]
            )
        
        submitted = st.form_submit_button("🔍 فحص وتحويل", use_container_width=True)
        
        if submitted:
            if not from_account or not to_account:
                st.error("⚠️ يرجى إدخال أرقام الحسابات")
            elif from_account == to_account:
                st.error("⚠️ لا يمكن التحويل لنفس الحساب")
            elif amount <= 0:
                st.error("⚠️ المبلغ يجب أن يكون أكبر من صفر")
            else:
                process_transfer(from_account, to_account, amount, transfer_type)


def process_transfer(from_account: str, to_account: str, amount: float, transfer_type: str):
    """Process transfer with fraud detection"""
    
    # Check if accounts exist
    from_acc = get_account(from_account)
    to_acc = get_account(to_account)
    
    if not from_acc:
        st.error(f"❌ الحساب المرسل غير موجود: {from_account}")
        return
    
    if not to_acc:
        st.error(f"❌ الحساب المستلم غير موجود: {to_account}")
        return
    
    # Check balance
    if from_acc["balance"] < amount:
        st.error(f"❌ الرصيد غير كافي. الرصيد الحالي: ${from_acc['balance']:,.2f}")
        return
    
    # Show processing
    with st.spinner("🔄 جاري فحص المعاملة..."):
        # Build transaction data for fraud detection
        transaction_data = {
            'step': 1,  # Current time step
            'type': transfer_type,
            'amount': amount,
            'oldbalanceOrg': from_acc["balance"],
            'newbalanceOrig': from_acc["balance"] - amount,
            'oldbalanceDest': to_acc["balance"],
            'newbalanceDest': to_acc["balance"] + amount
        }
        
        # Load model and predict
        model = load_model_by_name('Random Forest')
        if model is None:
            st.error("❌ تعذر تحميل النموذج")
            return
        
        # Predict fraud (pass transaction_data dict, not features)
        prediction_result = predict_fraud(model, transaction_data)
        if prediction_result is None:
            st.error("❌ فشل في التنبؤ")
            return
        
        fraud_prob = prediction_result.get('probability', 0.0) * 100  # Convert to percentage
        risk_level = prediction_result.get('risk_level', 'غير محدد')
        
        # Translate risk level to Arabic
        risk_level_ar = {
            "Low": "منخفض",
            "Medium": "متوسط",
            "High": "عالي",
            "Critical": "حرج"
        }.get(risk_level, risk_level)
        
        # Create transfer record
        status = "pending" if fraud_prob > FRAUD_THRESHOLD else "approved"
        transfer = create_transfer(
            from_account=from_account,
            to_account=to_account,
            amount=amount,
            fraud_probability=fraud_prob,
            risk_level=risk_level_ar,
            status=status
        )
        
        # Display results
        st.markdown("---")
        st.subheader("📊 نتائج الفحص")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("احتمالية الاحتيال", f"{fraud_prob:.2f}%")
        with col2:
            risk_color = {
                "منخفض": "🟢",
                "متوسط": "🟡",
                "عالي": "🟠",
                "حرج": "🔴"
            }.get(risk_level_ar, "⚪")
            st.metric("مستوى الخطورة", f"{risk_color} {risk_level_ar}")
        with col3:
            st.metric("رقم التحويل", transfer["transfer_id"])
        
        # Show recommendation
        if fraud_prob > FRAUD_THRESHOLD:
            st.warning(f"⚠️ **معاملة مشبوهة** - احتمالية احتيال: {fraud_prob:.2f}%")
            st.info("📢 تم إرسال إشعار للمسؤول. يرجى انتظار الموافقة.")
            
            # Create notification for admin
            message = (
                f"🚨 معاملة مشبوهة تم اكتشافها!\n"
                f"- رقم التحويل: {transfer['transfer_id']}\n"
                f"- من: {from_account} ({from_acc['owner_name']})\n"
                f"- إلى: {to_account} ({to_acc['owner_name']})\n"
                f"- المبلغ: ${amount:,.2f}\n"
                f"- احتمالية احتيال: {fraud_prob:.2f}%\n"
                f"- مستوى الخطورة: {risk_level}"
            )
            create_notification(transfer["transfer_id"], message, "fraud_alert")
            
        else:
            st.success(f"✅ **معاملة آمنة** - احتمالية احتيال: {fraud_prob:.2f}%")
            
            # Auto-approve and execute immediately for safe transactions
            # Store transfer_id in session state for execution
            transfer_id_key = f"auto_execute_{transfer['transfer_id']}"
            if transfer_id_key not in st.session_state:
                # Execute transfer automatically for safe transactions
                if execute_transfer(transfer["transfer_id"]):
                    st.success("✅ تم تنفيذ التحويل بنجاح!")
                    st.balloons()
                    st.session_state[transfer_id_key] = True
                else:
                    st.error("❌ فشل تنفيذ التحويل")
                    st.session_state[transfer_id_key] = False


def show_transfer_history():
    """Show transfer history"""
    st.subheader("📜 سجل التحويلات")
    
    transfers = get_all_transfers(limit=50)
    
    if not transfers:
        st.info("لا توجد تحويلات حتى الآن")
        return
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "تصفية حسب الحالة",
            ["الكل", "قيد الانتظار", "موافق عليها", "مرفوضة", "مكتملة"]
        )
    with col2:
        risk_filter = st.selectbox(
            "تصفية حسب الخطورة",
            ["الكل", "حرج", "عالي", "متوسط", "منخفض"]
        )
    with col3:
        search_term = st.text_input("🔍 بحث (رقم التحويل أو الحساب)")
    
    # Apply filters
    filtered_transfers = transfers
    if status_filter != "الكل":
        status_map = {
            "قيد الانتظار": "pending",
            "موافق عليها": "approved",
            "مرفوضة": "rejected",
            "مكتملة": "completed"
        }
        filtered_transfers = [t for t in filtered_transfers if t.get("status") == status_map[status_filter]]
    
    if risk_filter != "الكل":
        risk_map = {
            "حرج": "حرج",
            "عالي": "عالي",
            "متوسط": "متوسط",
            "منخفض": "منخفض"
        }
        filtered_transfers = [t for t in filtered_transfers if t.get("risk_level") == risk_map[risk_filter]]
    
    if search_term:
        filtered_transfers = [
            t for t in filtered_transfers
            if search_term.lower() in t.get("transfer_id", "").lower() or
            search_term in t.get("from_account", "") or
            search_term in t.get("to_account", "")
        ]
    
    # Display transfers
    if not filtered_transfers:
        st.info("لا توجد نتائج")
        return
    
    # Create DataFrame for display
    df_data = []
    for transfer in filtered_transfers:
        status_icons = {
            "pending": "⏳",
            "approved": "✅",
            "rejected": "❌",
            "completed": "✔️"
        }
        status_text = {
            "pending": "قيد الانتظار",
            "approved": "موافق عليها",
            "rejected": "مرفوضة",
            "completed": "مكتملة"
        }
        
        df_data.append({
            "رقم التحويل": transfer["transfer_id"],
            "من": transfer["from_account"],
            "إلى": transfer["to_account"],
            "المبلغ": f"${transfer['amount']:,.2f}",
            "احتمالية الاحتيال": f"{transfer['fraud_probability']:.2f}%",
            "مستوى الخطورة": transfer["risk_level"],
            "الحالة": f"{status_icons.get(transfer['status'], '')} {status_text.get(transfer['status'], transfer['status'])}",
            "التاريخ": transfer["created_at"][:10] if transfer.get("created_at") else ""
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

