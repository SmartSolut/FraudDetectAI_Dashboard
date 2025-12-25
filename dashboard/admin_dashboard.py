"""
Admin Dashboard
===============
Admin control panel for managing transfers and notifications
"""

import streamlit as st
import pandas as pd
from datetime import datetime
try:
    from database import (
        get_all_transfers, get_pending_transfers, get_transfer,
        update_transfer_status, execute_transfer, create_notification,
        get_all_notifications, get_unread_notifications, mark_notification_read,
        mark_all_notifications_read, get_statistics, get_account
    )
    from auth import require_admin, get_username
except ImportError:
    from .database import (
        get_all_transfers, get_pending_transfers, get_transfer,
        update_transfer_status, execute_transfer, create_notification,
        get_all_notifications, get_unread_notifications, mark_notification_read,
        mark_all_notifications_read, get_statistics, get_account
    )
    from .auth import require_admin, get_username

def show_admin_dashboard():
    """Show admin dashboard"""
    require_admin()
    
    st.title("👨‍💼 لوحة تحكم المسؤول")
    st.markdown("---")
    
    # Statistics
    stats = get_statistics()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("إجمالي التحويلات", stats["total_transfers"])
    with col2:
        st.metric("⏳ قيد الانتظار", stats["pending_transfers"], 
                 delta=f"{stats['pending_transfers']} معاملة" if stats['pending_transfers'] > 0 else None)
    with col3:
        st.metric("✅ مكتملة", stats["completed_transfers"])
    with col4:
        st.metric("🚨 مشبوهة", stats["suspicious_transfers"])
    with col5:
        unread_count = stats["unread_notifications"]
        st.metric("📢 إشعارات غير مقروءة", unread_count,
                 delta=f"{unread_count} جديد" if unread_count > 0 else None)
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["⏳ التحويلات المعلقة", "📢 الإشعارات", "📜 جميع التحويلات", "📊 الإحصائيات"])
    
    with tab1:
        show_pending_transfers()
    
    with tab2:
        show_notifications()
    
    with tab3:
        show_all_transfers()
    
    with tab4:
        show_statistics()


def show_pending_transfers():
    """Show pending transfers that need admin approval"""
    st.subheader("⏳ التحويلات المعلقة - تحتاج موافقة")
    
    pending = get_pending_transfers()
    
    if not pending:
        st.success("✅ لا توجد تحويلات معلقة")
        return
    
    st.info(f"يوجد {len(pending)} تحويل معلق يحتاج إلى مراجعة")
    
    for transfer in pending:
        with st.container():
            st.markdown("---")
            
            # Get account details
            from_acc = get_account(transfer["from_account"])
            to_acc = get_account(transfer["to_account"])
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"### 🔢 رقم التحويل: {transfer['transfer_id']}")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(f"**من:** {transfer['from_account']}")
                    if from_acc:
                        st.caption(f"👤 {from_acc['owner_name']}")
                        st.caption(f"💰 الرصيد: ${from_acc['balance']:,.2f}")
                
                with col_b:
                    st.markdown(f"**إلى:** {transfer['to_account']}")
                    if to_acc:
                        st.caption(f"👤 {to_acc['owner_name']}")
                        st.caption(f"💰 الرصيد: ${to_acc['balance']:,.2f}")
                
                with col_c:
                    st.markdown(f"**المبلغ:** ${transfer['amount']:,.2f}")
                    st.caption(f"📅 {transfer['created_at'][:19] if transfer.get('created_at') else ''}")
                
                # Risk indicators
                fraud_prob = transfer.get("fraud_probability", 0)
                risk_level = transfer.get("risk_level", "غير محدد")
                
                risk_colors = {
                    "حرج": "🔴",
                    "عالي": "🟠",
                    "متوسط": "🟡",
                    "منخفض": "🟢"
                }
                
                st.markdown(f"**احتمالية الاحتيال:** {fraud_prob:.2f}% | **مستوى الخطورة:** {risk_colors.get(risk_level, '⚪')} {risk_level}")
            
            with col2:
                st.markdown("### الإجراء")
                
                col_approve, col_reject = st.columns(2)
                
                with col_approve:
                    if st.button("✅ موافقة", key=f"approve_{transfer['transfer_id']}", use_container_width=True):
                        approve_transfer(transfer["transfer_id"])
                
                with col_reject:
                    if st.button("❌ رفض", key=f"reject_{transfer['transfer_id']}", use_container_width=True):
                        reject_transfer(transfer["transfer_id"])


def approve_transfer(transfer_id: str):
    """Approve and execute transfer"""
    transfer = get_transfer(transfer_id)
    if not transfer:
        st.error("❌ التحويل غير موجود")
        return
    
    # Update status
    update_transfer_status(transfer_id, "approved", get_username())
    
    # Execute transfer
    if execute_transfer(transfer_id):
        st.success(f"✅ تمت الموافقة وتنفيذ التحويل {transfer_id} بنجاح!")
        st.rerun()
    else:
        st.error("❌ فشل تنفيذ التحويل")


def reject_transfer(transfer_id: str):
    """Reject transfer"""
    transfer = get_transfer(transfer_id)
    if not transfer:
        st.error("❌ التحويل غير موجود")
        return
    
    update_transfer_status(transfer_id, "rejected", get_username())
    
    # Create notification
    message = f"❌ تم رفض التحويل {transfer_id} من قبل المسؤول"
    create_notification(transfer_id, message, "transfer_rejected")
    
    st.success(f"✅ تم رفض التحويل {transfer_id}")
    st.rerun()


def show_notifications():
    """Show notifications"""
    st.subheader("📢 الإشعارات")
    
    notifications = get_all_notifications(limit=100)
    unread = get_unread_notifications()
    
    if not notifications:
        st.info("لا توجد إشعارات")
        return
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"إجمالي الإشعارات: {len(notifications)} | غير مقروءة: {len(unread)}")
    with col2:
        if st.button("✅ تحديد الكل كمقروء", use_container_width=True):
            mark_all_notifications_read()
            st.rerun()
    
    st.markdown("---")
    
    for notification in notifications:
        is_read = notification.get("read", False)
        notif_type = notification.get("type", "info")
        
        # Notification styling
        if not is_read:
            st.markdown("""
            <div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px; border-right: 4px solid #2196f3; margin-bottom: 10px;'>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color: #f5f5f5; padding: 15px; border-radius: 10px; border-right: 4px solid #9e9e9e; margin-bottom: 10px; opacity: 0.7;'>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            if notif_type == "fraud_alert":
                st.markdown(f"### 🚨 {notification['message']}")
            elif notif_type == "transfer_rejected":
                st.markdown(f"### ❌ {notification['message']}")
            else:
                st.markdown(f"### ℹ️ {notification['message']}")
            
            created_at = notification.get("created_at", "")
            if created_at:
                st.caption(f"📅 {created_at[:19]}")
        
        with col2:
            if not is_read:
                if st.button("✅ قرأت", key=f"read_{notification['notification_id']}", use_container_width=True):
                    mark_notification_read(notification["notification_id"])
                    st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)


def show_all_transfers():
    """Show all transfers with filters"""
    st.subheader("📜 جميع التحويلات")
    
    transfers = get_all_transfers(limit=200)
    
    if not transfers:
        st.info("لا توجد تحويلات")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "تصفية حسب الحالة",
            ["الكل", "قيد الانتظار", "موافق عليها", "مرفوضة", "مكتملة"],
            key="admin_status_filter"
        )
    with col2:
        risk_filter = st.selectbox(
            "تصفية حسب الخطورة",
            ["الكل", "حرج", "عالي", "متوسط", "منخفض"],
            key="admin_risk_filter"
        )
    with col3:
        search_term = st.text_input("🔍 بحث", key="admin_search")
    
    # Apply filters
    filtered = transfers
    if status_filter != "الكل":
        status_map = {
            "قيد الانتظار": "pending",
            "موافق عليها": "approved",
            "مرفوضة": "rejected",
            "مكتملة": "completed"
        }
        filtered = [t for t in filtered if t.get("status") == status_map[status_filter]]
    
    if risk_filter != "الكل":
        filtered = [t for t in filtered if t.get("risk_level") == risk_filter]
    
    if search_term:
        filtered = [
            t for t in filtered
            if search_term.lower() in t.get("transfer_id", "").lower() or
            search_term in t.get("from_account", "") or
            search_term in t.get("to_account", "")
        ]
    
    # Display as DataFrame
    if filtered:
        df_data = []
        for transfer in filtered:
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
                "تاريخ الإنشاء": transfer["created_at"][:19] if transfer.get("created_at") else "",
                "موافق عليه من": transfer.get("approved_by", "-")
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("لا توجد نتائج")


def show_statistics():
    """Show detailed statistics"""
    st.subheader("📊 إحصائيات مفصلة")
    
    stats = get_statistics()
    transfers = get_all_transfers()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 إحصائيات التحويلات")
        st.metric("إجمالي التحويلات", stats["total_transfers"])
        st.metric("قيد الانتظار", stats["pending_transfers"])
        st.metric("مكتملة", stats["completed_transfers"])
        st.metric("مرفوضة", stats["rejected_transfers"])
        st.metric("المبلغ الإجمالي المحول", f"${stats['total_amount']:,.2f}")
    
    with col2:
        st.markdown("### 🚨 إحصائيات الأمان")
        st.metric("معاملات مشبوهة", stats["suspicious_transfers"])
        st.metric("إشعارات غير مقروءة", stats["unread_notifications"])
        
        # Calculate percentages
        if stats["total_transfers"] > 0:
            suspicious_pct = (stats["suspicious_transfers"] / stats["total_transfers"]) * 100
            st.metric("نسبة المعاملات المشبوهة", f"{suspicious_pct:.2f}%")
    
    # Status distribution
    if transfers:
        st.markdown("### 📊 توزيع الحالات")
        status_counts = {}
        for transfer in transfers:
            status = transfer.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        status_df = pd.DataFrame({
            "الحالة": list(status_counts.keys()),
            "العدد": list(status_counts.values())
        })
        st.bar_chart(status_df.set_index("الحالة"))

