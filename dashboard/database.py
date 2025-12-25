"""
Database System (Simulated)
============================
Simple file-based database for accounts, transfers, and notifications
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import os

# Database files
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'transfer_system'
DATA_DIR.mkdir(parents=True, exist_ok=True)

ACCOUNTS_FILE = DATA_DIR / 'accounts.json'
TRANSFERS_FILE = DATA_DIR / 'transfers.json'
NOTIFICATIONS_FILE = DATA_DIR / 'notifications.json'

# Initialize default admin account
DEFAULT_ADMIN = {
    "username": "admin",
    "password": "admin123",  # In production, use hashed passwords
    "role": "admin",
    "name": "System Administrator"
}

# Initialize default user accounts
DEFAULT_ACCOUNTS = {
    "accounts": [
        {
            "account_id": "ACC001",
            "account_number": "1234567890",
            "balance": 100000.0,
            "owner_name": "أحمد محمد",
            "status": "active"
        },
        {
            "account_id": "ACC002",
            "account_number": "0987654321",
            "balance": 50000.0,
            "owner_name": "فاطمة علي",
            "status": "active"
        },
        {
            "account_id": "ACC003",
            "account_number": "1122334455",
            "balance": 75000.0,
            "owner_name": "محمد خالد",
            "status": "active"
        }
    ]
}


def init_database():
    """Initialize database files if they don't exist"""
    if not ACCOUNTS_FILE.exists():
        with open(ACCOUNTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_ACCOUNTS, f, ensure_ascii=False, indent=2)
    
    if not TRANSFERS_FILE.exists():
        with open(TRANSFERS_FILE, 'w', encoding='utf-8') as f:
            json.dump({"transfers": []}, f, ensure_ascii=False, indent=2)
    
    if not NOTIFICATIONS_FILE.exists():
        with open(NOTIFICATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump({"notifications": []}, f, ensure_ascii=False, indent=2)


def load_json(file_path: Path) -> dict:
    """Load JSON file"""
    if not file_path.exists():
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def save_json(file_path: Path, data: dict):
    """Save JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving {file_path}: {e}")


# Account Management
def get_account(account_number: str) -> Optional[Dict]:
    """Get account by account number"""
    data = load_json(ACCOUNTS_FILE)
    accounts = data.get("accounts", [])
    for account in accounts:
        if account.get("account_number") == account_number:
            return account
    return None


def get_all_accounts() -> List[Dict]:
    """Get all accounts"""
    data = load_json(ACCOUNTS_FILE)
    return data.get("accounts", [])


def update_account_balance(account_number: str, new_balance: float):
    """Update account balance"""
    data = load_json(ACCOUNTS_FILE)
    accounts = data.get("accounts", [])
    for account in accounts:
        if account.get("account_number") == account_number:
            account["balance"] = new_balance
            save_json(ACCOUNTS_FILE, data)
            return True
    return False


def create_account(account_number: str, initial_balance: float, owner_name: str) -> bool:
    """Create new account"""
    data = load_json(ACCOUNTS_FILE)
    accounts = data.get("accounts", [])
    
    # Check if account exists
    if get_account(account_number):
        return False
    
    new_account = {
        "account_id": f"ACC{len(accounts) + 1:03d}",
        "account_number": account_number,
        "balance": initial_balance,
        "owner_name": owner_name,
        "status": "active"
    }
    accounts.append(new_account)
    data["accounts"] = accounts
    save_json(ACCOUNTS_FILE, data)
    return True


# Transfer Management
def create_transfer(from_account: str, to_account: str, amount: float, 
                   fraud_probability: float, risk_level: str, status: str = "pending") -> Dict:
    """Create new transfer record"""
    data = load_json(TRANSFERS_FILE)
    transfers = data.get("transfers", [])
    
    transfer = {
        "transfer_id": f"TXN{len(transfers) + 1:06d}",
        "from_account": from_account,
        "to_account": to_account,
        "amount": amount,
        "fraud_probability": fraud_probability,
        "risk_level": risk_level,
        "status": status,  # pending, approved, rejected, completed
        "created_at": datetime.now().isoformat(),
        "approved_at": None,
        "approved_by": None
    }
    
    transfers.append(transfer)
    data["transfers"] = transfers
    save_json(TRANSFERS_FILE, data)
    
    return transfer


def get_transfer(transfer_id: str) -> Optional[Dict]:
    """Get transfer by ID"""
    data = load_json(TRANSFERS_FILE)
    transfers = data.get("transfers", [])
    for transfer in transfers:
        if transfer.get("transfer_id") == transfer_id:
            return transfer
    return None


def get_all_transfers(limit: int = 100) -> List[Dict]:
    """Get all transfers (most recent first)"""
    data = load_json(TRANSFERS_FILE)
    transfers = data.get("transfers", [])
    # Sort by created_at descending
    transfers.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return transfers[:limit]


def get_pending_transfers() -> List[Dict]:
    """Get all pending transfers"""
    transfers = get_all_transfers()
    return [t for t in transfers if t.get("status") == "pending"]


def update_transfer_status(transfer_id: str, status: str, approved_by: str = None):
    """Update transfer status"""
    data = load_json(TRANSFERS_FILE)
    transfers = data.get("transfers", [])
    for transfer in transfers:
        if transfer.get("transfer_id") == transfer_id:
            transfer["status"] = status
            if status in ["approved", "rejected"]:
                transfer["approved_at"] = datetime.now().isoformat()
                transfer["approved_by"] = approved_by
            save_json(TRANSFERS_FILE, data)
            return True
    return False


def execute_transfer(transfer_id: str) -> bool:
    """Execute approved transfer (update account balances)"""
    transfer = get_transfer(transfer_id)
    if not transfer or transfer.get("status") != "approved":
        return False
    
    from_account = get_account(transfer["from_account"])
    to_account = get_account(transfer["to_account"])
    
    if not from_account or not to_account:
        return False
    
    amount = transfer["amount"]
    
    # Check if sender has sufficient balance
    if from_account["balance"] < amount:
        return False
    
    # Update balances
    update_account_balance(from_account["account_number"], from_account["balance"] - amount)
    update_account_balance(to_account["account_number"], to_account["balance"] + amount)
    
    # Update transfer status
    update_transfer_status(transfer_id, "completed")
    
    return True


# Notification Management
def create_notification(transfer_id: str, message: str, notification_type: str = "fraud_alert") -> Dict:
    """Create new notification"""
    data = load_json(NOTIFICATIONS_FILE)
    notifications = data.get("notifications", [])
    
    notification = {
        "notification_id": f"NOT{len(notifications) + 1:06d}",
        "transfer_id": transfer_id,
        "message": message,
        "type": notification_type,
        "read": False,
        "created_at": datetime.now().isoformat()
    }
    
    notifications.append(notification)
    data["notifications"] = notifications
    save_json(NOTIFICATIONS_FILE, data)
    
    return notification


def get_all_notifications(limit: int = 50) -> List[Dict]:
    """Get all notifications (most recent first)"""
    data = load_json(NOTIFICATIONS_FILE)
    notifications = data.get("notifications", [])
    notifications.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return notifications[:limit]


def get_unread_notifications() -> List[Dict]:
    """Get unread notifications"""
    notifications = get_all_notifications()
    return [n for n in notifications if not n.get("read", False)]


def mark_notification_read(notification_id: str):
    """Mark notification as read"""
    data = load_json(NOTIFICATIONS_FILE)
    notifications = data.get("notifications", [])
    for notification in notifications:
        if notification.get("notification_id") == notification_id:
            notification["read"] = True
            save_json(NOTIFICATIONS_FILE, data)
            return True
    return False


def mark_all_notifications_read():
    """Mark all notifications as read"""
    data = load_json(NOTIFICATIONS_FILE)
    notifications = data.get("notifications", [])
    for notification in notifications:
        notification["read"] = True
    save_json(NOTIFICATIONS_FILE, data)


# Statistics
def get_statistics() -> Dict:
    """Get system statistics"""
    transfers = get_all_transfers()
    notifications = get_all_notifications()
    
    total_transfers = len(transfers)
    pending_transfers = len([t for t in transfers if t.get("status") == "pending"])
    completed_transfers = len([t for t in transfers if t.get("status") == "completed"])
    rejected_transfers = len([t for t in transfers if t.get("status") == "rejected"])
    
    total_amount = sum([t.get("amount", 0) for t in transfers if t.get("status") == "completed"])
    suspicious_transfers = len([t for t in transfers if t.get("fraud_probability", 0) > 70])
    
    unread_notifications = len([n for n in notifications if not n.get("read", False)])
    
    return {
        "total_transfers": total_transfers,
        "pending_transfers": pending_transfers,
        "completed_transfers": completed_transfers,
        "rejected_transfers": rejected_transfers,
        "total_amount": total_amount,
        "suspicious_transfers": suspicious_transfers,
        "unread_notifications": unread_notifications
    }


# Initialize database on import
init_database()

