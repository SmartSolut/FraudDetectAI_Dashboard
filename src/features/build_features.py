"""
Feature Engineering Script
===========================
تجهيز وبناء الخصائص (Features) من البيانات المنظفة
يضيف خصائص جديدة لتحسين أداء النماذج

الخصائص المضافة:
- deltaOrg: الفرق في رصيد المرسل
- deltaDest: الفرق في رصيد المستقبل
- hour_of_day: ساعة اليوم من خطوة الوقت
- is_flagged: علامة الاحتيال المحتملة
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path


def add_time_features(df):
    """إضافة خصائص زمنية"""
    if 'step' in df.columns:
        # كل step = ساعة واحدة
        df['hour_of_day'] = df['step'] % 24
        df['day_of_month'] = (df['step'] // 24) % 30
        df['is_weekend'] = ((df['step'] // 24) % 7).isin([5, 6]).astype(int)
    return df


def add_transaction_features(df):
    """إضافة خصائص متعلقة بالمعاملات"""
    # نسبة المبلغ إلى الرصيد الأصلي
    df['amount_to_oldbalance_ratio'] = np.where(
        df['oldbalanceOrg'] > 0,
        df['amount'] / df['oldbalanceOrg'],
        0
    )
    
    # هل الحساب أصبح فارغاً بعد المعاملة؟
    df['account_emptied'] = (
        (df['oldbalanceOrg'] > 0) & 
        (df['newbalanceOrig'] == 0)
    ).astype(int)
    
    # هل الحساب المستقبل كان فارغاً؟
    df['dest_was_empty'] = (df['oldbalanceDest'] == 0).astype(int)
    
    return df


def add_error_features(df):
    """إضافة خصائص الأخطاء (anomalies)"""
    # خطأ في الرصيد = الفرق بين الرصيد المتوقع والفعلي
    expected_orig = df['oldbalanceOrg'] - df['amount']
    df['error_balance_orig'] = np.abs(expected_orig - df['newbalanceOrig'])
    
    expected_dest = df['oldbalanceDest'] + df['amount']
    df['error_balance_dest'] = np.abs(expected_dest - df['newbalanceDest'])
    
    # علامة وجود خطأ
    df['has_balance_error'] = (
        (df['error_balance_orig'] > 0) | 
        (df['error_balance_dest'] > 0)
    ).astype(int)
    
    return df


def encode_categorical(df):
    """تحويل المتغيرات الفئوية إلى أرقام"""
    if 'type' in df.columns:
        # One-Hot Encoding لنوع المعاملة
        df = pd.get_dummies(df, columns=['type'], prefix='type')
    
    return df


def build_features(df):
    """تطبيق جميع عمليات Feature Engineering"""
    print("Building features...")
    
    # الخصائص الزمنية
    df = add_time_features(df)
    print("  Time features added")
    
    # خصائص المعاملات
    df = add_transaction_features(df)
    print("  Transaction features added")
    
    # خصائص الأخطاء
    df = add_error_features(df)
    print("  Error features added")
    
    # تحويل الفئات
    df = encode_categorical(df)
    print("  Categorical encoding done")
    
    print(f"Final shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    
    return df


if __name__ == '__main__':
    # Default values for easy execution
    DEFAULT_INFILE = 'data/processed/paysim_clean_1M.parquet'
    DEFAULT_OUTFILE = 'data/processed/paysim_features'
    
    parser = argparse.ArgumentParser(
        description='Feature Engineering for Fraud Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default values (press F5)
  python src/features/build_features.py
  
  # Or specify custom paths
  python src/features/build_features.py --infile data/processed/paysim_clean.parquet --outfile data/processed/paysim_features
        """
    )
    parser.add_argument(
        '--infile',
        default=DEFAULT_INFILE,
        help=f'Input parquet file (cleaned data). Default: {DEFAULT_INFILE}'
    )
    parser.add_argument(
        '--outfile',
        default=DEFAULT_OUTFILE,
        help=f'Output file path (without extension). Default: {DEFAULT_OUTFILE}'
    )
    
    args = parser.parse_args()
    
    # Display settings
    print("\n" + "=" * 70)
    print("Configuration:")
    print("=" * 70)
    print(f"Input file: {args.infile}")
    print(f"Output:     {args.outfile}.parquet / .csv")
    print("=" * 70 + "\n")
    
    # قراءة البيانات المنظفة
    print(f"Reading data from: {args.infile}")
    
    try:
        df = pd.read_parquet(args.infile)
        print(f"   Shape before: {df.shape}")
    except FileNotFoundError:
        print(f"   [ERROR] File not found: {args.infile}")
        print("   [INFO] Make sure the file exists or run load_data.py first")
        exit(1)
    except Exception as e:
        print(f"   [ERROR] Error loading file: {e}")
        exit(1)
    
    # بناء الخصائص
    df = build_features(df)
    
    # حفظ النتيجة
    Path(os.path.dirname(args.outfile)).mkdir(parents=True, exist_ok=True)
    
    # حفظ كـ Parquet (للأداء)
    parquet_path = f"{args.outfile}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Features saved to: {parquet_path}")
    
    # حفظ كـ CSV (للمراجعة والتحليل)
    csv_path = f"{args.outfile}.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV version saved to: {csv_path}")
    
    # File size information
    parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)  # MB
    csv_size = os.path.getsize(csv_path) / (1024 * 1024)  # MB
    
    print(f"\nFile sizes:")
    print(f"   Parquet: {parquet_size:.2f} MB")
    print(f"   CSV: {csv_size:.2f} MB")
    print(f"   Compression ratio: {(csv_size/parquet_size):.1f}x")
    
    # عرض إحصائيات الخصائص الجديدة
    print("\n=== Feature Engineering Summary ===")
    print(f"Original columns: {len([col for col in df.columns if col in ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud', 'deltaOrg', 'deltaDest']])}")
    print(f"New features added: {len(df.columns) - 11}")
    print(f"Total columns: {len(df.columns)}")
    
    # عرض الخصائص الجديدة
    new_features = [col for col in df.columns if col not in ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud', 'deltaOrg', 'deltaDest']]
    print(f"\nNew features: {', '.join(new_features)}")
    
    # عرض عينة من البيانات
    print(f"\n=== Sample Data (First 3 rows) ===")
    print(df.head(3).to_string())
    
    print("\n[SUCCESS] Process completed successfully!")
    print(f"[FILES] Files created:")
    print(f"   - {parquet_path} (for processing)")
    print(f"   - {csv_path} (for inspection)")

