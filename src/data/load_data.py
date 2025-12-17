"""
Data Loading and Cleaning Script for PaySim Dataset
==================================================
تحميل وتنظيف بيانات PaySim من Kaggle

الوظائف:
- حذف الأعمدة الحساسة (nameOrig, nameDest)
- إزالة القيم السالبة أو غير المنطقية
- إنشاء أعمدة جديدة (deltaOrg, deltaDest)
- تحويل الأعمدة النصية إلى فئات
- حفظ بصيغتي Parquet و CSV
"""

import pandas as pd
import argparse
import os
import numpy as np
from pathlib import Path


def remove_sensitive_columns(df):
    """حذف الأعمدة الحساسة"""
    print("[SECURITY] Removing sensitive columns...")
    sensitive_cols = ['nameOrig', 'nameDest']
    
    # فحص وجود الأعمدة قبل الحذف
    existing_sensitive = [col for col in sensitive_cols if col in df.columns]
    if existing_sensitive:
        df = df.drop(columns=existing_sensitive)
        print(f"   [OK] Removed: {existing_sensitive}")
    else:
        print("   [INFO] No sensitive columns found")
    
    return df


def remove_invalid_values(df):
    """إزالة القيم السالبة أو غير المنطقية"""
    print("[CLEANING] Removing invalid values...")
    
    initial_shape = df.shape[0]
    
    # إزالة القيم السالبة في المبالغ والأرصدة
    numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                   'oldbalanceDest', 'newbalanceDest']
    
    for col in numeric_cols:
        if col in df.columns:
            # إزالة القيم السالبة
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                print(f"   [WARNING] Found {negative_count} negative values in {col}")
                df = df[df[col] >= 0]
    
    # إزالة القيم غير المنطقية (مبالغ كبيرة جداً)
    if 'amount' in df.columns:
        # إزالة المبالغ التي تزيد عن 10 مليون (قيمة عشوائية للحد الأقصى)
        large_amounts = (df['amount'] > 10_000_000).sum()
        if large_amounts > 0:
            print(f"   [WARNING] Found {large_amounts} extremely large amounts")
            df = df[df['amount'] <= 10_000_000]
    
    # إزالة الصفوف التي تحتوي على قيم NaN
    nan_count = df.isnull().any(axis=1).sum()
    if nan_count > 0:
        print(f"   [WARNING] Found {nan_count} rows with NaN values")
        df = df.dropna()
    
    removed_count = initial_shape - df.shape[0]
    if removed_count > 0:
        print(f"   [OK] Removed {removed_count} invalid rows")
    else:
        print("   [OK] No invalid values found")
    
    return df


def create_balance_deltas(df):
    """إنشاء أعمدة deltaOrg و deltaDest للتحقق من سلامة الأرصدة"""
    print("[BALANCE] Creating balance delta columns...")
    
    # حساب الفرق المتوقع في رصيد المرسل
    # الرصيد الجديد = الرصيد القديم - المبلغ المحول
    expected_orig = df['oldbalanceOrg'] - df['amount']
    df['deltaOrg'] = df['newbalanceOrig'] - expected_orig
    
    # حساب الفرق المتوقع في رصيد المستقبل  
    # الرصيد الجديد = الرصيد القديم + المبلغ المستلم
    expected_dest = df['oldbalanceDest'] + df['amount']
    df['deltaDest'] = df['newbalanceDest'] - expected_dest
    
    # إحصائيات الأخطاء
    org_errors = (df['deltaOrg'] != 0).sum()
    dest_errors = (df['deltaDest'] != 0).sum()
    
    print(f"   [STATS] Balance errors - Origin: {org_errors}, Destination: {dest_errors}")
    print("   [OK] Delta columns created successfully")
    
    return df


def convert_to_categorical(df):
    """تحويل الأعمدة النصية إلى فئات (Categorical)"""
    print("[CATEGORICAL] Converting text columns to categorical...")
    
    # الأعمدة التي يجب تحويلها إلى فئات
    categorical_cols = ['type']
    
    for col in categorical_cols:
        if col in df.columns:
            # تحويل إلى فئة
            df[col] = df[col].astype('category')
            
            # عرض إحصائيات الفئات
            unique_values = df[col].nunique()
            print(f"   [INFO] {col}: {unique_values} unique categories")
            print(f"      Values: {list(df[col].cat.categories)}")
    
    print("   [OK] Categorical conversion completed")
    return df


def add_data_quality_checks(df):
    """إضافة فحوصات جودة البيانات"""
    print("[QUALITY] Performing data quality checks...")
    
    # فحص التوزيع
    if 'isFraud' in df.columns:
        fraud_rate = df['isFraud'].mean() * 100
        print(f"   [STATS] Fraud rate: {fraud_rate:.2f}%")
        
        fraud_count = df['isFraud'].sum()
        total_count = len(df)
        print(f"   [STATS] Fraud cases: {fraud_count:,} out of {total_count:,}")
    
    # فحص أنواع المعاملات
    if 'type' in df.columns:
        type_counts = df['type'].value_counts()
        print(f"   [STATS] Transaction types distribution:")
        for trans_type, count in type_counts.items():
            percentage = (count / len(df)) * 100
            print(f"      {trans_type}: {count:,} ({percentage:.1f}%)")
    
    # فحص المبالغ
    if 'amount' in df.columns:
        print(f"   [STATS] Amount statistics:")
        print(f"      Min: ${df['amount'].min():,.2f}")
        print(f"      Max: ${df['amount'].max():,.2f}")
        print(f"      Mean: ${df['amount'].mean():,.2f}")
        print(f"      Median: ${df['amount'].median():,.2f}")
    
    return df


def clean_data(df):
    """تطبيق جميع عمليات التنظيف"""
    print("[PROCESS] Starting data cleaning process...")
    print("=" * 60)
    
    # 1. حذف الأعمدة الحساسة
    df = remove_sensitive_columns(df)
    
    # 2. إزالة القيم غير الصحيحة
    df = remove_invalid_values(df)
    
    # 3. إنشاء أعمدة الأرصدة
    df = create_balance_deltas(df)
    
    # 4. تحويل إلى فئات
    df = convert_to_categorical(df)
    
    # 5. فحوصات الجودة
    df = add_data_quality_checks(df)
    
    print("=" * 60)
    print("[SUCCESS] Data cleaning completed successfully!")
    return df


def save_data(df, parquet_path, csv_path):
    """حفظ البيانات بصيغتي Parquet و CSV"""
    print("[SAVE] Saving cleaned data...")
    
    # إنشاء المجلدات إذا لم تكن موجودة
    Path(os.path.dirname(parquet_path)).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(csv_path)).mkdir(parents=True, exist_ok=True)
    
    # حفظ بصيغة Parquet (للأداء)
    df.to_parquet(parquet_path, index=False)
    print(f"   [OK] Parquet saved: {parquet_path}")
    
    # حفظ بصيغة CSV (للعرض على المشرفة)
    df.to_csv(csv_path, index=False)
    print(f"   [OK] CSV saved: {csv_path}")
    
    # معلومات الملفات
    parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)  # MB
    csv_size = os.path.getsize(csv_path) / (1024 * 1024)  # MB
    
    print(f"   [INFO] File sizes:")
    print(f"      Parquet: {parquet_size:.2f} MB")
    print(f"      CSV: {csv_size:.2f} MB")
    print(f"      Compression ratio: {(csv_size/parquet_size):.1f}x")


def main():
    """البرنامج الرئيسي"""
    # القيم الافتراضية (عند التشغيل بزر Run)
    DEFAULT_CSV = 'data/raw/dataset_FraudDetectAI_1M.csv'
    DEFAULT_OUT = 'data/processed/paysim_clean_1M'
    
    parser = argparse.ArgumentParser(
        description='Load and clean PaySim dataset for fraud detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default values (press F5) - uses 1M sample
  python src/data/load_data.py
  
  # Or specify parameters for full dataset
  python src/data/load_data.py --csv data/raw/dataset_FraudDetectAI.csv --out data/processed/paysim_clean
        """
    )
    
    parser.add_argument(
        '--csv',
        default=DEFAULT_CSV,
        help=f'Path to input CSV file (PaySim dataset). Default: {DEFAULT_CSV}'
    )
    
    parser.add_argument(
        '--out',
        default=DEFAULT_OUT,
        help=f'Output file path (without extension). Default: {DEFAULT_OUT}'
    )
    
    args = parser.parse_args()
    
    # عرض الإعدادات
    print("\n" + "=" * 70)
    print("Configuration:")
    print("=" * 70)
    print(f"Input CSV: {args.csv}")
    print(f"Output:    {args.out}.parquet / .csv")
    print("=" * 70 + "\n")
    
    # قراءة البيانات
    print("Loading PaySim dataset...")
    print(f"   Source: {args.csv}")
    
    try:
        df = pd.read_csv(args.csv)
        print(f"   [OK] Loaded successfully: {df.shape[0]:,} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print(f"   [ERROR] File not found: {args.csv}")
        print("   [INFO] Make sure the file exists in data/raw/ directory")
        return
    except Exception as e:
        print(f"   [ERROR] Error loading file: {e}")
        return
    
    # تنظيف البيانات
    df_cleaned = clean_data(df)
    
    # تحديد مسارات الملفات
    parquet_path = f"{args.out}.parquet"
    csv_path = f"{args.out}.csv"
    
    # حفظ البيانات
    save_data(df_cleaned, parquet_path, csv_path)
    
    print("\n[SUCCESS] Process completed successfully!")
    print(f"[FILES] Files created:")
    print(f"   - {parquet_path} (for processing)")
    print(f"   - {csv_path} (for inspection)")


if __name__ == '__main__':
    # فحص إذا كان يتم التشغيل من VS Code أو بيئة تفاعلية
    import sys
    
    # إذا لم تكن هناك معاملات سطر الأوامر، استخدم القيم الافتراضية
    if len(sys.argv) == 1:
        print("[INFO] No arguments provided, using default values...")
        print("[INFO] To use custom paths, run:")
        print("       python src/data/load_data.py --csv your_file.csv --out your_output")
        print()
    
    main()
