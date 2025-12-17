"""
Sample Data for Training
=========================
أخذ عينة من البيانات للتدريب (أول مليون صف)
"""

import pandas as pd
import argparse
import os
from pathlib import Path


def sample_data(input_file, output_file, n_rows=1000000):
    """
    أخذ عينة من البيانات
    
    Args:
        input_file: مسار ملف البيانات الأصلي
        output_file: مسار ملف الحفظ
        n_rows: عدد الصفوف المطلوبة
    """
    print("=" * 70)
    print("[SAMPLE] Starting data sampling process...")
    print("=" * 70)
    
    # قراءة البيانات
    print(f"\n[LOAD] Reading data from: {input_file}")
    
    if input_file.endswith('.parquet'):
        df_full = pd.read_parquet(input_file)
    else:
        df_full = pd.read_csv(input_file)
    
    total_rows = len(df_full)
    print(f"[INFO] Total rows in original file: {total_rows:,}")
    print(f"[INFO] Columns: {df_full.shape[1]}")
    
    # التحقق من العدد المطلوب
    if n_rows > total_rows:
        print(f"[WARNING] Requested {n_rows:,} rows but file has only {total_rows:,}")
        n_rows = total_rows
    
    # أخذ أول n صف
    print(f"\n[SAMPLE] Taking first {n_rows:,} rows...")
    df_sample = df_full.head(n_rows)
    
    # إحصائيات العينة
    print(f"\n[STATS] Sample statistics:")
    print(f"   Rows: {len(df_sample):,}")
    print(f"   Columns: {df_sample.shape[1]}")
    print(f"   Percentage: {(len(df_sample)/total_rows)*100:.2f}% of original data")
    
    # إحصائيات الاحتيال
    if 'isFraud' in df_sample.columns:
        fraud_count = df_sample['isFraud'].sum()
        fraud_rate = df_sample['isFraud'].mean() * 100
        print(f"\n[FRAUD] Fraud statistics:")
        print(f"   Fraud cases: {fraud_count:,}")
        print(f"   Fraud rate: {fraud_rate:.2f}%")
        print(f"   Normal cases: {len(df_sample) - fraud_count:,}")
    
    # حفظ العينة
    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
    
    print(f"\n[SAVE] Saving sample to: {output_file}")
    
    if output_file.endswith('.parquet'):
        df_sample.to_parquet(output_file, index=False)
    else:
        df_sample.to_csv(output_file, index=False)
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"[OK] File saved successfully!")
    print(f"[INFO] File size: {file_size:.2f} MB")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] Sampling completed!")
    print("=" * 70)
    
    return df_sample


def main():
    # القيم الافتراضية (عند التشغيل بزر Run)
    DEFAULT_INPUT = 'data/raw/dataset_FraudDetectAI.csv'
    DEFAULT_OUTPUT = 'data/raw/dataset_FraudDetectAI_1M.csv'
    DEFAULT_ROWS = 1000000
    
    parser = argparse.ArgumentParser(
        description='Sample first N rows from dataset for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default values (press F5)
  python scripts/sample_data.py
  
  # Or specify parameters
  python scripts/sample_data.py --input data/processed/paysim_clean.parquet --output data/processed/paysim_sample_1M.parquet --rows 1000000
        """
    )
    
    parser.add_argument(
        '--input',
        default=DEFAULT_INPUT,
        help=f'Input file path (CSV or Parquet). Default: {DEFAULT_INPUT}'
    )
    
    parser.add_argument(
        '--output',
        default=DEFAULT_OUTPUT,
        help=f'Output file path. Default: {DEFAULT_OUTPUT}'
    )
    
    parser.add_argument(
        '--rows',
        type=int,
        default=DEFAULT_ROWS,
        help=f'Number of rows to sample (default: {DEFAULT_ROWS:,})'
    )
    
    args = parser.parse_args()
    
    # عرض الإعدادات
    print("\n" + "=" * 70)
    print("Configuration:")
    print("=" * 70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Rows:   {args.rows:,}")
    print("=" * 70 + "\n")
    
    # أخذ العينة
    sample_data(args.input, args.output, args.rows)


if __name__ == '__main__':
    main()

