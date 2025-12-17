"""
Split Large Dataset into Smaller Parts
=======================================
تقسيم البيانات الكبيرة إلى ملفات أصغر لسهولة الفتح في Excel
"""

import pandas as pd
import argparse
import os
from pathlib import Path


def split_data(input_file, output_dir, num_parts=2, max_rows_per_file=1000000):
    """
    تقسيم البيانات إلى ملفات أصغر
    
    Args:
        input_file: مسار ملف البيانات الأصلي
        output_dir: مجلد حفظ الملفات المقسمة
        num_parts: عدد الأجزاء المطلوبة
        max_rows_per_file: الحد الأقصى للصفوف في كل ملف
    """
    print("=" * 70)
    print("[SPLIT] Starting data splitting process...")
    print("=" * 70)
    
    # قراءة البيانات
    print(f"\n[LOAD] Reading data from: {input_file}")
    
    if input_file.endswith('.parquet'):
        df = pd.read_parquet(input_file)
    else:
        df = pd.read_csv(input_file)
    
    total_rows = len(df)
    print(f"[INFO] Total rows: {total_rows:,}")
    print(f"[INFO] Total columns: {df.shape[1]}")
    
    # إنشاء المجلد
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # حساب عدد الصفوف في كل جزء
    rows_per_part = total_rows // num_parts
    print(f"\n[CALC] Rows per part: {rows_per_part:,}")
    
    # التقسيم
    print(f"\n[SPLIT] Splitting into {num_parts} parts...")
    print("-" * 70)
    
    for i in range(num_parts):
        start_idx = i * rows_per_part
        
        # الجزء الأخير يحتوي على الصفوف المتبقية
        if i == num_parts - 1:
            end_idx = total_rows
        else:
            end_idx = (i + 1) * rows_per_part
        
        # قص البيانات
        df_part = df.iloc[start_idx:end_idx]
        
        # اسم الملف
        output_file = os.path.join(output_dir, f'paysim_clean_part{i+1}.csv')
        
        # حفظ
        df_part.to_csv(output_file, index=False)
        
        print(f"[OK] Part {i+1}/{num_parts}")
        print(f"     File: {output_file}")
        print(f"     Rows: {len(df_part):,} (from {start_idx:,} to {end_idx:,})")
        print(f"     Size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        print()
    
    print("=" * 70)
    print(f"[SUCCESS] Data split completed!")
    print(f"[INFO] Files saved in: {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Split large dataset into smaller parts for easy viewing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split into 2 parts
  python scripts/split_data.py --input data/processed/paysim_clean.csv --output data/processed/split --parts 2
  
  # Split into 7 parts (each part ~1M rows for Excel)
  python scripts/split_data.py --input data/processed/paysim_clean.csv --output data/processed/split --parts 7
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input file path (CSV or Parquet)'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for split files'
    )
    
    parser.add_argument(
        '--parts',
        type=int,
        default=2,
        help='Number of parts to split into (default: 2)'
    )
    
    args = parser.parse_args()
    
    # التقسيم
    split_data(args.input, args.output, args.parts)


if __name__ == '__main__':
    main()

