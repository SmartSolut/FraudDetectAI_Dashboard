"""
Feature Engineering Simple GUI
==============================
Simple GUI using Tkinter for Feature Engineering
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import os
from pathlib import Path
import threading

def add_time_features(df):
    """Add temporal features"""
    if 'step' in df.columns:
        df['hour_of_day'] = df['step'] % 24
        df['day_of_month'] = (df['step'] // 24) % 30
        df['is_weekend'] = ((df['step'] // 24) % 7).isin([5, 6]).astype(int)
    return df

def add_transaction_features(df):
    """Add transaction-related features"""
    df['amount_to_oldbalance_ratio'] = np.where(
        df['oldbalanceOrg'] > 0,
        df['amount'] / df['oldbalanceOrg'],
        0
    )
    
    df['account_emptied'] = (
        (df['oldbalanceOrg'] > 0) & 
        (df['newbalanceOrig'] == 0)
    ).astype(int)
    
    df['dest_was_empty'] = (df['oldbalanceDest'] == 0).astype(int)
    
    return df

def add_error_features(df):
    """Add error detection features (anomalies)"""
    expected_orig = df['oldbalanceOrg'] - df['amount']
    df['error_balance_orig'] = np.abs(expected_orig - df['newbalanceOrig'])
    
    expected_dest = df['oldbalanceDest'] + df['amount']
    df['error_balance_dest'] = np.abs(expected_dest - df['newbalanceDest'])
    
    df['has_balance_error'] = (
        (df['error_balance_orig'] > 0) | 
        (df['error_balance_dest'] > 0)
    ).astype(int)
    
    return df

def encode_categorical(df):
    """Convert categorical variables to numeric"""
    if 'type' in df.columns:
        df = pd.get_dummies(df, columns=['type'], prefix='type')
    return df

def build_features(df):
    """Apply all Feature Engineering operations"""
    df = add_time_features(df)
    df = add_transaction_features(df)
    df = add_error_features(df)
    df = encode_categorical(df)
    return df

class FeatureEngineeringGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Feature Engineering - Fraud Detection")
        self.root.geometry("800x600")
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_name = tk.StringVar(value="paysim_features")
        self.save_parquet = tk.BooleanVar(value=True)
        self.save_csv = tk.BooleanVar(value=True)
        
        self.setup_ui()
    
    def setup_ui(self):
        # Title
        title_label = tk.Label(
            self.root, 
            text="🔧 Feature Engineering - Fraud Detection",
            font=("Arial", 16, "bold"),
            fg="blue"
        )
        title_label.pack(pady=10)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(self.root, text="Settings", padding=10)
        settings_frame.pack(fill="x", padx=10, pady=5)
        
        # Input file selection
        ttk.Label(settings_frame, text="Cleaned Data File:").grid(row=0, column=0, sticky="w", pady=5)
        file_frame = ttk.Frame(settings_frame)
        file_frame.grid(row=0, column=1, sticky="ew", pady=5)
        
        self.file_entry = ttk.Entry(file_frame, textvariable=self.input_file, width=50)
        self.file_entry.pack(side="left", fill="x", expand=True)
        
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side="right", padx=(5, 0))
        
        # Output file name
        ttk.Label(settings_frame, text="Output File Name:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(settings_frame, textvariable=self.output_name, width=30).grid(row=1, column=1, sticky="w", pady=5)
        
        # Save options
        ttk.Label(settings_frame, text="Save Options:").grid(row=2, column=0, sticky="w", pady=5)
        check_frame = ttk.Frame(settings_frame)
        check_frame.grid(row=2, column=1, sticky="w", pady=5)
        
        ttk.Checkbutton(check_frame, text="Parquet", variable=self.save_parquet).pack(side="left")
        ttk.Checkbutton(check_frame, text="CSV", variable=self.save_csv).pack(side="left", padx=(10, 0))
        
        # Control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=20)
        
        ttk.Button(
            button_frame, 
            text="🔍 Check Data", 
            command=self.check_data,
            style="Accent.TButton"
        ).pack(side="left", padx=5)
        
        ttk.Button(
            button_frame, 
            text="⚡ Build Features", 
            command=self.run_feature_engineering,
            style="Accent.TButton"
        ).pack(side="left", padx=5)
        
        ttk.Button(
            button_frame, 
            text="📊 Show Results", 
            command=self.show_results,
            style="Accent.TButton"
        ).pack(side="left", padx=5)
        
        # Results area
        self.results_frame = ttk.LabelFrame(self.root, text="Results", padding=10)
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.results_frame, mode='indeterminate')
        self.progress.pack(fill="x", pady=5)
        
        # Text area
        self.text_area = tk.Text(self.results_frame, height=15, wrap="word")
        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.text_area.yview)
        self.text_area.configure(yscrollcommand=scrollbar.set)
        
        self.text_area.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Data variables
        self.data = None
        self.features_data = None
        
        # Set default file
        self.input_file.set("data/processed/paysim_clean_1M.parquet")
    
    def browse_file(self):
        """Browse data file"""
        file_path = filedialog.askopenfilename(
            title="Select Cleaned Data File",
            filetypes=[
                ("Parquet files", "*.parquet"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.input_file.set(file_path)
    
    def log_message(self, message):
        """Add message to results area"""
        self.text_area.insert(tk.END, message + "\n")
        self.text_area.see(tk.END)
        self.root.update()
    
    def check_data(self):
        """Check data"""
        try:
            self.log_message("🔍 Checking data...")
            
            file_path = self.input_file.get()
            if not file_path:
                messagebox.showerror("Error", "Please select a data file")
                return
            
            # Read data
            if file_path.endswith('.parquet'):
                self.data = pd.read_parquet(file_path)
            else:
                self.data = pd.read_csv(file_path)
            
            self.log_message(f"✅ Data loaded successfully!")
            self.log_message(f"📊 Size: {self.data.shape[0]:,} rows × {self.data.shape[1]} columns")
            self.log_message(f"📋 Columns: {list(self.data.columns)}")
            
            # Show sample
            self.log_message("\n📄 Data sample:")
            self.log_message(str(self.data.head(3).to_string()))
            
        except Exception as e:
            self.log_message(f"❌ Error checking data: {str(e)}")
            messagebox.showerror("Error", f"Error checking data: {str(e)}")
    
    def run_feature_engineering_thread(self):
        """Run Feature Engineering in separate thread"""
        try:
            self.log_message("🚀 Starting feature engineering...")
            self.progress.start()
            
            # Read data if not loaded
            if self.data is None:
                file_path = self.input_file.get()
                if file_path.endswith('.parquet'):
                    self.data = pd.read_parquet(file_path)
                else:
                    self.data = pd.read_csv(file_path)
            
            self.log_message("📊 Original data:")
            self.log_message(f"   Size: {self.data.shape[0]:,} rows × {self.data.shape[1]} columns")
            
            # Build features
            self.log_message("🔧 Building temporal features...")
            self.data = add_time_features(self.data)
            
            self.log_message("💰 Building transaction features...")
            self.data = add_transaction_features(self.data)
            
            self.log_message("⚠️ Building error features...")
            self.data = add_error_features(self.data)
            
            self.log_message("🔢 Encoding categorical data...")
            self.data = encode_categorical(self.data)
            
            # Save results
            self.log_message("💾 Saving results...")
            
            if self.save_parquet.get():
                parquet_path = f"data/processed/{self.output_name.get()}.parquet"
                Path(os.path.dirname(parquet_path)).mkdir(parents=True, exist_ok=True)
                self.data.to_parquet(parquet_path, index=False)
                self.log_message(f"✅ Parquet saved: {parquet_path}")
            
            if self.save_csv.get():
                csv_path = f"data/processed/{self.output_name.get()}.csv"
                self.data.to_csv(csv_path, index=False)
                self.log_message(f"✅ CSV saved: {csv_path}")
            
            # Show results
            self.log_message(f"\n🎉 Completed successfully!")
            self.log_message(f"📊 New size: {self.data.shape[0]:,} rows × {self.data.shape[1]} columns")
            
            # Show new features
            original_cols = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                           'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud', 
                           'deltaOrg', 'deltaDest']
            new_features = [col for col in self.data.columns if col not in original_cols]
            
            self.log_message(f"🆕 New features ({len(new_features)}):")
            for i, feature in enumerate(new_features, 1):
                self.log_message(f"   {i}. {feature}")
            
            self.features_data = self.data.copy()
            
        except Exception as e:
            self.log_message(f"❌ Error building features: {str(e)}")
            messagebox.showerror("Error", f"Error building features: {str(e)}")
        finally:
            self.progress.stop()
    
    def run_feature_engineering(self):
        """Run Feature Engineering"""
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select a data file first")
            return
        
        # Run in separate thread to avoid freezing the UI
        thread = threading.Thread(target=self.run_feature_engineering_thread)
        thread.daemon = True
        thread.start()
    
    def show_results(self):
        """Show results"""
        if self.features_data is not None:
            self.log_message("\n📊 Sample of data with features:")
            self.log_message(str(self.features_data.head(5).to_string()))
        else:
            self.log_message("❌ No data to display. Please run feature engineering first.")

def main():
    root = tk.Tk()
    app = FeatureEngineeringGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
