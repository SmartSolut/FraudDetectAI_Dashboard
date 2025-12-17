import pandas as pd, argparse, matplotlib.pyplot as plt, os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--infile',required=True);a=p.parse_args()
df=pd.read_parquet(a.infile);out=Path('reports/figures');out.mkdir(parents=True,exist_ok=True)
df['amount'].hist(bins=50);plt.title('Amount Distribution');plt.savefig(out/'hist_amount.png');plt.close()
if 'type' in df.columns:
    df['type'].value_counts().plot(kind='bar');plt.title('Transaction Type');plt.savefig(out/'type_distribution.png');plt.close()
    df.groupby('type')['isFraud'].mean().plot(kind='bar');plt.title('Fraud by Type');plt.savefig(out/'fraud_rate_by_type.png');plt.close()
if 'step' in df.columns:
    df.groupby('step')['isFraud'].mean().plot();plt.title('Fraud Over Time');plt.savefig(out/'fraud_over_time.png');plt.close()
print('EDA images saved to',out)
