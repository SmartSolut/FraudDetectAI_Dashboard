import pandas as pd, argparse, os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--infile',required=True)
    p.add_argument('--modeldir',required=True)
    a=p.parse_args()
    df=pd.read_parquet(a.infile)
    y=df['isFraud'];X=df.drop(columns=['isFraud'])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    models={
        'LogReg':LogisticRegression(max_iter=200,class_weight='balanced'),
        'RF':RandomForestClassifier(n_estimators=200,class_weight='balanced_subsample',random_state=42),
        'XGB':XGBClassifier(eval_metric='logloss',scale_pos_weight=max(1,int((len(y_train)-y_train.sum())/max(1,y_train.sum()))))
    }
    os.makedirs(a.modeldir,exist_ok=True)
    results=[]
    for n,m in models.items():
        m.fit(X_train,y_train)
        pr=average_precision_score(y_test,m.predict_proba(X_test)[:,1])
        roc=roc_auc_score(y_test,m.predict_proba(X_test)[:,1])
        results.append({'model':n,'roc_auc':roc,'pr_auc':pr})
        print(f"{n}: ROC-AUC={roc:.4f}, PR-AUC={pr:.4f}")
    pd.DataFrame(results).to_csv(os.path.join(a.modeldir,'baseline_metrics.csv'),index=False)
    print('Saved metrics to',a.modeldir)
