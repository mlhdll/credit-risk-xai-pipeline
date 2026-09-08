"""
Credit Risk Prediction & Explainable AI (XAI) Training Pipeline
Performs:
1. Data Cleaning & Domain Hygiene
2. Feature Engineering & Preprocessing (Zero Data Leakage)
3. 5-Fold Stratified Cross-Validation Benchmarking
4. Champion Model Selection (LightGBM) & Evaluation on Held-Out Test Set
5. SHAP (XAI) Interpretability (Global & Local Adverse Action Analysis)
6. Publication-grade Visualizations saved to images/
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    balanced_accuracy_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)
import lightgbm as lgb
import xgboost as xgb
import shap

from pipeline import FinancialFeatureEngineer, get_preprocessor

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette("muted")


import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def run_pipeline():
    print("=" * 70)
    print("[*] CREDIT RISK PREDICTION & XAI PRODUCTION PIPELINE")
    print("=" * 70)

    # 1. Load Data
    data_path = "data/credit_risk_dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}!")

    df = pd.read_csv(data_path)
    print(f"[*] Raw Dataset Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # 2. Domain Data Hygiene (Aykırı Değer Temizliği)
    initial_rows = len(df)
    # Remove biological/domain impossibilities (age > 100 or employment > 60 years)
    df_clean = df[(df['person_age'] <= 100) & (df['person_emp_length'].fillna(0) <= 60)].copy()
    print(f"[*] Data Hygiene: Removed {initial_rows - len(df_clean)} domain anomaly rows (age > 100, emp_len > 60)")

    # 3. Feature Engineering
    engineer = FinancialFeatureEngineer()
    df_feat = engineer.transform(df_clean)
    print(f"[*] Feature Engineering completed: 3 new financial ratios generated")

    # 4. Target & Features Separation
    target_col = 'loan_status'
    X = df_feat.drop(columns=[target_col])
    y = df_feat[target_col]

    print(f"[*] Target Distribution: {y.value_counts().to_dict()} (Default Rate: {y.mean():.2%})")

    # Generate EDA Distribution Chart
    os.makedirs("images", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Target distribution
    sns.countplot(x=y, ax=axes[0], palette=['#2b5c8f', '#d95f02'])
    axes[0].set_title("Target Distribution (0: Paid, 1: Default)", fontsize=13, fontweight='bold')
    axes[0].set_xticklabels(['Non-Default (78.2%)', 'Default (21.8%)'])
    axes[0].set_xlabel("")
    
    # Plot 2: Loan Amount by Status
    sns.boxplot(x=y, y=df_clean['loan_amnt'], ax=axes[1], palette=['#2b5c8f', '#d95f02'])
    axes[1].set_title("Loan Amount Distribution by Status", fontsize=13, fontweight='bold')
    axes[1].set_xticklabels(['Non-Default', 'Default'])
    axes[1].set_ylabel("Loan Amount ($)")
    
    # Plot 3: Loan Percent Income vs Interest Rate
    sample_sub = df_clean.sample(min(2000, len(df_clean)), random_state=42)
    sns.scatterplot(
        data=sample_sub, 
        x='loan_percent_income', 
        y='loan_int_rate', 
        hue='loan_status', 
        alpha=0.6, 
        palette=['#2b5c8f', '#d95f02'],
        ax=axes[2]
    )
    axes[2].set_title("Loan % Income vs Interest Rate", fontsize=13, fontweight='bold')
    axes[2].set_xlabel("Loan Percent of Income")
    axes[2].set_ylabel("Interest Rate (%)")
    
    plt.tight_layout()
    plt.savefig("images/01_eda_distributions.png", dpi=300)
    plt.close()
    print("[+] Saved EDA Chart: images/01_eda_distributions.png")

    # 5. Train-Test Split (Strict Zero-Leakage Hold-Out Test Set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Split Data: Train = {X_train.shape[0]:,} samples | Test = {X_test.shape[0]:,} samples")

    # 6. Build Preprocessor and Transform Training Set
    preprocessor, num_cols, ord_cols, cat_cols = get_preprocessor()
    preprocessor.fit(X_train)

    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    print(f"[*] Preprocessing complete: Processed feature dimension = {X_train_trans.shape[1]}")

    # 7. Model Benchmarking with 5-Fold Stratified Cross-Validation
    print("\n" + "-" * 70)
    print("📊 5-FOLD STRATIFIED CROSS-VALIDATION BENCHMARKING")
    print("-" * 70)

    # Imbalance ratio for scale_pos_weight
    neg_count, pos_count = np.bincount(y_train)
    scale_pos = neg_count / pos_count

    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(scale_pos_weight=scale_pos, eval_metric='logloss', random_state=42, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(scale_pos_weight=scale_pos, random_state=42, verbose=-1, n_jobs=-1)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_results = {}

    for name, model in models.items():
        auc_scores = []
        prauc_scores = []
        f1_scores = []
        recall_scores = []

        for train_idx, val_idx in cv.split(X_train_trans, y_train):
            X_fold_tr, y_fold_tr = X_train_trans[train_idx], y_train.iloc[train_idx]
            X_fold_val, y_fold_val = X_train_trans[val_idx], y_train.iloc[val_idx]

            model.fit(X_fold_tr, y_fold_tr)
            y_prob = model.predict_proba(X_fold_val)[:, 1]
            y_pred = (y_prob >= 0.50).astype(int)

            auc_scores.append(roc_auc_score(y_fold_val, y_prob))
            prauc_scores.append(average_precision_score(y_fold_val, y_prob))
            f1_scores.append(f1_score(y_fold_val, y_pred))
            recall_scores.append(recall_score(y_fold_val, y_pred))

        benchmark_results[name] = {
            "ROC-AUC": f"{np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}",
            "PR-AUC": f"{np.mean(prauc_scores):.4f} ± {np.std(prauc_scores):.4f}",
            "F1-Score": f"{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}",
            "Recall": f"{np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}"
        }
        print(f"| {name:<20} | ROC-AUC: {benchmark_results[name]['ROC-AUC']} | PR-AUC: {benchmark_results[name]['PR-AUC']} | F1: {benchmark_results[name]['F1-Score']} |")

    # 8. Train Champion Model on Full Train Set & Evaluate on Test Set
    champion_model = models["LightGBM"]
    champion_model.fit(X_train_trans, y_train)

    y_test_prob = champion_model.predict_proba(X_test_trans)[:, 1]
    y_test_pred = (y_test_prob >= 0.50).astype(int)

    test_roc_auc = roc_auc_score(y_test, y_test_prob)
    test_pr_auc = average_precision_score(y_test, y_test_prob)
    test_f1 = f1_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)

    print("\n" + "=" * 70)
    print("[*] FINAL CHAMPION (LightGBM) TEST EVALUATION (Zero Data Leakage)")
    print("=" * 70)
    print(f"[*] Test ROC-AUC:            {test_roc_auc:.4f}")
    print(f"[*] Test PR-AUC:             {test_pr_auc:.4f}")
    print(f"[*] Test F1-Score (Default): {test_f1:.4f}")
    print(f"[*] Test Recall (Default):   {test_recall:.4f}")
    print(f"[*] Test Precision:          {test_precision:.4f}")
    print(f"[*] Test Balanced Accuracy:  {test_balanced_acc:.4f}")

    # 9. ROC and Precision-Recall Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    axes[0].plot(fpr, tpr, color='#2b5c8f', lw=2.5, label=f'LightGBM (AUC = {test_roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Chance')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate (Recall)', fontsize=12)
    axes[0].set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold')
    axes[0].legend(loc="lower right")
    
    # PR Curve
    prec, rec, _ = precision_recall_curve(y_test, y_test_prob)
    axes[1].plot(rec, prec, color='#d95f02', lw=2.5, label=f'LightGBM (PR-AUC = {test_pr_auc:.4f})')
    axes[1].axhline(y=y_test.mean(), color='gray', linestyle='--', label=f'Baseline ({y_test.mean():.2%})')
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall (PR) Curve', fontsize=13, fontweight='bold')
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("images/02_roc_pr_curves.png", dpi=300)
    plt.close()
    print("[+] Saved ROC & PR Curves: images/02_roc_pr_curves.png")

    # 10. Confusion Matrix & Financial Cost Matrix Plot
    cm = confusion_matrix(y_test, y_test_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
                xticklabels=['Non-Default', 'Default'], yticklabels=['Non-Default', 'Default'])
    axes[0].set_title("Confusion Matrix (Count)", fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Actual Label')
    axes[0].set_xlabel('Predicted Label')

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1], cbar=False,
                xticklabels=['Non-Default', 'Default'], yticklabels=['Non-Default', 'Default'])
    axes[1].set_title("Normalized Confusion Matrix (Recall)", fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Actual Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig("images/03_confusion_matrix.png", dpi=300)
    plt.close()
    print("[+] Saved Confusion Matrix: images/03_confusion_matrix.png")

    # 11. SHAP Explainability Analysis (XAI)
    print("\n" + "-" * 70)
    print("[*] COMPUTING SHAP VALUES (TreeExplainer)")
    print("-" * 70)

    explainer = shap.TreeExplainer(champion_model)
    # Sample 1500 instances from test set for fast yet highly accurate SHAP distribution
    test_sample_idx = np.random.choice(len(X_test_trans), size=min(1500, len(X_test_trans)), replace=False)
    X_test_sample = X_test_trans[test_sample_idx]
    
    shap_values = explainer(X_test_sample)
    shap_values.feature_names = list(feature_names)

    # SHAP Beeswarm Plot
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.title("SHAP Global Feature Importance (Beeswarm)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("images/04_shap_summary_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Saved SHAP Beeswarm: images/04_shap_summary_beeswarm.png")

    # SHAP Feature Importance Bar Plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=12, show=False)
    plt.title("SHAP Mean |SHAP Value| (Global Feature Impact)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("images/05_shap_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Saved SHAP Bar Plot: images/05_shap_feature_importance.png")

    # SHAP Local Adverse Action Waterfall Plot (High-Risk Applicant vs Low-Risk Applicant)
    high_risk_idx = np.argmax(y_test_prob[test_sample_idx])
    low_risk_idx = np.argmin(y_test_prob[test_sample_idx])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    shap.plots.waterfall(shap_values[high_risk_idx], max_display=10, show=False)
    plt.title(f"Adverse Action Notice: High-Risk Applicant Explanation (Default Prob: {y_test_prob[test_sample_idx][high_risk_idx]:.1%})", 
              fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("images/06_adverse_action_waterfall.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Saved SHAP Adverse Action Waterfall: images/06_adverse_action_waterfall.png")

    # 12. Save Models and Metadata
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {"preprocessor": preprocessor, "model": champion_model, "feature_names": feature_names},
        "models/credit_risk_pipeline.joblib"
    )
    print("[+] Model Pipeline saved: models/credit_risk_pipeline.joblib")

    metrics_payload = {
        "benchmark_cv": benchmark_results,
        "champion_model": "LightGBM",
        "test_metrics": {
            "roc_auc": round(test_roc_auc, 4),
            "pr_auc": round(test_pr_auc, 4),
            "f1_score": round(test_f1, 4),
            "recall": round(test_recall, 4),
            "precision": round(test_precision, 4),
            "balanced_accuracy": round(test_balanced_acc, 4)
        },
        "confusion_matrix": cm.tolist()
    }
    with open("models/metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=4)
    print("[+] Metrics JSON saved: models/metrics.json")
    print("=" * 70)
    print("[*] PIPELINE EXECUTION FINISHED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    np.random.seed(42)
    run_pipeline()
