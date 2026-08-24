"""
Model Training and Validation Pipeline for Hallucination Risk Classifier
Executes:
1. Feature matrix construction from benchmark dataset
2. Baseline comparison (Lexical-Only, NLI-Only, Standard SelfCheckGPT)
3. Proposed Feature-Fused Classifiers (Logistic Regression, Random Forest, MLP)
4. 5-Fold Cross-Validation & Test Set Evaluation (Precision, Recall, F1, ROC-AUC)
5. Feature Importance analysis (Odds ratios & tree importance)
6. Export trained models to models/ for R Shiny & Python dashboard interop
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from src.feature_extraction import FeatureExtractor

def load_data_and_extract_features(dataset_path: str = "data/financial_qa_dataset.json"):
    """
    Loads benchmark QA dataset and extracts all numerical, entity, lexical, and NLI features.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        items = json.load(f)
        
    fe = FeatureExtractor()
    records = []
    
    print(f"[Train/Eval] Extracting features for {len(items)} financial queries...")
    for idx, item in enumerate(items):
        feats = fe.extract_features(item)
        feats["id"] = item["id"]
        feats["category"] = item["category"]
        feats["target"] = item["is_hallucination"]
        feats["hallucination_type"] = item.get("hallucination_type", "None")
        records.append(feats)
        
    df = pd.DataFrame(records)
    print(f"[Train/Eval] Feature matrix shape: {df.shape}")
    return df, items

def evaluate_classifier(model, X_train, y_train, X_test, y_test, model_name: str) -> Dict[str, Any]:
    """
    Trains model and computes evaluation metrics on train and test sets.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = 0.5
        
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    return {
        "model_name": model_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm,
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist()
    }

def run_cross_validation(X, y, feature_names):
    """
    Runs 5-fold cross validation to evaluate model stability.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        
        lr = LogisticRegression(C=1.0, random_state=42)
        lr.fit(X_tr_scaled, y_tr)
        y_pred = lr.predict(X_val_scaled)
        y_proba = lr.predict_proba(X_val_scaled)[:, 1]
        
        cv_results.append({
            "fold": fold + 1,
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "auc": roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) > 1 else 0.5
        })
        
    cv_df = pd.DataFrame(cv_results)
    return {
        "mean_accuracy": float(cv_df["accuracy"].mean()),
        "mean_precision": float(cv_df["precision"].mean()),
        "mean_recall": float(cv_df["recall"].mean()),
        "mean_f1": float(cv_df["f1"].mean()),
        "mean_auc": float(cv_df["auc"].mean()),
        "folds": cv_results
    }

def run_pipeline():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    df, raw_items = load_data_and_extract_features()
    
    feature_cols = [
        "numeric_cv", "numeric_mismatch_rate",
        "entity_jaccard_distance", "entity_mismatch_count",
        "token_jaccard_distance", "length_dispersion",
        "nli_contradiction_mean", "nli_contradiction_max",
        "nli_contradiction_std", "nli_entailment_mean"
    ]
    
    X = df[feature_cols].values
    y = df["target"].values
    
    # 70/30 Stratified Train/Test Split
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.30, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler
    joblib.dump(scaler, "models/feature_scaler.joblib")
    
    # -------------------------------------------------------------
    # 1. BASELINE MODELS
    # -------------------------------------------------------------
    # Baseline 1: Lexical-Only (Decision based on token_jaccard_distance threshold)
    lex_idx = feature_cols.index("token_jaccard_distance")
    y_pred_lex = (X_test[:, lex_idx] > 0.5).astype(int)
    baseline_lex = {
        "model_name": "Baseline 1: Lexical-Only (Token Jaccard)",
        "accuracy": float(accuracy_score(y_test, y_pred_lex)),
        "precision": float(precision_score(y_test, y_pred_lex, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_lex, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred_lex, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, X_test[:, lex_idx])),
        "confusion_matrix": confusion_matrix(y_test, y_pred_lex).tolist()
    }
    
    # Baseline 2: NLI-Only (Decision based on nli_contradiction_mean threshold)
    nli_idx = feature_cols.index("nli_contradiction_mean")
    y_pred_nli = (X_test[:, nli_idx] > 0.4).astype(int)
    baseline_nli = {
        "model_name": "Baseline 2: NLI-Only (Contradiction Mean)",
        "accuracy": float(accuracy_score(y_test, y_pred_nli)),
        "precision": float(precision_score(y_test, y_pred_nli, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_nli, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred_nli, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, X_test[:, nli_idx])),
        "confusion_matrix": confusion_matrix(y_test, y_pred_nli).tolist()
    }
    
    # Baseline 3: Standard SelfCheckGPT (Combination of Lexical + NLI mean threshold)
    sc_score = 0.5 * X_test[:, lex_idx] + 0.5 * X_test[:, nli_idx]
    y_pred_sc = (sc_score > 0.45).astype(int)
    baseline_sc = {
        "model_name": "Baseline 3: Standard SelfCheckGPT",
        "accuracy": float(accuracy_score(y_test, y_pred_sc)),
        "precision": float(precision_score(y_test, y_pred_sc, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_sc, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred_sc, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, sc_score)),
        "confusion_matrix": confusion_matrix(y_test, y_pred_sc).tolist()
    }

    # -------------------------------------------------------------
    # 2. PROPOSED FEATURE-FUSED ML MODELS
    # -------------------------------------------------------------
    # Proposed 1: Logistic Regression (Primary interpretable classifier)
    lr_model = LogisticRegression(C=1.5, penalty="l2", random_state=42)
    eval_lr = evaluate_classifier(lr_model, X_train_scaled, y_train, X_test_scaled, y_test, "Proposed Model 1: Feature-Fused Logistic Regression")
    joblib.dump(lr_model, "models/fusion_classifier.joblib")
    
    # Proposed 2: Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    eval_rf = evaluate_classifier(rf_model, X_train, y_train, X_test, y_test, "Proposed Model 2: Feature-Fused Random Forest")
    joblib.dump(rf_model, "models/rf_classifier.joblib")

    # Proposed 3: Multi-Layer Perceptron (Neural Net)
    mlp_model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
    eval_mlp = evaluate_classifier(mlp_model, X_train_scaled, y_train, X_test_scaled, y_test, "Proposed Model 3: Feature-Fused MLP Neural Net")
    joblib.dump(mlp_model, "models/mlp_classifier.joblib")

    # -------------------------------------------------------------
    # 3. FEATURE IMPORTANCE ANALYSIS
    # -------------------------------------------------------------
    lr_coefs = lr_model.coef_[0]
    rf_importances = rf_model.feature_importances_
    
    feature_importance_list = []
    for f_name, coef, imp in zip(feature_cols, lr_coefs, rf_importances):
        feature_importance_list.append({
            "feature": f_name,
            "logistic_regression_coefficient": float(coef),
            "odds_ratio": float(np.exp(coef)),
            "random_forest_importance": float(imp)
        })
    feature_importance_df = pd.DataFrame(feature_importance_list).sort_values(by="logistic_regression_coefficient", ascending=False)

    # -------------------------------------------------------------
    # 4. CROSS VALIDATION RESULTS
    # -------------------------------------------------------------
    cv_summary = run_cross_validation(X, y, feature_cols)

    # -------------------------------------------------------------
    # 5. CATEGORY-WISE RISK BREAKDOWN
    # -------------------------------------------------------------
    df["predicted_risk"] = lr_model.predict_proba(scaler.transform(X))[:, 1]
    df["predicted_label"] = (df["predicted_risk"] > 0.5).astype(int)
    
    category_metrics = []
    for cat, group in df.groupby("category"):
        cat_acc = accuracy_score(group["target"], group["predicted_label"])
        mean_risk = group["predicted_risk"].mean()
        actual_hallucination_rate = group["target"].mean()
        category_metrics.append({
            "category": cat,
            "sample_count": len(group),
            "actual_hallucination_rate": float(actual_hallucination_rate),
            "mean_predicted_risk_score": float(mean_risk),
            "category_accuracy": float(cat_acc)
        })

    # Save Results Summary JSON
    all_results = {
        "models_evaluation": [
            baseline_lex,
            baseline_nli,
            baseline_sc,
            eval_lr,
            eval_rf,
            eval_mlp
        ],
        "cross_validation": cv_summary,
        "feature_importance": feature_importance_df.to_dict(orient="records"),
        "category_breakdown": category_metrics,
        "feature_cols": feature_cols
    }
    
    with open("models/results_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        
    df.to_csv("data/processed_features_benchmark.csv", index=False)
    
    print("\n" + "="*80)
    print("                      EXPERIMENTAL EVALUATION RESULTS")
    print("="*80)
    res_df = pd.DataFrame([
        {
            "Model": m["model_name"],
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1-Score": f"{m['f1_score']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}"
        }
        for m in all_results["models_evaluation"]
    ])
    print(res_df.to_string(index=False))
    print("\nFeature Importance (Logistic Regression Coefficients & Odds Ratios):")
    print(feature_importance_df.to_string(index=False))
    print("="*80)
    
    return all_results

if __name__ == "__main__":
    run_pipeline()
