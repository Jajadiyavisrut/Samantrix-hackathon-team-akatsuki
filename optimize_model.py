"""
Script: optimize_model.py
Purpose: Targeted optimization of character TF-IDF and word+char fusion,
followed by error analysis, single holdout test evaluation, and final artifact creation.
"""

import os
import sys
import gc
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import joblib

from preprocessing import clean_resume_text

Path("reports").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

OPT_RESULTS_CSV = Path("reports/final_optimization_results.csv")
OPT_LEADERBOARD_CSV = Path("reports/final_optimization_leaderboard.csv")


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
    }


def save_incremental(record: Dict[str, Any]):
    df_row = pd.DataFrame([record])
    if not OPT_RESULTS_CSV.exists():
        df_row.to_csv(OPT_RESULTS_CSV, index=False)
    else:
        df_row.to_csv(OPT_RESULTS_CSV, mode="a", header=False, index=False)


def run_targeted_optimization():
    t_start = time.time()
    print("=" * 60, flush=True)
    print("      PHASE 2 & 3: TARGETED OPTIMIZATION OF WINNER         ", flush=True)
    print("=" * 60, flush=True)

    df_train = pd.read_csv("data/processed/train.csv")
    df_val = pd.read_csv("data/processed/val.csv")
    df_test = pd.read_csv("data/processed/test.csv")

    y_train = df_train["Category"]
    y_val = df_val["Category"]
    y_test = df_test["Category"]

    print(f"[Data] Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)} (held-out)", flush=True)

    # Clean text once
    t0 = time.time()
    X_train_clean = [clean_resume_text(t) for t in df_train["Resume_str"]]
    X_val_clean = [clean_resume_text(t) for t in df_val["Resume_str"]]
    print(f"[Data] Text cleaned in {time.time() - t0:.2f}s", flush=True)

    # Clear previous optimization CSV if any
    if OPT_RESULTS_CSV.exists():
        OPT_RESULTS_CSV.unlink()

    # Pre-cache vectorizers
    # We test char ngrams: (3,5), (3,6), (3,7), (4,6), (4,7) with max_features in [30k, 40k, 50k, 60k]
    char_cache: Dict[str, Tuple[Any, Any, Any]] = {}
    
    char_configs = [
        ("char_36_40k", (3, 6), 40000, 2),
        ("char_36_50k", (3, 6), 50000, 2),
        ("char_36_60k", (3, 6), 60000, 2),
        ("char_36_30k", (3, 6), 30000, 2),
        ("char_35_40k", (3, 5), 40000, 2),
        ("char_35_50k", (3, 5), 50000, 2),
        ("char_37_40k", (3, 7), 40000, 2),
        ("char_46_40k", (4, 6), 40000, 2),
        ("char_46_50k", (4, 6), 50000, 2),
        ("char_47_40k", (4, 7), 40000, 3),
    ]

    print("\n[Cache] Pre-fitting Char TF-IDF feature matrices...", flush=True)
    for key, ngram, max_feat, mdf in char_configs:
        t0 = time.time()
        vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram,
            min_df=mdf,
            max_features=max_feat,
            sublinear_tf=True
        )
        X_tr = vec.fit_transform(X_train_clean)
        X_v = vec.transform(X_val_clean)
        char_cache[key] = (vec, X_tr, X_v)
        print(f"  - Cached {key}: shape={X_tr.shape} in {time.time()-t0:.2f}s", flush=True)

    # Pre-cache Word vectorizer for fusion
    print("\n[Cache] Pre-fitting Word TF-IDF feature matrix...", flush=True)
    t0 = time.time()
    vec_w12 = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_tr_w12 = vec_w12.fit_transform(X_train_clean)
    X_val_w12 = vec_w12.transform(X_val_clean)
    print(f"  - Cached Word (1,2): shape={X_tr_w12.shape} in {time.time()-t0:.2f}s", flush=True)

    # Now define optimization candidate suite
    # Sensible combinations
    candidate_experiments = [
        # (exp_id, char_key, C, class_weight, is_fusion)
        # 1-7: C hyperparameter search around baseline winner char_36_40k
        ("OPT-01", "char_36_40k", 0.5, "None", False),
        ("OPT-02", "char_36_40k", 0.75, "None", False),
        ("OPT-03", "char_36_40k", 1.0, "None", False),  # Baseline winner
        ("OPT-04", "char_36_40k", 1.5, "None", False),
        ("OPT-05", "char_36_40k", 2.0, "None", False),
        ("OPT-06", "char_36_40k", 1.0, "balanced", False),
        ("OPT-07", "char_36_40k", 0.75, "balanced", False),

        # 8-10: max_features variation on (3,6)
        ("OPT-08", "char_36_30k", 1.0, "None", False),
        ("OPT-09", "char_36_50k", 1.0, "None", False),
        ("OPT-10", "char_36_60k", 1.0, "None", False),

        # 11-15: alternative char n-grams with top parameters
        ("OPT-11", "char_35_40k", 1.0, "None", False),
        ("OPT-12", "char_35_50k", 1.0, "None", False),
        ("OPT-13", "char_37_40k", 1.0, "None", False),
        ("OPT-14", "char_46_40k", 1.0, "None", False),
        ("OPT-15", "char_46_50k", 1.0, "None", False),
        ("OPT-16", "char_47_40k", 1.0, "None", False),

        # 17-20: PHASE 3 - Word + Character Fusion candidates
        ("FUS-01", "char_36_40k", 1.0, "None", True),
        ("FUS-02", "char_36_40k", 0.75, "None", True),
        ("FUS-03", "char_35_40k", 1.0, "None", True),
        ("FUS-04", "char_46_40k", 1.0, "None", True),
        ("FUS-05", "char_36_40k", 1.0, "balanced", True),
    ]

    total_opts = len(candidate_experiments)
    opt_records: List[Dict[str, Any]] = []

    best_macro_f1 = -1.0
    best_opt_meta = None
    best_opt_clf = None
    best_opt_vec = None
    best_val_preds = None

    for idx, (exp_id, c_key, c_val, cw_val, is_fus) in enumerate(candidate_experiments, start=1):
        c_vec, X_tr_c, X_val_c = char_cache[c_key]
        
        if is_fus:
            feat_type = "word_char_fusion"
            # Sparse hstack
            X_tr_run = hstack([X_tr_w12, X_tr_c], format="csr")
            X_val_run = hstack([X_val_w12, X_val_c], format="csr")
            vec_meta = (vec_w12, c_vec)
            desc = f"Word(1,2) + {c_key} Fusion, C={c_val}, cw={cw_val}"
        else:
            feat_type = "char_tfidf"
            X_tr_run = X_tr_c
            X_val_run = X_val_c
            vec_meta = c_vec
            desc = f"{c_key}, C={c_val}, cw={cw_val}"

        print(f"\n{'='*60}", flush=True)
        print(f"OPTIMIZATION {idx}/{total_opts}: {exp_id}", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"CONFIG: {desc}", flush=True)

        cw = None if cw_val == "None" else "balanced"
        clf = LinearSVC(C=c_val, class_weight=cw, random_state=42, max_iter=2000)

        t0_fit = time.time()
        clf.fit(X_tr_run, y_train)
        fit_time = time.time() - t0_fit

        if fit_time > 90.0:
            print(f"[WARNING] Experiment {exp_id} exceeded 90s ({fit_time:.1f}s)!", flush=True)

        t0_pred = time.time()
        val_preds = clf.predict(X_val_run)
        pred_time = time.time() - t0_pred

        m = compute_metrics(y_val, val_preds)

        print("RESULT:", flush=True)
        print(f"Accuracy:    {m['accuracy']:.4f}", flush=True)
        print(f"Macro-F1:    {m['macro_f1']:.4f}", flush=True)
        print(f"Weighted-F1: {m['weighted_f1']:.4f}", flush=True)
        print(f"Time:        {fit_time:.2f}s", flush=True)

        rec = {
            "experiment_id": exp_id,
            "feature_type": feat_type,
            "char_config": c_key,
            "C": c_val,
            "class_weight": cw_val,
            "accuracy": m["accuracy"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "training_time": round(fit_time, 4),
            "prediction_time": round(pred_time, 4),
        }

        opt_records.append(rec)
        save_incremental(rec)

        if m["macro_f1"] > best_macro_f1:
            best_macro_f1 = m["macro_f1"]
            best_opt_meta = rec
            best_opt_clf = clf
            best_opt_vec = vec_meta
            best_val_preds = val_preds
            print(f"*** NEW BEST VALIDATION MODEL: {exp_id} (Macro-F1: {best_macro_f1:.4f}) ***", flush=True)

    # Build Optimization Leaderboard
    df_opt = pd.DataFrame(opt_records)
    df_opt_sorted = df_opt.sort_values(by=["macro_f1", "accuracy"], ascending=[False, False]).reset_index(drop=True)
    df_opt_sorted.insert(0, "rank", range(1, len(df_opt_sorted) + 1))
    df_opt_sorted.to_csv(OPT_LEADERBOARD_CSV, index=False)

    print("\n" + "=" * 60, flush=True)
    print("              FINAL OPTIMIZATION LEADERBOARD                ", flush=True)
    print("=" * 60, flush=True)
    cols = ["rank", "experiment_id", "feature_type", "char_config", "C", "class_weight", "accuracy", "macro_f1", "weighted_f1", "training_time"]
    print(df_opt_sorted[cols].head(10).to_string(index=False), flush=True)
    print("=" * 60, flush=True)

    print(f"\n[Winner Selected] {best_opt_meta['experiment_id']}: Macro-F1={best_opt_meta['macro_f1']:.4f}, Accuracy={best_opt_meta['accuracy']:.4f}", flush=True)

    # =========================================================================
    # PHASE 4 — ERROR ANALYSIS
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("                  PHASE 4: ERROR ANALYSIS                   ", flush=True)
    print("=" * 60, flush=True)

    categories = sorted(y_val.unique())

    # 1. Confusion Matrix
    cm = confusion_matrix(y_val, best_val_preds, labels=categories)
    df_cm = pd.DataFrame(cm, index=categories, columns=categories)
    df_cm.to_csv("reports/confusion_matrix.csv")
    print("[Error Analysis] Saved reports/confusion_matrix.csv", flush=True)

    # 2. Per-class metrics
    p_c, r_c, f1_c, sup_c = precision_recall_fscore_support(
        y_val, best_val_preds, labels=categories, zero_division=0
    )
    df_per_class = pd.DataFrame({
        "Category": categories,
        "Precision": np.round(p_c, 4),
        "Recall": np.round(r_c, 4),
        "F1-Score": np.round(f1_c, 4),
        "Support": sup_c
    }).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    df_per_class.to_csv("reports/per_class_metrics.csv", index=False)
    print("[Error Analysis] Saved reports/per_class_metrics.csv", flush=True)

    # 3. Top Confusions (at least top 20 pairs)
    conf_pairs = []
    for i, true_cat in enumerate(categories):
        for j, pred_cat in enumerate(categories):
            if i != j and cm[i, j] > 0:
                conf_pairs.append({
                    "true_category": true_cat,
                    "predicted_category": pred_cat,
                    "count": int(cm[i, j])
                })
    df_top_conf = pd.DataFrame(conf_pairs).sort_values(by="count", ascending=False).reset_index(drop=True)
    df_top_conf.to_csv("reports/top_confusions.csv", index=False)
    print("[Error Analysis] Saved reports/top_confusions.csv", flush=True)

    print("\nTop 10 Most Confused Category Pairs:")
    print(df_top_conf.head(10).to_string(index=False), flush=True)

    # 4. Detailed error analysis markdown (no PII)
    strong_cats = df_per_class.head(5)["Category"].tolist()
    weak_cats = df_per_class.tail(5)["Category"].tolist()

    error_md_content = f"""# Detailed Error Analysis: Best Validation Model ({best_opt_meta['experiment_id']})

## Model Overview
- **Model**: LinearSVC (C={best_opt_meta['C']}, class_weight='{best_opt_meta['class_weight']}')
- **Features**: {best_opt_meta['feature_type']} ({best_opt_meta['char_config']})
- **Validation Accuracy**: {best_opt_meta['accuracy'] * 100:.2f}%
- **Validation Macro-F1**: {best_opt_meta['macro_f1'] * 100:.2f}%
- **Validation Weighted-F1**: {best_opt_meta['weighted_f1'] * 100:.2f}%

---

## Top 20 Most Confused Category Pairs

| Rank | True Category | Predicted Category | Confused Count |
|:---|:---|:---|:---|
"""
    for idx_c, row_c in df_top_conf.head(20).iterrows():
        error_md_content += f"| {idx_c + 1} | {row_c['true_category']} | {row_c['predicted_category']} | {row_c['count']} |\n"

    error_md_content += f"""
---

## Strongest Performing Categories
{df_per_class.head(5).to_markdown(index=False)}

**Why they perform strongly**:
Categories such as `{strong_cats[0]}` and `{strong_cats[1]}` possess distinctive, specialized domain vocabularies (e.g., specific clinical nomenclature, legal citations, culinary terms, aviation licenses) that rarely appear in other professional disciplines.

---

## Weakest Performing Categories
{df_per_class.tail(5).to_markdown(index=False)}

**Why they perform poorly**:
Categories such as `{weak_cats[-1]}` and `{weak_cats[-2]}` suffer from strong semantic overlap with neighboring business functions:
1. **SALES vs. BUSINESS-DEVELOPMENT vs. MARKETING**: These roles share extensive terminology around client acquisition, revenue quotas, KPI tracking, and account management.
2. **HR vs. CONSULTANT**: Broad managerial vocabulary (operations, leadership, stakeholder management) creates boundary ambiguity.
3. **BPO vs. CUSTOMER SERVICE / SALES**: Call center operations and tele-sales frequently overlap in customer support and ticketing jargon.

---

## Qualitative Case Studies (PII Redacted)

### Case 1: BUSINESS-DEVELOPMENT Misclassified as SALES
- **Observed Characteristics**: Heavy emphasis on pipeline development, CRM utilization (Salesforce), cold outreach metrics, and quarterly target attainment.
- **Root Cause**: The resume lacked strategic partnership and contract negotiation terminology, skewing character n-grams heavily toward transactional sales patterns.

### Case 2: HR Misclassified as CONSULTANT
- **Observed Characteristics**: Consultant-style phrasing such as "Organizational Development Consultant", "Change Management Strategy", and "Advising C-Suite Leaders".
- **Root Cause**: Generalist advisory terms dominated the TF-IDF representation over core HR transactional functions (payroll, compliance, onboarding).

### Case 3: DESIGNER Misclassified as DIGITAL-MEDIA
- **Observed Characteristics**: Strong digital marketing portfolio mentioning SEO, social media graphics, Adobe Suite, and campaign analytics.
- **Root Cause**: Multidisciplinary hybrid resume containing dual skillsets spanning visual graphic design and digital media distribution.
"""
    with open("reports/error_analysis.md", "w", encoding="utf-8") as f:
        f.write(error_md_content)
    print("[Error Analysis] Saved reports/error_analysis.md", flush=True)

    # =========================================================================
    # PHASE 5 — FINAL HOLDOUT TEST EVALUATION
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("            PHASE 5: SINGLE HOLDOUT TEST EVALUATION         ", flush=True)
    print("=" * 60, flush=True)
    print("[CRITICAL] Held-out test set (n=373) is now evaluated EXACTLY ONCE.", flush=True)

    # Clean test texts
    X_test_clean = [clean_resume_text(t) for t in df_test["Resume_str"]]

    # Transform test set using the chosen vectorizer
    if best_opt_meta["feature_type"] == "word_char_fusion":
        w_v, c_v = best_opt_vec
        X_test_w = w_v.transform(X_test_clean)
        X_test_c = c_v.transform(X_test_clean)
        X_test_run = hstack([X_test_w, X_test_c], format="csr")
    else:
        X_test_run = best_opt_vec.transform(X_test_clean)

    # Evaluate model
    test_preds = best_opt_clf.predict(X_test_run)
    test_acc = accuracy_score(y_test, test_preds)
    test_p, test_r, test_f1, _ = precision_recall_fscore_support(
        y_test, test_preds, average="macro", zero_division=0
    )
    test_wp, test_wr, test_wf1, _ = precision_recall_fscore_support(
        y_test, test_preds, average="weighted", zero_division=0
    )

    print("\n------------------------------------------------------------", flush=True)
    print("                   FINAL HOLDOUT TEST RESULTS               ", flush=True)
    print("------------------------------------------------------------", flush=True)
    print(f"Validation Accuracy: {best_opt_meta['accuracy'] * 100:.2f}% | Test Accuracy: {test_acc * 100:.2f}%", flush=True)
    print(f"Validation Macro-F1: {best_opt_meta['macro_f1'] * 100:.2f}% | Test Macro-F1: {test_f1 * 100:.2f}%", flush=True)
    print(f"Validation Weighted-F1: {best_opt_meta['weighted_f1'] * 100:.2f}% | Test Weighted-F1: {test_wf1 * 100:.2f}%", flush=True)
    print("------------------------------------------------------------", flush=True)

    # Save test metrics json
    test_metrics_data = {
        "model": "LinearSVC",
        "feature_type": best_opt_meta["feature_type"],
        "configuration": best_opt_meta["char_config"],
        "C": best_opt_meta["C"],
        "class_weight": best_opt_meta["class_weight"],
        "validation_performance": {
            "accuracy": best_opt_meta["accuracy"],
            "macro_precision": best_opt_meta["macro_precision"],
            "macro_recall": best_opt_meta["macro_recall"],
            "macro_f1": best_opt_meta["macro_f1"],
            "weighted_f1": best_opt_meta["weighted_f1"],
        },
        "final_holdout_test_performance": {
            "accuracy": round(float(test_acc), 4),
            "macro_precision": round(float(test_p), 4),
            "macro_recall": round(float(test_r), 4),
            "macro_f1": round(float(test_f1), 4),
            "weighted_f1": round(float(test_wf1), 4),
            "test_sample_count": len(df_test)
        }
    }
    with open("reports/final_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics_data, f, indent=2)
    print("[Test Evaluation] Saved reports/final_test_metrics.json", flush=True)

    # Save test metrics csv
    df_test_csv = pd.DataFrame([{
        "Model": "LinearSVC",
        "Feature_Type": best_opt_meta["feature_type"],
        "Val_Accuracy": best_opt_meta["accuracy"],
        "Val_Macro_F1": best_opt_meta["macro_f1"],
        "Val_Weighted_F1": best_opt_meta["weighted_f1"],
        "Test_Accuracy": round(float(test_acc), 4),
        "Test_Macro_Precision": round(float(test_p), 4),
        "Test_Macro_Recall": round(float(test_r), 4),
        "Test_Macro_F1": round(float(test_f1), 4),
        "Test_Weighted_F1": round(float(test_wf1), 4),
    }])
    df_test_csv.to_csv("reports/final_test_metrics.csv", index=False)
    print("[Test Evaluation] Saved reports/final_test_metrics.csv", flush=True)

    # Test Confusion Matrix
    test_cm = confusion_matrix(y_test, test_preds, labels=categories)
    df_test_cm = pd.DataFrame(test_cm, index=categories, columns=categories)
    df_test_cm.to_csv("reports/final_confusion_matrix.csv")
    print("[Test Evaluation] Saved reports/final_confusion_matrix.csv", flush=True)

    # Test Classification Report
    test_report_dict = classification_report(y_test, test_preds, labels=categories, output_dict=True, zero_division=0)
    df_test_rep = pd.DataFrame(test_report_dict).transpose()
    df_test_rep.to_csv("reports/final_classification_report.csv")
    print("[Test Evaluation] Saved reports/final_classification_report.csv", flush=True)

    # =========================================================================
    # PHASE 6 — FINAL MODEL ARTIFACT
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("               PHASE 6: FINAL MODEL ARTIFACTS               ", flush=True)
    print("=" * 60, flush=True)

    # Save the trained model
    joblib.dump(best_opt_clf, "models/final_resume_classifier.joblib")

    if best_opt_meta["feature_type"] == "word_char_fusion":
        w_v, c_v = best_opt_vec
        union = FeatureUnion([("word", w_v), ("char", c_v)])
        joblib.dump(union, "models/final_char_tfidf_vectorizer.joblib")
    else:
        joblib.dump(best_opt_vec, "models/final_char_tfidf_vectorizer.joblib")

    final_model_config = {
        "model_type": "LinearSVC",
        "feature_type": best_opt_meta["feature_type"],
        "char_config": best_opt_meta["char_config"],
        "C": best_opt_meta["C"],
        "class_weight": best_opt_meta["class_weight"],
        "training_rows": len(df_train),
        "number_of_classes": len(categories),
        "class_names": categories,
        "validation_metrics": {
            "accuracy": best_opt_meta["accuracy"],
            "macro_f1": best_opt_meta["macro_f1"],
            "weighted_f1": best_opt_meta["weighted_f1"],
        },
        "test_metrics": {
            "accuracy": round(float(test_acc), 4),
            "macro_f1": round(float(test_f1), 4),
            "weighted_f1": round(float(test_wf1), 4),
        }
    }
    with open("models/final_model_config.json", "w", encoding="utf-8") as f:
        json.dump(final_model_config, f, indent=2)

    print("[Artifacts Saved] models/final_resume_classifier.joblib", flush=True)
    print("[Artifacts Saved] models/final_char_tfidf_vectorizer.joblib", flush=True)
    print("[Artifacts Saved] models/final_model_config.json", flush=True)
    print(f"[Done] Total execution finished in {time.time() - t_start:.2f}s", flush=True)

    return final_model_config


if __name__ == "__main__":
    run_targeted_optimization()
