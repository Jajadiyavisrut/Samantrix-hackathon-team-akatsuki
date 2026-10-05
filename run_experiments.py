"""
Script: run_experiments.py
Purpose: Ultra-robust, high-visibility, fast NLP experiment runner (15 targeted experiments).
Avoids exhaustive grids, caches sparse matrices, flushes all prints immediately,
saves incrementally after every single experiment, and outputs final leaderboard.
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

from preprocessing import clean_resume_text

# Ensure required directories exist
Path("reports").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

CSV_REPORT = Path("reports/model_experiments.csv")
LEADERBOARD_CSV = Path("reports/model_leaderboard.csv")
BEST_CONFIG_JSON = Path("reports/BEST_MODEL_CONFIG.json")


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Computes standard evaluation metrics."""
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


def save_incremental_record(record: Dict[str, Any]):
    """Appends an experiment record to reports/model_experiments.csv immediately."""
    df_row = pd.DataFrame([record])
    if not CSV_REPORT.exists():
        df_row.to_csv(CSV_REPORT, index=False)
    else:
        df_row.to_csv(CSV_REPORT, mode="a", header=False, index=False)


def run_fast_experiments():
    start_total_time = time.time()

    print("=" * 60, flush=True)
    print("      INITIALIZING FAST RESUME CLASSIFIER EXPERIMENTS       ", flush=True)
    print("=" * 60, flush=True)

    # Load data splits
    train_path = Path("data/processed/train.csv")
    val_path = Path("data/processed/val.csv")

    if not (train_path.exists() and val_path.exists()):
        print("[Setup] Data splits missing, generating...", flush=True)
        from train import ensure_data_splits
        ensure_data_splits()

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    y_train = df_train["Category"]
    y_val = df_val["Category"]

    print(f"[Setup] Loaded Train: {len(df_train)} samples, Val: {len(df_val)} samples.", flush=True)

    # Preprocess text once on Train and Val (Variant C: domain-preserving cleaner)
    t0_prep = time.time()
    print("[Setup] Preprocessing texts via domain-preserving pipeline...", flush=True)
    X_train_clean = [clean_resume_text(t) for t in df_train["Resume_str"]]
    X_val_clean = [clean_resume_text(t) for t in df_val["Resume_str"]]
    print(f"[Setup] Text preprocessing finished in {time.time() - t0_prep:.2f}s", flush=True)

    # Always initialize fresh or clean reports/model_experiments.csv
    if CSV_REPORT.exists():
        # Preserve backup of previous experiments if any
        prev_df = pd.read_csv(CSV_REPORT)
        print(f"[Setup] Backing up previous {len(prev_df)} experiment records...", flush=True)
        prev_df.to_csv(Path("reports/model_experiments_prev.csv"), index=False)
        CSV_REPORT.unlink()

    all_records: List[Dict[str, Any]] = []
    TOTAL_EXPERIMENTS = 15
    exp_num = 1

    best_macro_f1 = -1.0
    best_config_meta = None
    best_model_obj = None
    best_vectorizer_obj = None

    # Cache for reusable matrices (key -> (vectorizer, X_tr, X_val))
    feature_cache: Dict[str, Tuple[Any, Any, Any]] = {}

    def log_and_record(
        exp_id: str,
        name: str,
        clf,
        feat_type: str,
        w_ng: str,
        c_ng: str,
        min_df: int,
        c_val: float,
        cw: str,
        X_tr,
        X_val,
        vec_obj
    ):
        nonlocal exp_num, best_macro_f1, best_config_meta, best_model_obj, best_vectorizer_obj

        print(f"\n{'='*60}", flush=True)
        print(f"STARTING EXPERIMENT {exp_num}/{TOTAL_EXPERIMENTS}: {exp_id} - {name}", flush=True)
        print(f"{'='*60}", flush=True)

        t0_fit = time.time()
        clf.fit(X_tr, y_train)
        train_time = time.time() - t0_fit

        if train_time > 90.0:
            print(f"[WARNING] Experiment {exp_id} took {train_time:.1f}s (>90s threshold)!", flush=True)

        t0_pred = time.time()
        preds = clf.predict(X_val)
        pred_time = time.time() - t0_pred
        m = compute_metrics(y_val, preds)

        rec = {
            "experiment_id": exp_id,
            "preprocessing": "Variant C (Domain Punctuation & Tech Tokens)",
            "feature_type": feat_type,
            "word_ngram": str(w_ng),
            "char_ngram": str(c_ng),
            "min_df": min_df,
            "max_df": 0.95,
            "sublinear_tf": True,
            "classifier": clf.__class__.__name__,
            "C": c_val,
            "class_weight": cw,
            "accuracy": m["accuracy"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "training_time": round(train_time, 4),
            "prediction_time": round(pred_time, 4),
        }

        all_records.append(rec)
        save_incremental_record(rec)

        print(f"\nCOMPLETED {exp_num}/{TOTAL_EXPERIMENTS}: {exp_id} - {name}", flush=True)
        print(f"Accuracy:    {m['accuracy']:.4f}", flush=True)
        print(f"Macro-F1:    {m['macro_f1']:.4f}", flush=True)
        print(f"Weighted-F1: {m['weighted_f1']:.4f}", flush=True)
        print(f"Time:        {train_time:.2f} seconds", flush=True)

        if m["macro_f1"] > best_macro_f1:
            best_macro_f1 = m["macro_f1"]
            best_config_meta = rec
            best_model_obj = clf
            best_vectorizer_obj = vec_obj
            print(f"*** NEW BEST MODEL: {exp_id} (Macro-F1: {best_macro_f1:.4f}) ***", flush=True)

        exp_num += 1

    # =========================================================================
    # PRE-FIT CACHES (Word & Char Vectorizers)
    # =========================================================================
    print("\n[Cache] Pre-fitting feature extractors once for maximum speed...", flush=True)

    # 1. Word (1,2)
    t0 = time.time()
    vec_w12 = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
    X_tr_w12 = vec_w12.fit_transform(X_train_clean)
    X_val_w12 = vec_w12.transform(X_val_clean)
    feature_cache["word_12"] = (vec_w12, X_tr_w12, X_val_w12)
    print(f"  - Cached Word (1,2) [features={X_tr_w12.shape[1]}] in {time.time()-t0:.2f}s", flush=True)

    # 2. Word (1,3)
    t0 = time.time()
    vec_w13 = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, max_df=0.95, sublinear_tf=True)
    X_tr_w13 = vec_w13.fit_transform(X_train_clean)
    X_val_w13 = vec_w13.transform(X_val_clean)
    feature_cache["word_13"] = (vec_w13, X_tr_w13, X_val_w13)
    print(f"  - Cached Word (1,3) [features={X_tr_w13.shape[1]}] in {time.time()-t0:.2f}s", flush=True)

    # 3. Char (3,5) with max_features=35000 for high speed & memory safety
    t0 = time.time()
    vec_c35 = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=35000, sublinear_tf=True)
    X_tr_c35 = vec_c35.fit_transform(X_train_clean)
    X_val_c35 = vec_c35.transform(X_val_clean)
    feature_cache["char_35"] = (vec_c35, X_tr_c35, X_val_c35)
    print(f"  - Cached Char (3,5) [features={X_tr_c35.shape[1]}] in {time.time()-t0:.2f}s", flush=True)

    # 4. Char (3,6) with max_features=40000
    t0 = time.time()
    vec_c36 = TfidfVectorizer(analyzer="char", ngram_range=(3, 6), min_df=2, max_features=40000, sublinear_tf=True)
    X_tr_c36 = vec_c36.fit_transform(X_train_clean)
    X_val_c36 = vec_c36.transform(X_val_clean)
    feature_cache["char_36"] = (vec_c36, X_tr_c36, X_val_c36)
    print(f"  - Cached Char (3,6) [features={X_tr_c36.shape[1]}] in {time.time()-t0:.2f}s", flush=True)

    # 5. Char (4,6) with max_features=40000
    t0 = time.time()
    vec_c46 = TfidfVectorizer(analyzer="char", ngram_range=(4, 6), min_df=2, max_features=40000, sublinear_tf=True)
    X_tr_c46 = vec_c46.fit_transform(X_train_clean)
    X_val_c46 = vec_c46.transform(X_val_clean)
    feature_cache["char_46"] = (vec_c46, X_tr_c46, X_val_c46)
    print(f"  - Cached Char (4,6) [features={X_tr_c46.shape[1]}] in {time.time()-t0:.2f}s", flush=True)

    # 6. Char (5,7) with min_df=3, max_features=35000 (prevents 95s bottleneck!)
    t0 = time.time()
    vec_c57 = TfidfVectorizer(analyzer="char", ngram_range=(5, 7), min_df=3, max_features=35000, sublinear_tf=True)
    X_tr_c57 = vec_c57.fit_transform(X_train_clean)
    X_val_c57 = vec_c57.transform(X_val_clean)
    feature_cache["char_57"] = (vec_c57, X_tr_c57, X_val_c57)
    print(f"  - Cached Char (5,7) [features={X_tr_c57.shape[1]}] in {time.time()-t0:.2f}s", flush=True)

    # =========================================================================
    # GROUP A — WORD TF-IDF
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("                  GROUP A — WORD TF-IDF                     ", flush=True)
    print("=" * 60, flush=True)

    # A1: Word (1,2) + LinearSVC(C=1)
    log_and_record(
        exp_id="A1",
        name="Word (1,2) + LinearSVC(C=1.0)",
        clf=LinearSVC(C=1.0, random_state=42, max_iter=2000),
        feat_type="word_tfidf",
        w_ng="(1,2)",
        c_ng="None",
        min_df=2,
        c_val=1.0,
        cw="None",
        X_tr=X_tr_w12,
        X_val=X_val_w12,
        vec_obj=vec_w12
    )

    # A2: Word (1,2) + LogisticRegression(C=2)
    log_and_record(
        exp_id="A2",
        name="Word (1,2) + LogisticRegression(C=2.0)",
        clf=LogisticRegression(C=2.0, max_iter=500, random_state=42),
        feat_type="word_tfidf",
        w_ng="(1,2)",
        c_ng="None",
        min_df=2,
        c_val=2.0,
        cw="None",
        X_tr=X_tr_w12,
        X_val=X_val_w12,
        vec_obj=vec_w12
    )

    # A3: Word (1,3) + LinearSVC(C=1)
    log_and_record(
        exp_id="A3",
        name="Word (1,3) + LinearSVC(C=1.0)",
        clf=LinearSVC(C=1.0, random_state=42, max_iter=2000),
        feat_type="word_tfidf",
        w_ng="(1,3)",
        c_ng="None",
        min_df=2,
        c_val=1.0,
        cw="None",
        X_tr=X_tr_w13,
        X_val=X_val_w13,
        vec_obj=vec_w13
    )

    # A4: Word (1,3) + LogisticRegression(C=2)
    log_and_record(
        exp_id="A4",
        name="Word (1,3) + LogisticRegression(C=2.0)",
        clf=LogisticRegression(C=2.0, max_iter=500, random_state=42),
        feat_type="word_tfidf",
        w_ng="(1,3)",
        c_ng="None",
        min_df=2,
        c_val=2.0,
        cw="None",
        X_tr=X_tr_w13,
        X_val=X_val_w13,
        vec_obj=vec_w13
    )

    # =========================================================================
    # GROUP B — CHARACTER TF-IDF
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("               GROUP B — CHARACTER TF-IDF                   ", flush=True)
    print("=" * 60, flush=True)

    # B1: Char (3,5) + LinearSVC
    log_and_record(
        exp_id="B1",
        name="Char (3,5) + LinearSVC(C=1.0)",
        clf=LinearSVC(C=1.0, random_state=42, max_iter=2000),
        feat_type="char_tfidf",
        w_ng="None",
        c_ng="(3,5)",
        min_df=2,
        c_val=1.0,
        cw="None",
        X_tr=X_tr_c35,
        X_val=X_val_c35,
        vec_obj=vec_c35
    )

    # B2: Char (3,6) + LinearSVC
    log_and_record(
        exp_id="B2",
        name="Char (3,6) + LinearSVC(C=1.0)",
        clf=LinearSVC(C=1.0, random_state=42, max_iter=2000),
        feat_type="char_tfidf",
        w_ng="None",
        c_ng="(3,6)",
        min_df=2,
        c_val=1.0,
        cw="None",
        X_tr=X_tr_c36,
        X_val=X_val_c36,
        vec_obj=vec_c36
    )

    # B3: Char (4,6) + LinearSVC
    log_and_record(
        exp_id="B3",
        name="Char (4,6) + LinearSVC(C=1.0)",
        clf=LinearSVC(C=1.0, random_state=42, max_iter=2000),
        feat_type="char_tfidf",
        w_ng="None",
        c_ng="(4,6)",
        min_df=2,
        c_val=1.0,
        cw="None",
        X_tr=X_tr_c46,
        X_val=X_val_c46,
        vec_obj=vec_c46
    )

    # B4: Char (5,7) + LinearSVC
    log_and_record(
        exp_id="B4",
        name="Char (5,7) + LinearSVC(C=1.0)",
        clf=LinearSVC(C=1.0, random_state=42, max_iter=2000),
        feat_type="char_tfidf",
        w_ng="None",
        c_ng="(5,7)",
        min_df=3,
        c_val=1.0,
        cw="None",
        X_tr=X_tr_c57,
        X_val=X_val_c57,
        vec_obj=vec_c57
    )

    # =========================================================================
    # GROUP C — WORD + CHARACTER FUSION
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("             GROUP C — WORD + CHARACTER FUSION              ", flush=True)
    print("=" * 60, flush=True)

    # Helper to construct sparse hstack matrices safely
    def make_fusion(w_key, c_key):
        _, X_tr_w, X_val_w = feature_cache[w_key]
        _, X_tr_c, X_val_c = feature_cache[c_key]
        X_tr_f = hstack([X_tr_w, X_tr_c], format="csr")
        X_val_f = hstack([X_val_w, X_val_c], format="csr")
        return X_tr_f, X_val_f

    # C1: word(1,2) + char(3,5) + LinearSVC
    X_tr_c1, X_val_c1 = make_fusion("word_12", "char_35")
    log_and_record(
        exp_id="C1",
        name="Word (1,2) + Char (3,5) + LinearSVC(C=1.0)",
        clf=LinearSVC(C=1.0, random_state=42, max_iter=2000),
        feat_type="word_char_fusion",
        w_ng="(1,2)",
        c_ng="(3,5)",
        min_df=2,
        c_val=1.0,
        cw="None",
        X_tr=X_tr_c1,
        X_val=X_val_c1,
        vec_obj=(vec_w12, vec_c35)
    )

    # C2: word(1,2) + char(3,6) + LinearSVC
    X_tr_c2, X_val_c2 = make_fusion("word_12", "char_36")
    log_and_record(
        exp_id="C2",
        name="Word (1,2) + Char (3,6) + LinearSVC(C=1.0)",
        clf=LinearSVC(C=1.0, random_state=42, max_iter=2000),
        feat_type="word_char_fusion",
        w_ng="(1,2)",
        c_ng="(3,6)",
        min_df=2,
        c_val=1.0,
        cw="None",
        X_tr=X_tr_c2,
        X_val=X_val_c2,
        vec_obj=(vec_w12, vec_c36)
    )

    # C3: word(1,3) + char(3,5) + LinearSVC
    X_tr_c3, X_val_c3 = make_fusion("word_13", "char_35")
    log_and_record(
        exp_id="C3",
        name="Word (1,3) + Char (3,5) + LinearSVC(C=1.0)",
        clf=LinearSVC(C=1.0, random_state=42, max_iter=2000),
        feat_type="word_char_fusion",
        w_ng="(1,3)",
        c_ng="(3,5)",
        min_df=2,
        c_val=1.0,
        cw="None",
        X_tr=X_tr_c3,
        X_val=X_val_c3,
        vec_obj=(vec_w13, vec_c35)
    )

    # C4: word(1,2) + char(4,6) + LinearSVC
    X_tr_c4, X_val_c4 = make_fusion("word_12", "char_46")
    log_and_record(
        exp_id="C4",
        name="Word (1,2) + Char (4,6) + LinearSVC(C=1.0)",
        clf=LinearSVC(C=1.0, random_state=42, max_iter=2000),
        feat_type="word_char_fusion",
        w_ng="(1,2)",
        c_ng="(4,6)",
        min_df=2,
        c_val=1.0,
        cw="None",
        X_tr=X_tr_c4,
        X_val=X_val_c4,
        vec_obj=(vec_w12, vec_c46)
    )

    # =========================================================================
    # GROUP D — STRONG VARIANTS
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("                GROUP D — STRONG VARIANTS                   ", flush=True)
    print("=" * 60, flush=True)

    # D1: best word configuration with class_weight balanced
    # (Word 1,2 with class_weight='balanced')
    log_and_record(
        exp_id="D1",
        name="Best Word (1,2) + LinearSVC(C=1.0, balanced)",
        clf=LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000),
        feat_type="word_tfidf",
        w_ng="(1,2)",
        c_ng="None",
        min_df=2,
        c_val=1.0,
        cw="balanced",
        X_tr=X_tr_w12,
        X_val=X_val_w12,
        vec_obj=vec_w12
    )

    # D2: best word configuration without class_weight (C=2.0 fine-tuning)
    log_and_record(
        exp_id="D2",
        name="Best Word (1,2) + LinearSVC(C=2.0)",
        clf=LinearSVC(C=2.0, random_state=42, max_iter=2000),
        feat_type="word_tfidf",
        w_ng="(1,2)",
        c_ng="None",
        min_df=2,
        c_val=2.0,
        cw="None",
        X_tr=X_tr_w12,
        X_val=X_val_w12,
        vec_obj=vec_w12
    )

    # D3: best word+char configuration with class_weight balanced
    # (C3 fusion with class_weight='balanced')
    log_and_record(
        exp_id="D3",
        name="Best Word+Char (1,3+3,5) + LinearSVC(C=1.0, balanced)",
        clf=LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000),
        feat_type="word_char_fusion",
        w_ng="(1,3)",
        c_ng="(3,5)",
        min_df=2,
        c_val=1.0,
        cw="balanced",
        X_tr=X_tr_c3,
        X_val=X_val_c3,
        vec_obj=(vec_w13, vec_c35)
    )

    # Clean memory
    gc.collect()

    # =========================================================================
    # LEADERBOARD & SELECTION
    # =========================================================================
    df_results = pd.DataFrame(all_records)
    df_sorted = df_results.sort_values(by=["macro_f1", "accuracy"], ascending=[False, False]).reset_index(drop=True)
    df_sorted.insert(0, "rank", range(1, len(df_sorted) + 1))

    # Save to model_leaderboard.csv
    df_sorted.to_csv(LEADERBOARD_CSV, index=False)

    print("\n" + "=" * 60, flush=True)
    print("                        TOP 10 MODELS                       ", flush=True)
    print("=" * 60, flush=True)
    display_df = df_sorted.rename(columns={
        "classifier": "model",
        "feature_type": "features",
        "accuracy": "accuracy",
        "macro_f1": "macro_f1",
        "weighted_f1": "weighted_f1",
        "training_time": "training_time"
    })
    display_cols = ["rank", "model", "features", "accuracy", "macro_f1", "weighted_f1", "training_time"]
    print(display_df[display_cols].head(10).to_string(index=False), flush=True)
    print("=" * 60, flush=True)

    # Save Best Model Config
    with open(BEST_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(best_config_meta, f, indent=2)
    print(f"\n[Best Model Config] Persisted to {BEST_CONFIG_JSON}", flush=True)

    # Serialize best model and vectorizers
    if isinstance(best_vectorizer_obj, tuple):
        w_v, c_v = best_vectorizer_obj
        union = FeatureUnion([("word", w_v), ("char", c_v)])
        joblib.dump(best_model_obj, Path("models/best_tfidf_model.joblib"))
        joblib.dump(union, Path("models/best_feature_vectorizer.joblib"))
    else:
        joblib.dump(best_model_obj, Path("models/best_tfidf_model.joblib"))
        joblib.dump(best_vectorizer_obj, Path("models/best_feature_vectorizer.joblib"))

    print(f"[Model Saved] Saved best model to models/best_tfidf_model.joblib", flush=True)
    print(f"[Total Time] Fast experimentation finished in {time.time() - start_total_time:.2f}s", flush=True)

    return df_sorted, best_config_meta


if __name__ == "__main__":
    run_fast_experiments()
