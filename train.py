"""
Module: train.py
Purpose: Production-grade training and evaluation pipeline for ResumeForge 2026.
Imports cleanly from local modules without any package prefix.
Can be executed directly from project root: python train.py
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from preprocessing import ResumeTextPreprocessor, clean_raw_data
from features import build_word_char_union
from evaluation import evaluate_predictions, benchmark_model_latency
from models_utils import run_error_analysis


def ensure_data_splits(
    raw_csv: str = "data/raw/Resume.csv",
    clean_csv: str = "data/processed/clean_resumes.csv",
    train_csv: str = "data/processed/train.csv",
    val_csv: str = "data/processed/val.csv",
    test_csv: str = "data/processed/test.csv",
    random_state: int = 42,
):
    """Ensures cleaned data and strict 70/15/15 stratified splits exist."""
    if not os.path.exists(clean_csv):
        print("[Train Pipeline] Clean dataset missing, generating...")
        clean_raw_data(raw_path=raw_csv, processed_path=clean_csv)

    if not (os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv)):
        print("[Train Pipeline] Splits missing, performing stratified 70/15/15 split...")
        df = pd.read_csv(clean_csv)
        # 70% Train, 30% Temp
        df_train, df_temp = train_test_split(
            df, test_size=0.30, random_state=random_state, stratify=df["Category"]
        )
        # 15% Val, 15% Test
        df_val, df_test = train_test_split(
            df_temp, test_size=0.50, random_state=random_state, stratify=df_temp["Category"]
        )
        os.makedirs("data/processed", exist_ok=True)
        df_train.to_csv(train_csv, index=False)
        df_val.to_csv(val_csv, index=False)
        df_test.to_csv(test_csv, index=False)
        print(f"[Train Pipeline] Generated: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    return train_csv, val_csv, test_csv


def run_training_pipeline(
    models_dir: str = "models",
    reports_dir: str = "reports",
    figures_dir: str = "reports/figures",
    C: float = 1.0,
    class_weight: str = "balanced",
):
    """
    Executes production model retraining on combined Train+Validation sets (2,108 resumes),
    evaluates strictly ONCE on the isolated holdout Test set (373 resumes),
    and serializes production model artifacts.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    train_csv, val_csv, test_csv = ensure_data_splits()

    print("[Train Pipeline] Loading dataset partitions...")
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    # Combine Train + Val for production training (n=2,108)
    df_train_full = pd.concat([df_train, df_val], ignore_index=True)
    print(f"[Train Pipeline] Combined Train+Val samples: {len(df_train_full)} | Test samples: {len(df_test)}")

    target_classes = sorted(df_train_full["Category"].unique())

    # Preprocess text
    preprocessor = ResumeTextPreprocessor(lower=True)
    print("[Train Pipeline] Preprocessing training texts...")
    X_train_clean = preprocessor.transform(df_train_full["Resume_str"])
    y_train_full = df_train_full["Category"]

    print("[Train Pipeline] Preprocessing isolated test texts...")
    X_test_clean = preprocessor.transform(df_test["Resume_str"])
    y_test = df_test["Category"]

    # Build and fit Word+Char FeatureUnion vectorizer
    print("[Train Pipeline] Extracting dual-channel Word+Char TF-IDF features...")
    vectorizer = build_word_char_union(
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
        word_max_features=25000,
        char_max_features=35000,
        min_df=2,
    )
    t0_feat = time.perf_counter()
    X_train_vec = vectorizer.fit_transform(X_train_clean)
    X_test_vec = vectorizer.transform(X_test_clean)
    print(f"[Train Pipeline] Vectorization complete in {time.perf_counter() - t0_feat:.2f}s ({X_train_vec.shape[1]} features)")

    # Build Calibrated Linear Classifier
    print(f"[Train Pipeline] Fitting CalibratedClassifierCV(LinearSVC(C={C}, class_weight='{class_weight}')) via 5-fold CV...")
    base_svc = LinearSVC(C=C, class_weight=class_weight, random_state=42, max_iter=3000)
    final_clf = CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=5)

    t0_train = time.perf_counter()
    final_clf.fit(X_train_vec, y_train_full)
    train_time = time.perf_counter() - t0_train
    print(f"[Train Pipeline] Model successfully trained in {train_time:.2f}s")

    # Single-run evaluation on untouched test set
    print("\n========================================================")
    print("  EVALUATION ON ISOLATED TEST SET (STRICT SINGLE RUN)")
    print("========================================================")
    t0_test = time.perf_counter()
    test_preds = final_clf.predict(X_test_vec)
    test_latency = ((time.perf_counter() - t0_test) / X_test_vec.shape[0]) * 1000.0

    metrics = evaluate_predictions(y_test, test_preds, target_names=target_classes)

    print(f"\n>>> FINAL TEST ACCURACY : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f">>> FINAL TEST MACRO-F1 : {metrics['macro_f1']:.4f}")
    print(f">>> FINAL TEST WEIGHTED-F1: {metrics['weighted_f1']:.4f}")
    print(f">>> Inference Latency   : {test_latency:.3f} ms / sample\n")
    print(metrics["classification_report_text"])

    # Save artifacts to models/
    model_save_path = os.path.join(models_dir, "final_classifier.joblib")
    vec_save_path = os.path.join(models_dir, "feature_vectorizer.joblib")
    meta_save_path = os.path.join(models_dir, "model_metadata.joblib")

    joblib.dump(final_clf, model_save_path)
    joblib.dump(vectorizer, vec_save_path)
    metadata = {
        "model_type": "CalibratedClassifierCV(LinearSVC)",
        "C": C,
        "class_weight": class_weight,
        "classes": target_classes,
        "n_classes": len(target_classes),
        "feature_dim": X_train_vec.shape[1],
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_weighted_f1": metrics["weighted_f1"],
        "train_samples": len(df_train_full),
        "test_samples": len(df_test),
        "latency_ms": test_latency,
    }
    joblib.dump(metadata, meta_save_path)
    print(f"[Train Pipeline] Production artifacts persisted to {models_dir}/")

    # Generate Test Confusion Matrix
    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=target_classes, yticklabels=target_classes)
    plt.title(f"Final Test Confusion Matrix (n={len(df_test)}, Accuracy={metrics['accuracy']:.3f}, Macro-F1={metrics['macro_f1']:.3f})", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Category", fontsize=12)
    plt.ylabel("True Category", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    test_cm_fig = os.path.join(figures_dir, "09_final_test_confusion_matrix.png")
    plt.savefig(test_cm_fig, dpi=300)
    plt.close()

    # Save metrics JSON & Markdown report
    with open(os.path.join(reports_dir, "FINAL_TEST_METRICS.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Run full error analysis on test predictions
    run_error_analysis(
        y_true=y_test,
        y_pred=test_preds,
        texts=df_test["Resume_str"],
        model_name="Calibrated LinearSVC (Word+Char TF-IDF)",
        output_report_path=os.path.join(reports_dir, "ERROR_ANALYSIS.md"),
        output_fig_dir=figures_dir,
    )

    return metrics


if __name__ == "__main__":
    run_training_pipeline()
