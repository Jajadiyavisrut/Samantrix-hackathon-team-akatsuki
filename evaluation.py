"""
Module: evaluation.py
Purpose: Standardized, leak-free evaluation metrics and reporting for ResumeForge 2026.
Computes Accuracy, Macro/Weighted Precision, Recall, F1-scores, and classification tables.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list = None,
) -> Dict[str, Any]:
    """
    Computes full suite of classification metrics:
    Accuracy, Macro-Precision, Macro-Recall, Macro-F1, Weighted-F1,
    per-class metrics, and confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report_dict = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=target_names if target_names else None)

    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": cm.tolist() if isinstance(cm, np.ndarray) else cm,
    }


def benchmark_model_latency(
    model,
    sample_texts: list,
    n_iterations: int = 5,
) -> float:
    """Measures average inference latency per sample in milliseconds."""
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        _ = model.predict(sample_texts)
    total_time = time.perf_counter() - start_time
    latency_ms_per_sample = (total_time / (len(sample_texts) * n_iterations)) * 1000.0
    return float(latency_ms_per_sample)


if __name__ == "__main__":
    y_t = np.array(["IT", "HR", "SALES"])
    y_p = np.array(["IT", "HR", "IT"])
    res = evaluate_predictions(y_t, y_p)
    print("[Evaluation] Self-test accuracy:", res["accuracy"])
