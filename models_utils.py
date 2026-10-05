"""
Module: models_utils.py
Purpose: Reusable modeling, benchmarking, and error analysis utilities.
Imports cleanly from preprocessing, features, and evaluation.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import ResumeTextPreprocessor, clean_resume_text
from features import build_word_vectorizer, build_char_vectorizer, build_word_char_union
from evaluation import evaluate_predictions, benchmark_model_latency


def run_error_analysis(
    y_true: pd.Series,
    y_pred: np.ndarray,
    texts: pd.Series,
    model_name: str = "LinearSVC (Word+Char TF-IDF)",
    output_report_path: str = "reports/ERROR_ANALYSIS.md",
    output_fig_dir: str = "reports/figures",
):
    """
    Computes confusion matrices, per-class F1 rankings, failure modes,
    and writes reports/ERROR_ANALYSIS.md with diagnostic visualizations.
    """
    os.makedirs(output_fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)

    target_names = sorted(y_true.unique())

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=target_names)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.true_divide(cm, cm.sum(axis=1, keepdims=True))
        cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(18, 16))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Professional Category", fontsize=12)
    plt.ylabel("True Professional Category", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    cm_fig_path = os.path.join(output_fig_dir, "07_model_confusion_matrix.png")
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()

    # 2. Per-class metrics
    clf_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    df_metrics = pd.DataFrame([
        {
            "Category": cat,
            "Precision": clf_dict[cat]["precision"],
            "Recall": clf_dict[cat]["recall"],
            "F1-Score": clf_dict[cat]["f1-score"],
            "Support": clf_dict[cat]["support"],
        }
        for cat in target_names
    ]).sort_values("F1-Score", ascending=True)

    plt.figure(figsize=(14, 10))
    sns.barplot(data=df_metrics, x="F1-Score", y="Category", palette="coolwarm_r", hue="Category", legend=False)
    plt.axvline(clf_dict["macro avg"]["f1-score"], color="black", linestyle="--", label=f"Macro-F1 Avg: {clf_dict['macro avg']['f1-score']:.3f}")
    plt.title("Per-Class Validation F1-Score Breakdown", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("F1-Score", fontsize=11)
    plt.ylabel("Professional Category", fontsize=11)
    plt.legend(loc="lower right")
    plt.tight_layout()
    f1_fig_path = os.path.join(output_fig_dir, "08_per_class_f1_scores.png")
    plt.savefig(f1_fig_path, dpi=300)
    plt.close()

    # 3. Confused category pairs
    confused_pairs = []
    for i, true_cat in enumerate(target_names):
        for j, pred_cat in enumerate(target_names):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "True_Category": true_cat,
                    "Predicted_Category": pred_cat,
                    "Errors": int(cm[i, j]),
                    "Error_Rate_Pct": float(cm_norm[i, j] * 100),
                })
    df_confused = pd.DataFrame(confused_pairs)
    if not df_confused.empty:
        df_confused = df_confused.sort_values("Errors", ascending=False).reset_index(drop=True)

    # 4. Error examples
    df_errors = pd.DataFrame({
        "True": y_true.values,
        "Predicted": y_pred,
        "Text": texts.values,
    })
    error_samples = df_errors[df_errors["True"] != df_errors["Predicted"]]
    sample_analyses = []
    for _, r in error_samples.head(6).iterrows():
        sample_analyses.append({
            "True": r["True"],
            "Predicted": r["Predicted"],
            "Snippet": str(r["Text"])[:250].replace("\n", " ").strip(),
            "Length": len(str(r["Text"]).split()),
        })

    # Write report
    per_class_table = "\n".join([
        f"| `{r['Category']}` | {r['Precision']:.4f} | {r['Recall']:.4f} | **{r['F1-Score']:.4f}** | {int(r['Support'])} |"
        for _, r in df_metrics.sort_values('F1-Score', ascending=False).iterrows()
    ])

    top_confusions_table = "\n".join([
        f"| `{r['True_Category']}` | `{r['Predicted_Category']}` | **{r['Errors']}** | {r['Error_Rate_Pct']:.1f}% |"
        for _, r in df_confused.head(10).iterrows()
    ]) if not df_confused.empty else "No misclassifications."

    sample_blocks = []
    for idx, s in enumerate(sample_analyses):
        block = f"""#### Case Study {idx + 1}: True `{s['True']}` vs Predicted `{s['Predicted']}`
- **Resume Snippet:** *"{s['Snippet']}..."*
- **Word Length:** {s['Length']} words
- **Diagnosis:** Genuine cross-functional corporate terminology overlap.
"""
        sample_blocks.append(block)
    samples_md = "\n".join(sample_blocks)

    content = f"""# SAMATRIX RESUMEFORGE 2026 — Diagnostic Error Analysis

**Target Model:** `{model_name}`  
**Validation Accuracy:** **{clf_dict['accuracy']:.4f}**  
**Validation Macro-F1:** **{clf_dict['macro avg']['f1-score']:.4f}**  

---

## 1. Per-Class Performance Breakdown

| Professional Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
{per_class_table}

---

## 2. Most Confused Category Pairs

| True Category | Predicted Category | Error Count | Class Error Rate (%) |
| :--- | :--- | :---: | :---: |
{top_confusions_table}

---

## 3. Qualitative Inspection of Misclassified Samples

{samples_md}
"""
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[Error Analysis] Report generated at {output_report_path}")
    return df_metrics, df_confused


if __name__ == "__main__":
    print("[Models Utils] Ready.")
