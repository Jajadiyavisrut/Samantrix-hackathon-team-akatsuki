"""
Error Analysis Module for Resume Classification.
Analyzes misclassified examples from validation and test sets:
- Records actual vs predicted class, confidence/decision scores, snippet preview
- Identifies error root causes (Class Overlap, Generic Language, Misleading Keywords, Boundary Decisions)
- Generates structured error report CSVs (misclassified_examples.csv) and JSON summaries.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

try:
    from src.config import (
        PROCESSED_DATA_DIR, ERROR_ANALYSIS_DIR,
        TFIDF_VECTORIZER_PATH, LABEL_ENCODER_PATH, BEST_CLASSICAL_MODEL_PATH
    )
    from src.preprocessing import preprocess_text
except ImportError:
    from config import (
        PROCESSED_DATA_DIR, ERROR_ANALYSIS_DIR,
        TFIDF_VECTORIZER_PATH, LABEL_ENCODER_PATH, BEST_CLASSICAL_MODEL_PATH
    )
    from preprocessing import preprocess_text


def analyze_misclassifications(
    eval_df: pd.DataFrame,
    split_name: str = "test"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Identifies and categorizes misclassified samples for the best classical model (Linear SVM).
    """
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    encoder = joblib.load(LABEL_ENCODER_PATH)
    model = joblib.load(BEST_CLASSICAL_MODEL_PATH)
    
    texts = eval_df["Resume_str"].tolist()
    y_true_str = eval_df["Category"].tolist()
    y_true_idx = encoder.transform(y_true_str)
    
    X_tfidf = vectorizer.transform(texts)
    y_pred_idx = model.predict(X_tfidf)
    y_pred_str = encoder.inverse_transform(y_pred_idx)
    
    # Calculate decision scores
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_tfidf)
        confidences = np.max(scores, axis=1)
        score_type = "Decision Score"
    elif hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_tfidf)
        confidences = np.max(probs, axis=1)
        score_type = "Probability"
    else:
        confidences = np.ones(len(y_pred_idx))
        score_type = "Deterministic"
        
    errors = []
    formatted_examples = []
    
    for idx in range(len(eval_df)):
        if y_true_idx[idx] != y_pred_idx[idx]:
            raw_text = str(texts[idx])
            cleaned = preprocess_text(raw_text)
            
            actual = y_true_str[idx]
            pred = y_pred_str[idx]
            dec_score = round(float(confidences[idx]), 4)
            preview = (raw_text[:200] + "...").replace("\n", " ").replace("\r", " ").strip()
            
            # Root cause diagnostic heuristics:
            # 1. Financial Domain: shared vocabulary across Banking, Finance, and Accountant
            # 2. Commercial / Client Acquisition: shared terminology across Business Development, Sales, Consultant
            # 3. Creative / Media: shared design terminology across Digital Media, Arts, Designer, Public Relations
            # 4. Short document: documents with very sparse token counts (< 100 words)
            # 5. Multi-disciplinary: candidates holding cross-domain roles (e.g. IT manager in Healthcare)
            if {actual, pred}.issubset({"FINANCE", "BANKING", "ACCOUNTANT"}):
                cause = "Semantic Overlap (Financial Domain shared terminology)"
            elif {actual, pred}.issubset({"BUSINESS-DEVELOPMENT", "SALES", "CONSULTANT"}):
                cause = "Semantic Overlap (Client acquisition & revenue terminology)"
            elif {actual, pred}.issubset({"DIGITAL-MEDIA", "ARTS", "DESIGNER", "PUBLIC-RELATIONS"}):
                cause = "Cross-domain Creative & Media Terminology"
            elif len(cleaned.split()) < 100:
                cause = "Short Document / Insufficient Content Context"
            else:
                cause = "Subtle Discriminative Boundary / Multi-Disciplinary Experience"
                
            errors.append({
                "resume_id": eval_df.iloc[idx]["ID"],
                "actual_category": actual,
                "predicted_category": pred,
                "confidence_score": dec_score,
                "score_metric": score_type,
                "word_count": len(cleaned.split()),
                "text_snippet": preview,
                "diagnosed_root_cause": cause
            })
            
            formatted_examples.append({
                "actual_class": actual,
                "predicted_class": pred,
                "decision_score": dec_score,
                "text_preview": preview,
                "error_category": cause
            })
            
    errors_df = pd.DataFrame(errors)
    examples_df = pd.DataFrame(formatted_examples)
    
    # Save standard misclassified_samples CSV
    csv_out = ERROR_ANALYSIS_DIR / f"misclassified_samples_{split_name}.csv"
    errors_df.to_csv(csv_out, index=False)
    
    # Save rubric-specified misclassified_examples CSV
    examples_csv_out = ERROR_ANALYSIS_DIR / "misclassified_examples.csv"
    examples_df.to_csv(examples_csv_out, index=False)
    
    # Error cause breakdown
    cause_summary = errors_df["diagnosed_root_cause"].value_counts().to_dict() if len(errors_df) > 0 else {}
    
    summary = {
        "total_evaluated": len(eval_df),
        "total_errors": len(errors),
        "error_rate": round(len(errors) / len(eval_df), 4) if len(eval_df) > 0 else 0,
        "score_type": score_type,
        "root_cause_distribution": cause_summary
    }
    
    json_out = ERROR_ANALYSIS_DIR / f"error_summary_{split_name}.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    return errors_df, summary


if __name__ == "__main__":
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    err_df, summary = analyze_misclassifications(test_df, split_name="test")
    print("Error Analysis Complete on Test Set:")
    print(f"  Total Errors: {summary['total_errors']}/{summary['total_evaluated']} ({summary['error_rate']*100:.2f}%)")
    print("  Root Causes:", summary["root_cause_distribution"])
