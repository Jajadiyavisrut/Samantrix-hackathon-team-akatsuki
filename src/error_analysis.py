"""
Error Analysis Module for Resume Classification.
Analyzes misclassified examples from validation and test sets:
- Records actual vs predicted class, confidence/decision scores, snippet preview
- Identifies error root causes (Class Overlap, Generic Language, Misleading Keywords, Boundary Decisions)
- Generates structured error report CSV and Markdown summary.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any

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
    Identifies and categorizes misclassified samples for the best classical model.
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
    
    # Calculate confidence or decision score
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_tfidf)
        confidences = np.max(probs, axis=1)
        score_type = "probability"
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_tfidf)
        confidences = np.max(scores, axis=1)
        score_type = "decision_margin"
    else:
        confidences = [1.0] * len(y_pred_idx)
        score_type = "uncalibrated"
        
    errors = []
    for idx in range(len(eval_df)):
        if y_true_idx[idx] != y_pred_idx[idx]:
            raw_text = str(texts[idx])
            cleaned = preprocess_text(raw_text)
            
            # Root cause heuristic diagnosis
            actual = y_true_str[idx]
            pred = y_pred_str[idx]
            
            # Check for specific known semantic overlaps
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
                "confidence_score": round(float(confidences[idx]), 4),
                "score_metric": score_type,
                "word_count": len(cleaned.split()),
                "text_snippet": (raw_text[:250] + "...").replace("\n", " ").strip(),
                "diagnosed_root_cause": cause
            })
            
    errors_df = pd.DataFrame(errors)
    
    # Save CSV
    csv_out = ERROR_ANALYSIS_DIR / f"misclassified_samples_{split_name}.csv"
    errors_df.to_csv(csv_out, index=False)
    
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
