"""
Production Inference Pipeline Module for Resume Classification.
Loads serialized models and provides unified prediction interface for:
- Raw resume text string
- PDF files (path or binary buffer)
Outputs predicted category, decision confidence/scores, and top feature contributors.
"""
import os
import joblib
import numpy as np
from typing import Dict, Any, Union, Optional
from pathlib import Path

try:
    from src.config import (
        TFIDF_VECTORIZER_PATH, LABEL_ENCODER_PATH, BEST_CLASSICAL_MODEL_PATH
    )
    from src.preprocessing import preprocess_text
    from src.pdf_extractor import extract_text_from_pdf
except ImportError:
    from config import (
        TFIDF_VECTORIZER_PATH, LABEL_ENCODER_PATH, BEST_CLASSICAL_MODEL_PATH
    )
    from preprocessing import preprocess_text
    from pdf_extractor import extract_text_from_pdf


class ResumeClassificationPipeline:
    """
    End-to-End Production Inference Pipeline.
    """
    def __init__(
        self,
        vectorizer_path: Optional[Path] = None,
        encoder_path: Optional[Path] = None,
        model_path: Optional[Path] = None
    ):
        v_path = vectorizer_path or TFIDF_VECTORIZER_PATH
        e_path = encoder_path or LABEL_ENCODER_PATH
        m_path = model_path or BEST_CLASSICAL_MODEL_PATH
        
        if not os.path.exists(v_path):
            raise FileNotFoundError(f"Vectorizer artifact not found: {v_path}")
        if not os.path.exists(e_path):
            raise FileNotFoundError(f"Encoder artifact not found: {e_path}")
        if not os.path.exists(m_path):
            raise FileNotFoundError(f"Model artifact not found: {m_path}")
            
        self.vectorizer = joblib.load(v_path)
        self.encoder = joblib.load(e_path)
        self.model = joblib.load(m_path)
        self.classes = list(self.encoder.classes_)
        
    def predict_text(self, raw_text: str) -> Dict[str, Any]:
        """
        Execute prediction on raw resume text string with strict input validation.
        """
        if raw_text is None or not isinstance(raw_text, str) or len(raw_text.strip()) == 0:
            return {
                "success": False,
                "error": "Input resume text is empty or contains only whitespace.",
                "predicted_category": None,
                "confidence": 0.0,
                "score_type": None
            }
            
        # 1. Reproducible preprocessing
        cleaned_text = preprocess_text(raw_text)
        tokens = cleaned_text.split()
        
        if len(tokens) == 0:
            return {
                "success": False,
                "error": "Resume yielded 0 valid tokens after text cleaning.",
                "predicted_category": None,
                "confidence": 0.0,
                "score_type": None
            }
            
        if len(tokens) < 4:
            return {
                "success": False,
                "error": "Resume text is too short (fewer than 4 words) to perform meaningful category classification. Please provide a more complete resume.",
                "predicted_category": None,
                "confidence": 0.0,
                "score_type": None
            }
            
        # 2. TF-IDF Transformation
        X_tfidf = self.vectorizer.transform([raw_text])
        
        # 3. Model Prediction & Decision Scoring
        pred_idx = self.model.predict(X_tfidf)[0]
        pred_category = self.encoder.inverse_transform([pred_idx])[0]
        
        top_classes = []
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X_tfidf)[0]
            confidence = float(scores[pred_idx])
            score_type = "Decision Score"
            top_indices = scores.argsort()[-3:][::-1]
            top_classes = [
                {"category": self.classes[i], "score": round(float(scores[i]), 4), "metric": "decision_score"}
                for i in top_indices
            ]
        elif hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_tfidf)[0]
            confidence = float(probs[pred_idx])
            score_type = "Probability"
            top_indices = probs.argsort()[-3:][::-1]
            top_classes = [
                {"category": self.classes[i], "score": round(float(probs[i]), 4), "metric": "probability"}
                for i in top_indices
            ]
        else:
            confidence = 1.0
            score_type = "Deterministic"
            top_classes = [{"category": pred_category, "score": 1.0, "metric": "label"}]
            
        # Extract top discriminative terms present in this resume
        feature_names = self.vectorizer.get_feature_names_out()
        non_zero_indices = X_tfidf.nonzero()[1]
        term_scores = [(feature_names[i], float(X_tfidf[0, i])) for i in non_zero_indices]
        top_terms = sorted(term_scores, key=lambda x: x[1], reverse=True)[:8]
        
        return {
            "success": True,
            "predicted_category": pred_category,
            "confidence": round(confidence, 4),
            "score_type": score_type,
            "top_classes": top_classes,
            "top_terms": top_terms,
            "preprocessed_snippet": cleaned_text[:300] + "..." if len(cleaned_text) > 300 else cleaned_text,
            "token_count": len(tokens)
        }
        
    def predict_pdf(self, pdf_source: Union[str, bytes]) -> Dict[str, Any]:
        """
        Execute prediction on PDF resume (file path or bytes).
        """
        success, text, err = extract_text_from_pdf(pdf_source)
        if not success:
            return {
                "success": False,
                "error": err,
                "predicted_category": None,
                "confidence": 0.0,
                "score_type": None
            }
            
        res = self.predict_text(text)
        if res["success"]:
            res["extracted_char_count"] = len(text)
        return res


if __name__ == "__main__":
    try:
        pipeline = ResumeClassificationPipeline()
        sample = "Full stack software developer with experience in React, Node.js, Python, PostgreSQL, and AWS cloud infrastructure."
        out = pipeline.predict_text(sample)
        print("Sample Text Prediction Result:")
        for k, v in out.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Pipeline test note: {e}")
