"""
Module: src.inference.predictor
Purpose: Production-ready, calibrated inference pipeline for ResumeForge 2026.
Provides predict_resume(text) with input validation, preprocessing,
probability/confidence estimation, and model-faithful feature explainability.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import clean_resume_text


DEFAULT_MODEL_DIR = "models"


class ResumePredictor:
    """
    Encapsulates trained model, vectorizer, and label encoder for inference.
    Supports probability/calibrated confidence and feature-level attribution.
    """
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.vectorizer = None
        self.classes_ = None
        self.is_loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        model_path = os.path.join(self.model_dir, "final_classifier.joblib")
        vec_path = os.path.join(self.model_dir, "feature_vectorizer.joblib")
        meta_path = os.path.join(self.model_dir, "model_metadata.joblib")

        if not os.path.exists(model_path) or not os.path.exists(vec_path):
            # Model not yet saved (will be loaded after training)
            return

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vec_path)
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            self.classes_ = meta.get("classes", getattr(self.model, "classes_", None))
        else:
            self.classes_ = getattr(self.model, "classes_", None)

        self.is_loaded = True
        print(f"[Predictor] Loaded model and vectorizer from {self.model_dir}")

    def predict_resume(self, text: str, top_k_features: int = 8) -> Dict[str, Any]:
        """
        End-to-end inference for a single resume text string.

        Returns:
            dict containing:
                - success: bool
                - predicted_category: str
                - confidence: float (0.0 to 1.0)
                - confidence_type: str ('calibrated_probability' or 'decision_score')
                - top_3_predictions: list of tuples (category, confidence)
                - contributing_features: list of dicts (feature, weight)
                - cleaned_word_count: int
                - error_message: str (if any)
        """
        if not self.is_loaded:
            self._load_artifacts()
            if not self.is_loaded:
                return {
                    "success": False,
                    "error_message": "Model artifacts not found in models/ directory. Run training first.",
                }

        # 1. Input validation
        if not text or not isinstance(text, str) or not text.strip():
            return {
                "success": False,
                "error_message": "Input text is empty or invalid. Please provide resume text.",
            }

        cleaned_text = clean_resume_text(text)
        cleaned_words = cleaned_text.split()
        if len(cleaned_words) < 5:
            return {
                "success": False,
                "error_message": "Input text is too short to be a valid resume (fewer than 5 words).",
            }

        # 2. Feature transformation
        X_vec = self.vectorizer.transform([cleaned_text])

        # 3. Prediction & Confidence Estimation
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_vec)[0]
            pred_idx = np.argmax(probs)
            pred_class = self.classes_[pred_idx]
            conf = float(probs[pred_idx])
            conf_type = "calibrated_probability"

            top_3_indices = probs.argsort()[-3:][::-1]
            top_3 = [(self.classes_[i], float(probs[i])) for i in top_3_indices]
        elif hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(X_vec)[0]
            # Softmax calibration of raw decision margins
            exp_scores = np.exp(decision - np.max(decision))
            probs = exp_scores / np.sum(exp_scores)
            pred_idx = np.argmax(decision)
            pred_class = self.classes_[pred_idx]
            conf = float(probs[pred_idx])
            conf_type = "calibrated_decision_softmax"

            top_3_indices = decision.argsort()[-3:][::-1]
            top_3 = [(self.classes_[i], float(probs[i])) for i in top_3_indices]
        else:
            pred_class = self.model.predict(X_vec)[0]
            conf = 1.0
            conf_type = "uncalibrated_discrete"
            top_3 = [(pred_class, 1.0)]

        # 4. Feature-Level Explainability
        contributing_features = self._explain_prediction(X_vec, pred_class, top_k=top_k_features)

        return {
            "success": True,
            "predicted_category": str(pred_class),
            "confidence": round(conf, 4),
            "confidence_type": conf_type,
            "top_3_predictions": top_3,
            "contributing_features": contributing_features,
            "cleaned_word_count": len(cleaned_words),
            "error_message": None,
        }

    def _explain_prediction(self, X_vec, pred_class: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Computes exact linear feature attribution: score_j = x_j * W_{pred, j}.
        """
        try:
            # Check if underlying estimator has coef_
            clf = self.model
            coef = None
            if hasattr(clf, "calibrated_classifiers_") and len(clf.calibrated_classifiers_) > 0:
                # Average coefficients across calibrated folds
                fold_coefs = []
                for cc in clf.calibrated_classifiers_:
                    sub = getattr(cc, "estimator", getattr(cc, "base_estimator", None))
                    if sub is not None and hasattr(sub, "coef_"):
                        fold_coefs.append(sub.coef_)
                if fold_coefs:
                    coef = np.mean(fold_coefs, axis=0)
            elif hasattr(clf, "base_estimator_") and hasattr(clf.base_estimator_, "coef_"):
                coef = clf.base_estimator_.coef_
            elif hasattr(clf, "coef_"):
                coef = clf.coef_
            elif hasattr(clf, "estimator") and hasattr(clf.estimator, "coef_"):
                coef = clf.estimator.coef_

            if coef is None:
                return []

            class_idx = list(self.classes_).index(pred_class)
            weights = coef[class_idx]

            # Multiply sample sparse vector with class weights
            feature_names = self._get_feature_names()
            if feature_names is None or len(feature_names) != X_vec.shape[1]:
                return []

            # Non-zero indices in sample
            row, cols = X_vec.nonzero()
            contributions = []
            for col in cols:
                val = X_vec[0, col]
                contrib = val * weights[col]
                if contrib > 0:  # Positively contributing features
                    feat_name = feature_names[col]
                    # Clean up representation if char n-gram
                    is_char = feat_name.startswith("char_tfidf__")
                    clean_name = feat_name.split("__")[-1]
                    contributions.append({
                        "feature": clean_name,
                        "score": float(contrib),
                        "type": "subword_ngram" if is_char else "word_token",
                    })

            # Sort descending by contribution score
            contributions.sort(key=lambda x: x["score"], reverse=True)
            return contributions[:top_k]
        except Exception as e:
            # Safe fallback if feature names cannot be retrieved
            return []

    def _get_feature_names(self):
        try:
            if hasattr(self.vectorizer, "get_feature_names_out"):
                return self.vectorizer.get_feature_names_out()
            return None
        except Exception:
            return None


# Global singleton instance
_predictor_instance = None


def predict_resume(text: str) -> Dict[str, Any]:
    """Public convenience function for inference."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ResumePredictor()
    return _predictor_instance.predict_resume(text)
