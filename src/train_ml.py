"""
Classical Machine Learning Models Training & Tuning Module.
Implements:
1. Multinomial Naive Bayes
2. Logistic Regression (with balanced class weighting)
3. Linear Support Vector Machine (LinearSVC with balanced class weighting)
4. Hyperparameter tuning exclusively on Validation set.
5. Serialization of best classical model.
"""
import time
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

try:
    from src.config import (
        PROCESSED_DATA_DIR, BEST_CLASSICAL_MODEL_PATH, RANDOM_SEED
    )
    from src.features import build_tfidf_features, encode_labels
except ImportError:
    from config import (
        PROCESSED_DATA_DIR, BEST_CLASSICAL_MODEL_PATH, RANDOM_SEED
    )
    from features import build_tfidf_features, encode_labels


def train_and_evaluate_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    model_name: str
) -> Dict[str, Any]:
    """Train a single classical model and evaluate on validation set."""
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_val_pred = model.predict(X_val)
    
    acc = accuracy_score(y_val, y_val_pred)
    macro_p = precision_score(y_val, y_val_pred, average='macro', zero_division=0)
    macro_r = recall_score(y_val, y_val_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
    
    return {
        "model_name": model_name,
        "model": model,
        "train_time_sec": round(train_time, 4),
        "val_accuracy": round(acc, 4),
        "val_macro_precision": round(macro_p, 4),
        "val_macro_recall": round(macro_r, 4),
        "val_macro_f1": round(macro_f1, 4),
        "val_weighted_f1": round(weighted_f1, 4),
        "val_predictions": y_val_pred
    }


def train_classical_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame
) -> Tuple[Dict[str, Any], Any]:
    """
    Train all three required classical baselines on training split and tune/evaluate on validation split.
    """
    # 1. Build features (fit strictly on train)
    vectorizer, X_train, X_val, _ = build_tfidf_features(
        train_texts=train_df["Resume_str"].tolist(),
        val_texts=val_df["Resume_str"].tolist(),
        save_vectorizer=True
    )
    
    # 2. Encode labels
    encoder, y_train, y_val, _ = encode_labels(
        train_labels=train_df["Category"].tolist(),
        val_labels=val_df["Category"].tolist(),
        save_encoder=True
    )
    
    # Define models
    models_to_train = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(
            C=5.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),
        "Linear SVM (LinearSVC)": LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_SEED
        )
    }
    
    results = {}
    best_model_name = None
    best_macro_f1 = -1.0
    best_model_obj = None
    
    print("Training Classical Models on TF-IDF features:")
    for name, clf in models_to_train.items():
        res = train_and_evaluate_model(clf, X_train, y_train, X_val, y_val, name)
        results[name] = res
        print(f"  -> {name:25s} | Val Acc: {res['val_accuracy']:.4f} | Val Macro-F1: {res['val_macro_f1']:.4f} | Time: {res['train_time_sec']:.2f}s")
        
        if res["val_macro_f1"] > best_macro_f1:
            best_macro_f1 = res["val_macro_f1"]
            best_model_name = name
            best_model_obj = clf
            
    print(f"\nBest Classical Model on Validation Set: {best_model_name} (Macro-F1: {best_macro_f1:.4f})")
    
    # Save best classical model
    joblib.dump(best_model_obj, BEST_CLASSICAL_MODEL_PATH)
    
    return results, best_model_obj


if __name__ == "__main__":
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    results, best_clf = train_classical_models(train_df, val_df)
