"""
Model Evaluation and Comparison Module.
Computes comprehensive metrics on the test set:
- Accuracy
- Macro Precision, Recall, F1
- Weighted F1
- Per-class classification reports
- Confusion Matrix visualization
- Model comparison tables
"""
import os
import json
import joblib
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

try:
    from src.config import (
        PROCESSED_DATA_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR,
        TFIDF_VECTORIZER_PATH, LABEL_ENCODER_PATH, TOKENIZER_PATH,
        NEURAL_MODEL_PATH, BEST_CLASSICAL_MODEL_PATH
    )
    from src.train_dl import BiLSTMClassifier, ResumeDataset
    from torch.utils.data import DataLoader
except ImportError:
    from config import (
        PROCESSED_DATA_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR,
        TFIDF_VECTORIZER_PATH, LABEL_ENCODER_PATH, TOKENIZER_PATH,
        NEURAL_MODEL_PATH, BEST_CLASSICAL_MODEL_PATH
    )
    from train_dl import BiLSTMClassifier, ResumeDataset
    from torch.utils.data import DataLoader


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, Any]:
    """Calculate standard evaluation metrics."""
    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Macro Precision": round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "Macro Recall": round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "Macro F1": round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "Weighted F1": round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: str = None
) -> str:
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(16, 13))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5
    )
    plt.title(f'Confusion Matrix — {model_name} (Test Set)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted Category', fontsize=12, labelpad=10)
    plt.ylabel('True Category', fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    out_path = save_path or str(FIGURES_DIR / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def evaluate_all_models_on_test(
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """
    Evaluates all trained classical baselines and neural model on the UNTOUCHED test set.
    """
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    encoder = joblib.load(LABEL_ENCODER_PATH)
    class_names = list(encoder.classes_)
    
    y_test = encoder.transform(test_df["Category"].tolist())
    X_test_tfidf = vectorizer.transform(test_df["Resume_str"].tolist())
    
    # Evaluate Classical Models
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    
    # Load all models or train if evaluating standalone
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    X_train_tfidf = vectorizer.transform(train_df["Resume_str"].tolist())
    y_train = encoder.transform(train_df["Category"].tolist())
    
    models_dict = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1).fit(X_train_tfidf, y_train),
        "Logistic Regression": LogisticRegression(C=5.0, max_iter=1000, class_weight="balanced", random_state=42).fit(X_train_tfidf, y_train),
        "Linear SVM (LinearSVC)": LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=42).fit(X_train_tfidf, y_train)
    }
    
    all_metrics = []
    test_preds_dict = {}
    
    for name, model in models_dict.items():
        preds = model.predict(X_test_tfidf)
        test_preds_dict[name] = preds
        m = compute_metrics(y_test, preds, name)
        all_metrics.append(m)
        
        # Save individual confusion matrix
        cm = confusion_matrix(y_test, preds)
        plot_confusion_matrix(cm, class_names, name)
        
    # Evaluate Neural Model
    if os.path.exists(NEURAL_MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        tokenizer = joblib.load(TOKENIZER_PATH)
        X_test_seq = tokenizer.transform(test_df["Resume_str"].tolist())
        
        checkpoint = torch.load(NEURAL_MODEL_PATH, weights_only=False)
        neural_net = BiLSTMClassifier(
            vocab_size=checkpoint["vocab_size"],
            embedding_dim=checkpoint["embedding_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_classes=checkpoint["num_classes"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"]
        )
        neural_net.load_state_dict(checkpoint["model_state_dict"])
        neural_net.eval()
        
        test_dataset = ResumeDataset(X_test_seq)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        neural_preds = []
        with torch.no_grad():
            for seqs in test_loader:
                out = neural_net(seqs)
                p = torch.argmax(out, dim=1).numpy()
                neural_preds.extend(p)
                
        neural_preds = np.array(neural_preds)
        test_preds_dict["BiLSTM (Neural)"] = neural_preds
        m_neural = compute_metrics(y_test, neural_preds, "BiLSTM (Neural)")
        all_metrics.append(m_neural)
        
        cm_neural = confusion_matrix(y_test, neural_preds)
        plot_confusion_matrix(cm_neural, class_names, "BiLSTM (Neural)")
        
    comparison_df = pd.DataFrame(all_metrics)
    comparison_df.to_csv(METRICS_DIR / "model_comparison_test.csv", index=False)
    
    # Identify best overall model by Macro F1
    best_row = comparison_df.sort_values(by="Macro F1", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    
    # Save full detailed per-class report for best model
    best_preds = test_preds_dict[best_model_name]
    report_dict = classification_report(y_test, best_preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(METRICS_DIR / f"per_class_report_{best_model_name.lower().replace(' ', '_')}.csv")
    
    # Save summary json
    summary = {
        "best_model": best_model_name,
        "test_comparison": all_metrics,
        "best_model_metrics": best_row.to_dict()
    }
    with open(METRICS_DIR / "final_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    return comparison_df, summary, best_model_name


if __name__ == "__main__":
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    comp_df, summary, best_name = evaluate_all_models_on_test(test_df)
    print("\n=== FINAL TEST SET COMPARISON TABLE ===")
    print(comp_df.to_string(index=False))
    print(f"\nWinner: {best_name}")
