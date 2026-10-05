"""
Deep Learning Training Module for Resume Classification.
Implements a bidirectional LSTM neural classifier in PyTorch:
- Token sequences & learned word embeddings
- Bidirectional LSTM feature extraction
- Dropout regularization & Dense classification layer
- Validation early stopping
- Model checkpointing to models/neural_lstm_model.pt
"""
import os
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from src.config import (
        PROCESSED_DATA_DIR, NEURAL_MODEL_PATH, TOKENIZER_PATH, LABEL_ENCODER_PATH,
        MAX_VOCAB_SIZE, MAX_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS,
        DROPOUT, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE, RANDOM_SEED
    )
    from src.features import build_neural_sequences, encode_labels, NeuralTokenizer
except ImportError:
    from config import (
        PROCESSED_DATA_DIR, NEURAL_MODEL_PATH, TOKENIZER_PATH, LABEL_ENCODER_PATH,
        MAX_VOCAB_SIZE, MAX_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS,
        DROPOUT, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE, RANDOM_SEED
    )
    from features import build_neural_sequences, encode_labels, NeuralTokenizer

# Set deterministic PyTorch seeds
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


class ResumeDataset(Dataset):
    """PyTorch Dataset for Resume integer sequences and labels."""
    def __init__(self, sequences: np.ndarray, labels: np.ndarray = None):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM Neural Network for Text Classification.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(x))  # (batch_size, seq_len, emb_dim)
        lstm_out, (hn, cn) = self.lstm(embedded)   # lstm_out: (batch_size, seq_len, hidden_dim * 2)
        
        # Max-pooling across time dimension for richer sequence representation
        pooled, _ = torch.max(lstm_out, dim=1)      # (batch_size, hidden_dim * 2)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)                   # (batch_size, num_classes)
        return logits


def train_neural_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame
) -> Tuple[BiLSTMClassifier, Dict[str, Any]]:
    """
    Train BiLSTM Neural model with early stopping on validation macro-F1.
    """
    # 1. Build Tokenizer and Sequences
    tokenizer, X_train_seq, X_val_seq, _ = build_neural_sequences(
        train_texts=train_df["Resume_str"].tolist(),
        val_texts=val_df["Resume_str"].tolist(),
        save_tokenizer=True
    )
    
    # 2. Encode labels
    if os.path.exists(LABEL_ENCODER_PATH):
        encoder = joblib.load(LABEL_ENCODER_PATH)
        y_train = encoder.transform(train_df["Category"].tolist())
        y_val = encoder.transform(val_df["Category"].tolist())
    else:
        encoder, y_train, y_val, _ = encode_labels(
            train_labels=train_df["Category"].tolist(),
            val_labels=val_df["Category"].tolist(),
            save_encoder=True
        )
        
    num_classes = len(encoder.classes_)
    vocab_size = len(tokenizer.word2idx)
    
    train_dataset = ResumeDataset(X_train_seq, y_train)
    val_dataset = ResumeDataset(X_val_seq, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Neural Model on Device: {device} (Vocab size: {vocab_size}, Classes: {num_classes})")
    
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    best_val_macro_f1 = -1.0
    patience_counter = 0
    best_model_state = None
    best_metrics = {}
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Training loop
        model.train()
        train_loss = 0.0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            
        train_loss = train_loss / len(train_dataset)
        
        # Validation loop
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs = seqs.to(device)
                outputs = model(seqs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        val_macro_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        val_weighted_f1 = f1_score(val_targets, val_preds, average='weighted', zero_division=0)
        
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_macro_f1:.4f}")
        
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_metrics = {
                "val_accuracy": round(val_acc, 4),
                "val_macro_f1": round(val_macro_f1, 4),
                "val_weighted_f1": round(val_weighted_f1, 4),
                "val_precision": round(precision_score(val_targets, val_preds, average='macro', zero_division=0), 4),
                "val_recall": round(recall_score(val_targets, val_preds, average='macro', zero_division=0), 4),
                "epoch": epoch
            }
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Macro-F1: {best_val_macro_f1:.4f}")
                break
                
    # Restore best weights and save
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_classes": num_classes,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "metrics": best_metrics
    }, NEURAL_MODEL_PATH)
    
    print(f"Neural model checkpoint saved to: {NEURAL_MODEL_PATH}")
    return model, best_metrics


if __name__ == "__main__":
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    neural_model, metrics = train_neural_model(train_df, val_df)
