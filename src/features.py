"""
Feature Engineering Module for Resume Classification.
Implements:
1. TF-IDF Feature Extraction with configurable n-grams (Unigrams vs Unigrams+Bigrams)
2. Strict training-only fitting to prevent data leakage
3. Inspection of top weighted TF-IDF terms per category
4. Token sequence encoding and vocabulary building for Neural (LSTM/GRU) models.
"""
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

try:
    from src.config import (
        TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_MAX_DF, TFIDF_NGRAM_RANGE,
        TFIDF_VECTORIZER_PATH, LABEL_ENCODER_PATH, TOKENIZER_PATH,
        MAX_VOCAB_SIZE, MAX_SEQ_LEN
    )
    from src.preprocessing import preprocess_text
except ImportError:
    from config import (
        TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_MAX_DF, TFIDF_NGRAM_RANGE,
        TFIDF_VECTORIZER_PATH, LABEL_ENCODER_PATH, TOKENIZER_PATH,
        MAX_VOCAB_SIZE, MAX_SEQ_LEN
    )
    from preprocessing import preprocess_text


class NeuralTokenizer:
    """
    Tokenizer and vocabulary builder for sequence models.
    Learned ONLY on training corpus.
    """
    def __init__(self, max_vocab: int = MAX_VOCAB_SIZE, max_len: int = MAX_SEQ_LEN):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.is_fitted = False
        
    def fit(self, texts: List[str]):
        from collections import Counter
        all_tokens = []
        for text in texts:
            cleaned = preprocess_text(text)
            all_tokens.extend(cleaned.split())
            
        counts = Counter(all_tokens)
        most_common = counts.most_common(self.max_vocab - 2)
        
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        self.is_fitted = True
        return self
        
    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before transform.")
            
        sequences = []
        for text in texts:
            cleaned = preprocess_text(text)
            tokens = cleaned.split()
            seq = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in tokens[:self.max_len]]
            # Pad with 0s if shorter than max_len
            if len(seq) < self.max_len:
                seq = seq + [0] * (self.max_len - len(seq))
            sequences.append(seq)
            
        return np.array(sequences, dtype=np.int64)


def build_tfidf_features(
    train_texts: List[str],
    val_texts: Optional[List[str]] = None,
    test_texts: Optional[List[str]] = None,
    ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
    max_features: int = TFIDF_MAX_FEATURES,
    min_df: int = TFIDF_MIN_DF,
    max_df: float = TFIDF_MAX_DF,
    save_vectorizer: bool = True
) -> Tuple[TfidfVectorizer, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fits TF-IDF vectorizer exclusively on training texts and transforms val/test data.
    """
    # Custom stop words avoiding technical domain removals
    custom_stop = [
        'state', 'city', 'name', 'company', 'url_ref', 'email_ref', 'phone_ref',
        'and', 'the', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from',
        'an', 'as', 'is', 'was', 'are', 'that', 'all', 'etc'
    ]
    
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text,
        stop_words=custom_stop,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True
    )
    
    # Fit strictly on train
    X_train = vectorizer.fit_transform(train_texts)
    
    X_val = vectorizer.transform(val_texts) if val_texts is not None else None
    X_test = vectorizer.transform(test_texts) if test_texts is not None else None
    
    if save_vectorizer:
        joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
        
    return vectorizer, X_train, X_val, X_test


def encode_labels(
    train_labels: List[str],
    val_labels: Optional[List[str]] = None,
    test_labels: Optional[List[str]] = None,
    save_encoder: bool = True
) -> Tuple[LabelEncoder, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fit label encoder on training classes and transform splits.
    """
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_labels)
    y_val = encoder.transform(val_labels) if val_labels is not None else None
    y_test = encoder.transform(test_labels) if test_labels is not None else None
    
    if save_encoder:
        joblib.dump(encoder, LABEL_ENCODER_PATH)
        
    return encoder, y_train, y_val, y_test


def build_neural_sequences(
    train_texts: List[str],
    val_texts: Optional[List[str]] = None,
    test_texts: Optional[List[str]] = None,
    save_tokenizer: bool = True
) -> Tuple[NeuralTokenizer, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fits NeuralTokenizer strictly on training texts and converts all splits to padded integer sequences.
    """
    tokenizer = NeuralTokenizer(max_vocab=MAX_VOCAB_SIZE, max_len=MAX_SEQ_LEN)
    tokenizer.fit(train_texts)
    
    X_train_seq = tokenizer.transform(train_texts)
    X_val_seq = tokenizer.transform(val_texts) if val_texts is not None else None
    X_test_seq = tokenizer.transform(test_texts) if test_texts is not None else None
    
    if save_tokenizer:
        joblib.dump(tokenizer, TOKENIZER_PATH)
        
    return tokenizer, X_train_seq, X_val_seq, X_test_seq


if __name__ == "__main__":
    train_sample = ["Senior Python developer with AWS and SQL experience.", "Executive chef managing kitchen operations and menus."]
    val_sample = ["Experienced Java and C++ engineer."]
    
    vec, X_tr, X_va, _ = build_tfidf_features(train_sample, val_sample, save_vectorizer=False)
    print("TF-IDF Train shape:", X_tr.shape)
    print("TF-IDF Val shape:  ", X_va.shape)
    
    tok, seq_tr, seq_va, _ = build_neural_sequences(train_sample, val_sample, save_tokenizer=False)
    print("Neural seq Train shape:", seq_tr.shape)
    print("Neural seq Val shape:  ", seq_va.shape)
