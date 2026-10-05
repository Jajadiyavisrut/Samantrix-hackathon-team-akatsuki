"""
Configuration module for Resume Classification System.
Defines paths, random seeds, split ratios, and hyperparameters.
"""
import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
ERROR_ANALYSIS_DIR = REPORTS_DIR / "error_analysis"

# Raw Data Files
RAW_CSV_PATH = RAW_DATA_DIR / "Resume.csv"
RAW_XLSX_PATH = RAW_DATA_DIR / "Resume.xlsx"

# Processed Data Files
PROCESSED_CSV_PATH = PROCESSED_DATA_DIR / "cleaned_resumes.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

# Model & Artifact Paths
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
BEST_CLASSICAL_MODEL_PATH = MODELS_DIR / "best_classical_model.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
TOKENIZER_PATH = MODELS_DIR / "neural_tokenizer.joblib"
NEURAL_MODEL_PATH = MODELS_DIR / "neural_lstm_model.pt"
FINAL_PIPELINE_METADATA_PATH = MODELS_DIR / "pipeline_metadata.json"

# Random Seed & Reproducibility
RANDOM_SEED = 42

# Train / Val / Test Split Ratios (70% Train, 15% Val, 15% Test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# TF-IDF Feature Parameters
TFIDF_MAX_FEATURES = 10000
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.85
TFIDF_NGRAM_RANGE = (1, 2)

# Neural Model Hyperparameters (Optimized for Fast, Accurate Training)
MAX_VOCAB_SIZE = 8000
MAX_SEQ_LEN = 150
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.2
BATCH_SIZE = 64
LEARNING_RATE = 0.003
NUM_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 3

# Ensure required directories exist
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
          REPORTS_DIR, FIGURES_DIR, METRICS_DIR, ERROR_ANALYSIS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
