import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

nb.metadata = {
    'language_info': {
        'name': 'python',
        'version': '3.14.2'
    },
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    }
}

cells = []

# Section 1: Problem Understanding
cells.append(nbf.v4.new_markdown_cell('''# SAMATRIX RESUMEFORGE 2026: End-to-End Resume Classification System
**Lead ML/NLP Engineer**

---
## 1. Problem Understanding
- **Task**: Multiclass Text Classification of professional resumes across 24 distinct categories.
- **Input**: Raw text or extracted text from multi-page PDF resumes.
- **Output**: Predicted professional category along with decision confidence / margin score.
- **Evaluation Metric**: **Macro-F1** is the primary metric to account for class imbalance, supported by Accuracy, Weighted-F1, Precision, Recall, and Confusion Matrix heatmaps.
- **Pipeline Flow**:
  `Raw Resume / PDF Document` -> `Text Extraction` -> `Reproducible Preprocessing` -> `Feature Engineering (TF-IDF / Learned Embeddings)` -> `Model Inference` -> `Predicted Category & Decision Metric`
'''))

# Section 2: Environment Setup & Data Loading
cells.append(nbf.v4.new_markdown_cell('''## 2. Dataset Loading & Inspection'''))
cells.append(nbf.v4.new_code_cell('''import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to sys.path
notebook_dir = Path.cwd()
project_root = notebook_dir.parent if notebook_dir.name == 'notebooks' else notebook_dir
sys.path.insert(0, str(project_root))

from src.data_loader import load_raw_csv, inspect_dataset_sources
from src.data_validation import run_data_quality_audit, clean_and_split_data
from src.preprocessing import preprocess_text
from src.features import build_tfidf_features, encode_labels, build_neural_sequences
from src.train_ml import train_classical_models
from src.train_dl import train_neural_model
from src.evaluate import evaluate_all_models_on_test, plot_confusion_matrix
from src.error_analysis import analyze_misclassifications
from src.inference import ResumeClassificationPipeline

# Load dataset
df_raw = load_raw_csv(project_root / 'data/raw/Resume.csv')
print(f"Dataset Shape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}")
df_raw.head(3)
'''))

# Section 3: Data Quality Check
cells.append(nbf.v4.new_markdown_cell('''## 3. Data Quality & Validation Audit
Checking for missing text, empty resumes, exact duplicate texts, and label conflicts.
'''))
cells.append(nbf.v4.new_code_cell('''audit = run_data_quality_audit(df_raw)
print(f"Total Records: {audit['total_records']}")
print(f"Total Classes: {audit['num_classes']}")
print(f"Missing Values: {audit['missing_values']}")
print(f"Empty Resumes (<1 char): {audit['empty_resume_count']} (IDs: {audit['empty_resume_ids']})")
print(f"Duplicate Exact Texts: {audit['duplicate_exact_text_count']}")
print(f"Duplicate Text Label Conflicts: {audit['duplicate_texts_with_conflicting_labels']}")
print(f"Char Length Stats: {audit['text_char_length']}")
print(f"Word Count Stats: {audit['text_word_count']}")
'''))

# Section 4: EDA
cells.append(nbf.v4.new_markdown_cell('''## 4. Exploratory Data Analysis (EDA)
Visualizing class distribution, resume length characteristics, top vocabulary, n-grams, and class-discriminative terms.
'''))
cells.append(nbf.v4.new_code_cell('''from src.eda import (
    plot_class_distribution, plot_resume_lengths,
    plot_top_frequent_words, generate_wordclouds,
    plot_ngrams_analysis, plot_class_discriminative_terms
)

# 1. Class distribution
plot_class_distribution(df_raw, save_path=str(project_root / 'reports/figures/01_class_distribution.png'))

# 2. Length distributions
plot_resume_lengths(df_raw, save_path=str(project_root / 'reports/figures/02_resume_length_distribution.png'))

# 3. Top frequent terms & N-grams
df_clean = df_raw.copy()
df_clean['cleaned_text'] = df_clean['Resume_str'].fillna('').apply(preprocess_text)
df_clean = df_clean[df_clean['cleaned_text'].str.strip().str.len() > 0]

plot_top_frequent_words(df_clean, save_path=str(project_root / 'reports/figures/03_top_frequent_words.png'))
plot_ngrams_analysis(df_clean, save_path=str(project_root / 'reports/figures/06_ngram_analysis.png'))
plot_class_discriminative_terms(df_clean, save_path=str(project_root / 'reports/figures/07_class_discriminative_terms.png'))

print("All EDA visualizations successfully generated and saved to reports/figures/")
'''))

# Section 5: Preprocessing & Data Splitting
cells.append(nbf.v4.new_markdown_cell('''## 5. Text Preprocessing & Stratified Splitting
- Preprocessing preserves technical tokens (C++, C#, .NET, Python, AWS, SQL) while removing markup and normalizing contact placeholders.
- Stratified 70/15/15 train/val/test split prevents train/test leakage.
'''))
cells.append(nbf.v4.new_code_cell('''df_cleaned, train_df, val_df, test_df = clean_and_split_data(df_raw)
print("Cleaned Total:", len(df_cleaned))
print("Train Split (70%):", len(train_df))
print("Val Split   (15%):", len(val_df))
print("Test Split  (15%):", len(test_df))
'''))

# Section 6: Classical ML Models
cells.append(nbf.v4.new_markdown_cell('''## 6. Feature Engineering & Classical ML Models (TF-IDF Baselines)
- TF-IDF unigram + bigram features (10,000 max features, sublinear term frequency).
- Models: Multinomial Naive Bayes, Logistic Regression (balanced), Linear SVM (LinearSVC, balanced).
'''))
cells.append(nbf.v4.new_code_cell('''classical_results, best_clf = train_classical_models(train_df, val_df)
'''))

# Section 7: Deep Learning Model
cells.append(nbf.v4.new_markdown_cell('''## 7. Deep Learning Model (BiLSTM Neural Classifier)
- Padded integer token sequences with learned word embedding layer.
- Bidirectional LSTM layer for sequence feature extraction with dropout regularization.
- Validation early stopping.
'''))
cells.append(nbf.v4.new_code_cell('''neural_model, neural_metrics = train_neural_model(train_df, val_df)
print("Neural Model Training Metrics:", neural_metrics)
'''))

# Section 8: Test Set Evaluation & Comparison
cells.append(nbf.v4.new_markdown_cell('''## 8. Model Evaluation & Comparison on Untouched Test Set'''))
cells.append(nbf.v4.new_code_cell('''comp_df, summary, best_model_name = evaluate_all_models_on_test(test_df)
print("=== FINAL MODEL COMPARISON ON TEST SET ===")
comp_df
'''))

# Section 9: Error Analysis
cells.append(nbf.v4.new_markdown_cell('''## 9. Diagnostic Error Analysis'''))
cells.append(nbf.v4.new_code_cell('''errors_df, error_summary = analyze_misclassifications(test_df, split_name='test')
print("Total Test Misclassifications:", error_summary['total_errors'], "/", error_summary['total_evaluated'])
print("Root Causes:", error_summary['root_cause_distribution'])
errors_df.head(5)
'''))

# Section 10: Production Inference Demonstration
cells.append(nbf.v4.new_markdown_cell('''## 10. End-to-End Inference Demonstration (Text & PDF)'''))
cells.append(nbf.v4.new_code_cell('''pipeline = ResumeClassificationPipeline()

test_samples = [
    ('Information Technology', 'Experienced Software Engineer with proficiency in Python, Django, REST APIs, PostgreSQL, Docker, AWS EC2, and CI/CD pipelines.'),
    ('Chef / Culinary', 'Head Executive Chef with 12 years managing fine dining culinary operations, kitchen inventory, pastry creation, and banquet menu planning.'),
    ('Aviation', 'FAA Certified Airline Transport Pilot with 5000 flight hours across multi-engine turboprops and commercial Boeing aircraft.'),
    ('Healthcare', 'Registered Nurse (RN) specializing in ICU patient care, vital signs monitoring, medical dosage administration, and trauma triage.')
]

for label, text in test_samples:
    res = pipeline.predict_text(text)
    print(f"Domain: {label:25s} -> Predicted: {res['predicted_category']:22s} | Score: {res['confidence']}")
'''))

nb.cells = cells

with open('notebooks/resume_classification_analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Notebook created successfully at notebooks/resume_classification_analysis.ipynb')
