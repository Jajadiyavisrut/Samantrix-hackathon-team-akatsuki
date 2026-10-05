"""
Script: notebooks/generate_all_notebooks.py
Purpose: Generates clean, production-grade Jupyter notebooks for all 5 required phases:
01_data_audit.ipynb
02_eda.ipynb
03_model_experiments.ipynb
04_error_analysis.ipynb
05_final_model.ipynb
All notebooks dynamically locate project root with pathlib and use root-level imports.
"""

import os
from pathlib import Path
import nbformat as nbf


def make_notebook(cells, filepath):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    with open(filepath, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[Notebooks] Generated {filepath}")


def create_01_data_audit():
    cells = [
        nbf.v4.new_markdown_cell("""# Notebook 01: Forensic Data & Workspace Audit
**Samatrix ResumeForge 2026**
**Lead ML Engineer & Data Scientist**

This notebook performs a complete audit of the raw data, checks shapes, column types, missing values, empty/corrupted records, and duplicate texts.
"""),
        nbf.v4.new_code_cell("""import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

# Dynamically ensure project root is on sys.path
root_dir = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Load raw dataset
raw_csv_path = root_dir / "data" / "raw" / "Resume.csv"
if not raw_csv_path.exists():
    raw_csv_path = root_dir / "Resume" / "Resume.csv"

raw_df = pd.read_csv(raw_csv_path)
print("Raw Dataset Dimensions:", raw_df.shape)
print("Columns:", raw_df.columns.tolist())
raw_df.head()
"""),
        nbf.v4.new_code_cell("""# Check Missing Values and Data Types
print("Missing values per column:\\n", raw_df.isnull().sum())
print("\\nData types:\\n", raw_df.dtypes)
"""),
        nbf.v4.new_code_cell("""# Check Duplicates in Resume_str
dup_mask = raw_df.duplicated(subset=['Resume_str'], keep=False)
print("Number of rows involved in duplicate resume text:", dup_mask.sum())
raw_df[dup_mask].sort_values('Resume_str')[['ID', 'Category', 'Resume_str']]
"""),
        nbf.v4.new_code_cell("""# Check Corrupted / Empty Resumes
word_lens = raw_df['Resume_str'].apply(lambda x: len(str(x).split()))
empty_rows = raw_df[word_lens < 10]
print("Corrupted / Empty rows count:", len(empty_rows))
empty_rows[['ID', 'Category', 'Resume_str']]
"""),
        nbf.v4.new_code_cell("""# Run Automated Cleaning and Save Clean Data
from clean_data import clean_raw_data

clean_df = clean_raw_data(
    raw_path='../data/raw/Resume.csv',
    processed_path='../data/processed/clean_resumes.csv',
    report_path='../reports/DATA_CLEANING_REPORT.md'
)
print("Clean dataset size:", len(clean_df))
""")
    ]
    make_notebook(cells, "notebooks/01_data_audit.ipynb")


def create_02_eda():
    cells = [
        nbf.v4.new_markdown_cell("""# Notebook 02: Exploratory Data Analysis (EDA)
**Samatrix ResumeForge 2026**

This notebook performs statistical and visual exploration of professional categories, word/char length distributions, n-grams, wordclouds, and category similarity matrices.
"""),
        nbf.v4.new_code_cell("""import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

root_dir = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

clean_csv = root_dir / "data" / "processed" / "clean_resumes.csv"
if not clean_csv.exists():
    from clean_data import clean_raw_data
    clean_raw_data(processed_path=str(clean_csv))

df = pd.read_csv(clean_csv)
print("Total records:", len(df))
print("Categories:", df['Category'].nunique())
"""),
        nbf.v4.new_code_cell("""# Category Distribution
plt.figure(figsize=(12, 6))
df['Category'].value_counts().plot(kind='bar', color='#2b5c8f')
plt.title("Class Distribution Across 24 Categories")
plt.xlabel("Category")
plt.ylabel("Number of Resumes")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
"""),
        nbf.v4.new_code_cell("""# Word and Character Length Analysis
df['word_len'] = df['Resume_str'].apply(lambda x: len(str(x).split()))
df['char_len'] = df['Resume_str'].str.len()
print("Word length summary:\\n", df['word_len'].describe())
print("\\nCharacter length summary:\\n", df['char_len'].describe())
"""),
        nbf.v4.new_code_cell("""# Trigger Full EDA Pipeline and Generate Figures
from eda import run_full_eda

run_full_eda(
    data_path=str(clean_csv),
    output_fig_dir=str(root_dir / "reports" / "figures"),
    output_report_path=str(root_dir / "reports" / "EDA_SUMMARY.md")
)
""")
    ]
    make_notebook(cells, "notebooks/02_eda.ipynb")


def create_03_model_experiments():
    cells = [
        nbf.v4.new_markdown_cell("""# Notebook 03: NLP Model Experiments & Benchmarking
**Samatrix ResumeForge 2026**

This notebook benchmarks multiple feature extraction methods (Word TF-IDF, Char TF-IDF, Word+Char FeatureUnion) and models (LinearSVC, Logistic Regression, ComplementNB, SGDClassifier, MLP, Transformer Embeddings).
"""),
        nbf.v4.new_code_cell("""import os, sys
from pathlib import Path
import pandas as pd

root_dir = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

train_df = pd.read_csv(root_dir / "data" / "processed" / "train.csv")
val_df = pd.read_csv(root_dir / "data" / "processed" / "val.csv")
print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
"""),
        nbf.v4.new_code_cell("""# Load Model Benchmark Summary Leaderboard
report_csv = root_dir / "reports" / "model_comparison.csv"
if report_csv.exists():
    df_results = pd.read_csv(report_csv)
    display(df_results)
else:
    print("Benchmark summary generated during training.")
""")
    ]
    make_notebook(cells, "notebooks/03_model_experiments.ipynb")


def create_04_error_analysis():
    cells = [
        nbf.v4.new_markdown_cell("""# Notebook 04: Diagnostic Error Analysis
**Samatrix ResumeForge 2026**

Examines the confusion matrix, per-class F1 performance, and investigates specific misclassified resumes to evaluate domain overlap.
"""),
        nbf.v4.new_code_cell("""import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

root_dir = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from preprocessing import clean_resume_text
from model_utils import run_error_analysis

# Load Model and Validation Data
val_df = pd.read_csv(root_dir / "data" / "processed" / "val.csv")
clf = joblib.load(root_dir / "models" / "final_classifier.joblib")
vec = joblib.load(root_dir / "models" / "feature_vectorizer.joblib")

val_cleaned = val_df['Resume_str'].apply(clean_resume_text)
X_val = vec.transform(val_cleaned)
y_val = val_df['Category']
y_pred = clf.predict(X_val)

# Run Detailed Error Analysis
run_error_analysis(
    y_true=y_val,
    y_pred=y_pred,
    texts=val_df['Resume_str'],
    output_report_path=str(root_dir / "reports" / "ERROR_ANALYSIS.md"),
    output_fig_dir=str(root_dir / "reports" / "figures")
)
""")
    ]
    make_notebook(cells, "notebooks/04_error_analysis.ipynb")


def create_05_final_model():
    cells = [
        nbf.v4.new_markdown_cell("""# Notebook 05: Final Model Retraining & Production Evaluation
**Samatrix ResumeForge 2026**

Retrains the winning Word+Char Calibrated Linear architecture on combined Train+Validation partitions (2,108 resumes), evaluates strictly ONCE on the isolated Test set (373 resumes), and persists production artifacts.
"""),
        nbf.v4.new_code_cell("""import os, sys
from pathlib import Path

root_dir = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from train import run_training_pipeline

metrics = run_training_pipeline(
    models_dir=str(root_dir / "models"),
    reports_dir=str(root_dir / "reports"),
    figures_dir=str(root_dir / "reports" / "figures"),
    C=1.0,
    class_weight='balanced'
)
print("Final Test Accuracy:", metrics['accuracy'])
print("Final Test Macro-F1:", metrics['macro_f1'])
"""),
        nbf.v4.new_code_cell("""# Test Inference API on a New Sample
from predict import predict_resume

sample_text = \"\"\"
Senior Cloud Architect & Machine Learning Engineer with 8 years of experience.
Specialized in Python, PyTorch, TensorFlow, Kubernetes, Docker, AWS SageMaker, and CI/CD pipelines.
Designed enterprise-scale deep learning microservices and distributed data pipelines.
\"\"\"

result = predict_resume(sample_text)
print("Predicted Domain:", result['predicted_category'])
print("Confidence:", result['confidence'])
print("Top Alternative:", result['top_3_predictions'])
print("Contributing Features:", result['contributing_features'])
""")
    ]
    make_notebook(cells, "notebooks/05_final_model.ipynb")


if __name__ == "__main__":
    os.makedirs("notebooks", exist_ok=True)
    create_01_data_audit()
    create_02_eda()
    create_03_model_experiments()
    create_04_error_analysis()
    create_05_final_model()
    print("[Notebooks] All 5 Jupyter notebooks generated successfully with root-level imports.")
