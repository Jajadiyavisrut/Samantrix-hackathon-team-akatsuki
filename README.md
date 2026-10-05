# SAMATRIX RESUMEFORGE 2026 — End-to-End NLP Resume Classifier

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Demo-Streamlit-red.svg)](app/app.py)
[![Clean Architecture](https://img.shields.io/badge/Design-Production--Grade-brightgreen.svg)]()

> **Autonomous AI-Driven Resume Categorization & Feature Attribution System.**  
> Built for the SAMATRIX RESUMEFORGE 2026 Hackathon.

---

## 1. Problem Statement & Objectives

Automated recruitment workflows require screening thousands of resumes across diverse professional domains. Existing systems often suffer from:
- Brittleness when encountering technical symbols (`C++`, `C#`, `.NET`).
- Data leakage between training and testing partitions.
- Opaque, black-box decision boundaries.
- Computational latency unsuitable for production microservices.

**ResumeForge 2026** delivers:
1. **Strict Zero-Leakage Architecture:** Preprocessing and vectorizers fit exclusively on training data; isolated test set evaluated strictly once.
2. **Subword & Technical Token Preservation:** Domain-aware regex preservation for modern developer toolchains.
3. **High-Performance Representation:** Dual-channel Word + Character N-Gram FeatureUnion capturing both high-level semantic qualifications and fine-grained subword terminology.
4. **Exact Linear Feature Attribution:** Mathematical feature-level explainability for every prediction.
5. **Interactive Streamlit Web Interface:** Real-time PDF/TXT parsing and domain prediction.

---

## 2. Workspace & Project Architecture

```
SAMATRIX HACKATHON/
│
├── data/
│   ├── raw/
│   │   └── Resume.csv                  # Immutable raw source dataset
│   └── processed/
│       ├── clean_resumes.csv           # Deduplicated, anomaly-free dataset (2,481 rows)
│       ├── train.csv                   # 70% Stratified Training partition (1,736 rows)
│       ├── val.csv                     # 15% Stratified Validation partition (372 rows)
│       └── test.csv                    # 15% Stratified Holdout Test partition (373 rows)
│
├── notebooks/
│   ├── 01_data_audit.ipynb             # Data integrity, schema, and duplicate forensics
│   ├── 02_eda.ipynb                    # Distributions, n-grams, wordclouds, similarity matrices
│   ├── 03_model_experiments.ipynb      # Multi-model benchmarking across Word, Char, and Union TF-IDF
│   ├── 04_error_analysis.ipynb         # Confusion matrix, per-class F1, misclassification diagnosis
│   └── 05_final_model.ipynb            # Retraining on Train+Val, isolated test evaluation
│
├── src/
│   ├── data/
│   │   ├── clean_data.py               # Data deduplication & corruption removal pipeline
│   │   ├── split_data.py               # 70/15/15 stratified leakage-free data splitter
│   │   └── eda.py                      # Publication-grade exploratory analysis generator
│   ├── preprocessing/
│   │   └── text_cleaner.py             # Domain-aware normalizer (protects C++, C#, .NET, etc.)
│   ├── features/
│   │   └── feature_builder.py          # Word, Char, and FeatureUnion vectorizers
│   ├── models/
│   │   ├── train_experiments.py        # Benchmark suite (Linear, Naive Bayes, MLP, Transformer)
│   │   ├── train_final.py              # Production model retrainer & single-run test evaluator
│   │   └── error_analysis.py           # Diagnostic confusion matrix and error logger
│   ├── evaluation/
│   │   └── metrics.py                  # Standardized Accuracy, Macro/Weighted F1, latency benchmark
│   └── inference/
│       └── predictor.py                # Public inference API with calibrated confidence & attribution
│
├── models/
│   ├── final_classifier.joblib         # Calibrated production classifier
│   ├── feature_vectorizer.joblib       # Fitted Word+Char FeatureUnion
│   └── model_metadata.joblib           # Class taxonomy and model performance metadata
│
├── reports/
│   ├── figures/                        # High-resolution 300 DPI analytical charts
│   ├── DATA_AUDIT.md                   # Forensic dataset audit report
│   ├── DATA_CLEANING_REPORT.md         # Deduplication & anomaly removal audit trail
│   ├── EDA_SUMMARY.md                  # Exploratory data analysis insights
│   ├── MODEL_EXPERIMENTS_SUMMARY.md    # Multi-model comparative benchmark report
│   ├── ERROR_ANALYSIS.md               # Diagnostic confusion matrix and failure mode study
│   ├── FINAL_TEST_REPORT.md            # Unbiased final test set evaluation
│   └── FINAL_REPORT.md                 # Complete hackathon executive submission
│
├── app/
│   └── app.py                          # Production Streamlit web application
│
├── requirements.txt                    # Python environment dependencies
├── README.md                           # Documentation & execution guide
└── .gitignore                          # Standard exclusion rules
```

---

## 3. Dataset & Professional Categories

The canonical dataset contains **2,481 clean resumes** across **24 professional domains**:

| Category | Clean Count | % of Dataset | Category | Clean Count | % of Dataset |
| :--- | :---: | :---: | :--- | :---: | :---: |
| `INFORMATION-TECHNOLOGY` | 120 | 4.84% | `CONSULTANT` | 115 | 4.64% |
| `BUSINESS-DEVELOPMENT` | 119 | 4.80% | `CONSTRUCTION` | 112 | 4.51% |
| `ADVOCATE` | 118 | 4.76% | `PUBLIC-RELATIONS` | 111 | 4.47% |
| `CHEF` | 118 | 4.76% | `HR` | 110 | 4.43% |
| `ENGINEERING` | 118 | 4.76% | `DESIGNER` | 107 | 4.31% |
| `ACCOUNTANT` | 118 | 4.76% | `ARTS` | 103 | 4.15% |
| `FINANCE` | 117 | 4.71% | `TEACHER` | 102 | 4.11% |
| `FITNESS` | 117 | 4.71% | `APPAREL` | 97 | 3.91% |
| `SALES` | 116 | 4.68% | `DIGITAL-MEDIA` | 96 | 3.87% |
| `AVIATION` | 116 | 4.68% | `AGRICULTURE` | 63 | 2.54% |
| `BANKING` | 115 | 4.64% | `AUTOMOBILE` | 36 | 1.45% |
| `HEALTHCARE` | 115 | 4.64% | `BPO` | 22 | 0.89% |

---

## 4. Pipeline Execution & Quickstart

### Step 1: Environment Installation
```bash
pip install -r requirements.txt
```

### Step 2: Run End-to-End Pipeline
Execute each stage independently or run the master script:
```bash
# 1. Clean raw data & deduplicate
python -m src.data.clean_data

# 2. Perform stratified split (70% train, 15% val, 15% test)
python -m src.data.split_data

# 3. Generate complete EDA and figures
python -m src.data.eda

# 4. Run model benchmark suite
python -m src.models.train_experiments

# 5. Train final model & evaluate on isolated test set
python -m src.models.train_final
```

### Step 3: Launch Streamlit Web Application
```bash
streamlit run app/app.py
```

---

## 5. Model Inference Example

```python
from predict import predict_resume

text = \"\"\"
Experienced Full Stack Engineer with proficiency in Python, C++, C#, Docker,
microservices, PostgreSQL, and Kubernetes. Led backend engineering teams.
\"\"\"

result = predict_resume(text)
print("Predicted Domain:", result["predicted_category"])
print("Confidence:", result["confidence"])
print("Key Driving Features:", result["contributing_features"])
```
