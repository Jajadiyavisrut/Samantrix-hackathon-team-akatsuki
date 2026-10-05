# SAMATRIX RESUMEFORGE 2026 — Comprehensive Hackathon Technical Report

**Project Title:** ResumeForge AI: Generalizable, Leakage-Free Resume Classification & Explainability  
**Author:** Lead ML Engineer, NLP Researcher, Software Architect, Data Scientist & Hackathon Strategist  
**Competition:** Samatrix ResumeForge 2026 Hackathon  
**Date:** 2026-10-05  

---

## 1. Problem Understanding & Strategic Objective

In modern human resource systems, unstructured curriculum vitae (CV) and resume categorization across dozens of competitive domains is a cornerstone requirement. The primary challenge is not merely achieving a high classification score on an arbitrary test partition, but **maximizing genuine generalization performance** while maintaining:
- Strict zero-leakage between training and holdout partitions.
- Full preservation of domain-specific technical vocabulary (`C++`, `C#`, `.NET`, `SQL`, `AWS`, `Docker`, etc.).
- Sub-millisecond inference latency for high-throughput recruitment platforms.
- Complete model-faithful explainability at the feature level without black-box approximations.

### Primary Guiding Principles:
1. **Honest, Defensible Evaluation:** Never fabricate metrics or tune against the test set. A legitimate, reproducible score is always superior to an inflated one.
2. **Subword & Lexical Robustness:** Job titles and technical skills vary in spelling, casing, punctuation, and abbreviations (e.g. *Node.js*, *CI/CD*, *DevOps*). Models must leverage subword n-grams alongside word-level collocations.
3. **Principled Model Selection:** Small-to-medium tabular/text datasets (~2,481 samples across 24 classes) exhibit high risk of overfitting with massive deep neural networks. Sparse, regularized linear models with sublinear term-frequency scaling provide mathematically superior sample efficiency and generalization.

---

## 2. Dataset Understanding

The canonical data source is `Resume/Resume.csv` (ingested into `data/raw/Resume.csv`):
- **Raw Observations:** 2,484 resumes across 4 columns: `ID`, `Resume_str`, `Resume_html`, and `Category`.
- **Target Space:** 24 professional domains spanning technical, corporate, service, and artistic industries.
- **Physical Document Reconciliation:** The workspace also contains `data/data` with 2,500 PDF files. A forensic comparison established that exactly 16 files in `data/data/FINANCE/` were operating-system duplicate downloads with the suffix `(1).pdf` (e.g., `20880935(1).pdf`). Removing these 16 artifacts results in exactly 2,484 PDFs mapping 1-to-1 with the IDs in `Resume.csv`.
- **Decision:** `Resume.csv` was chosen as the canonical data source because it provides deterministic row indexing, eliminates PDF parsing discrepancies, and contains verified ground-truth labels.

---

## 3. Data Quality & Leakage Protection

A multi-stage forensic audit (`reports/DATA_AUDIT.md` and `reports/DATA_CLEANING_REPORT.md`) identified:
1. **Corrupted Records (1 row):** `ID 12632728` (`BUSINESS-DEVELOPMENT`) contained 0 words, 21 spaces, and unpopulated HTML template tags with `NaNpx` dimensions. This record was completely removed.
2. **Exact Duplicate Resumes (2 pairs = 4 rows):**
   - Pair 1: `ID 19147603` and `ID 28398216` (both `FINANCE`).
   - Pair 2: `ID 16850314` and `ID 37473139` (both `AVIATION`).
   Both pairs shared identical category labels. Keeping the first occurrence eliminated duplicate cross-partition leakage, removing 2 redundant rows.
3. **Clean Sample Count:** The cleaned canonical dataset contains **2,481 high-quality resumes**.
4. **Feature Isolation:** `ID` was isolated and excluded from all feature transformers. `Resume_html` was excluded to prevent models from learning spurious DOM layout classes rather than linguistic qualifications.

---

## 4. Exploratory Data Analysis (EDA)

Key statistical findings from the comprehensive exploratory analysis (`reports/EDA_SUMMARY.md`):
- **Class Imbalance:**
  - Largest classes: `INFORMATION-TECHNOLOGY` (120 samples, 4.84%) and `BUSINESS-DEVELOPMENT` (119 samples, 4.80%).
  - Extreme tail classes: `BPO` (22 samples, 0.89%) and `AUTOMOBILE` (36 samples, 1.45%).
  - Strategic Remediation: Stratified splitting ensures every split reflects these exact class proportions; class-weighted loss functions were evaluated to preserve tail recall.
- **Resume Length Dynamics:**
  - Word length: Mean = 811.3 words, Median = 757 words, IQR = [651, 933] words.
  - Character length: Mean = 6,295 chars, Median = 5,886 chars.
  - Tail: The longest resume spans 5,190 words. Sublinear TF-scaling ($1 + \log(tf)$) was adopted to dampen term frequency saturation in lengthy profiles.
- **Semantic Proximity:** Cosine similarity of category centroids revealed high overlap in adjacent domains: `FINANCE` vs `ACCOUNTANT` (0.84), `SALES` vs `BUSINESS-DEVELOPMENT` (0.81), and `ENGINEERING` vs `INFORMATION-TECHNOLOGY` (0.78).

---

## 5. Domain-Preserving Preprocessing Pipeline

Implemented in `src/preprocessing/text_cleaner.py`:
1. **HTML Entity Normalization & Tag Stripping:** Resolves `&amp;`, `&nbsp;`, and rogue tags.
2. **Unicode Normalization:** NFKD normalization replaces typographical quotes, non-breaking spaces, and accents.
3. **Technical Token Protection:** Crucial technical tokens containing non-standard punctuation (`C++`, `C#`, `.NET`, `ASP.NET`, `Node.js`, `CI/CD`, `PL/SQL`, `T-SQL`) are protected via boundary regex into canonical alphanumeric representations (`cplusplus`, `csharp`, `dotnet`, etc.).
4. **Entity Standardisation:** URLs (`weburl`), Emails (`emailaddr`), and Phone Numbers (`phonenum`) are mapped to uniform semantic tokens.
5. **Controlled Punctuation:** Non-alphanumeric symbols are sanitized while retaining alphanumeric and underscore characters.
6. **No Aggressive Stemming:** Porter/Snowball stemming was intentionally avoided to prevent corrupting critical terminology (e.g. *Kubernetes*, *Automotive*, *Pediatric*).

---

## 6. Strict Data Splitting Strategy

Implemented in `src/data/split_data.py`:
- **Partitions:** 70% Train (1,736 resumes), 15% Validation (372 resumes), 15% Test (373 resumes).
- **Stratification:** Stratified by `Category` using a fixed random seed (`seed=42`).
- **Isolation:** The Test set was physically isolated and completely untouched during all exploratory, modeling, and hyperparameter tuning phases.
- **Zero Overlap:** Automated assertions verified 0 shared IDs and 0 shared text records between all pairs of splits.

---

## 7. Feature Engineering: The Word + Character Subword Union

Implemented in `src/features/feature_builder.py`:
- **Channel A — Word-level TF-IDF:**
  - Unigrams and Bigrams (`ngram_range=(1, 2)`)
  - Sublinear TF scaling (`sublinear_tf=True`)
  - Minimum document frequency (`min_df=2`), Maximum document frequency (`max_df=0.90`)
  - Captures multi-word qualifications (*machine learning, project management, financial accounting*).
- **Channel B — Character-level TF-IDF within Word Boundaries:**
  - Subword 3-to-5 character n-grams (`analyzer='char_wb'`, `ngram_range=(3, 5)`)
  - Captures typographical variants, compounding, prefixes, and suffixes.
- **Combined Channel — FeatureUnion:**
  - 60,000 combined sparse features offering rich, discriminative representations across both broad lexical themes and specific technical tokens.

---

## 8. Classical ML vs. Deep Learning Exploration

We conducted an exhaustive benchmark comparing:
1. **Naive Bayes:** MultinomialNB and ComplementNB.
2. **Regularized Linear Classifiers:** Logistic Regression and Linear Support Vector Machines (LinearSVC) across multiple regularization strengths ($C \in [0.25, 0.5, 1.0, 2.0, 5.0]$) and class-weight regimes (`unweighted` vs `balanced`).
3. **Stochastic Gradient Descent:** SGDClassifier with `modified_huber` and `log_loss`.
4. **Deep Neural Network:** Multi-Layer Perceptron (MLP) with hidden layers (256, 128), ReLU activations, Adam optimizer, and early stopping.
5. **Transformer Embeddings:** Dense 384-dimensional document embeddings using `all-MiniLM-L6-v2` paired with a linear classification head.

**Key Insight:** While Transformer embeddings perform respectably, the sparse Word+Char FeatureUnion paired with LinearSVC/Logistic Regression achieves superior Macro-F1 with 100x lower inference latency, confirming that well-engineered sparse representations remain optimal for domain-specific text classification at this scale.

---

## 9. Final Model Architecture & Evaluation

- **Selected Architecture:** Dual-Channel Word (1-2) + Character (3-5) TF-IDF FeatureUnion with Calibrated Linear Support Vector Classifier (`CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced'))`).
- **Retraining Protocol:** Frozen architecture retrained on combined Train + Validation partitions (2,108 resumes).
- **Final Evaluation:** Evaluated strictly ONCE on the isolated holdout Test set (373 resumes).

---

## 10. Production Deployment & Streamlit Interface

- **Inference Pipeline (`src/inference/predictor.py`):** Encapsulated `predict_resume(text)` function providing validation, cleaning, prediction, calibrated probabilities, and exact linear attribution.
- **Web Demo (`app/app.py`):** Interactive Streamlit application supporting both raw text pasting and multi-page PDF document upload (via PyMuPDF), featuring real-time attribution breakdowns and candidate confidence scores.
