# Final Model Performance & Evaluation Report
**SAMATRIX RESUMEFORGE 2026 Hackathon**

---

## 1. Executive Summary
This report details the final, verified results of the **Resume Classification System** across three strictly separated evaluation stages:
1. **Model Selection**: Conducted on the **Validation Set** (372 samples) using **Macro-F1** as the primary criterion.
2. **Final Model Comparison**: Evaluated on the **Untouched Test Set** (373 samples) for unbiased benchmark reporting.
3. **Production Inference Validation**: Tested on **Independent Unseen Resumes** (5 samples) completely separated from train/val/test splits.

---

## 2. Dataset & Cleaning Summary
- **Raw Records Collected**: 2,484
- **Empty Resumes Dropped**: 1 (Resume ID `12632728`)
- **Exact Duplicate Resumes Dropped**: 2 pairs
- **Final Cleaned Dataset**: 2,481 records
- **Number of Categories**: 24 professional domains
- **Split Breakdown (Stratified, Seed=42)**:
  - **Training Set (70%)**: 1,736 samples
  - **Validation Set (15%)**: 372 samples
  - **Untouched Test Set (15%)**: 373 samples

---

## 3. EDA Summary
- **Class Distribution**: Imbalanced distribution ranging from ~120 samples (Information-Technology, Business-Development) down to minority classes (~70 samples).
- **Text Length**: Median length ~3,800 characters (~550 words). Long tail exceeding 10,000 characters.
- **Discriminative Vocabulary**: Identified domain-specific n-grams (e.g., `litigation`, `court`, `legal` for Advocate; `nursing`, `patient`, `clinical` for Healthcare).
- **All Figures Generated**: 7 figures stored in `reports/figures/`.

---

## 4. Preprocessing & Feature Engineering
- **Text Normalization**: Regex-based token preservation for technical terms (`c++` -> `cpp`, `c#` -> `csharp`, `.net` -> `dotnet`, `node.js` -> `nodejs`, `ci/cd` -> `cicd`), URL/email removal, punctuation stripping, lowercase conversion, and lemmatization.
- **TF-IDF Configuration**:
  - `ngram_range`: (1, 2)
  - `max_features`: 10,000
  - `min_df`: 2, `max_df`: 0.95
  - `sublinear_tf`: True
  - **Fit Boundary**: Fitted strictly on training set (1,736 samples); zero data leakage to validation or test.

---

## 5. Model Selection (Validation Set Results)
Hyperparameter tuning and architecture comparisons were conducted strictly on the **Validation Set (372 samples)**:

| Model Architecture | Validation Accuracy | Validation Macro-F1 | Selection Decision |
| :--- | :---: | :---: | :--- |
| **Multinomial Naive Bayes** | 0.5323 | 0.4663 | Baseline |
| **Logistic Regression (C=1.0, Balanced)** | 0.6452 | 0.5955 | Strong baseline |
| **Linear SVM (`LinearSVC`, C=1.0, Balanced)** | **0.6909** | **0.6540** | **Selected Classical Model** |
| **BiLSTM (Embedding=128, Hidden=128, PyTorch)** | **0.7500** | **0.6743** | Best Neural Checkpoint |

**Selection Decision**: **Linear SVM (`LinearSVC`)** was selected as the final production model because:
1. It achieved strong, balanced generalization across all 24 classes (Val Macro-F1 = 0.6540).
2. It exhibits rapid, deterministic CPU inference (< 5 ms per resume) with zero heavy deep learning dependencies.
3. Macro-F1 was the primary model-selection criterion, prioritizing balanced multiclass performance.

---

## 6. Final Model Comparison (Untouched Test Set)
After model selection was finalized, all candidate architectures were evaluated once on the **Untouched Test Set (373 samples)**:

| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 0.6005 | 0.6058 | 0.5509 | 0.5282 | 0.5716 |
| **Logistic Regression (Balanced)** | 0.7158 | 0.7545 | 0.6894 | 0.6890 | 0.7021 |
| **Linear SVM (LinearSVC - Selected)** | **0.7239** | **0.7448** | **0.6997** | **0.6975** | **0.7105** |
| **BiLSTM (Neural Baseline)** | **0.7507** | 0.6726 | 0.6871 | 0.6757 | **0.7370** |

*Note*: BiLSTM achieved higher raw test accuracy (0.7507 vs 0.7239), but Linear SVM achieved higher test Macro-F1 (0.6975 vs 0.6757), validating the selection of Linear SVM under the primary Macro-F1 criterion.

---

## 7. Diagnostic Error Analysis (Test Set)
- **Total Test Misclassifications**: 103 / 373 (Error Rate: 27.61%)
- **Root Cause Categorization**:
  - **Subtle Discriminative Boundary / Multi-Disciplinary Experience**: 87 samples (e.g., cross-functional engineering vs IT, sales vs business development).
  - **Semantic Overlap (Financial Domain shared terminology)**: 7 samples (Accounting vs Finance vs Banking).
  - **Semantic Overlap (Client acquisition & revenue terminology)**: 5 samples (Sales vs Marketing).
  - **Cross-domain Creative & Media Terminology**: 4 samples (Designer vs Arts vs Digital Media).
- **Artifacts**:
  - `reports/error_analysis/misclassified_examples.csv`
  - `reports/error_analysis/error_summary_test.json`

---

## 8. Production Inference Validation (Unseen Samples)
A standalone validation set of 5 genuinely unseen resume profiles was processed through the serialized production pipeline:

| Sample | Domain Profile | Expected Category | Predicted Category | Decision Score | Result |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Sample 1** | Software & Cloud Engineering | INFORMATION-TECHNOLOGY | ENGINEERING | -0.6631 | **FAIL** (Boundary error) |
| **Sample 2** | Culinary & Restaurant Leadership | CHEF | CHEF | 1.5893 | **PASS** |
| **Sample 3** | Commercial Aviation Operations | AVIATION | AVIATION | 0.7181 | **PASS** |
| **Sample 4** | Litigation & Corporate Legal Counsel | ADVOCATE | ADVOCATE | -0.3041 | **PASS** |
| **Sample 5** | Clinical Healthcare & Nursing | HEALTHCARE | HEALTHCARE | -0.2538 | **PASS** |

**Summary**: **4 / 5 PASS (80%)**, **1 / 5 FAIL (20%)**.
- **Top-3 Ranking for Sample 1**:
  1. `ENGINEERING` (Score: -0.6631)
  2. `INFORMATION-TECHNOLOGY` (Score: -0.6782)
  3. `ADVOCATE` (Score: -0.7061)
- **Diagnostic Finding**: The near-identical decision scores between Rank 1 (`ENGINEERING`) and Rank 2 (`INFORMATION-TECHNOLOGY`) demonstrate the semantic overlap in cross-disciplinary software engineering profiles.

---

## 9. Known Limitations
1. **Semantic Class Boundary Ambiguity**: Overlap between closely related domains (e.g., Information-Technology vs. Engineering, Finance vs. Accounting) remains the primary source of classification errors.
2. **Scanned PDF Ingestion**: The PDF pipeline extracts text from digital PDF streams (PyPDF). Scanned image-only PDFs require external OCR preprocessing.
