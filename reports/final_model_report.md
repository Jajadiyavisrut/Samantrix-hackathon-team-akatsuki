# Final Model Performance & Evaluation Report
**SAMATRIX RESUMEFORGE 2026 Hackathon**

---

## 1. Executive Summary
This report details the final, verified results of the **Resume Classification System** across three strictly separated evaluation stages:
1. **Model Tuning & Validation Selection**: Conducted on the **Validation Set** (372 samples) comparing 15 configurations across Word TF-IDF, Char-wb TF-IDF, Section-Aware TF-IDF, and Neural architectures.
2. **Final Model Comparison**: Evaluated on the **Untouched Test Set** (373 samples) for unbiased benchmark reporting.
3. **Production Inference Validation**: Tested on **Independent Unseen Resumes** (5 samples) and a real-world multidisciplinary case study (`Visrut-Jajadiya_Resume.pdf`).

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

## 3. Stage 4 Validation Tuning & Model Selection

Systematic validation search executed during pipeline execution (`reports/metrics/validation_tuning_experiments.csv`):

| Model Architecture | Feature Representation | Hyperparameters | Val Accuracy | Val Macro-F1 | Validation Rank |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **BiLSTM (Neural)** | Token Sequences | `Embedding 128, Hidden 128` | **0.7500** | **0.6743** | **Rank 1 (Overall)** |
| **Linear SVM (LinearSVC)** | Word (1,2) 10k | `C=3.0, balanced` | **0.7070** | **0.6717** | **Rank 2 (Best Classical)** |
| **Linear SVM (LinearSVC)** | Word (1,2) 10k | `C=1.5, balanced` | 0.6989 | 0.6619 | Rank 3 |
| **Linear SVM (LinearSVC)** | Word (1,2) 10k | `C=2.0, balanced` | 0.6989 | 0.6619 | Rank 3 |
| **Linear SVM (LinearSVC)** | Section-Aware Word | `C=3.0, balanced` | 0.7043 | 0.6606 | Rank 5 |
| **Linear SVM (LinearSVC)** | Section-Aware Word | `C=2.0, balanced` | 0.6989 | 0.6552 | Rank 6 |
| **Linear SVM (LinearSVC)** | Word (1,2) 10k | `C=1.0, balanced [Baseline]` | 0.6909 | 0.6496 | Rank 7 |
| **Linear SVM (LinearSVC)** | Section-Aware Word | `C=1.0, balanced` | 0.6855 | 0.6455 | Rank 8 |
| **Linear SVM (LinearSVC)** | Word (1,2) 10k | `C=0.5, balanced` | 0.6747 | 0.6346 | Rank 9 |
| **Linear SVM (LinearSVC)** | Word(10k)+Char(15k) | `C=3.0, balanced` | 0.6801 | 0.6272 | Rank 10 |
| **Linear SVM (LinearSVC)** | Word(10k)+Char(15k) | `C=1.0, balanced` | 0.6694 | 0.6157 | Rank 11 |
| **Logistic Regression** | Word (1,2) 10k | `C=10.0, balanced` | 0.6559 | 0.6056 | Rank 12 |
| **Logistic Regression** | Word (1,2) 10k | `C=5.0, balanced` | 0.6425 | 0.5922 | Rank 13 |
| **Linear SVM (LinearSVC)** | Char-wb (3,5) 15k | `C=2.0, balanced` | 0.6371 | 0.5868 | Rank 14 |
| **Linear SVM (LinearSVC)** | Char-wb (3,5) 15k | `C=1.0, balanced` | 0.6398 | 0.5864 | Rank 15 |
| **Logistic Regression** | Word (1,2) 10k | `C=1.0, balanced` | 0.6075 | 0.5580 | Rank 16 |
| **Multinomial Naive Bayes** | Word (1,2) 10k | `alpha=0.1` | 0.5296 | 0.4627 | Rank 17 |

**Model Selection Summary**:
1. **Best Overall Validation Model**: **BiLSTM** (Validation Macro-F1: `0.6743`, Accuracy: `0.7500`).
2. **Selected Production Model**: **Linear SVM (`LinearSVC, C=3.0, balanced`)** (Validation Macro-F1: `0.6717`, Accuracy: `0.7070`). Selected for deployment due to high Macro-F1, fast inference (<5ms per resume), and lightweight CPU operation without PyTorch dependencies.

---

## 4. Final Model Comparison (Untouched Test Set: 373 Samples)

| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 0.5952 | 0.5938 | 0.5461 | 0.5232 | 0.5662 |
| **Logistic Regression (Balanced)** | 0.7105 | 0.7333 | 0.6840 | 0.6811 | 0.6980 |
| **Linear SVM (LinearSVC - Selected)** | **0.7212** | **0.7332** | **0.6891** | **0.6884** | **0.7086** |
| **BiLSTM (Neural Baseline)** | **0.7507** | 0.6726 | 0.6871 | 0.6757 | **0.7370** |

---

## 5. Diagnostic Error Analysis & Multidisciplinary Case Study

### A. Test Set Diagnostic Breakdown
- **Total Misclassifications**: 104 / 373 (27.88%)
- **Root Causes**:
  - **Subtle Discriminative Boundary / Multi-Disciplinary Experience**: 90 samples
  - **Financial Domain Semantic Overlap**: 6 samples (Accounting vs. Finance vs. Banking)
  - **Client Acquisition Overlap**: 4 samples (Sales vs. Marketing vs. Business Development)
  - **Creative & Media Terminology**: 4 samples (Designer vs. Digital Media)

### B. Multidisciplinary Case Study: `Visrut-Jajadiya_Resume.pdf`
- **Candidate Profile**: Full Stack / Web Developer with Python, JavaScript, React, SQL, PHP, NLP, and one secondary project (*"WellFit Smart Fitness Assistant"*).
- **Flat Bag-of-Words Prediction**: `FITNESS` (-0.3418) vs `INFORMATION-TECHNOLOGY` (-0.3531) — project vocabulary hijack due to rare IDF fitness terms.
- **Section-Aware Prediction**: `INFORMATION-TECHNOLOGY` (-0.4378) vs `FITNESS` (-0.6074) — successfully prioritizes primary career evidence over secondary projects.
