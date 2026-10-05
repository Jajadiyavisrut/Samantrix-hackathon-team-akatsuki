# Resume Classification System
**SAMATRIX RESUMEFORGE 2026 Hackathon**

An end-to-end, production-grade Multiclass Resume Classification system that predicts candidate professional domains across 24 categories from raw text or multi-page PDF documents.

---

## 1. Problem Statement
- **Task**: Multiclass Text Classification of professional resumes.
- **Input**: Raw text or extracted text from multi-page PDF files.
- **Output**: Predicted professional category with decision confidence scores.
- **Classes (24 Domains)**: Information Technology, Business Development, Advocate, Chef, Engineering, Accountant, Finance, Fitness, Aviation, Sales, Banking, Healthcare, Consultant, Construction, Public Relations, HR, Designer, Arts, Teacher, Apparel, Digital Media, Agriculture, Automobile, BPO.
- **Success Metrics**: **Macro-F1** (primary optimization metric for class imbalance), Weighted-F1, Accuracy, Precision, Recall, and Confusion Matrix diagnostics.

---

## 2. Dataset
- **Raw Data Source**: `Resume.csv` containing 2,484 records across 4 columns (`ID`, `Resume_str`, `Resume_html`, `Category`).
- **PDF Dataset**: 2,500 category-organized PDF documents across 24 folder categories used to test and verify PDF extraction.

---

## 3. Data Quality Findings
- **Total Records**: 2,484
- **Unique IDs**: 2,484 (0 duplicate IDs)
- **Empty Resumes**: 1 empty document (ID `12632728`, 0 chars) successfully identified and removed.
- **Exact Duplicate Text**: 2 duplicate pairs (4 records total) removed to prevent train/test contamination.
- **Label Inconsistencies**: 0 conflicting labels detected for duplicate texts.
- **Text Statistics**:
  - Minimum Char Length: 770 | Maximum: 38,813 | Mean: 6,281 | Median: 5,874
  - Minimum Word Count: 104 | Maximum: 5,190 | Mean: 811 | Median: 757
- **Final Cleaned Records**: 2,481 records.

---

## 4. EDA Findings
Generated all 7 required visualization artifacts in `reports/figures/`:
1. **Class Distribution** (`01_class_distribution.png`): Reveals class imbalance ranging from 120 resumes (Information-Technology, Business-Development) down to 22 resumes (BPO) and 36 (Automobile).
2. **Resume Length Distribution** (`02_resume_length_distribution.png`): Demonstrates log-normal length distribution with median length of ~757 words.
3. **Top Frequent Words** (`03_top_frequent_words.png`): Identifies high-frequency professional terminology (experience, management, sales, customer, project, development).
4. **Corpus WordClouds** (`04_wordcloud_overall.png` & `05_wordcloud_classes.png`): Visualizes overall and domain-specific terminology (e.g. chef, culinary, kitchen, legal, court, aviation, pilot, flight).
5. **N-Gram Analysis** (`06_ngram_analysis.png`): Unigrams, bigrams, and trigrams (e.g., "customer service", "project management", "team member").
6. **Class-Discriminative TF-IDF Terms** (`07_class_discriminative_terms.png`): Shows highly discriminative domain terms across representative classes.

---

## 5. Preprocessing
A central reproducible function (`src/preprocessing.py`) is used identically across training, validation, testing, and inference:
- Normalizes Unicode artifacts (NFKD) and line breaks.
- Strips HTML tags.
- Converts URLs to `url_ref`, emails to `email_ref`, and phone numbers to `phone_ref`.
- **Preserves essential technical tokens** (`C++` -> `cpp`, `C#` -> `csharp`, `.NET` -> `dotnet`, `Node.js` -> `nodejs`, `CI/CD` -> `cicd`, `Python`, `SQL`, `AWS`, etc.) via specialized token protection.
- Eliminates noise while retaining domain vocabulary.

---

## 6. Feature Engineering
- **Classical ML**: Word + Bigram TF-IDF (`ngram_range=(1,2)`, `max_features=10000`, `min_df=3`, `max_df=0.85`, sublinear term frequency scaling).
- **Deep Learning**: Padded integer sequence representations (`max_seq_len=150`, `vocab_size=8000`) with learned dense word embeddings.
- **Strict Leakage Prevention**: All vectorizers and tokenizers are fitted strictly on the Training set.

---

## 7. Classical ML Models
Trained with `class_weight='balanced'` and tuned on the Validation set:
- **Multinomial Naive Bayes**
- **Logistic Regression**
- **Linear Support Vector Machine (LinearSVC)**

---

## 8. Deep Learning Model
- **Architecture**: Bidirectional LSTM (BiLSTM) Classifier in PyTorch (`vocab_size=8000`, `embedding_dim=64`, `hidden_dim=64`, `dropout=0.2`, `num_layers=1`).
- **Training Strategy**: Cross-Entropy Loss with AdamW optimizer, validation macro-F1 early stopping.

---

## 9. Evaluation (Untouched Test Set Comparison)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 0.6005 | 0.6058 | 0.5509 | 0.5282 | 0.5716 |
| **Logistic Regression** | 0.7158 | 0.7545 | 0.6894 | 0.6890 | 0.7021 |
| **Linear SVM (LinearSVC)** | **0.7239** | **0.7448** | **0.6997** | **0.6975** | **0.7105** |
| **BiLSTM (Neural)** | 0.7507 | 0.6726 | 0.6871 | 0.6757 | 0.7370 |

*Detailed per-class metrics and confusion matrix heatmaps are saved in `reports/metrics/` and `reports/figures/`.*

---

## 10. Error Analysis
Diagnostic error analysis conducted on test misclassifications (`reports/error_analysis/`):
- **Total Test Misclassifications**: 103 / 373 (27.61% test error rate).
- **Primary Root Causes**:
  1. *Subtle Discriminative Boundaries (84.5%)*: Multi-disciplinary resumes containing cross-industry roles (e.g., IT Project Managers in Healthcare).
  2. *Financial Domain Semantic Overlap (6.8%)*: Shared vocabulary between `FINANCE`, `BANKING`, and `ACCOUNTANT`.
  3. *Client Acquisition Overlap (4.9%)*: Shared vocabulary between `BUSINESS-DEVELOPMENT`, `SALES`, and `CONSULTANT`.
  4. *Creative & Media Overlap (3.9%)*: Shared vocabulary between `DESIGNER`, `ARTS`, and `DIGITAL-MEDIA`.

---

## 11. Final Model Selection
- **Selected Final Model**: **Linear Support Vector Classifier (LinearSVC)** on TF-IDF word+bigram features.
- **Model Selection Criterion**: **Validation Macro-F1 (0.6540)** and balanced multiclass generalization across 24 domains.
- **Test Set Validation**: Evaluated once on the untouched test set, confirming the highest **Macro-F1 (0.6975)** among evaluated models. While BiLSTM achieved higher overall accuracy (0.7507 vs 0.7239), Linear SVM achieved higher Macro-F1 (0.6975 vs 0.6757), validating its selection under the primary balanced multiclass criterion.

---

## 12. Independent Unseen Production Validation
A standalone set of 5 genuinely unseen resume profiles was evaluated through the serialized inference pipeline:
- **Result**: 4 / 5 samples passed (CHEF, AVIATION, ADVOCATE, HEALTHCARE).
- **Known Limitation / Boundary Ambiguity**: 1 Software/Cloud Engineering resume expected as `INFORMATION-TECHNOLOGY` was predicted as `ENGINEERING` with nearly identical top decision scores (`-0.6631` vs `-0.6782`), demonstrating semantic boundary overlap between adjacent technical domains.

---

## 12. Project Structure
```
resume-classification/
│
├── data/
│   ├── raw/
│   │   ├── Resume.csv
│   │   └── Resume.xlsx
│   ├── processed/
│   │   ├── cleaned_resumes.csv
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── README.md
│
├── notebooks/
│   └── resume_classification_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── features.py
│   ├── train_ml.py
│   ├── train_dl.py
│   ├── evaluate.py
│   ├── error_analysis.py
│   ├── inference.py
│   └── pdf_extractor.py
│
├── models/
│   ├── best_classical_model.joblib
│   ├── label_encoder.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── neural_tokenizer.joblib
│   └── neural_lstm_model.pt
│
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── error_analysis/
│
├── tests/
│   └── test_pipeline.py
│
├── app/
│   └── app.py
│
├── requirements.txt
├── README.md
├── generate_notebook.py
└── run_pipeline.py
```

---

## 13. Installation
```bash
cd resume-classification
pip install -r requirements.txt
```

---

## 14. Training & Reproduction
To run the complete pipeline from data validation to model evaluation:
```bash
python run_pipeline.py
```

---

## 15. Running Tests
```bash
python -m unittest tests/test_pipeline.py
```

---

## 16. Running the Streamlit Demo
```bash
streamlit run app/app.py
```

---

## 17. Example Prediction
```python
from src.inference import ResumeClassificationPipeline

pipeline = ResumeClassificationPipeline()

# Predict raw text
sample = "Senior Chef with expertise in French pastry, banquet catering, menu planning, and kitchen operations."
result = pipeline.predict_text(sample)

print("Predicted Category:", result["predicted_category"]) # -> CHEF
print("Decision Score:", result["confidence"])
```
