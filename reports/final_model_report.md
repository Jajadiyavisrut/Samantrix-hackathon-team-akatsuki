# Final Model Performance & Evaluation Report
**SAMATRIX RESUMEFORGE 2026 Hackathon**

---

## 1. Executive Summary
This report details the final, verified performance metrics of the **Resume Classification System** evaluated on the untouched test set (373 samples). The objective is multiclass classification of professional resumes across 24 categories.

---

## 2. Dataset & Split Statistics
- **Total Raw Records**: 2,484
- **Cleaned & Deduplicated Records**: 2,481
- **Number of Classes**: 24 categories
- **Training Split (70%)**: 1,736 samples
- **Validation Split (15%)**: 372 samples
- **Untouched Test Split (15%)**: 373 samples

---

## 3. Final Model Comparison (Test Set)

All metrics were evaluated exclusively on the untouched test set following hyperparameter tuning on validation data:

| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 0.6005 | 0.6058 | 0.5509 | 0.5282 | 0.5716 |
| **Logistic Regression (Balanced)** | 0.7158 | 0.7545 | 0.6894 | 0.6890 | 0.7021 |
| **Linear SVM (LinearSVC, Balanced)** | **0.7239** | **0.7448** | **0.6997** | **0.6975** | **0.7105** |
| **Bidirectional LSTM (PyTorch)** | 0.7507 | 0.6726 | 0.6871 | 0.6757 | 0.7370 |

---

## 4. Final Selected Model & Justification
- **Selected Model**: **Linear Support Vector Classifier (`LinearSVC`)**
- **Feature Representation**: Word + Bigram TF-IDF (10,000 max features, sublinear term frequency scaling)
- **Selection Metric**: **Macro-F1 (0.6975)**
- **Technical Justification**:
  1. Achieved the highest **Macro-F1 (0.6975)** across all classical baselines and neural architectures.
  2. Balanced class weighting provides superior sensitivity on minority domains (such as BPO and Automobile).
  3. Ultra-fast CPU inference (< 5 ms per resume) with zero complex hardware dependencies.
  4. Robust margin-based decision boundaries that generalize effectively on sparse, high-dimensional text representations.

---

## 5. Artifact Locations
- **Fitted TF-IDF Vectorizer**: `models/tfidf_vectorizer.joblib`
- **Fitted Label Encoder**: `models/label_encoder.joblib`
- **Trained Linear SVM Model**: `models/best_classical_model.joblib`
- **Trained Neural BiLSTM Checkpoint**: `models/neural_lstm_model.pt`
- **Detailed Per-Class Report**: `reports/metrics/per_class_report_linear_svm_(linearsvc).csv`
- **Test Metrics Summary**: `reports/metrics/model_comparison_test.csv`
- **Confusion Matrix Heatmaps**:
  - Linear SVM: `reports/figures/confusion_matrix_linear_svm_(linearsvc).png`
  - BiLSTM: `reports/figures/confusion_matrix_bilstm_(neural).png`
  - Logistic Regression: `reports/figures/confusion_matrix_logistic_regression.png`
  - Naive Bayes: `reports/figures/confusion_matrix_multinomial_naive_bayes.png`
- **Error Analysis Diagnostics**: `reports/error_analysis/misclassified_samples_test.csv` & `reports/error_analysis/error_summary_test.json`
