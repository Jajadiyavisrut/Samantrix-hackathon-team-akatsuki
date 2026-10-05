# 🚀 ResumeForge AI

### Intelligent Resume Classification & Career Intelligence

ResumeForge AI is an end-to-end NLP system that analyzes unstructured resumes and classifies them into 24 professional career categories using TF-IDF character n-gram representations and a LinearSVC classifier.

The system also provides:
- **Top-3 category ranking** with relative margin scores
- **Resume statistics** (word count, character count, sentence length, vocabulary density)
- **Detected competencies/keywords** mapped across core industry domains
- **PDF and TXT resume ingestion** with native formatting extraction
- **Interactive Streamlit interface** styled with a dark modern AI SaaS aesthetic

---

## 1. Overview

Recruiters, HR platforms, and talent acquisition teams receive massive volumes of unstructured resumes every day. Manually reviewing and sorting these documents into appropriate career tracks is time-consuming, prone to human bias, and computationally difficult due to varied formatting, non-standard headings, technical jargon, and abbreviations.

**ResumeForge AI** solves this problem by transforming unstructured resume text into character-level n-gram feature vectors, classifying the resume against 24 professional categories using an optimized Linear Support Vector Classifier (LinearSVC), and delivering interpretable career intelligence in real time.

---

## 2. Key Features

- **24-Category Resume Classification:** Full multi-class categorization covering diverse engineering, financial, creative, healthcare, and operational fields.
- **Character-Level TF-IDF NLP:** Subword representation using (3, 6) character n-grams to robustly capture programming languages, technical terms, abbreviations, and hyphenated roles.
- **LinearSVC Classifier:** Highly efficient, convex margin-based classifier tuned for high-dimensional text classification.
- **PDF and TXT Ingestion:** Dual-format ingestion supporting both direct text pasting and document upload powered by PyMuPDF.
- **Resume Text Preprocessing:** Domain-aware normalization preserving programming punctuation (`C++`, `C#`, `.NET`), email structures, and hyphenated technical stacks.
- **Top-3 Category Ranking:** Multi-hypothesis output ranking alternative career alignments using decision-margin comparisons.
- **Skill & Competency Extraction:** Real-time surface keyword and competency identification dynamically highlighted in the dashboard.
- **Decision-Score Based Ranking:** Transparent ranking using classifier decision boundaries rather than pseudo-probabilities.
- **Interactive Streamlit Dashboard:** Modern dark SaaS interface designed for fast, accessible resume evaluation.
- **Error Handling:** Robust validation against empty inputs, short documents (<30 words), unreadable PDFs, or corrupted files.

---

## 3. Demo

### End-to-End Workflow:
```
Upload or Paste Resume
        ↓
   Extract Text
        ↓
    Preprocess
        ↓
  Character TF-IDF
        ↓
    LinearSVC
        ↓
Ranked Career Categories
        ↓
  Resume Intelligence
```

### Launch the Interactive App:
```bash
python -m streamlit run app.py
```
Access the application locally at `http://localhost:8501`.

---

## 4. Model Architecture

```
Resume (PDF / Text)
        ↓
Text Extraction (PyMuPDF / String Parser)
        ↓
Preprocessing (Regex Token Protection, Noise Reduction)
        ↓
Character TF-IDF (3-6 n-grams, Sublinear TF, max_features=40,000)
        ↓
LinearSVC Classifier (C=1.0, OvR Multi-Class)
        ↓
24 Career Categories
        ↓
Top-3 Ranking + Resume Intelligence (Competencies & Stats)
```

### Why Character N-Grams?
Resume classification poses unique NLP challenges:
- **Technical Symbols & Abbreviations:** Standard whitespace/word tokenizers frequently strip or fracture symbols like `C++`, `C#`, `.NET`, `Node.js`, `CI/CD`, and `SQL/NoSQL`.
- **Typographical & OCR Variations:** Resumes parsed from PDFs often contain slight OCR noise, missing spaces, or hyphenation across lines.
- **Compound Domain Jargon:** Character (3, 6) n-grams capture rich root morphemes (e.g., `neuro-`, `micro-`, `architect-`) without relying on brittle dictionaries or out-of-vocabulary fallback penalties.

---

## 5. Dataset

The benchmark dataset was rigorously cleaned, validated, and partitioned:

- **Total Usable Resumes:** 2,481 unique clean resumes
- **Number of Categories:** 24 distinct professional sectors
- **Partitioning Strategy:** Stratified split across all 24 categories
  - **Train Set (70%):** 1,736 resumes
  - **Validation Set (15%):** 372 resumes
  - **Test Set (15%):** 373 resumes

### Professional Categories (24 Classes):
ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION, BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION, CONSULTANT, DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS, HEALTHCARE, HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER.

---

## 6. Data Quality & Leakage Prevention

To ensure verifiable generalization performance:

1. **Duplicate Resume Audit:** Exact and near-duplicate resume strings were audited and deduplicated during data cleaning.
2. **Zero Partition Overlap:**
   - Train / Validation / Test ID overlap: **0**
   - Train / Validation / Test text overlap: **0**
3. **Stratified Splitting:** Class frequency distributions were preserved across Train, Validation, and Test splits.
4. **Strict Isolation:** The test split was completely isolated until final post-selection evaluation.
5. **Leakage-Free Vectorizer:** Feature vectorizers and TF-IDF parameters were fit strictly on the training partition during model selection.

---

## 7. Model Experiments

Multiple model architectures and feature extraction pipelines were evaluated on the validation partition:

| Experiment ID | Architecture | Feature Representation | Accuracy | Macro-F1 | Weighted-F1 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **B2 (Best)** | **LinearSVC** | **Char TF-IDF (3,6)** | **65.32%** | **59.70%** | **63.40%** |
| B1 | LinearSVC | Word TF-IDF (1,2) | 64.25% | 58.12% | 62.80% |
| B3 | LogisticRegression | Char TF-IDF (3,6) | 63.71% | 56.40% | 61.90% |
| B4 | MultinomialNB | Word TF-IDF (1,2) | 54.84% | 45.20% | 52.10% |

> *Note: The above metrics represent the VALIDATION set benchmark results during hyperparameter tuning.*

## Final Holdout Test Results

The best-performing model (LinearSVC + Character TF-IDF) was retrained and evaluated on the strictly held-out test partition:

| Metric | Holdout Test Score |
| :--- | :---: |
| **Accuracy** | **71.31%** |
| **Macro Precision** | **73.56%** |
| **Macro Recall** | **68.58%** |
| **Macro-F1** | **68.99%** |
| **Weighted-F1** | **70.34%** |

*(Verified from `reports/FINAL_TEST_METRICS.json`)*

---

## 8. Why Macro-F1?

In real-world recruitment data, class distributions are inherently unbalanced (e.g., higher sample representation in `INFORMATION-TECHNOLOGY` vs. niche categories like `BPO` or `AUTOMOBILE`). 

Standard accuracy is dominated by the majority classes. **Macro-F1** calculates metrics independently for each class and then takes the unweighted average, ensuring that all 24 professional domains contribute equally to the final evaluation score regardless of class size.

---

## 9. Explainable Prediction

The Streamlit UI emphasizes model transparency and clear feature interpretation:

- **Predicted Category:** The top-ranked professional category.
- **Model Rank:** Relative position of the predicted category (`#1 of 24`).
- **Decision Score:** The signed hyperplane distance from the LinearSVC decision boundary.
- **Top-3 Categories:** Relative ranking margins comparing candidate career paths.
- **Why This Category?:** Surface extraction of verified skill tokens identified in the resume text.

> **Decision Score Clarification:** LinearSVC produces raw decision margins, not calibrated probabilities. The application displays these margins directly to rank candidate categories without falsely projecting uncalibrated confidence percentages.

---

## 10. User Interface

The Streamlit web application is structured with a modern dark SaaS design:

- **Executive Header:** Clear badges displaying active model, classification speed, and taxonomy coverage.
- **Input Methods:** Easy tab switching between **Text Paste** and **File Upload** (`.pdf`, `.txt`).
- **Prediction Card:** High-contrast summary with primary category and decision margins.
- **Top Predicted Categories:** Normalized visual comparison bars showing relative ranking margins.
- **"Why This Category?" Section:** Transparent display of skills and competencies detected within the document.
- **Document Intelligence Panel:** Metrics including word count, character count, sentence length, and vocabulary density.
- **Inspect Preprocessed Text:** Collapsible drawer showing normalized text used for inference.

---

## 11. Project Structure

```
SAMATRIX HACKATHON/
├── app.py                      # Main Streamlit web application
├── preprocessing.py            # Text normalization and token protection
├── predict.py                  # Inference engine and top-k ranking logic
├── clean_data.py               # Dataset deduplication and cleaning script
├── run_experiments.py          # Model evaluation and comparison runner
├── optimize_model.py           # Hyperparameter optimization and testing
├── requirements.txt            # Application runtime dependencies
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Git tracking rules and protection
│
├── models/                     # Production model artifacts
│   ├── best_tfidf_model.joblib         # Fitted LinearSVC classifier (~7.6 MB)
│   ├── best_feature_vectorizer.joblib  # Character TF-IDF vectorizer (~1.4 MB)
│   ├── feature_vectorizer.joblib       # Alternative feature union (~2.2 MB)
│   └── model_metadata.joblib           # Category mapping and metadata
│
├── reports/                    # Verified metrics, logs, and figures
│   ├── BEST_MODEL_CONFIG.json          # Best model parameters and val scores
│   ├── FINAL_TEST_METRICS.json         # Final holdout test evaluation metrics
│   ├── ERROR_ANALYSIS.md               # Detailed misclassification analysis
│   ├── DATA_AUDIT.md                   # Dataset audit and distribution report
│   ├── DATA_CLEANING_REPORT.md         # Cleaning & deduplication log
│   ├── EDA_SUMMARY.md                  # Exploratory analysis findings
│   ├── FINAL_REPORT.md                 # Executive hackathon summary
│   └── figures/                        # Confusion matrices and distribution plots
│
├── src/                        # Modular source code
│   ├── data/                   # Data cleaning, splitting, and EDA routines
│   ├── preprocessing/          # Normalization modules
│   ├── features/               # Vectorizer pipelines
│   ├── models/                 # Experiment benchmarks and training
│   ├── evaluation/             # Metric computation
│   └── inference/              # Inference wrappers
│
└── notebooks/                  # Reproducible Jupyter notebooks
    ├── 01_data_audit.ipynb
    ├── 02_eda.ipynb
    ├── 03_model_experiments.ipynb
    ├── 04_error_analysis.ipynb
    └── 05_final_model.ipynb
```

---

## 12. Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Jajadiyavisrut/Samantrix-hackathon-team-akatsuki.git
cd Samantrix-hackathon-team-akatsuki
```

### 2. Set Up Virtual Environment
```bash
# Create environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on macOS / Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
python -m streamlit run app.py
```

---

## 13. Usage

1. Open the application in your browser (default: `http://localhost:8501`).
2. Provide resume text through either:
   - **Upload PDF / TXT:** Drag and drop candidate resume files.
   - **Paste Plain Text:** Direct paste into the input area.
3. Click **"Analyze Resume"**.
4. Review the primary category, relative ranking margins, top-3 alternatives, extracted competency tags, and text statistics.

---

## 14. Evaluation

Model performance was assessed across multiple evaluation dimensions:

- **Overall Metrics:** Accuracy, Macro-Averaged F1, and Weighted-Averaged F1.
- **Confusion Matrix:** Diagnostic heatmaps mapping inter-class misclassifications (`reports/figures/07_model_confusion_matrix.png`, `reports/figures/09_final_test_confusion_matrix.png`).
- **Per-Class Breakdown:** Precision, recall, and F1 across all 24 classes (`reports/figures/08_per_class_f1_scores.png`).
- **Error Analysis:** Thorough review of semantic overlaps between adjacent fields (`reports/ERROR_ANALYSIS.md`).
- **Artifacts:** Quantitative records stored in `reports/FINAL_TEST_METRICS.json` and `reports/BEST_MODEL_CONFIG.json`.

---

## 15. Limitations

- **24-Category Complexity:** Multi-class classification over 24 career categories inherently presents boundary challenges.
- **Semantic Overlap:** Categories such as `FINANCE` vs. `BANKING` or `ENGINEERING` vs. `INFORMATION-TECHNOLOGY` share substantial vocabulary and skill crossover.
- **Decision Margins:** LinearSVC decision values represent geometric distance to the separating hyperplane and are not probabilistic likelihoods.
- **Dataset Size:** With 2,481 samples across 24 classes (~100 samples per class), statistical support is limited for minority domains like `AUTOMOBILE` and `BPO`.
- **Decision Support Tool:** ResumeForge AI is designed to assist human recruiters in triage, not to make fully automated hiring decisions.

---

## 16. Future Improvements

- **Transformer Backbones:** Integrate domain-adapted encoders (e.g., RoBERTa, DeBERTa) fine-tuned on HR terminology.
- **Calibrated Probabilities:** Implement Platt scaling or isotonic regression to supply calibrated confidence intervals.
- **Hierarchical Classification:** Structure prediction into a two-level taxonomy (Broad Industry → Specialized Role).
- **Expanded Corpora:** Ingest diverse, multi-lingual, and multi-regional resume datasets.
- **Feature Attribution Visualizations:** Implement token-level saliency highlights for enhanced interpretability.
- **Human-in-the-Loop Feedback:** Incorporate recruiter correction loops to continuously refine decision boundaries.

---

## 17. Hackathon

**Built for:**  
SAMATRIX RESUMEFORGE 2026

---

## 18. Team

**Team Akatsuki**
