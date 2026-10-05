# SAMATRIX RESUMEFORGE 2026 — Comprehensive Data & Workspace Audit

**Date:** 2026-10-05  
**Author:** Lead ML Engineer & Hackathon Strategist  
**Dataset Name:** ResumeForge 2026 (Resume Classification)  
**Artifact Status:** Verified & Canonicalized  

---

## 1. Executive Summary

A full forensic audit of the `SAMATRIX HACKATHON` workspace was conducted across all files, directories, tabular formats (`.csv`, `.xlsx`), and binary assets (`.pdf`).

Key findings:
1. **Canonical Tabular Dataset:** `Resume/Resume.csv` (and `data/raw/Resume.csv`) contains **2,484 rows** and **4 columns** across **24 distinct job categories**.
2. **Directory Dataset (`data/data`):** Contains **2,500 PDF files** distributed across 24 category subfolders. Crucially, 16 files in `data/data/FINANCE/` are operating-system duplicate downloads bearing the suffix `(1).pdf` (e.g., `20880935(1).pdf`). Removing these 16 artifacts results in exactly 2,484 unique PDFs corresponding 1-to-1 with the IDs in `Resume.csv`.
3. **Corrupted/Empty Records:** Exactly **1 row** (`ID: 12632728`, Category: `BUSINESS-DEVELOPMENT`) is corrupted. Its text contains 21 whitespace characters and 0 words, and its HTML snippet consists of unpopulated template tags with `NaNpx` dimensions.
4. **Exact Duplicates:** Exactly **4 rows** represent **2 duplicate pairs**:
   - `ID: 19147603` and `ID: 28398216` (both categorized as `FINANCE`)
   - `ID: 16850314` and `ID: 37473139` (both categorized as `AVIATION`)
   Both pairs share identical category labels (no label conflict). Retaining the first occurrence removes 2 redundant rows.
5. **Effective Clean Sample Size:** Following removal of the 1 corrupted resume and deduplication of the 2 duplicate entries, the clean canonical dataset contains **2,481 high-quality resumes**.

---

## 2. Dataset Dimensions & Schema

| Column Name | Data Type | Null Count | Unique Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| `ID` | `int64` | 0 (0.00%) | 2,484 | Unique resume identifier. **Must NOT be used as a feature** to avoid identifier leakage. |
| `Resume_str` | `object` (string) | 0 (0.00%) | 2,482 | Plain text extracted from the resume document. Canonical input feature. |
| `Resume_html` | `object` (string) | 0 (0.00%) | 2,484 | HTML-rendered resume structure. Contains DOM tags, inline styles, and font markers. |
| `Category` | `object` (string) | 0 (0.00%) | 24 | Target ground-truth professional domain label. |

---

## 3. Category Count & Class Distribution

The dataset spans **24 target categories**. The distribution exhibits mild-to-moderate class imbalance, particularly in the tail classes (`BPO` with 22 samples, `AUTOMOBILE` with 36 samples, `AGRICULTURE` with 63 samples), while the remaining 21 classes maintain roughly 96 to 120 samples each.

### Class Breakdown (Raw vs Clean)

| Rank | Category | Raw Count | Clean Count | % of Clean Dataset | Cumulative % | Imbalance Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | `INFORMATION-TECHNOLOGY` | 120 | 120 | 4.84% | 4.84% | Balanced |
| 2 | `BUSINESS-DEVELOPMENT` | 120 | 119 | 4.80% | 9.63% | Balanced (-1 empty) |
| 3 | `ADVOCATE` | 118 | 118 | 4.76% | 14.39% | Balanced |
| 4 | `CHEF` | 118 | 118 | 4.76% | 19.15% | Balanced |
| 5 | `ENGINEERING` | 118 | 118 | 4.76% | 23.90% | Balanced |
| 6 | `ACCOUNTANT` | 118 | 118 | 4.76% | 28.66% | Balanced |
| 7 | `FINANCE` | 118 | 117 | 4.71% | 33.37% | Balanced (-1 duplicate) |
| 8 | `FITNESS` | 117 | 117 | 4.71% | 38.09% | Balanced |
| 9 | `SALES` | 116 | 116 | 4.68% | 42.76% | Balanced |
| 10 | `AVIATION` | 117 | 116 | 4.68% | 47.44% | Balanced (-1 duplicate) |
| 11 | `BANKING` | 115 | 115 | 4.64% | 52.08% | Balanced |
| 12 | `HEALTHCARE` | 115 | 115 | 4.64% | 56.71% | Balanced |
| 13 | `CONSULTANT` | 115 | 115 | 4.64% | 61.35% | Balanced |
| 14 | `CONSTRUCTION` | 112 | 112 | 4.51% | 65.86% | Balanced |
| 15 | `PUBLIC-RELATIONS` | 111 | 111 | 4.47% | 70.33% | Balanced |
| 16 | `HR` | 110 | 110 | 4.43% | 74.77% | Balanced |
| 17 | `DESIGNER` | 107 | 107 | 4.31% | 79.08% | Balanced |
| 18 | `ARTS` | 103 | 103 | 4.15% | 83.23% | Balanced |
| 19 | `TEACHER` | 102 | 102 | 4.11% | 87.34% | Balanced |
| 20 | `APPAREL` | 97 | 97 | 3.91% | 91.25% | Moderate representation |
| 21 | `DIGITAL-MEDIA` | 96 | 96 | 3.87% | 95.12% | Moderate representation |
| 22 | `AGRICULTURE` | 63 | 63 | 2.54% | 97.66% | Minor class |
| 23 | `AUTOMOBILE` | 36 | 36 | 1.45% | 99.11% | Tail class |
| 24 | `BPO` | 22 | 22 | 0.89% | 100.00% | Extreme tail class |
| **Total** | | **2,484** | **2,481** | **100.00%** | | |

---

## 4. Text Length & Distribution Statistics

Statistical analysis of `Resume_str` across character and word dimensions:

| Metric | Character Count (`char_len`) | Word Count (`word_len`) |
| :--- | :---: | :---: |
| **Mean** | 6,295.31 | 811.33 |
| **Std Dev** | 2,769.25 | 371.01 |
| **Minimum** | 21 (corrupted) / 1,024 (clean min) | 0 (corrupted) / 123 (clean min) |
| **25th Percentile (Q1)** | 5,160.00 | 651.00 |
| **50th Percentile (Median)** | 5,886.50 | 757.00 |
| **75th Percentile (Q3)** | 7,227.25 | 933.00 |
| **99th Percentile** | 16,840.10 | 2,192.50 |
| **Maximum** | 38,842.00 | 5,190.00 |

### Observations:
- Typical resumes contain between 650 and 930 words, providing rich lexical and domain-specific terminology for NLP modeling.
- The maximum resume has 5,190 words (38,842 characters), indicating detailed curriculum vitae profiles with extensive project listings.
- No extreme truncation is needed, but sublinear TF-scaling ($1 + \log(tf)$) will be critical to prevent long CVs from dominating feature weights.

---

## 5. Duplicate & Anomaly Investigation

### Exact Text Duplicate Analysis
- `df.duplicated(subset=['Resume_str'])` flagged 4 rows (2 pairs).
- **Pair 1:**
  - `ID 19147603` (FINANCE)
  - `ID 28398216` (FINANCE)
  - Text: *"FINANCE OFFICER Professional Summary To obtain a position with..."*
- **Pair 2:**
  - `ID 16850314` (AVIATION)
  - `ID 37473139` (AVIATION)
  - Text: *"STOREKEEPER II Professional Summary The purpose of this position is..."*
- **Label Consistency:** Both pairs have 100% internal label agreement.
- **Action:** Deduplicated by keeping the earliest entry for each pair. This eliminates data leakage across prospective train/test splits.

### Corrupted / Empty Resumes
- `ID 12632728` in `BUSINESS-DEVELOPMENT`:
  - `Resume_str`: 21 whitespace characters (`'                     '`).
  - `Resume_html`: `<div class="fontsize fontface ... style="padding-top:NaNpx;"> </div>...`
  - Zero informational content.
  - **Action:** Removed unconditionally from the dataset.

---

## 6. PDF Folder vs Tabular Dataset Reconciliation

- `data/data` directory: 24 subfolders matching the 24 categories.
- Total PDF files: 2,500.
- Analysis revealed that the 16 additional files were all located in `data/data/FINANCE/` with duplicate filename patterns `* (1).pdf`.
- For example, `20880935(1).pdf` is bit-for-bit identical to `20880935.pdf`.
- Excluding the 16 duplicate downloads gives exactly 2,484 PDFs matching the 2,484 IDs in `Resume.csv`.
- Therefore, **the folder dataset does NOT contain additional unique resumes**. Combining them blindly would introduce severe duplicate leakage.

---

## 7. Potential Leakage Vectors & Protective Measures

1. **Identifier Leakage:** `ID` column is an arbitrary database integer or timestamp index. It MUST be excluded from all feature pipelines.
2. **Target Leakage:** `Category` must only be used as $y$, never transformed or concatenated with text.
3. **HTML Metadata Leakage:** `Resume_html` contains layout classes and internal metadata. Using raw HTML could cause classifiers to overfit to template artifacts rather than professional skills. `Resume_str` provides clean, extracted linguistic content and serves as our primary feature representation.
4. **Duplicate Leakage Across Splits:** Deduplication MUST precede data splitting so that no duplicate pair spans the train and test sets.
5. **TF-IDF Vocabulary Leakage:** All vectorizers, normalizers, and scalers MUST be fit strictly on the training partition and only transform the validation/test partitions.

---

## 8. Canonical Data Source Recommendation

**Canonical Source Selected:** `data/raw/Resume.csv` (mirroring `Resume/Resume.csv`) with the clean extraction pipeline generating `data/processed/clean_resumes.csv`.

**Justification:**
1. Structured tabular format guarantees row-level deterministic indexing.
2. Eliminates PDF parsing overhead and font-extraction non-determinism during experimentation.
3. Provides exact alignment between raw text and target categories with 0 missing labels.
