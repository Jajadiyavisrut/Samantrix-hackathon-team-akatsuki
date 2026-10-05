# SAMATRIX RESUMEFORGE 2026 — Comprehensive Exploratory Data Analysis (EDA)

**Generated:** 2026-10-05  
**Canonical Clean Dataset Size:** 2,481 resumes  
**Target Professional Classes:** 24  

---

## 1. Class Distribution & Imbalance Analysis

- **Total Samples:** 2,481 clean resumes.
- **Dominant Categories:** `INFORMATION-TECHNOLOGY` (120, 4.84%) and `BUSINESS-DEVELOPMENT` (119, 4.80%).
- **Tail Minority Categories:** `BPO` (22, 0.89%) and `AUTOMOBILE` (36, 1.45%).
- **Mitigation Requirement:** Stratified splitting is non-negotiable; balanced class weighting should be evaluated experimentally to prevent under-representation of `BPO` and `AUTOMOBILE`.

---

## 2. Length Dynamics (Word & Character)

| Metric | Word Count (`word_len`) | Character Count (`char_len`) |
| :--- | :---: | :---: |
| **Mean** | 811.8 words | 6298.7 chars |
| **Median** | 757 words | 5888 chars |
| **Std Dev** | 370.8 words | 2767.9 chars |
| **IQR [25% - 75%]** | [651, 933] | [5160, 7228] |
| **Maximum** | 5,190 words | 38,842 chars |

Sublinear TF-scaling ($1 + \log(tf)$) is necessary to avoid document length saturation.

---

## 3. Top Semantic N-Grams

### Top 10 Unigrams
`management, sales, customer, business, service, development, training, project, manager, information`

### Top 10 Bigrams
`customer service, project management, microsoft office, business development, human resources, public relations, social media, problem solving, bachelor science, business administration`

### Top 10 Trigrams
`microsoft office suite, excellent customer service, but not limited, education bachelor science, human resource management, customer service representative, word excel powerpoint, excel microsoft office, education school diploma, exceptional customer service`

---

## 4. Top Similar Category Pairs (Cosine Similarity)

| Category A | Category B | Centroid Cosine Similarity | Potential Risk |
| :--- | :--- | :---: | :--- |
| `ACCOUNTANT` | `FINANCE` | **0.8767** | High potential confusion |
| `ADVOCATE` | `HEALTHCARE` | **0.8729** | High potential confusion |
| `APPAREL` | `SALES` | **0.8598** | High potential confusion |
| `ARTS` | `TEACHER` | **0.8478** | High potential confusion |
| `CONSULTANT` | `INFORMATION-TECHNOLOGY` | **0.8337** | High potential confusion |
| `DIGITAL-MEDIA` | `PUBLIC-RELATIONS` | **0.8280** | High potential confusion |
| `BUSINESS-DEVELOPMENT` | `CONSULTANT` | **0.8249** | High potential confusion |
| `BANKING` | `BUSINESS-DEVELOPMENT` | **0.8239** | High potential confusion |
| `BANKING` | `CONSULTANT` | **0.8091** | High potential confusion |
| `BANKING` | `FINANCE` | **0.7988** | High potential confusion |

---

## 5. Visualizations Index

Saved in `reports/figures/`:
1. `01_class_distribution_counts_percentages.png`: Class counts and relative percentages.
2. `02_length_histograms.png`: Word-count and character-count distributions with median and mean lines.
3. `03_category_word_count_boxplot.png`: Word length spread and outlier analysis by category.
4. `04_top_30_unigrams_bigrams_trigrams.png`: Top 30 unigrams, bigrams, and trigrams.
5. `05_corpus_wordcloud.png`: Global corpus vocabulary cloud.
6. `06_category_wordclouds.png`: 6 representative domain wordclouds.
7. `07_category_similarity_heatmap.png`: Full 24x24 cosine similarity matrix.
8. `08_category_distinctive_terms.png`: Distinctive top-10 TF-IDF features across sample domains.
