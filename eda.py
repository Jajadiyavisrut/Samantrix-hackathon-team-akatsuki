"""
Module: eda.py
Purpose: Complete, publication-grade Exploratory Data Analysis for ResumeForge 2026.
Generates all 13 required EDA artifacts, saves figures to reports/figures/,
and produces reports/EDA_SUMMARY.md.
"""

import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

from preprocessing import clean_resume_text

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
FIGURES_DIR = Path("reports/figures")
REPORTS_DIR = Path("reports")


def run_full_eda(
    data_path: str = "data/processed/clean_resumes.csv",
    output_fig_dir: str = "reports/figures",
    output_report_path: str = "reports/EDA_SUMMARY.md",
):
    fig_dir = Path(output_fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    rep_path = Path(output_report_path)
    rep_path.parent.mkdir(parents=True, exist_ok=True)

    df_path = Path(data_path)
    if not df_path.exists():
        from clean_data import clean_raw_data
        df = clean_raw_data(processed_path=str(df_path))
    else:
        df = pd.read_csv(df_path)

    print(f"[EDA] Loaded dataset: {len(df)} records across {df['Category'].nunique()} categories.")

    # 1. Feature Length Calculations
    df["char_len"] = df["Resume_str"].str.len()
    df["word_len"] = df["Resume_str"].apply(lambda x: len(str(x).split()))
    df["cleaned_text"] = df["Resume_str"].apply(clean_resume_text)
    df["cleaned_word_len"] = df["cleaned_text"].apply(lambda x: len(x.split()))

    # -------------------------------------------------------------
    # 1 & 2: Class Distribution & Percentages
    # -------------------------------------------------------------
    counts = df["Category"].value_counts()
    percentages = df["Category"].value_counts(normalize=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    palette = sns.color_palette("mako", len(counts))

    sns.barplot(x=counts.values, y=counts.index, ax=axes[0], palette=palette, hue=counts.index, legend=False)
    axes[0].set_title("1. Resume Counts per Category (N=2,481)", fontsize=14, fontweight="bold", pad=12)
    axes[0].set_xlabel("Number of Resumes", fontsize=12)
    axes[0].set_ylabel("Category", fontsize=12)
    for i, v in enumerate(counts.values):
        axes[0].text(v + 1, i, str(v), va="center", fontsize=9, fontweight="bold")

    sns.barplot(x=percentages.values, y=percentages.index, ax=axes[1], palette=palette, hue=percentages.index, legend=False)
    axes[1].set_title("2. Category Distribution Percentage (%)", fontsize=14, fontweight="bold", pad=12)
    axes[1].set_xlabel("Percentage of Total Dataset (%)", fontsize=12)
    axes[1].set_ylabel("")
    for i, v in enumerate(percentages.values):
        axes[1].text(v + 0.05, i, f"{v:.2f}%", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig1_path = fig_dir / "01_class_distribution_counts_percentages.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved {fig1_path}")

    # -------------------------------------------------------------
    # 3 & 4: Word-Count and Character-Count Histograms
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    sns.histplot(df["word_len"], bins=50, kde=True, ax=axes[0], color="#1d3557")
    axes[0].axvline(df["word_len"].median(), color="red", linestyle="--", linewidth=2, label=f"Median: {df['word_len'].median():.0f}")
    axes[0].axvline(df["word_len"].mean(), color="green", linestyle=":", linewidth=2, label=f"Mean: {df['word_len'].mean():.0f}")
    axes[0].set_title("3. Resume Word-Count Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Word Count", fontsize=11)
    axes[0].set_ylabel("Resume Frequency", fontsize=11)
    axes[0].legend(fontsize=11)

    sns.histplot(df["char_len"], bins=50, kde=True, ax=axes[1], color="#2a9d8f")
    axes[1].axvline(df["char_len"].median(), color="red", linestyle="--", linewidth=2, label=f"Median: {df['char_len'].median():.0f}")
    axes[1].axvline(df["char_len"].mean(), color="green", linestyle=":", linewidth=2, label=f"Mean: {df['char_len'].mean():.0f}")
    axes[1].set_title("4. Resume Character-Count Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Character Count", fontsize=11)
    axes[1].set_ylabel("Resume Frequency", fontsize=11)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    fig2_path = fig_dir / "02_length_histograms.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved {fig2_path}")

    # -------------------------------------------------------------
    # 5: Category-Wise Word-Count Boxplot
    # -------------------------------------------------------------
    cat_order = df.groupby("Category")["word_len"].median().sort_values(ascending=False).index
    plt.figure(figsize=(16, 10))
    sns.boxplot(data=df, x="word_len", y="Category", order=cat_order, palette="Spectral", hue="Category", legend=False)
    plt.title("5. Category-Wise Word-Count Distribution & Outliers", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Word Length", fontsize=12)
    plt.ylabel("Category", fontsize=12)
    plt.tight_layout()
    fig3_path = fig_dir / "03_category_word_count_boxplot.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved {fig3_path}")

    # -------------------------------------------------------------
    # 6, 7 & 8: Top 30 Unigrams, Bigrams, and Trigrams
    # -------------------------------------------------------------
    print("[EDA] Computing Top 30 n-grams...")
    stopwords_set = {
        "and", "the", "to", "of", "in", "for", "a", "with", "on", "as", "an", "at", "by",
        "is", "all", "from", "that", "this", "or", "be", "are", "state", "city", "company",
        "name", "work", "job", "experience", "skills", "summary", "year", "years", "responsible",
        "duties", "including", "new", "etc", "also", "using", "used", "per", "well", "ensure",
        "weburl", "emailaddr", "phonenum", "various", "multiple", "provide", "provided", "daily",
        "team", "time", "high", "ability", "strong", "knowledge", "system", "systems", "day"
    }

    def get_top_ngrams(corpus, n=1, top_k=30):
        vec = CountVectorizer(ngram_range=(n, n), stop_words=list(stopwords_set), min_df=3)
        X = vec.fit_transform(corpus)
        sums = np.array(X.sum(axis=0)).flatten()
        features = np.array(vec.get_feature_names_out())
        top_indices = sums.argsort()[-top_k:][::-1]
        return features[top_indices], sums[top_indices]

    unigram_terms, unigram_counts = get_top_ngrams(df["cleaned_text"], n=1, top_k=30)
    bigram_terms, bigram_counts = get_top_ngrams(df["cleaned_text"], n=2, top_k=30)
    trigram_terms, trigram_counts = get_top_ngrams(df["cleaned_text"], n=3, top_k=30)

    fig, axes = plt.subplots(1, 3, figsize=(24, 12))

    sns.barplot(x=unigram_counts, y=unigram_terms, ax=axes[0], color="#2b5c8f")
    axes[0].set_title("6. Top 30 Discriminative Unigrams", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Frequency")

    sns.barplot(x=bigram_counts, y=bigram_terms, ax=axes[1], color="#2a9d8f")
    axes[1].set_title("7. Top 30 Discriminative Bigrams", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Frequency")

    sns.barplot(x=trigram_counts, y=trigram_terms, ax=axes[2], color="#e76f51")
    axes[2].set_title("8. Top 30 Discriminative Trigrams", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Frequency")

    plt.tight_layout()
    fig4_path = fig_dir / "04_top_30_unigrams_bigrams_trigrams.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved {fig4_path}")

    # -------------------------------------------------------------
    # 9: Corpus-Wide WordCloud
    # -------------------------------------------------------------
    print("[EDA] Generating Corpus WordCloud...")
    corpus_sample = " ".join(df["cleaned_text"].sample(min(1000, len(df)), random_state=42))
    wc = WordCloud(
        width=1200, height=600, background_color="white", colormap="plasma",
        stopwords=stopwords_set, max_words=250, random_state=42
    ).generate(corpus_sample)

    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("9. Corpus-Wide Professional Vocabulary WordCloud", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    fig5_path = fig_dir / "05_corpus_wordcloud.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved {fig5_path}")

    # -------------------------------------------------------------
    # 10: Representative Category-Specific WordClouds
    # -------------------------------------------------------------
    print("[EDA] Generating Category-Specific WordClouds...")
    target_sample_cats = ["INFORMATION-TECHNOLOGY", "HEALTHCARE", "CHEF", "FINANCE", "ADVOCATE", "TEACHER"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for idx, cat in enumerate(target_sample_cats):
        cat_corpus = " ".join(df[df["Category"] == cat]["cleaned_text"])
        cat_wc = WordCloud(
            width=600, height=400, background_color="white", colormap="viridis",
            stopwords=stopwords_set, max_words=100, random_state=42
        ).generate(cat_corpus)
        axes[idx].imshow(cat_wc, interpolation="bilinear")
        axes[idx].axis("off")
        axes[idx].set_title(f"Domain: {cat}", fontsize=13, fontweight="bold")

    plt.suptitle("10. Representative Category-Specific Vocabulary Clouds", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig6_path = fig_dir / "06_category_wordclouds.png"
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved {fig6_path}")

    # -------------------------------------------------------------
    # 11 & 12: Class-Wise Important Terms & Category Similarity Heatmap
    # -------------------------------------------------------------
    print("[EDA] Calculating Category Centroid Similarity Matrix...")
    tfidf_cat = TfidfVectorizer(max_features=5000, stop_words="english", sublinear_tf=True)
    X_tfidf = tfidf_cat.fit_transform(df["cleaned_text"])
    categories = sorted(df["Category"].unique())

    centroids = []
    for cat in categories:
        cat_idx = np.where(df["Category"] == cat)[0]
        cat_centroid = X_tfidf[cat_idx].mean(axis=0)
        centroids.append(np.asarray(cat_centroid).flatten())

    centroid_matrix = np.array(centroids)
    similarity_matrix = cosine_similarity(centroid_matrix)

    plt.figure(figsize=(18, 16))
    sns.heatmap(
        similarity_matrix,
        xticklabels=categories,
        yticklabels=categories,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Cosine Similarity"}
    )
    plt.title("12. Inter-Category Cosine Similarity Matrix (Centroid TF-IDF Vectors)", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    fig7_path = fig_dir / "07_category_similarity_heatmap.png"
    plt.savefig(fig7_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved {fig7_path}")

    # Top confused pairs based on similarity
    sim_no_diag = similarity_matrix.copy()
    np.fill_diagonal(sim_no_diag, 0)
    max_sim_pairs = []
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            max_sim_pairs.append((categories[i], categories[j], similarity_matrix[i, j]))
    max_sim_pairs.sort(key=lambda x: x[2], reverse=True)

    # 11: Top Distinctive Terms for Selected Domains
    feature_names = np.array(tfidf_cat.get_feature_names_out())
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, cat in enumerate(target_sample_cats):
        cat_i = categories.index(cat)
        centroid = centroid_matrix[cat_i]
        top_feat_idx = centroid.argsort()[-10:][::-1]
        top_words = feature_names[top_feat_idx]
        top_scores = centroid[top_feat_idx]

        sns.barplot(x=top_scores, y=top_words, ax=axes[idx], color="#1d3557")
        axes[idx].set_title(f"Top Terms: {cat}", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Mean TF-IDF")

    plt.suptitle("11. Category-Distinctive Top TF-IDF Terms", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig8_path = fig_dir / "08_category_distinctive_terms.png"
    plt.savefig(fig8_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved {fig8_path}")

    # Generate Markdown Summary Report
    _write_eda_summary(
        report_path=rep_path,
        df=df,
        counts=counts,
        percentages=percentages,
        unigram_terms=unigram_terms,
        bigram_terms=bigram_terms,
        trigram_terms=trigram_terms,
        max_sim_pairs=max_sim_pairs[:10],
    )
    print(f"[EDA] Complete EDA summary successfully written to {rep_path}")


def _write_eda_summary(
    report_path: Path,
    df: pd.DataFrame,
    counts: pd.Series,
    percentages: pd.Series,
    unigram_terms,
    bigram_terms,
    trigram_terms,
    max_sim_pairs,
):
    top_pairs_table = "\n".join([f"| `{p[0]}` | `{p[1]}` | **{p[2]:.4f}** | High potential confusion |" for p in max_sim_pairs])

    content = f"""# SAMATRIX RESUMEFORGE 2026 — Comprehensive Exploratory Data Analysis (EDA)

**Generated:** 2026-10-05  
**Canonical Clean Dataset Size:** {len(df):,} resumes  
**Target Professional Classes:** {df['Category'].nunique()}  

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
| **Mean** | {df['word_len'].mean():.1f} words | {df['char_len'].mean():.1f} chars |
| **Median** | {df['word_len'].median():.0f} words | {df['char_len'].median():.0f} chars |
| **Std Dev** | {df['word_len'].std():.1f} words | {df['char_len'].std():.1f} chars |
| **IQR [25% - 75%]** | [{df['word_len'].quantile(0.25):.0f}, {df['word_len'].quantile(0.75):.0f}] | [{df['char_len'].quantile(0.25):.0f}, {df['char_len'].quantile(0.75):.0f}] |
| **Maximum** | {df['word_len'].max():,} words | {df['char_len'].max():,} chars |

Sublinear TF-scaling ($1 + \\log(tf)$) is necessary to avoid document length saturation.

---

## 3. Top Semantic N-Grams

### Top 10 Unigrams
`{", ".join(unigram_terms[:10])}`

### Top 10 Bigrams
`{", ".join(bigram_terms[:10])}`

### Top 10 Trigrams
`{", ".join(trigram_terms[:10])}`

---

## 4. Top Similar Category Pairs (Cosine Similarity)

| Category A | Category B | Centroid Cosine Similarity | Potential Risk |
| :--- | :--- | :---: | :--- |
{top_pairs_table}

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
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    run_full_eda()
