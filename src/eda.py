"""
Comprehensive Exploratory Data Analysis (EDA) Module.
Generates all visualizations and quantitative statistics required by the Hackathon guidelines:
1. Class Distribution (imbalance bar chart)
2. Resume Length Distribution (characters & word tokens)
3. Missing / Empty Resumes summary
4. Top Most Frequent Words (cleaned)
5. Overall and Class-Wise WordClouds
6. N-gram Analysis (Unigrams, Bigrams, Trigrams)
7. Class-Specific Discriminative Vocabulary (TF-IDF analysis)
Saves all figures to reports/figures/ and generates textual summary.
"""
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

try:
    from src.config import RAW_CSV_PATH, FIGURES_DIR, PROCESSED_DATA_DIR
    from src.preprocessing import preprocess_text
    from src.data_loader import load_raw_csv
except ImportError:
    from config import RAW_CSV_PATH, FIGURES_DIR, PROCESSED_DATA_DIR
    from preprocessing import preprocess_text
    from data_loader import load_raw_csv

# Configure plotting aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def plot_class_distribution(df: pd.DataFrame, save_path: str = None) -> str:
    """Plot horizontal bar chart of class counts clearly showing distribution and imbalance."""
    counts = df['Category'].value_counts()
    
    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("viridis", len(counts))
    bars = plt.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1], edgecolor='black', alpha=0.85)
    
    plt.title('Resume Dataset: Class Distribution (24 Categories)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Number of Resumes', fontsize=12, labelpad=10)
    plt.ylabel('Category', fontsize=12)
    
    # Add count labels on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1.5, bar.get_y() + bar.get_height()/2, f'{int(width)}',
                 ha='left', va='center', fontsize=10, color='#333333', fontweight='semibold')
                 
    plt.xlim(0, max(counts.values) + 15)
    plt.tight_layout()
    
    out_path = save_path or str(FIGURES_DIR / '01_class_distribution.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_resume_lengths(df: pd.DataFrame, save_path: str = None) -> str:
    """Plot distributions of character and word lengths."""
    char_lens = df['Resume_str'].fillna('').apply(lambda x: len(str(x)))
    word_lens = df['Resume_str'].fillna('').apply(lambda x: len(str(x).split()))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Character count histogram
    sns.histplot(char_lens, bins=40, kde=True, ax=axes[0], color='#2b5c8f', edgecolor='black', alpha=0.7)
    axes[0].set_title('Resume Character Length Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Character Count', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].axvline(char_lens.median(), color='red', linestyle='--', label=f'Median: {int(char_lens.median())}')
    axes[0].legend(frameon=True)
    
    # Word count histogram
    sns.histplot(word_lens, bins=40, kde=True, ax=axes[1], color='#287271', edgecolor='black', alpha=0.7)
    axes[1].set_title('Resume Word Count Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Word Count', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].axvline(word_lens.median(), color='red', linestyle='--', label=f'Median: {int(word_lens.median())}')
    axes[1].legend(frameon=True)
    
    plt.tight_layout()
    out_path = save_path or str(FIGURES_DIR / '02_resume_length_distribution.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_top_frequent_words(df: pd.DataFrame, top_n: int = 25, save_path: str = None) -> str:
    """Plot top N most frequent words across preprocessed corpus."""
    stop_words = set([
        'and', 'the', 'to', 'of', 'in', 'a', 'for', 'with', 'on', 'as', 'by', 'at', 'an', 'be', 'this',
        'which', 'or', 'from', 'is', 'are', 'was', 'were', 'that', 'it', 'will', 'have', 'has', 'had',
        'state', 'city', 'name', 'company', 'url_ref', 'email_ref', 'phone_ref'
    ])
    
    words = []
    for text in df['cleaned_text']:
        tokens = text.split()
        words.extend([w for w in tokens if len(w) > 2 and w not in stop_words])
        
    counts = Counter(words).most_common(top_n)
    word_labels, freqs = zip(*counts)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(freqs), y=list(word_labels), hue=list(word_labels), palette='mako', legend=False)
    plt.title(f'Top {top_n} Most Frequent Meaningful Words in Resumes', fontsize=13, fontweight='bold')
    plt.xlabel('Frequency', fontsize=11)
    plt.ylabel('Token', fontsize=11)
    plt.tight_layout()
    
    out_path = save_path or str(FIGURES_DIR / '03_top_frequent_words.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def generate_wordclouds(df: pd.DataFrame, save_path_overall: str = None, save_path_classes: str = None):
    """Generate overall wordcloud and multi-panel class-wise wordclouds for key categories."""
    stop_words = set([
        'state', 'city', 'name', 'company', 'url_ref', 'email_ref', 'phone_ref',
        'and', 'the', 'for', 'with', 'skills', 'experience', 'management', 'work'
    ])
    
    # 1. Overall WordCloud
    all_text = " ".join(df['cleaned_text'].dropna())
    wc = WordCloud(width=1000, height=500, background_color='white',
                   stopwords=stop_words, colormap='Dark2', max_words=150, random_state=42).generate(all_text)
                   
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Overall Resume Corpus WordCloud', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    
    out_overall = save_path_overall or str(FIGURES_DIR / '04_wordcloud_overall.png')
    plt.savefig(out_overall, dpi=300)
    plt.close()
    
    # 2. Class-Wise WordCloud Panel for Representative Categories
    sample_categories = ['INFORMATION-TECHNOLOGY', 'CHEF', 'AVIATION', 'HEALTHCARE', 'ADVOCATE', 'HR']
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, cat in enumerate(sample_categories):
        cat_text = " ".join(df[df['Category'] == cat]['cleaned_text'].dropna())
        cat_wc = WordCloud(width=500, height=350, background_color='white',
                           stopwords=stop_words, colormap='tab10', max_words=80, random_state=42).generate(cat_text)
        axes[i].imshow(cat_wc, interpolation='bilinear')
        axes[i].set_title(f'Category: {cat}', fontsize=12, fontweight='bold')
        axes[i].axis('off')
        
    plt.tight_layout()
    out_classes = save_path_classes or str(FIGURES_DIR / '05_wordcloud_classes.png')
    plt.savefig(out_classes, dpi=300)
    plt.close()


def plot_ngrams_analysis(df: pd.DataFrame, save_path: str = None) -> str:
    """Analyze and plot top unigrams, bigrams, and trigrams."""
    custom_stop = ['state', 'city', 'name', 'company', 'url_ref', 'email_ref', 'phone_ref', 'and', 'the', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from', 'an', 'as', 'is', 'was', 'are', 'that', 'all']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Unigrams
    vec1 = CountVectorizer(ngram_range=(1, 1), stop_words=custom_stop, max_features=15)
    bag1 = vec1.fit_transform(df['cleaned_text'])
    sum1 = bag1.sum(axis=0)
    words1 = [(word, sum1[0, idx]) for word, idx in vec1.vocabulary_.items()]
    words1 = sorted(words1, key=lambda x: x[1], reverse=True)
    sns.barplot(x=[w[1] for w in words1], y=[w[0] for w in words1], hue=[w[0] for w in words1], ax=axes[0], palette='Blues_r', legend=False)
    axes[0].set_title('Top 15 Unigrams', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Count')
    
    # Bigrams
    vec2 = CountVectorizer(ngram_range=(2, 2), stop_words=custom_stop, max_features=15)
    bag2 = vec2.fit_transform(df['cleaned_text'])
    sum2 = bag2.sum(axis=0)
    words2 = [(word, sum2[0, idx]) for word, idx in vec2.vocabulary_.items()]
    words2 = sorted(words2, key=lambda x: x[1], reverse=True)
    sns.barplot(x=[w[1] for w in words2], y=[w[0] for w in words2], hue=[w[0] for w in words2], ax=axes[1], palette='Greens_r', legend=False)
    axes[1].set_title('Top 15 Bigrams', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Count')
    
    # Trigrams
    vec3 = CountVectorizer(ngram_range=(3, 3), stop_words=custom_stop, max_features=15)
    bag3 = vec3.fit_transform(df['cleaned_text'])
    sum3 = bag3.sum(axis=0)
    words3 = [(word, sum3[0, idx]) for word, idx in vec3.vocabulary_.items()]
    words3 = sorted(words3, key=lambda x: x[1], reverse=True)
    sns.barplot(x=[w[1] for w in words3], y=[w[0] for w in words3], hue=[w[0] for w in words3], ax=axes[2], palette='Purples_r', legend=False)
    axes[2].set_title('Top 15 Trigrams', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Count')
    
    plt.tight_layout()
    out_path = save_path or str(FIGURES_DIR / '06_ngram_analysis.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_class_discriminative_terms(df: pd.DataFrame, save_path: str = None) -> str:
    """
    Extract discriminative top TF-IDF keywords per class to show class-specific vocabulary.
    """
    # Group text by category
    cat_grouped = df.groupby('Category')['cleaned_text'].apply(lambda x: ' '.join(x))
    
    custom_stop = ['state', 'city', 'name', 'company', 'url_ref', 'email_ref', 'phone_ref', 'english', 'skills', 'experience', 'responsible', 'including', 'years', 'work', 'working', 'team', 'management']
    
    tfidf = TfidfVectorizer(max_features=2500, stop_words=custom_stop, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(cat_grouped)
    feature_names = np.array(tfidf.get_feature_names_out())
    
    # Select 6 key categories to display
    display_cats = ['INFORMATION-TECHNOLOGY', 'CHEF', 'ADVOCATE', 'BANKING', 'FITNESS', 'AVIATION']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, cat in enumerate(display_cats):
        if cat in cat_grouped.index:
            row_idx = cat_grouped.index.get_loc(cat)
            row_vec = tfidf_matrix[row_idx].toarray().flatten()
            top_indices = row_vec.argsort()[-8:][::-1]
            top_terms = feature_names[top_indices]
            top_scores = row_vec[top_indices]
            
            sns.barplot(x=top_scores, y=top_terms, hue=top_terms, ax=axes[i], palette='crest', legend=False)
            axes[i].set_title(f'Category: {cat}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('TF-IDF Score')
            
    plt.tight_layout()
    out_path = save_path or str(FIGURES_DIR / '07_class_discriminative_terms.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def run_full_eda() -> dict:
    """Run full EDA pipeline, generating and saving all figures and metrics."""
    df = load_raw_csv()
    
    # Filter empty resumes for text-based analysis
    df['cleaned_text'] = df['Resume_str'].fillna('').apply(preprocess_text)
    df_valid = df[df['cleaned_text'].str.strip().str.len() > 0].copy()
    
    print("Generating EDA Visualizations...")
    p1 = plot_class_distribution(df)
    print(f"  Saved: {p1}")
    p2 = plot_resume_lengths(df)
    print(f"  Saved: {p2}")
    p3 = plot_top_frequent_words(df_valid)
    print(f"  Saved: {p3}")
    generate_wordclouds(df_valid)
    print(f"  Saved: WordClouds")
    p4 = plot_ngrams_analysis(df_valid)
    print(f"  Saved: {p4}")
    p5 = plot_class_discriminative_terms(df_valid)
    print(f"  Saved: {p5}")
    
    return {
        "num_records": len(df),
        "num_classes": df['Category'].nunique(),
        "figures_generated": 7
    }


if __name__ == "__main__":
    summary = run_full_eda()
    print("EDA Complete:", summary)
