"""
Module: src.features.feature_builder
Purpose: Modular, leak-free feature extraction pipelines using Word TF-IDF,
Character n-gram TF-IDF, and combined FeatureUnion representations.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import ResumeTextPreprocessor


def build_word_vectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.90,
    sublinear_tf=True,
    max_features=25000,
):
    """Constructs a Word-level TF-IDF vectorizer."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        max_features=max_features,
        token_pattern=r"(?u)\b\w+\b",  # handles alphanumeric including single tokens
    )


def build_char_vectorizer(
    ngram_range=(3, 5),
    min_df=3,
    max_df=0.90,
    sublinear_tf=True,
    max_features=35000,
    analyzer="char_wb",
):
    """Constructs a Character n-gram TF-IDF vectorizer within word boundaries."""
    return TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        max_features=max_features,
    )


def build_word_char_union(
    word_ngram_range=(1, 2),
    char_ngram_range=(3, 5),
    word_max_features=25000,
    char_max_features=35000,
    min_df=2,
):
    """Combines Word TF-IDF and Character TF-IDF via FeatureUnion."""
    word_vec = build_word_vectorizer(
        ngram_range=word_ngram_range,
        min_df=min_df,
        max_features=word_max_features,
    )
    char_vec = build_char_vectorizer(
        ngram_range=char_ngram_range,
        min_df=min_df,
        max_features=char_max_features,
    )
    return FeatureUnion(
        transformer_list=[
            ("word_tfidf", word_vec),
            ("char_tfidf", char_vec),
        ]
    )


def create_feature_pipeline(feature_type: str = "word_char"):
    """
    Factory creating end-to-end preprocessing + feature extraction pipeline.
    Options: 'word', 'char', 'word_char'.
    """
    preprocessor = ResumeTextPreprocessor(lower=True)

    if feature_type == "word":
        extractor = build_word_vectorizer()
    elif feature_type == "char":
        extractor = build_char_vectorizer()
    elif feature_type == "word_char":
        extractor = build_word_char_union()
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("vectorizer", extractor),
    ])
