"""
Module: preprocessing.py
Purpose: Robust, leakage-safe text preprocessing for resume classification.
Preserves technical vocabularies (C++, C#, .NET, SQL, AWS, etc.) while
normalizing noise, Unicode, HTML artifacts, and entities.
Also provides data cleaning & deduplication utilities.
"""

import os
import re
import html
import unicodedata
import pandas as pd
import numpy as np
from typing import List, Union
from sklearn.base import BaseEstimator, TransformerMixin

# Regular expressions for entity normalization
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\(?\b[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
MULTIPLE_SPACES_PATTERN = re.compile(r"[ \t]+")
MULTIPLE_NEWLINES_PATTERN = re.compile(r"\n+")

# Specific preservation map for key technical symbols before punctuation stripping
TECH_TOKEN_PRESERVE = {
    "c++": "token_cpp_token",
    "c#": "token_csharp_token",
    ".net": "token_dotnet_token",
    "asp.net": "token_aspdotnet_token",
    "node.js": "token_nodejs_token",
    "vue.js": "token_vuejs_token",
    "react.js": "token_reactjs_token",
    "angular.js": "token_angularjs_token",
    "d3.js": "token_d3js_token",
    "three.js": "token_threejs_token",
    "ci/cd": "token_cicd_token",
    "tcp/ip": "token_tcpip_token",
    "pl/sql": "token_plsql_token",
    "t-sql": "token_tsql_token",
}

CANONICAL_TECH_MAP = {
    "token_cpp_token": "cplusplus",
    "token_csharp_token": "csharp",
    "token_dotnet_token": "dotnet",
    "token_aspdotnet_token": "aspdotnet",
    "token_nodejs_token": "nodejs",
    "token_vuejs_token": "vuejs",
    "token_reactjs_token": "reactjs",
    "token_angularjs_token": "angularjs",
    "token_d3js_token": "d3js",
    "token_threejs_token": "threejs",
    "token_cicd_token": "cicd",
    "token_tcpip_token": "tcpip",
    "token_plsql_token": "plsql",
    "token_tsql_token": "tsql",
}


def clean_resume_text(text: str, lower: bool = True) -> str:
    """
    Cleans and normalizes resume text while strictly preserving critical domain vocabulary.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    if not text.strip():
        return ""

    # 1. HTML Unescape and strip any stray HTML tags
    text = html.unescape(text)
    text = HTML_TAG_PATTERN.sub(" ", text)

    # 2. Unicode normalization
    text = unicodedata.normalize("NFKD", text)

    # 3. Lowercase if requested
    if lower:
        text = text.lower()

    # Pad with space for clean boundary detection
    text = f" {text} "

    # 4. Protect key technical symbols
    for raw_token, protected_token in TECH_TOKEN_PRESERVE.items():
        escaped_raw = re.escape(raw_token)
        text = re.sub(rf"(?<=\s){escaped_raw}(?=[\s.,;:!?\)\/])", f" {protected_token} ", text)

    # 5. Entity normalization
    text = URL_PATTERN.sub(" weburl ", text)
    text = EMAIL_PATTERN.sub(" emailaddr ", text)
    text = PHONE_PATTERN.sub(" phonenum ", text)

    # 6. Controlled punctuation cleanup
    text = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219\u00B7\u25AA\u25AB\u2013\u2014]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)

    # 7. Map protected tokens to clean canonical feature strings
    for prot, canon in CANONICAL_TECH_MAP.items():
        text = text.replace(prot, canon)

    # 8. Collapse whitespace
    text = MULTIPLE_SPACES_PATTERN.sub(" ", text)
    text = MULTIPLE_NEWLINES_PATTERN.sub(" ", text)
    return text.strip()


class ResumeTextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for resume text preprocessing.
    """
    def __init__(self, lower: bool = True):
        self.lower = lower

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            return X.apply(lambda t: clean_resume_text(t, lower=self.lower)).tolist()
        elif isinstance(X, (list, tuple)):
            return [clean_resume_text(t, lower=self.lower) for t in X]
        else:
            return [clean_resume_text(str(X), lower=self.lower)]


def clean_raw_data(
    raw_path: str = "data/raw/Resume.csv",
    processed_path: str = "data/processed/clean_resumes.csv",
    report_path: str = "reports/DATA_CLEANING_REPORT.md",
) -> pd.DataFrame:
    """
    Cleans raw resume data by removing corrupted samples and exact duplicates.
    """
    if not os.path.exists(raw_path):
        # Fallback to Resume/Resume.csv if data/raw doesn't have it yet
        if os.path.exists("Resume/Resume.csv"):
            raw_path = "Resume/Resume.csv"
        else:
            raise FileNotFoundError(f"Raw data file not found at {raw_path}")

    df_raw = pd.read_csv(raw_path)
    initial_rows = len(df_raw)

    word_counts = df_raw["Resume_str"].apply(lambda x: len(str(x).split()))
    empty_mask = (df_raw["Resume_str"].str.strip() == "") | (word_counts < 10)
    df_valid = df_raw[~empty_mask].copy()

    duplicate_mask = df_valid.duplicated(subset=["Resume_str"], keep="first")
    df_clean = df_valid[~duplicate_mask].copy().reset_index(drop=True)

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_clean.to_csv(processed_path, index=False)
    print(f"[Preprocessing] Cleaned {initial_rows} raw rows -> {len(df_clean)} clean rows saved to {processed_path}")
    return df_clean


if __name__ == "__main__":
    test_str = "Expert in C++, C#, .NET, Python, SQL, and Docker. Contact: test@example.com, https://github.com"
    print("Cleaned test sample:", clean_resume_text(test_str))
