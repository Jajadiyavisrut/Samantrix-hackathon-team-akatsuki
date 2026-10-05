"""
Module: src.preprocessing.text_cleaner
Purpose: Robust, leakage-safe text preprocessing for resume classification.
Preserves technical vocabularies (C++, C#, .NET, SQL, AWS, etc.) while
normalizing noise, Unicode, HTML artifacts, and identifiers.
"""

import re
import html
import unicodedata
import pandas as pd
from typing import List, Union
from sklearn.base import BaseEstimator, TransformerMixin


# Regular expressions for entity normalization
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\(?\b[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
MULTIPLE_SPACES_PATTERN = re.compile(r"[ \t]+")
MULTIPLE_NEWLINES_PATTERN = re.compile(r"\n+")

# Specific preservation map for key technical symbols before general punctuation stripping
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

TECH_TOKEN_RESTORE = {v: k.replace("+", "plus").replace("#", "sharp").replace(".", "dot").replace("/", "slash").replace("-", "") for k, v in TECH_TOKEN_PRESERVE.items()}
# Specifically, token_cpp_token -> cplusplus, token_csharp_token -> csharp, token_dotnet_token -> dotnet
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

    Steps:
    1. Unescape HTML entities & strip residual HTML tags if any.
    2. NFKD Unicode normalization (handles accents, special quotes, dashes).
    3. Protect technical tokens (C++, C#, .NET, Node.js, PL/SQL, etc.).
    4. Replace URLs, Emails, and Phone Numbers with generic tokens.
    5. Replace bullets, non-standard hyphens, and decorative punctuation with whitespace.
    6. Normalize residual non-alphanumeric punctuation while keeping protected tokens.
    7. Restore protected tokens into canonical alphanumeric tokens (e.g. cplusplus, csharp, dotnet).
    8. Normalize multiple whitespace and newlines.
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

    # 6. Controlled punctuation cleanup:
    # Retain standard letters, numbers, and underscores (which form protected_token)
    # Convert hyphens, slashes, brackets, bullets, etc. to spaces
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
    Can be safely placed inside an sklearn Pipeline.
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
