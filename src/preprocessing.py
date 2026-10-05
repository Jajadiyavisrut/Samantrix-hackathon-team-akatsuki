"""
Reproducible Preprocessing Module for Resume Classification.
Preserves technical tokens (C++, C#, .NET, Python, AWS, SQL, etc.),
normalizes whitespace, unicode artifacts, and emails/URLs appropriately.
Used identically across training, evaluation, inference, and Streamlit app.
"""
import re
import unicodedata
from typing import List, Union


# Technical terms and compound tokens mapping to preserve them through punctuation stripping
TECH_NORMALIZATION_MAP = [
    (r"\bc\+\+(?!\w)", " cpp "),
    (r"\bc#(?!\w)", " csharp "),
    (r"(?<!\w)\.net\b", " dotnet "),
    (r"\bnode\.js\b", " nodejs "),
    (r"\bvue\.js\b", " vuejs "),
    (r"\breact\.js\b", " reactjs "),
    (r"\bangular\.js\b", " angularjs "),
    (r"\bci/cd\b", " cicd "),
    (r"\btcp/ip\b", " tcpip "),
    (r"\bpl/sql\b", " plsql "),
]


def preprocess_text(text: Union[str, float, None]) -> str:
    """
    Central, canonical, and reproducible text preprocessing function.
    
    Processing Steps:
    1. Handle null / non-string input safely.
    2. Normalize Unicode characters (NFKD decomposition to standard ASCII equivalents).
    3. Remove HTML/XML markup tags if present.
    4. Normalize URLs to placeholder 'url_ref'.
    5. Normalize Email addresses to 'email_ref'.
    6. Normalize Phone numbers to 'phone_ref'.
    7. Normalize and protect special technical tokens (C++, C#, .NET, Node.js, CI/CD, etc.).
    8. Remove unwanted punctuation while preserving alphanumeric characters and whitespace.
    9. Lowercase and collapse consecutive whitespace and newlines.
    
    Args:
        text: Raw resume text string.
        
    Returns:
        Cleaned, normalized text string.
    """
    if text is None or not isinstance(text, str):
        return ""
    
    # 1. Normalize unicode characters (smart quotes, em-dashes, full-width characters)
    text = unicodedata.normalize("NFKD", text)
    
    # 2. Strip HTML tags if any exist
    text = re.sub(r"<[^>]+>", " ", text)
    
    # 3. Lowercase text
    text_lower = text.lower()
    
    # 4. Standardize URLs
    text_lower = re.sub(r"https?://\S+|www\.\S+", " url_ref ", text_lower)
    
    # 5. Standardize Email addresses
    text_lower = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " email_ref ", text_lower)
    
    # 6. Standardize Phone numbers
    text_lower = re.sub(r"\(?\b[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b", " phone_ref ", text_lower)
    
    # 7. Normalize key technical compound tokens
    for pattern, replacement in TECH_NORMALIZATION_MAP:
        text_lower = re.sub(pattern, replacement, text_lower)
        
    # 8. Remove punctuation and non-alphanumeric symbols (keeping word chars and spaces)
    text_lower = re.sub(r"[^\w\s]", " ", text_lower)
    
    # 9. Normalize multiple whitespaces, tabs, and newlines to a single space
    cleaned = re.sub(r"\s+", " ", text_lower).strip()
    
    return cleaned


def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes preprocessed text into individual words/tokens.
    """
    cleaned = preprocess_text(text)
    if not cleaned:
        return []
    return cleaned.split()


if __name__ == "__main__":
    sample = "Senior C++ and C# Developer with 5+ years experience in .NET, Node.js, Python, & SQL. Contact: test@example.com (555) 123-4567. Website: https://github.com/test"
    print("Original:", sample)
    print("Cleaned: ", preprocess_text(sample))
    print("Tokens:  ", tokenize_text(sample))
