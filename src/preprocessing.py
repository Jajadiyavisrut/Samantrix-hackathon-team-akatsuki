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


# Section detection headers for section-aware text weighting
EXPERIENCE_SECTION_PATTERN = re.compile(
    r"\b(experience|employment history|work history|professional experience|career history|work experience|professional background)\b",
    re.IGNORECASE
)
SKILLS_SECTION_PATTERN = re.compile(
    r"\b(technical skills|skills|core competencies|areas of expertise|competencies|technologies|proficiencies|skills & tools)\b",
    re.IGNORECASE
)
SUMMARY_SECTION_PATTERN = re.compile(
    r"\b(summary|professional summary|profile|career profile|executive summary|about me)\b",
    re.IGNORECASE
)
PROJECTS_SECTION_PATTERN = re.compile(
    r"\b(projects|academic projects|personal projects|key projects|selected projects|project experience)\b",
    re.IGNORECASE
)
EDUCATION_SECTION_PATTERN = re.compile(
    r"\b(education|academic background|qualifications|academic credentials)\b",
    re.IGNORECASE
)
CERTIFICATIONS_SECTION_PATTERN = re.compile(
    r"\b(certifications|certificates|licenses|accreditations)\b",
    re.IGNORECASE
)


def preprocess_section_aware(
    text: Union[str, float, None],
    experience_boost: int = 2,
    skills_boost: int = 2
) -> str:
    """
    Section-aware text preprocessor.
    Identifies resume sections and weights primary professional evidence
    (Experience, Job Titles, Core Technical Skills, Profile Summary) higher than
    isolated secondary project descriptions.
    
    If no explicit section headers are detected, gracefully falls back to canonical preprocessing.
    """
    if text is None or not isinstance(text, str):
        return ""
        
    lines = text.split("\n")
    if len(lines) <= 2:
        return preprocess_text(text)
        
    current_section = "header"
    sections = {
        "header": [],
        "experience": [],
        "skills": [],
        "summary": [],
        "education": [],
        "certifications": [],
        "projects": [],
        "other": []
    }
    
    header_found = False
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
            
        line_lower = line_clean.lower()
        if len(line_clean) < 40:
            if EXPERIENCE_SECTION_PATTERN.match(line_lower):
                current_section = "experience"
                header_found = True
                continue
            elif SKILLS_SECTION_PATTERN.match(line_lower):
                current_section = "skills"
                header_found = True
                continue
            elif SUMMARY_SECTION_PATTERN.match(line_lower):
                current_section = "summary"
                header_found = True
                continue
            elif PROJECTS_SECTION_PATTERN.match(line_lower):
                current_section = "projects"
                header_found = True
                continue
            elif EDUCATION_SECTION_PATTERN.match(line_lower):
                current_section = "education"
                header_found = True
                continue
            elif CERTIFICATIONS_SECTION_PATTERN.match(line_lower):
                current_section = "certifications"
                header_found = True
                continue
                
        sections[current_section].append(line_clean)
        
    if not header_found:
        return preprocess_text(text)
        
    weighted_parts = []
    
    # 1. Header (Name, Contact, Job Title if top-of-resume): 2x
    header_text = preprocess_text(" ".join(sections["header"]))
    if header_text:
        weighted_parts.extend([header_text] * 2)
        
    # 2. Summary / Objective: 2x
    summary_text = preprocess_text(" ".join(sections["summary"]))
    if summary_text:
        weighted_parts.extend([summary_text] * 2)
        
    # 3. Professional Experience (Primary career signal): experience_boost x
    exp_text = preprocess_text(" ".join(sections["experience"]))
    if exp_text:
        weighted_parts.extend([exp_text] * experience_boost)
        
    # 4. Technical Skills: skills_boost x
    skills_text = preprocess_text(" ".join(sections["skills"]))
    if skills_text:
        weighted_parts.extend([skills_text] * skills_boost)
        
    # 5. Certifications: 2x
    cert_text = preprocess_text(" ".join(sections["certifications"]))
    if cert_text:
        weighted_parts.extend([cert_text] * 2)
        
    # 6. Education: 1x
    edu_text = preprocess_text(" ".join(sections["education"]))
    if edu_text:
        weighted_parts.append(edu_text)
        
    # 7. Projects (Secondary project domain): 1x
    proj_text = preprocess_text(" ".join(sections["projects"]))
    if proj_text:
        weighted_parts.append(proj_text)
        
    # 8. Other / Miscellaneous: 1x
    other_text = preprocess_text(" ".join(sections["other"]))
    if other_text:
        weighted_parts.append(other_text)
        
    combined = " ".join(weighted_parts).strip()
    return combined if combined else preprocess_text(text)


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
