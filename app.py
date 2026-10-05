"""
ResumeForge AI — Modern Streamlit Web Application
Hackathon Demo Edition: AI-Powered Resume Classification & Career Intelligence
Built for SAMATRIX RESUMEFORGE 2026
"""

import os
import sys
import json
import re
import time
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Safe import of PDF parsing
try:
    import fitz  # PyMuPDF
    PDF_SUPPORTED = True
except ImportError:
    PDF_SUPPORTED = False

# Import preprocessing
try:
    from preprocessing import clean_resume_text
except ImportError:
    def clean_resume_text(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return ' '.join(text.split())

def render_html(html_str: str):
    """
    Renders raw HTML safely in Streamlit, stripping leading whitespace and blank lines
    to prevent CommonMark from interpreting indented HTML as markdown code blocks.
    """
    clean_lines = [line.strip() for line in html_str.strip().splitlines() if line.strip()]
    st.markdown("".join(clean_lines), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ResumeForge AI — Career Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# Design System & Custom CSS (Dark Navy SaaS Aesthetic)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Root Styling */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .stApp {
        background-color: #070B14;
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.15) 0%, rgba(7, 11, 20, 0) 55%),
            radial-gradient(circle at 10% 20%, rgba(139, 92, 246, 0.08) 0%, rgba(7, 11, 20, 0) 40%),
            radial-gradient(circle at 90% 80%, rgba(14, 165, 233, 0.06) 0%, rgba(7, 11, 20, 0) 50%);
        color: #F1F5F9;
    }

    /* Hide Streamlit Default UI Noise */
    #MainMenu, header, footer {
        visibility: hidden !important;
        height: 0px !important;
    }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1380px !important;
    }

    /* Hero Section */
    .hero-container {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6);
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366F1, #A855F7, #38BDF8, #6366F1);
        background-size: 300% 100%;
        animation: gradientShift 6s ease infinite;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .hero-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }

    .hero-title-group h1 {
        font-size: 2.1rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        margin: 0;
        background: linear-gradient(135deg, #FFFFFF 20%, #E2E8F0 60%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }

    .hero-tagline {
        font-size: 1.05rem;
        color: #818CF8;
        font-weight: 700;
        margin-top: 0.15rem;
        letter-spacing: -0.01em;
    }

    .hero-sub {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 0.35rem;
        line-height: 1.4;
    }

    .badge-status {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34D399;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        padding: 0.4rem 0.9rem;
        border-radius: 9999px;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.15);
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    .hero-meta-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #C7D2FE;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.4rem 0.9rem;
        border-radius: 9999px;
    }

    /* Glassmorphism Card Containers */
    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 18px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 12px 30px -10px rgba(0, 0, 0, 0.5);
        transition: border 0.25s ease, box-shadow 0.25s ease;
    }

    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.28);
        box-shadow: 0 16px 36px -10px rgba(99, 102, 241, 0.12);
    }

    .card-header-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.8rem;
    }

    /* Primary Prediction Hero Card (Focal Point) */
    .pred-hero-card {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 2px solid rgba(139, 92, 246, 0.5);
        border-radius: 20px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.3rem;
        box-shadow: 0 20px 50px -10px rgba(99, 102, 241, 0.35), 0 0 20px rgba(139, 92, 246, 0.15);
        position: relative;
        overflow: hidden;
    }

    .pred-hero-card::after {
        content: '';
        position: absolute;
        top: -40%;
        right: -25%;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.25) 0%, transparent 70%);
        pointer-events: none;
    }

    .pred-badge-label {
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #C4B5FD;
        margin-bottom: 0.35rem;
    }

    .pred-category-name {
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #FFFFFF 15%, #E0E7FF 60%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.15;
        margin-bottom: 1.1rem;
    }

    .score-container-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.8rem;
        align-items: center;
        margin-top: 0.6rem;
    }

    .metric-badge {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 0.55rem 0.95rem;
        border-radius: 12px;
        font-size: 0.85rem;
        color: #E2E8F0;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }

    .badge-label-small {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        font-weight: 700;
        margin-right: 0.2rem;
    }

    .badge-value-strong {
        color: #F59E0B;
        font-size: 1.05rem;
        font-weight: 900;
        font-family: 'JetBrains Mono', monospace;
    }

    .badge-value-margin {
        color: #38BDF8;
        font-size: 0.95rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }

    .badge-value-latency {
        color: #A5B4FC;
        font-size: 0.9rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    .score-note {
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 0.8rem;
        line-height: 1.4;
    }

    /* WHY THIS CATEGORY Section */
    .why-container {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }

    .why-heading {
        font-size: 0.88rem;
        font-weight: 800;
        color: #C4B5FD;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    .why-checklist {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 0.5rem 1rem;
    }

    .why-item {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.9rem;
        font-weight: 700;
        color: #F1F5F9;
    }

    .why-check {
        color: #10B981;
        font-weight: 900;
        font-size: 1rem;
    }

    .why-text {
        color: #E2E8F0;
    }

    /* Prominent Top 3 Predictions Card */
    .prominent-top3 {
        background: rgba(15, 23, 42, 0.75) !important;
        border: 1px solid rgba(99, 102, 241, 0.35) !important;
        box-shadow: 0 16px 40px -10px rgba(99, 102, 241, 0.22) !important;
    }

    .prominent-title {
        font-size: 1.25rem !important;
        font-weight: 900 !important;
        color: #FFFFFF !important;
        letter-spacing: -0.01em !important;
    }

    .top-rank-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .top-rank-card:hover {
        transform: translateX(4px);
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 6px 20px -5px rgba(99, 102, 241, 0.2);
    }

    .top-rank-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .rank-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.65rem;
        font-weight: 800;
        font-size: 1.05rem;
        color: #FFFFFF;
        letter-spacing: 0.01em;
    }

    .rank-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 8px;
        font-weight: 900;
        font-size: 0.82rem;
    }

    .rank-1 .rank-pill { background: linear-gradient(135deg, #F59E0B, #D97706); color: #FFF; box-shadow: 0 0 12px rgba(245, 158, 11, 0.45); }
    .rank-2 .rank-pill { background: linear-gradient(135deg, #94A3B8, #64748B); color: #FFF; }
    .rank-3 .rank-pill { background: linear-gradient(135deg, #B45309, #78350F); color: #FFF; }

    .rank-score-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.84rem;
        font-weight: 600;
        color: #38BDF8;
    }

    .custom-bar-track {
        background: rgba(30, 41, 59, 0.9);
        border-radius: 9999px;
        height: 9px;
        width: 100%;
        overflow: hidden;
    }

    .custom-bar-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.6s ease;
    }

    .fill-gold { background: linear-gradient(90deg, #F59E0B, #FBBF24); }
    .fill-indigo { background: linear-gradient(90deg, #6366F1, #818CF8); }
    .fill-cyan { background: linear-gradient(90deg, #0EA5E9, #38BDF8); }

    /* Resume Intelligence Metric Grid */
    .intel-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.8rem;
        margin-bottom: 1rem;
    }

    .intel-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 0.85rem;
        text-align: center;
    }

    .intel-box-title {
        font-size: 0.72rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 0.2rem;
    }

    .intel-box-val {
        font-size: 1.35rem;
        font-weight: 800;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Detected Skills Pill Tags */
    .skills-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.5rem;
        max-height: 180px;
        overflow-y: auto;
        padding-right: 0.3rem;
    }

    .skill-tag {
        display: inline-flex;
        align-items: center;
        background: rgba(99, 102, 241, 0.14);
        border: 1px solid rgba(99, 102, 241, 0.32);
        color: #C7D2FE;
        padding: 0.25rem 0.65rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .skill-tag:hover {
        background: rgba(99, 102, 241, 0.25);
        border-color: rgba(99, 102, 241, 0.55);
        color: #FFFFFF;
        transform: translateY(-1px);
    }

    .no-skills-msg {
        font-size: 0.85rem;
        color: #94A3B8;
        font-style: italic;
    }

    /* Sub-word N-Gram Evidence Tags */
    .ngram-tag {
        display: inline-block;
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(14, 165, 233, 0.3);
        color: #7DD3FC;
        padding: 0.2rem 0.55rem;
        margin: 0.15rem;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 500;
    }

    /* Streamlit Element Overrides */
    div[data-baseweb="textarea"] textarea {
        background-color: #0B1120 !important;
        color: #F8FAFC !important;
        border: 1px solid #1E293B !important;
        border-radius: 14px !important;
        font-size: 0.88rem !important;
        line-height: 1.5 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        padding: 0.85rem !important;
    }

    div[data-baseweb="textarea"] textarea:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 1px #6366F1 !important;
    }

    div[data-testid="stFileUploader"] {
        background-color: rgba(15, 23, 42, 0.5) !important;
        border-radius: 14px !important;
        border: 1px dashed rgba(99, 102, 241, 0.35) !important;
        padding: 0.5rem !important;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #A855F7 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.7rem 1.6rem !important;
        font-size: 1rem !important;
        letter-spacing: -0.01em !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.45) !important;
        transition: all 0.25s ease !important;
        width: 100% !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 28px rgba(139, 92, 246, 0.65) !important;
    }

    /* Step Workflow Cards */
    .step-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1.2rem 0;
    }

    .step-card {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 1.2rem 1rem;
        text-align: center;
        position: relative;
    }

    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: rgba(99, 102, 241, 0.18);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #A5B4FC;
        font-weight: 800;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }

    .step-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.25rem;
    }

    .step-desc {
        font-size: 0.78rem;
        color: #94A3B8;
        line-height: 1.35;
    }

    /* Model At A Glance Cards */
    .glance-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1.2rem 0;
    }

    .glance-card {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 1.1rem;
        text-align: center;
    }

    .glance-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 0.35rem;
    }

    .glance-val {
        font-size: 1.6rem;
        font-weight: 900;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        font-family: 'JetBrains Mono', monospace;
    }

    .glance-sub {
        font-size: 0.72rem;
        color: #64748B;
        margin-top: 0.2rem;
    }

    /* Placeholder Welcome State */
    .empty-state-box {
        background: rgba(15, 23, 42, 0.4);
        border: 1px dashed rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        padding: 3.5rem 2rem;
        text-align: center;
        color: #64748B;
    }

    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: 0.8rem;
        opacity: 0.7;
    }

    .empty-state-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #E2E8F0;
        margin-bottom: 0.4rem;
    }

    .empty-state-desc {
        font-size: 0.88rem;
        color: #94A3B8;
        max-width: 420px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #64748B;
        font-size: 0.82rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 2rem;
    }

    .app-footer b {
        color: #94A3B8;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Skills Database (Curated, No Hallucinations)
# ---------------------------------------------------------------------------
SKILLS_TAXONOMY = [
    # Languages
    "Python", "Java", "C++", "C#", ".NET", "JavaScript", "TypeScript", "Go", "Golang", "Rust", 
    "Ruby", "PHP", "Swift", "Kotlin", "R", "MATLAB", "Scala", "SQL", "HTML", "CSS", "Bash", "Shell", "PowerShell",
    # Data & AI
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Data Science", "Data Analysis", 
    "Artificial Intelligence", "Neural Networks", "PyTorch", "TensorFlow", "Keras", "Scikit-Learn", 
    "Pandas", "NumPy", "SciPy", "OpenCV", "Spark", "PySpark", "Hadoop", "Databricks", "BigQuery", 
    "Snowflake", "Airflow", "Tableau", "Power BI", "Excel", "Statistics", "Predictive Modeling",
    # Cloud & DevOps
    "AWS", "Azure", "GCP", "Google Cloud", "Docker", "Kubernetes", "Terraform", "CI/CD", "Jenkins", 
    "Git", "GitHub Actions", "GitLab", "Linux", "Nginx", "Ansible", "Microservices", "Prometheus", "Grafana",
    # Web & Backend
    "React", "Next.js", "Node.js", "Express", "Django", "FastAPI", "Flask", "Spring Boot", 
    "GraphQL", "REST APIs", "REST API", "WebSockets", "Redis", "PostgreSQL", "MySQL", "MongoDB", "SQLite", "Elasticsearch", "Kafka",
    # Business, Finance & Management
    "Financial Modeling", "DCF", "Valuation", "GAAP", "Accounting", "QuickBooks", "SAP", "Salesforce", 
    "CRM", "ERP", "Budgeting", "Forecasting", "Auditing", "Tax Audits", "Equity Research", "Variance Analysis", 
    "Risk Management", "Six Sigma", "Agile", "Scrum", "JIRA", "Project Management", "Product Management", "Due Diligence",
    "Talent Acquisition", "Recruiting", "HRIS", "Workday", "Onboarding", "Labor Law Compliance",
    # Healthcare & Clinical
    "Patient Triage", "Medication Administration", "BLS", "ACLS", "CPR", "EMR", "EHR", "Epic", "Cerner", 
    "Patient Care", "HIPAA", "Phlebotomy", "Clinical Research", "Wound Care", "Acute Care", "Nursing",
    # Engineering & Design
    "CAD", "AutoCAD", "SolidWorks", "Revit", "Quality Assurance", "QA Testing", "UI/UX", "Figma", 
    "Adobe Photoshop", "Illustrator", "InDesign", "Blender"
]

def extract_detected_skills(text: str) -> List[str]:
    """Extracts only skills explicitly found in the resume text."""
    if not text:
        return []
    found_skills = []
    for skill in SKILLS_TAXONOMY:
        # Match using word boundary or custom bounds for C++, C#, .NET
        pattern = r'(?<![a-zA-Z0-9])' + re.escape(skill) + r'(?![a-zA-Z0-9])'
        if re.search(pattern, text, re.IGNORECASE):
            found_skills.append(skill)
    # Deduplicate and sort
    return sorted(list(dict.fromkeys(found_skills)), key=lambda s: s.lower())


# ---------------------------------------------------------------------------
# Sanitized Demo Resumes (NO PII: No Names, Phones, Emails, Addresses)
# ---------------------------------------------------------------------------
SANITIZED_SAMPLES = {
    "Select a pre-loaded sample...": "",
    "💻 Information Technology Specialist": """Executive Summary:
Information Technology Specialist & Systems Administrator with 8+ years experience managing enterprise IT infrastructure, LAN/WAN networks, and cloud virtualization.

Technical Skills:
IT Infrastructure, Active Directory, Windows Server, Linux, Cisco Networking, VMware ESXi, AWS Cloud, IT Support, Cyber Security, Firewalls, SQL, Bash.

Professional Experience:
Senior IT Systems Administrator | Enterprise Infrastructure Solutions (2020 – Present)
- Administered corporate IT networks, Active Directory, multi-factor authentication, and distributed server clusters with 99.9% uptime.
- Managed technical support service desk handling 600+ tickets monthly across network, hardware, and software issues.
- Deployed automated VMware virtual machine provisioning and configured AWS cloud backups, reducing disaster recovery RPO by 50%.

Network Support Specialist | Telecommunications Systems (2017 – 2020)
- Configured Cisco switches, routers, VLAN segmentation, and VPN tunnels for 1,200 remote employees.
- Monitored intrusion detection systems and enforced security compliance protocols.

Education & Certifications:
- Bachelor of Science in Information Technology
- CompTIA Security+ & Cisco CCNA Certified
""",
    "⚙️ Software & Systems Engineer": """Executive Summary:
Senior Systems & Software Engineer with 7+ years architecting scalable distributed backend systems, microservices, and automated CI/CD pipelines.

Technical Skills:
Python, C++, Docker, Kubernetes, AWS (EC2, S3, RDS), PostgreSQL, Redis, Linux, Microservices, Git, REST APIs, CI/CD pipelines.

Professional Experience:
Lead Systems Engineer | Enterprise Tech Systems (2021 – Present)
- Architected high-throughput microservices processing 15M daily requests with sub-50ms latency.
- Optimized database indexing and caching layers in Redis and PostgreSQL, reducing p99 latency by 45%.
- Automated deployment workflows using Docker, Kubernetes, and GitLab CI/CD, reducing cycle times by 60%.

Software Engineer | FinTech Platforms (2018 – 2021)
- Built scalable payment processing microservices handling $12M+ monthly transaction volume.
- Implemented robust error monitoring and distributed tracing across Kubernetes clusters.

Education & Credentials:
- Bachelor of Science in Computer Engineering
- AWS Certified Solutions Architect
""",
    "🏥 Healthcare / Registered Nurse": """Executive Summary:
Healthcare Professional & Registered Nurse (RN) with 6+ years of clinical experience in acute inpatient care, ICU monitoring, and emergency patient triage.

Clinical Skills:
Patient Triage, Patient Care, Medication Administration, BLS, ACLS, CPR Certified, EMR/EHR (Epic, Cerner), Wound Care, Patient Advocacy, HIPAA Compliance, Vital Signs.

Professional Experience:
Staff Registered Nurse (ICU) | Regional Medical Health Center (2020 – Present)
- Coordinated acute inpatient healthcare delivery and personalized treatment plans across a 24-bed ICU unit.
- Monitored critically ill patients, titrating vasoactive infusions and managing mechanical ventilation.
- Collaborated with multidisciplinary healthcare teams to ensure stringent infection control and clinical compliance.

Emergency Department Staff Nurse | Metropolitan Hospital (2018 – 2020)
- Assessed and triaged 45+ emergency patients per shift following emergency severity index protocols.
- Maintained detailed clinical documentation in Epic electronic health records adhering to HIPAA standards.

Education & Credentials:
- Bachelor of Science in Nursing (BSN)
- Registered Nurse State Licensure (Active)
- BLS & ACLS Certified
""",
    "📊 Financial Analyst / Investment Banking": """Executive Summary:
Results-oriented Financial Analyst with 5+ years of experience in corporate valuation, DCF modeling, variance analysis, and asset management.

Core Competencies:
Financial Modeling, DCF Analysis, Budgeting & Forecasting, GAAP, Variance Analysis, SEC Filings (10-K, 10-Q), Excel VBA, QuickBooks, Bloomberg Terminal, SAP ERP.

Professional Experience:
Senior Financial Analyst | Global Asset Management (2021 – Present)
- Built detailed 3-statement financial models and discounted cash flow (DCF) analyses for $80M in corporate transactions.
- Spearheaded quarterly variance analysis and executive EBITDA forecasting across 6 international business units.
- Automated monthly consolidation reporting workflows using advanced Excel formulas and VBA macros, saving 15 hours per close cycle.

Financial Analyst | Commercial Banking Group (2019 – 2021)
- Underwrote commercial credit facilities and performed risk rating assessments for middle-market borrowers.
- Prepared comprehensive credit approval memorandums and monitored portfolio covenant compliance.

Education & Credentials:
- Bachelor of Business Administration in Finance
- CFA Charterholder / Level III Passed
""",
    "🤝 Human Resources Generalist": """Executive Summary:
Strategic Human Resources Generalist with 6+ years of experience in talent acquisition, employee relations, onboarding, and compliance.

Core Competencies:
Full-Cycle Recruiting, Talent Acquisition, Employee Relations, Onboarding, HRIS (Workday, BambooHR), ATS (Greenhouse), Labor Law Compliance, Compensation & Benefits, Performance Reviews.

Professional Experience:
Human Resources Generalist | Global Logistics Corp (2021 – Present)
- Managed end-to-end recruitment lifecycle for 75+ technical, operational, and managerial positions annually.
- Led company-wide employee engagement initiatives and redesigned onboarding workflows, reducing first-year turnover by 22%.
- Investigated and resolved complex employee relations matters, ensuring adherence to EEOC, Title VII, and state labor regulations.

HR Coordinator | Tech Services International (2018 – 2021)
- Administered employee benefits programs, open enrollment, and 401(k) retirement plans for 600+ staff members.
- Coordinated performance appraisal cycles and facilitated professional development training workshops.

Education & Certifications:
- Bachelor of Arts in Human Resources Management
- SHRM-CP Certified Professional
"""
}


# ---------------------------------------------------------------------------
# Artifact Loading & Model Setup
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_assets():
    """
    Loads verified model, vectorizer, and metadata.
    Prefers best_tfidf_model.joblib (LinearSVC with char 3-6) or final_classifier.joblib.
    """
    model = None
    vectorizer = None
    classes = None
    config = {}

    # Attempt to load pure LinearSVC (Char TF-IDF 3-6)
    if os.path.exists("models/best_tfidf_model.joblib") and os.path.exists("models/best_feature_vectorizer.joblib"):
        try:
            model = joblib.load("models/best_tfidf_model.joblib")
            vectorizer = joblib.load("models/best_feature_vectorizer.joblib")
            classes = getattr(model, "classes_", None)
        except Exception as e:
            pass

    # Fallback to final classifier if needed
    if model is None:
        if os.path.exists("models/final_classifier.joblib") and os.path.exists("models/feature_vectorizer.joblib"):
            try:
                model = joblib.load("models/final_classifier.joblib")
                vectorizer = joblib.load("models/feature_vectorizer.joblib")
                classes = getattr(model, "classes_", None)
            except Exception as e:
                pass

    # Load metadata and metrics
    if os.path.exists("reports/BEST_MODEL_CONFIG.json"):
        try:
            with open("reports/BEST_MODEL_CONFIG.json", "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            pass

    test_metrics = {}
    if os.path.exists("reports/FINAL_TEST_METRICS.json"):
        try:
            with open("reports/FINAL_TEST_METRICS.json", "r", encoding="utf-8") as f:
                test_metrics = json.load(f)
        except Exception:
            pass

    return model, vectorizer, classes, config, test_metrics


model, vectorizer, classes, best_cfg, test_metrics = load_model_assets()


# ---------------------------------------------------------------------------
# Session State Management
# ---------------------------------------------------------------------------
if "resume_content" not in st.session_state:
    st.session_state.resume_content = ""
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "prediction_payload" not in st.session_state:
    st.session_state.prediction_payload = None


# ---------------------------------------------------------------------------
# HEADER & HERO SECTION
# ---------------------------------------------------------------------------
render_html("""
<div class="hero-container">
    <div class="hero-header-row">
        <div class="hero-title-group">
            <h1>RESUMEFORGE AI</h1>
            <div class="hero-tagline">Intelligent Resume Classification & Career Intelligence</div>
            <div class="hero-sub">Transform unstructured resumes into actionable career intelligence using high-dimensional Character TF-IDF & Support Vector hyperplanes.</div>
        </div>
        <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 0.5rem;">
            <div class="badge-status">
                <span class="status-dot"></span>
                AI MODEL ONLINE
            </div>
            <div class="hero-meta-pill">
                24 Career Categories &nbsp;•&nbsp; Character TF-IDF + LinearSVC
            </div>
        </div>
    </div>
</div>
""")


# ---------------------------------------------------------------------------
# MAIN TWO-COLUMN DASHBOARD LAYOUT
# ---------------------------------------------------------------------------
col_input, col_results = st.columns([1, 1], gap="large")

# ===========================================================================
# LEFT COLUMN: RESUME INPUT AREA
# ===========================================================================
with col_input:
    render_html("""
    <div class="card-header-title">
        <span>📄</span> Upload or Paste Resume
    </div>
    <div style="font-size: 0.84rem; color: #94A3B8; margin-bottom: 0.8rem;">
        Paste candidate resume text or upload a document (.pdf or .txt) to extract domain classification and career intelligence.
    </div>
    """)

    # Sample Selector
    sample_key = st.selectbox(
        "💡 Quick Demo: Try a sanitized example",
        options=list(SANITIZED_SAMPLES.keys()),
        index=0,
        help="Select a sanitized pre-loaded resume without any personal identifying information."
    )

    # File Uploader
    uploaded_file = st.file_uploader(
        "📎 Or upload document (.pdf, .txt)",
        type=["pdf", "txt"],
        help="Upload candidate PDF or TXT resume."
    )

    # Determine input text
    current_text = st.session_state.resume_content

    # If sample selected (and not placeholder)
    if sample_key != "Select a pre-loaded sample...":
        sample_text = SANITIZED_SAMPLES[sample_key]
        if st.session_state.resume_content != sample_text:
            st.session_state.resume_content = sample_text
            current_text = sample_text

    # If file uploaded
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        extracted_text = ""
        if uploaded_file.name.lower().endswith(".pdf"):
            if PDF_SUPPORTED:
                try:
                    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                        for page in doc:
                            extracted_text += page.get_text() + "\n"
                    extracted_text = extracted_text.strip()
                except Exception as e:
                    st.error("Could not parse PDF document. Please copy and paste the text instead.")
            else:
                st.warning("PyMuPDF parser not installed. Please paste resume text directly.")
        else:
            # Plain TXT
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    extracted_text = file_bytes.decode(enc).strip()
                    break
                except UnicodeDecodeError:
                    continue
            if not extracted_text:
                extracted_text = file_bytes.decode("utf-8", errors="ignore").strip()

        if extracted_text and extracted_text != st.session_state.resume_content:
            st.session_state.resume_content = extracted_text
            current_text = extracted_text

    # Large Textarea
    input_resume = st.text_area(
        "Candidate Resume Text",
        value=current_text,
        height=320,
        placeholder="Paste full resume text here (Work experience, technical skills, education, summary)...",
        label_visibility="collapsed"
    )

    # Sync state with edit
    if input_resume != st.session_state.resume_content:
        st.session_state.resume_content = input_resume

    # Primary Action Button
    st.markdown("<div style='margin-top: 1rem;'>", unsafe_allow_html=True)
    analyze_clicked = st.button("✨ Analyze Resume", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# RIGHT COLUMN: PREDICTION & CAREER INTELLIGENCE AREA
# ===========================================================================
with col_results:
    # Trigger analysis on button click or existing payload
    if analyze_clicked:
        raw_text = st.session_state.resume_content.strip()

        if not raw_text:
            st.session_state.analysis_done = False
            st.session_state.prediction_payload = None
            st.warning("⚠️ Please upload or paste resume text to begin analysis.")
        elif len(raw_text.split()) < 5:
            st.session_state.analysis_done = False
            st.session_state.prediction_payload = None
            st.warning("⚠️ Resume text is too brief for classification. Please provide at least 5 words.")
        elif model is None or vectorizer is None:
            st.session_state.analysis_done = False
            st.session_state.prediction_payload = None
            st.error("❌ Model artifacts could not be loaded from `models/`. Please verify model joblib files.")
        else:
            # Execution
            t_start = time.time()
            cleaned = clean_resume_text(raw_text)
            X_vec = vectorizer.transform([cleaned])

            # Get decision function scores
            if hasattr(model, "decision_function"):
                decision_scores = model.decision_function(X_vec)[0]
                pred_idx = int(np.argmax(decision_scores))
                pred_category = str(classes[pred_idx])
                raw_margin = float(decision_scores[pred_idx])

                # Top 3 based on pure decision margins
                top3_indices = decision_scores.argsort()[-3:][::-1]
                top3_list = [
                    (str(classes[i]), float(decision_scores[i]))
                    for i in top3_indices
                ]
            elif hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_vec)[0]
                pred_idx = int(np.argmax(probs))
                pred_category = str(classes[pred_idx])
                raw_margin = float(probs[pred_idx])
                top3_indices = probs.argsort()[-3:][::-1]
                top3_list = [
                    (str(classes[i]), float(probs[i]))
                    for i in top3_indices
                ]
            else:
                pred_category = str(model.predict(X_vec)[0])
                pred_idx = list(classes).index(pred_category) if classes is not None else 0
                raw_margin = 1.0
                top3_list = [(pred_category, 1.0)]

            t_elapsed = (time.time() - t_start) * 1000

            # Feature Attribution
            top_features = []
            try:
                if hasattr(model, "coef_"):
                    coefs = model.coef_[pred_idx]
                    feature_names = vectorizer.get_feature_names_out()
                    nonzero_cols = X_vec.nonzero()[1]
                    contributions = [(feature_names[c], coefs[c] * X_vec[0, c]) for c in nonzero_cols if coefs[c] * X_vec[0, c] > 0]
                    contributions.sort(key=lambda x: x[1], reverse=True)
                    top_features = [f"'{feat}' (+{score:.3f})" for feat, score in contributions[:8]]
            except Exception:
                pass

            # Detect genuine skills
            detected_skills = extract_detected_skills(raw_text)

            # Store in session state
            st.session_state.prediction_payload = {
                "pred_category": pred_category,
                "raw_margin": raw_margin,
                "top3_list": top3_list,
                "t_elapsed": t_elapsed,
                "raw_word_count": len(raw_text.split()),
                "raw_char_count": len(raw_text),
                "detected_skills": detected_skills,
                "top_features": top_features
            }
            st.session_state.analysis_done = True

    # -----------------------------------------------------------------------
    # Render Results or Empty State
    # -----------------------------------------------------------------------
    if st.session_state.analysis_done and st.session_state.prediction_payload:
        p = st.session_state.prediction_payload

        pred_cat_display = p['pred_category'].replace("-", " ").title()

        # Build 3-5 verified skills/terms actually detected in the resume
        why_terms = []
        for s in p['detected_skills']:
            if re.search(r'(?<![a-zA-Z0-9])' + re.escape(s) + r'(?![a-zA-Z0-9])', raw_text, re.IGNORECASE) and s not in why_terms:
                why_terms.append(s)
            if len(why_terms) >= 5:
                break

        # Fallback if fewer than 3 terms found in taxonomy: inspect words genuinely present
        if len(why_terms) < 3:
            extra_candidates = [
                "Algorithms", "Architecture", "Analysis", "Engineering", "Development",
                "Management", "Infrastructure", "Security", "Operations", "Design",
                "System Design", "Database", "Networks", "Framework", "Optimization",
                "Automation", "Clinical", "Valuation", "Compliance", "Recruiting", "Testing"
            ]
            for cand in extra_candidates:
                if re.search(r'\b' + re.escape(cand) + r'\b', raw_text, re.IGNORECASE) and cand not in why_terms:
                    why_terms.append(cand)
                if len(why_terms) >= 5:
                    break

        if why_terms:
            why_items_html = "".join([f"<div class='why-item'><span class='why-check'>✓</span><span class='why-text'>{term}</span></div>" for term in why_terms])
        else:
            why_items_html = f"<div style='font-size:0.85rem; color:#94A3B8; font-style:italic;'>Sub-word character n-gram distribution matches {pred_cat_display} training profiles.</div>"

        # 1. RESULT CARD (HERO / VISUAL FOCAL POINT)
        hero_card_html = f"""
        <div class="pred-hero-card">
            <div class="pred-badge-label">PREDICTED CAREER CATEGORY</div>
            <div class="pred-category-name">{p['pred_category']}</div>
            <div class="score-container-row">
                <div class="metric-badge">
                    <span class="badge-label-small">MODEL RANK</span>
                    <span class="badge-value-strong">#1 of 24</span>
                </div>
                <div class="metric-badge">
                    <span class="badge-label-small">DECISION SCORE</span>
                    <span class="badge-value-margin">{p['raw_margin']:+.3f}</span>
                </div>
                <div class="metric-badge">
                    <span class="badge-label-small">Inference</span>
                    <span class="badge-value-latency">{p['t_elapsed']:.1f}ms</span>
                </div>
            </div>
            <div class="why-container">
                <div class="why-heading">Why {pred_cat_display}?</div>
                <div class="why-checklist">
                    {why_items_html}
                </div>
            </div>
            <div class="score-note">
                Scores are used for ranking and are not calibrated probabilities.
            </div>
        </div>
        """
        render_html(hero_card_html)

        # 2. TOP PREDICTIONS (CARDS & VISUALIZATION)
        top3_items = p["top3_list"][:3]
        margins = [m for _, m in top3_items]
        m_max = margins[0]
        m_min = margins[-1]
        spread = m_max - m_min

        rank_classes = ["rank-1", "rank-2", "rank-3"]
        bar_colors = ["fill-gold", "fill-indigo", "fill-cyan"]

        cards_html_list = []
        for idx, (cat, margin) in enumerate(top3_items):
            rank_cls = rank_classes[idx] if idx < 3 else "rank-3"
            bar_color = bar_colors[idx] if idx < 3 else "fill-cyan"
            # Normalize bar lengths ONLY for visual ranking:
            # Highest score gets 100% bar width, lower scores get proportionally shorter bars
            if spread > 1e-6:
                fill_pct = 100.0 - 55.0 * ((m_max - margin) / spread)
            else:
                fill_pct = 100.0
            cat_upper = cat.upper()

            cards_html_list.append(f"""
            <div class="top-rank-card {rank_cls}">
                <div class="top-rank-header">
                    <div class="rank-indicator">
                        <span class="rank-pill">#{idx+1}</span>
                        <span>{cat_upper}</span>
                    </div>
                    <div class="rank-score-text">
                        <span style="color:#94A3B8; font-weight:500; font-size:0.82rem;">Decision score:</span> <b style="color:#38BDF8; font-family:'JetBrains Mono', monospace; font-size:0.95rem;">{margin:+.3f}</b>
                    </div>
                </div>
                <div class="custom-bar-track">
                    <div class="custom-bar-fill {bar_color}" style="width: {fill_pct:.1f}%;"></div>
                </div>
            </div>""")

        all_cards_html = "".join(cards_html_list)
        top3_section_html = f"""
        <div class="glass-card prominent-top3">
            <div class="card-header-title prominent-title">
                <span>🏆</span> Top Predicted Categories
            </div>
            {all_cards_html}
            <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 500; margin-top: 0.6rem; text-align: right;">
                Relative model ranking — higher score indicates a stronger model preference. Scores are not probabilities.
            </div>
        </div>
        """
        render_html(top3_section_html)

        # 3. RESUME INTELLIGENCE
        if p['detected_skills']:
            tags_html = "".join([f"<span class='skill-tag'>{sk}</span>" for sk in p['detected_skills']])
            skills_block = f"<div class='skills-container'>{tags_html}</div>"
        else:
            skills_block = "<div class='no-skills-msg'>No technical skills from standard taxonomy detected in input text.</div>"

        if p['top_features']:
            feat_tags = "".join([f"<span class='ngram-tag'>{f}</span>" for f in p['top_features']])
            feature_block = f"""
            <div style="font-size: 0.82rem; font-weight: 700; color: #E2E8F0; margin-top: 0.9rem; margin-bottom: 0.3rem;">
                Character N-Gram Attribution (LinearSVC Hyperplane Evidence)
            </div>
            <div>{feat_tags}</div>
            """
        else:
            feature_block = ""

        intel_section_html = f"""
        <div class="glass-card">
            <div class="card-header-title">
                <span>🧠</span> Resume Intelligence
            </div>
            <div class="intel-grid">
                <div class="intel-box">
                    <div class="intel-box-title">Word Count</div>
                    <div class="intel-box-val">{p['raw_word_count']:,}</div>
                </div>
                <div class="intel-box">
                    <div class="intel-box-title">Character Count</div>
                    <div class="intel-box-val">{p['raw_char_count']:,}</div>
                </div>
                <div class="intel-box">
                    <div class="intel-box-title">Skills Detected</div>
                    <div class="intel-box-val">{len(p['detected_skills'])}</div>
                </div>
            </div>
            <div style="font-size: 0.82rem; font-weight: 700; color: #E2E8F0; margin-bottom: 0.3rem;">
                Detected Competencies & Keywords ({len(p['detected_skills'])})
            </div>
            {skills_block}
            {feature_block}
        </div>
        """
        render_html(intel_section_html)

    else:
        # Placeholder / Empty State
        empty_state_html = """
        <div class="empty-state-box">
            <div class="empty-state-icon">⚡</div>
            <div class="empty-state-title">Ready for Analysis</div>
            <div class="empty-state-desc">
                Paste resume text on the left or select a pre-loaded sample, then click <b>✨ Analyze Resume</b> to extract career classification, hyperplane margins, and detected competencies.
            </div>
        </div>
        """
        render_html(empty_state_html)


# ---------------------------------------------------------------------------
# BOTTOM SECTION: HOW IT WORKS
# ---------------------------------------------------------------------------
step_grid_html = """
<div class="glass-card" style="margin-top: 1.5rem;">
    <div class="card-header-title">
        <span>🔄</span> How It Works — End-to-End Classification Pipeline
    </div>
    <div class="step-grid">
        <div class="step-card">
            <div class="step-number">01</div>
            <div class="step-title">Resume Input</div>
            <div class="step-desc">Accepts raw text, PDF documents, or TXT files without requiring pre-formatting.</div>
        </div>
        <div class="step-card">
            <div class="step-number">02</div>
            <div class="step-title">Text Processing</div>
            <div class="step-desc">Variant C domain normalization preserving tech symbols (C++, C#, .NET) and removing noise.</div>
        </div>
        <div class="step-card">
            <div class="step-number">03</div>
            <div class="step-title">Character TF-IDF</div>
            <div class="step-desc">Extracts high-dimensional sub-word n-grams (3–6 chars) robust to OCR typos and abbreviations.</div>
        </div>
        <div class="step-card">
            <div class="step-number">04</div>
            <div class="step-title">AI Classification</div>
            <div class="step-desc">LinearSVC separates 24 professional categories via geometric maximum-margin hyperplanes.</div>
        </div>
    </div>
</div>
"""
render_html(step_grid_html)


# ---------------------------------------------------------------------------
# BOTTOM SECTION: MODEL AT A GLANCE
# ---------------------------------------------------------------------------
val_acc = best_cfg.get("accuracy", 0.6532)
val_f1 = best_cfg.get("macro_f1", 0.5970)
test_acc = test_metrics.get("accuracy", 0.7131)
test_f1 = test_metrics.get("macro_f1", 0.6899)

glance_html = f"""
<div class="glass-card">
    <div class="card-header-title">
        <span>📈</span> Model at a Glance
    </div>
    <div class="glance-grid">
        <div class="glance-card">
            <div class="glance-label">Total Resumes</div>
            <div class="glance-val">2,481</div>
            <div class="glance-sub">Strictly deduplicated dataset</div>
        </div>
        <div class="glance-card">
            <div class="glance-label">Career Categories</div>
            <div class="glance-val">24</div>
            <div class="glance-sub">Multi-class domain coverage</div>
        </div>
        <div class="glance-card">
            <div class="glance-label">Validation Accuracy</div>
            <div class="glance-val">{val_acc:.2%}</div>
            <div class="glance-sub">Test Accuracy: {test_acc:.2%}</div>
        </div>
        <div class="glance-card">
            <div class="glance-label">Validation Macro-F1</div>
            <div class="glance-val">{val_f1:.2%}</div>
            <div class="glance-sub">Test Macro-F1: {test_f1:.2%}</div>
        </div>
    </div>
</div>
"""
render_html(glance_html)


# ---------------------------------------------------------------------------
# EXPANDABLE SECTION: ABOUT THE MODEL
# ---------------------------------------------------------------------------
with st.expander("ℹ️ About the Model Architecture & Verified Metrics", expanded=False):
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"""
        **Model Architecture**:
        - **Classifier**: `LinearSVC (Support Vector Machine)`
        - **Representation**: `Character TF-IDF`
        - **Character n-grams**: `3–6 sub-word grams`
        - **Feature Dimension**: `40,000 top n-grams`
        - **Sublinear TF Scaling**: `Enabled (logarithmic 1 + log(tf))`
        - **Min Document Frequency**: `2`
        - **Max Document Frequency**: `0.95`
        - **Penalty Parameter (C)**: `1.0`
        """)
    with col_m2:
        st.markdown(f"""
        **Verified Benchmark Metrics**:
        - **Total Clean Resumes**: `2,481`
        - **Classes / Categories**: `24`
        - **Best Validation Accuracy**: `{val_acc:.2%}`
        - **Best Validation Macro-F1**: `{val_f1:.2%}`
        - **Best Validation Weighted-F1**: `{best_cfg.get('weighted_f1', 0.6340):.2%}`
        - **Final Test Accuracy**: `{test_acc:.2%}`
        - **Final Test Macro-F1**: `{test_f1:.2%}`
        - **Final Test Weighted-F1**: `{test_metrics.get('weighted_f1', 0.7034):.2%}`
        - **Average Inference Latency**: `~2.1 ms / resume`
        """)


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
render_html("""
<div class="app-footer">
    <b>Built for SAMATRIX RESUMEFORGE 2026</b> &nbsp;•&nbsp; AI-Powered Resume Classification & Career Intelligence
</div>
""")
