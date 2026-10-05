"""
Streamlit Web Application for Resume Classification System.
SAMATRIX RESUMEFORGE 2026 Hackathon Compliant:
- Clean, focused interface
- Supports PDF file upload and raw-text input
- Real-time classification with Linear SVM / BiLSTM models
- Correctly labeled Decision Confidence / Scores
- Preprocessed text preview & Top discriminative keywords
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Add project root to path
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import ResumeClassificationPipeline
from src.pdf_extractor import extract_text_from_pdf
from src.preprocessing import preprocess_text

# Page Configuration
st.set_page_config(
    page_title="Resume Classification System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Pipeline Cache
@st.cache_resource
def load_pipeline():
    return ResumeClassificationPipeline()

pipeline = load_pipeline()

# Title & Description
st.title("📄 Resume Classification System")
st.markdown(
    """
    **SAMATRIX RESUMEFORGE 2026** — Multiclass Resume Category Classifier.  
    This system analyzes technical competencies, experience indicators, and domain-specific vocabulary 
    to classify resumes into **24 professional domains**.
    """
)

# Sidebar with Model & Architecture Information
with st.sidebar:
    st.header("⚙️ Model Architecture")
    st.markdown("""
    - **Classifier**: Linear Support Vector Machine (LinearSVC)
    - **Feature Representation**: Word & Bigram TF-IDF (10,000 max features)
    - **Alternative Model**: Bidirectional LSTM (PyTorch)
    - **Total Classes**: 24 Categories
    - **Class Balancing**: Balanced Class Weights
    """)
    st.divider()
    st.markdown("### 📊 Supported Domains")
    st.caption("Information Technology, HR, Finance, Advocate, Chef, Aviation, Banking, Engineering, Healthcare, Sales, Teacher, etc.")

# Main Input Section
input_mode = st.radio(
    "Choose Input Method:",
    ["Upload PDF Resume", "Paste Raw Resume Text"],
    horizontal=True
)

raw_text_content = ""
uploaded_file_name = None

if input_mode == "Upload PDF Resume":
    uploaded_file = st.file_uploader("Upload a Resume PDF file", type=["pdf"])
    if uploaded_file is not None:
        uploaded_file_name = uploaded_file.name
        with st.spinner("Extracting text from PDF..."):
            pdf_bytes = uploaded_file.read()
            success, extracted_text, err = extract_text_from_pdf(pdf_bytes)
            if success:
                raw_text_content = extracted_text
                st.success(f"Successfully extracted {len(raw_text_content)} characters from **{uploaded_file_name}**")
            else:
                st.error(f"Error parsing PDF: {err}")
else:
    raw_text_content = st.text_area(
        "Paste Resume Text Here:",
        height=250,
        placeholder="e.g. Lead Software Engineer with 8 years experience in Python, AWS, Docker, React, and PostgreSQL..."
    )

# Run Classification Button
if st.button("🚀 Classify Resume", type="primary"):
    if not raw_text_content or len(raw_text_content.strip()) == 0:
        st.warning("⚠️ Please provide resume text or upload a valid PDF document before classifying.")
    else:
        with st.spinner("Analyzing resume content and predicting category..."):
            result = pipeline.predict_text(raw_text_content)
            
        if result["success"]:
            st.divider()
            
            col1, col2, col3 = st.columns([1.5, 1, 1])
            
            with col1:
                st.subheader("🎯 Predicted Category")
                st.markdown(f"## :green[{result['predicted_category']}]")
                st.caption("Based on learned discriminative terminology and text patterns.")
                
            with col2:
                st.subheader(f"📊 {result['score_type']}")
                st.metric(
                    label=result['score_type'],
                    value=f"{result['confidence']:.4f}",
                    help="Decision score margin from Linear SVM hyperplanes."
                )
                
            with col3:
                st.subheader("📝 Token Statistics")
                st.metric(label="Extracted Word Tokens", value=result["token_count"])
                
            st.divider()
            
            # Top Alternative Candidate Categories
            st.subheader("🏆 Top Ranked Categories")
            top_df = pd.DataFrame(result["top_classes"])
            top_df.columns = ["Category", "Decision Score", "Score Metric"]
            st.dataframe(top_df, hide_index=True)
            
            # Discriminative Terms in this Resume
            if result.get("top_terms"):
                st.subheader("🔍 Top Weighted Discriminative Terms Found in Resume")
                term_cols = st.columns(min(4, len(result["top_terms"])))
                for idx, (term, score) in enumerate(result["top_terms"][:8]):
                    with term_cols[idx % len(term_cols)]:
                        st.badge(f"{term} ({score:.3f})")
                        
            # Preprocessed Text Preview
            with st.expander("🔎 View Normalized / Preprocessed Text Snippet"):
                st.text(result["preprocessed_snippet"])
                
        else:
            st.error(f"Classification Failed: {result.get('error')}")
