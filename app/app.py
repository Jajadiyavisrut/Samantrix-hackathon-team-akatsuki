"""
Streamlit Web Application: SAMATRIX RESUMEFORGE 2026
Production Demo for Intelligent Resume Domain Classification & Explainability.
"""

import os
import sys
import streamlit as st
import pandas as pd
import fitz  # PyMuPDF for high-fidelity PDF extraction

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from predict import ResumePredictor
from preprocessing import clean_resume_text

# Page Configuration
st.set_page_config(
    page_title="ResumeForge AI — Professional Category Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #e0f2fe;
        color: #0369a1;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .prob-bar {
        height: 8px;
        border-radius: 4px;
        background-color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Initializes and caches the model predictor."""
    return ResumePredictor(model_dir="models")


def extract_text_from_file(uploaded_file) -> str:
    """Extracts text from uploaded PDF or TXT files."""
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".pdf"):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            return text.strip()
        elif filename.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore").strip()
        else:
            st.error("Unsupported file format. Please upload a .pdf or .txt resume.")
            return ""
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""


def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/resume.png", width=120)
        st.title("ResumeForge 2026")
        st.caption("Autonomous NLP Classification Engine")
        st.markdown("---")
        st.markdown("### System Architecture")
        st.markdown("""
        - **Pipeline:** Word + Character Subword TF-IDF
        - **Classifier:** Calibrated Linear Classifier
        - **Target Categories:** 24 Disciplines
        - **Validation Protocol:** Zero-Leakage Stratification
        - **Explainability:** Exact Linear Feature Attribution
        """)
        st.markdown("---")
        st.markdown("### Benchmarks")
        st.info("**Leaderboard:** Word+Char Linear Model\n- Sub-millisecond latency\n- Balanced class weighting")

    # Header
    st.markdown('<div class="main-header">ResumeForge AI — Resume Category Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated candidate classification across 24 professional domains with exact feature-level explainability.</div>', unsafe_allow_html=True)

    predictor = load_predictor()

    # Input method selection
    input_mode = st.radio("Choose Input Method:", ["📄 Paste Resume Text", "📁 Upload Resume Document (.pdf, .txt)"], horizontal=True)

    resume_text = ""
    if input_mode == "📄 Paste Resume Text":
        resume_text = st.text_area(
            "Paste full candidate resume text here:",
            height=260,
            placeholder="e.g. Senior Software Engineer with 7+ years of experience in Python, C++, Docker, Kubernetes, microservices architecture...",
        )
    else:
        uploaded_file = st.file_uploader("Upload resume file (.pdf or .txt)", type=["pdf", "txt"])
        if uploaded_file is not None:
            with st.spinner("Extracting text from resume document..."):
                resume_text = extract_text_from_file(uploaded_file)
            if resume_text:
                st.success(f"Successfully extracted {len(resume_text.split())} words from {uploaded_file.name}")
                with st.expander("Preview Extracted Text"):
                    st.text(resume_text[:1200] + ("..." if len(resume_text) > 1200 else ""))

    # Classification Button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        classify_btn = st.button("🚀 Classify Resume", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("🔄 Reset", use_container_width=True)

    if clear_btn:
        st.rerun()

    if classify_btn:
        if not resume_text or not resume_text.strip():
            st.warning("Please provide resume text or upload a valid document to begin classification.")
            return

        with st.spinner("Executing preprocessing & inference..."):
            result = predictor.predict_resume(resume_text)

        if not result["success"]:
            st.error(f"Inference Warning: {result['error_message']}")
            return

        # Results Dashboard
        st.markdown("---")
        st.subheader("🎯 Classification Results")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.caption("PRIMARY PREDICTED DOMAIN")
            st.markdown(f"<h2 style='color:#1e40af; margin-top:0;'>{result['predicted_category']}</h2>", unsafe_allow_html=True)

            conf_pct = result["confidence"] * 100
            st.markdown(f"**Confidence:** `{conf_pct:.1f}%` ({result['confidence_type']})")
            st.progress(result["confidence"])

            st.markdown("---")
            st.markdown("**Top Alternative Predictions:**")
            for cat, prob in result["top_3_predictions"]:
                c_pct = prob * 100
                st.write(f"- **{cat}**: `{c_pct:.1f}%`")
            st.markdown('</div>', unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.caption("FEATURE-LEVEL EXPLAINABILITY (ATTRIBUTION)")
            st.markdown("Top discriminative tokens & subwords driving this prediction:")

            if result["contributing_features"]:
                feat_df = pd.DataFrame(result["contributing_features"])
                st.dataframe(
                    feat_df.rename(columns={"feature": "Token / N-Gram", "score": "Attribution Weight", "type": "Feature Type"}),
                    use_container_width=True,
                    height=210,
                )
                st.caption("Attribution weights represent exact mathematical contributions to the class decision margin.")
            else:
                st.write("Generic vocabulary distributed across multiple domain centroids.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Preprocessing Audit
        with st.expander("🔍 Preprocessing & Tokenization Audit"):
            st.write(f"- **Raw Word Count:** {len(resume_text.split())} words")
            st.write(f"- **Cleaned Normalized Words:** {result['cleaned_word_count']} words")
            st.write("- **Preserved Technical Tokens:** C++, C#, .NET, Python, SQL, Docker, etc.")
            st.write("- **Normalized Entities:** URLs, Email addresses, and Phone numbers")


if __name__ == "__main__":
    main()
