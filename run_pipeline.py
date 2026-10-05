"""
End-to-End Execution Pipeline for Resume Classification System.
Executes all stages in rigorous chronological order:
1. Data Gathering & Validation
2. Exploratory Data Analysis (EDA) & Visualizations
3. Reproducible Text Preprocessing & Splitting
4. TF-IDF & Deep Learning Feature Engineering
5. Classical Model Training (NB, Logistic Regression, Linear SVM)
6. Neural Deep Learning Model Training (BiLSTM with learned embeddings)
7. Final Untouched Test Set Evaluation & Comparison
8. Comprehensive Error Analysis
9. Model Serialization & Verification
"""
import os
import sys
import time
import pandas as pd
from pathlib import Path

# Add project root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.config import (
    RAW_CSV_PATH, PROCESSED_DATA_DIR, FIGURES_DIR, METRICS_DIR,
    ERROR_ANALYSIS_DIR, MODELS_DIR
)
from src.data_loader import inspect_dataset_sources, load_raw_csv
from src.data_validation import save_validation_artifacts, run_data_quality_audit
from src.eda import run_full_eda
from src.train_ml import train_classical_models
from src.train_dl import train_neural_model
from src.evaluate import evaluate_all_models_on_test
from src.error_analysis import analyze_misclassifications
from src.inference import ResumeClassificationPipeline


def run_pipeline():
    print("=" * 70)
    print("SAMATRIX RESUMEFORGE 2026 — RESUME CLASSIFICATION PIPELINE")
    print("=" * 70)
    
    start_total_time = time.time()
    
    # STAGE 1 & 2: Data Gathering & Source Inspection
    print("\n[STAGE 1/8] Inspecting Dataset Sources...")
    source_info = inspect_dataset_sources()
    print(f"  Raw CSV Shape: {source_info['csv_shape']}")
    print(f"  Categories detected: {source_info['csv_num_classes']}")
    
    # STAGE 3: Data Validation, Quality Check & Splitting
    print("\n[STAGE 2/8] Executing Data Quality Audit & Clean Splitting...")
    audit, cleaned_len, train_len, val_len, test_len = save_validation_artifacts()
    print(f"  Raw records: {audit['total_records']} -> Cleaned records: {cleaned_len}")
    print(f"  Split counts: Train={train_len} (70%), Val={val_len} (15%), Test={test_len} (15%)")
    print(f"  Empty documents removed: {audit['empty_resume_count']}, Duplicates removed: {audit['duplicate_exact_text_count']}")
    
    # STAGE 4: Full EDA & Visualizations
    print("\n[STAGE 3/8] Generating Comprehensive EDA Figures & Reports...")
    eda_summary = run_full_eda()
    print(f"  All {eda_summary['figures_generated']} EDA visualizations generated in reports/figures/")
    
    # Load processed splits
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    
    # STAGE 5: Classical ML Models Training
    print("\n[STAGE 4/8] Training Classical ML Baselines (Naive Bayes, Logistic Regression, Linear SVM)...")
    classical_results, best_classical_clf = train_classical_models(train_df, val_df)
    
    # STAGE 6: Deep Learning Model Training
    print("\n[STAGE 5/8] Training Neural Deep Learning Model (BiLSTM Text Classifier)...")
    neural_model, neural_metrics = train_neural_model(train_df, val_df)
    print(f"  BiLSTM Best Val Macro-F1: {neural_metrics.get('val_macro_f1', 0.0):.4f}")
    
    # STAGE 7: Final Untouched Test Set Evaluation
    print("\n[STAGE 6/8] Evaluating ALL Models on Untouched Test Set...")
    comp_df, eval_summary, best_model_name = evaluate_all_models_on_test(test_df)
    print("\n" + "-" * 70)
    print("FINAL TEST SET PERFORMANCE COMPARISON:")
    print("-" * 70)
    print(comp_df.to_string(index=False))
    print("-" * 70)
    print(f"Selected Final Model: {best_model_name}")
    
    # STAGE 8: Error Analysis on Selected Model
    print("\n[STAGE 7/8] Performing Diagnostic Error Analysis on Test Set...")
    errors_df, error_summary = analyze_misclassifications(test_df, split_name="test")
    print(f"  Test Misclassifications: {error_summary['total_errors']}/{error_summary['total_evaluated']} ({error_summary['error_rate']*100:.2f}%)")
    print("  Root Cause Diagnostic Breakdown:")
    for cause, count in error_summary["root_cause_distribution"].items():
        print(f"    * {cause}: {count}")
        
    # STAGE 9: Inference Pipeline Verification
    print("\n[STAGE 8/8] Testing Production Inference Pipeline...")
    pipeline = ResumeClassificationPipeline()
    
    sample_resumes = [
        ("Software Engineering", "Lead Python Developer with 8 years building cloud native APIs using FastAPI, Docker, Kubernetes, PostgreSQL, and AWS."),
        ("Culinary / Chef", "Executive Head Chef with expertise in French fine dining cuisine, pastry, banquet planning, kitchen staff management and menu curation."),
        ("Aviation", "Commercial Airline Captain and FAA certified flight instructor with 6500 flight hours across Boeing 737 and Airbus A320 aircraft."),
        ("Legal / Advocate", "Senior Litigation Associate Attorney specializing in corporate law, contract dispute arbitration, civil trials, and court appearances."),
        ("Healthcare", "Registered Nurse (RN) with 10 years experience in emergency room triage, patient vital monitoring, medication administration and trauma care.")
    ]
    
    print("\nVerifying Predictions on Test Samples:")
    for label_domain, sample_text in sample_resumes:
        pred_res = pipeline.predict_text(sample_text)
        print(f"  [{label_domain}] -> Predicted: {pred_res['predicted_category']} | {pred_res['score_type']}: {pred_res['confidence']}")
        
    elapsed = time.time() - start_total_time
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f}s!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
