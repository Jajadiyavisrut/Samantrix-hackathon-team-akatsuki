"""
Data Validation and Quality Audit module.
Performs thorough checks for missing values, duplicates, empty text, label anomalies,
and outputs a structured audit report.
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

try:
    from src.config import RAW_CSV_PATH, PROCESSED_DATA_DIR, REPORTS_DIR, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    from src.data_loader import load_raw_csv
except ImportError:
    from config import RAW_CSV_PATH, PROCESSED_DATA_DIR, REPORTS_DIR, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    from data_loader import load_raw_csv


def run_data_quality_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform a complete data quality audit on raw dataframe.
    """
    audit = {}
    
    total_records = len(df)
    audit["total_records"] = total_records
    audit["num_classes"] = int(df["Category"].nunique())
    audit["classes"] = sorted(df["Category"].unique().tolist())
    audit["class_distribution"] = df["Category"].value_counts().to_dict()
    
    # Missing values
    audit["missing_values"] = {col: int(df[col].isnull().sum()) for col in df.columns}
    audit["missing_Resume_str_count"] = int(df["Resume_str"].isnull().sum())
    audit["missing_Category_count"] = int(df["Category"].isnull().sum())
    
    # Text length stats
    cleaned_lens = df["Resume_str"].fillna("").apply(lambda x: len(str(x).strip()))
    word_counts = df["Resume_str"].fillna("").apply(lambda x: len(str(x).split()))
    
    audit["text_char_length"] = {
        "min": int(cleaned_lens.min()),
        "max": int(cleaned_lens.max()),
        "mean": float(round(cleaned_lens.mean(), 2)),
        "median": float(round(cleaned_lens.median(), 2)),
        "std": float(round(cleaned_lens.std(), 2))
    }
    
    audit["text_word_count"] = {
        "min": int(word_counts.min()),
        "max": int(word_counts.max()),
        "mean": float(round(word_counts.mean(), 2)),
        "median": float(round(word_counts.median(), 2)),
        "std": float(round(word_counts.std(), 2))
    }
    
    # Empty and very short resumes (< 10 words)
    empty_resumes = df[cleaned_lens == 0]
    audit["empty_resume_count"] = len(empty_resumes)
    audit["empty_resume_ids"] = empty_resumes["ID"].tolist()
    
    short_resumes = df[(word_counts < 10) & (cleaned_lens > 0)]
    audit["short_resume_count"] = len(short_resumes)
    audit["short_resume_ids"] = short_resumes["ID"].tolist()
    
    # Duplicates check
    audit["duplicate_id_count"] = int(df["ID"].duplicated().sum())
    audit["duplicate_exact_text_count"] = int(df["Resume_str"].duplicated().sum())
    
    dup_texts_all = df[df.duplicated(subset=["Resume_str"], keep=False)]
    audit["duplicate_records_total"] = len(dup_texts_all)
    
    # Duplicate text across different labels (leakage / label ambiguity check)
    dup_label_conflicts = df.groupby("Resume_str")["Category"].nunique()
    conflicting_texts = dup_label_conflicts[dup_label_conflicts > 1]
    audit["duplicate_texts_with_conflicting_labels"] = int(len(conflicting_texts))
    
    return audit


def clean_and_split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cleans raw data with explicitly justified rules and performs stratified 70/15/15 train/val/test split.
    
    Cleaning rules justified:
    1. Remove records with 0 characters / 0 words in Resume_str (empty document cannot be learned or classified).
    2. Deduplicate exact duplicate text (keeps first occurrence) to prevent test-set leakage.
    
    Returns:
        cleaned_df, train_df, val_df, test_df
    """
    initial_count = len(df)
    
    # 1. Remove empty resumes (e.g. ID 12632728 which has 0 chars)
    valid_mask = df["Resume_str"].fillna("").apply(lambda x: len(str(x).strip()) > 0)
    df_cleaned = df[valid_mask].copy()
    
    # 2. Remove duplicate text to prevent train/test contamination
    df_cleaned = df_cleaned.drop_duplicates(subset=["Resume_str"], keep="first").copy()
    
    # Stratified Train/Val/Test Split
    # Step 1: 70% train, 30% temp (val + test)
    from sklearn.model_selection import train_test_split
    
    train_df, temp_df = train_test_split(
        df_cleaned,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        stratify=df_cleaned["Category"]
    )
    
    # Step 2: Split temp into 50/50 (15% val, 15% test of original)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp_df["Category"]
    )
    
    return df_cleaned, train_df, val_df, test_df


def save_validation_artifacts():
    """
    Execute validation pipeline and save audit report and split CSVs.
    """
    df = load_raw_csv()
    audit = run_data_quality_audit(df)
    
    # Save audit report
    audit_path = REPORTS_DIR / "data_quality_report.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    
    # Save cleaned and split datasets
    cleaned_df, train_df, val_df, test_df = clean_and_split_data(df)
    
    cleaned_df.to_csv(PROCESSED_DATA_DIR / "cleaned_resumes.csv", index=False)
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    
    return audit, len(cleaned_df), len(train_df), len(val_df), len(test_df)


if __name__ == "__main__":
    audit, cleaned_len, train_len, val_len, test_len = save_validation_artifacts()
    print("Data Validation Complete:")
    print(f"  Raw count: {audit['total_records']}")
    print(f"  Cleaned count: {cleaned_len}")
    print(f"  Train: {train_len}, Val: {val_len}, Test: {test_len}")
    print(f"  Audit report saved to: reports/data_quality_report.json")
